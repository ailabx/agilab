"""
通用回测引擎：接收价格矩阵和每日目标权重矩阵，执行调仓并返回净值、权重。

价格矩阵 prices: DataFrame, index=date, columns=assets
目标权重矩阵 weights_target: DataFrame, index=date, columns=assets, 每行之和 <= 1
空仓用权重 0 表示。
"""
import pandas as pd


def backtest(prices: pd.DataFrame,
             weights_target: pd.DataFrame,
             fee: float,
             *,
             rebalance_dates=None):
    """
    极简回测：遍历每个交易日，根据目标权重和调仓信号执行交易。

    参数：
        prices: DataFrame, index=date, columns=assets，收盘价
        weights_target: DataFrame, index=date, columns=assets，每日目标权重
        fee: float, 交易成本（双边）
        rebalance_dates: set/Timestamp list，实际执行调仓的日期；
                         默认为 None，表示每天都按 weights_target 调仓

    返回：
        nav: Series, 每日组合净值
        weights: DataFrame, 每日实际持仓权重
    """
    assets = prices.columns
    nav = pd.Series(index=prices.index, dtype=float)
    weights = pd.DataFrame(index=prices.index, columns=assets, dtype=float)

    initial_capital = 10000.0
    cash = initial_capital
    positions = {asset: 0.0 for asset in assets}

    for i, date in enumerate(prices.index):
        price_today = prices.loc[date]

        # 当前市值（按当日收盘价更新）
        market_values = {asset: positions[asset] * price_today[asset] for asset in assets}
        total = cash + sum(market_values.values())

        # 目标权重
        target = weights_target.loc[date]

        # 只在调仓日执行调仓
        if rebalance_dates is None or date in rebalance_dates:
            target_values = {asset: total * target[asset] for asset in assets}
            turnover = sum(abs(market_values[asset] - target_values[asset]) for asset in assets) / 2
            total -= turnover * 2 * fee

            positions = {asset: total * target[asset] / price_today[asset] for asset in assets}
            invested = sum(positions[asset] * price_today[asset] for asset in assets)
            cash = total - invested

            market_values = {asset: positions[asset] * price_today[asset] for asset in assets}
            total = cash + sum(market_values.values())

        nav.loc[date] = total
        for asset in assets:
            weights.loc[date, asset] = market_values[asset] / total if total > 0 else 0.0

    return nav, weights
