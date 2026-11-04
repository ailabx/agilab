"""
极简再平衡回测：支持 N 个资产按目标权重周期性再平衡。

示例：
    python 05_rebalance.py --weights='050002.OF.csv:0.6,000218.OF.csv:0.4' --freq=M
"""
import argparse
import os

import pandas as pd

from utils import calc_metrics, plot_nav_and_weights


def load_fund_name_map(data_dir: str) -> dict:
    """从 basic.csv 读取基金名称映射，{ts_code: name}。"""
    basic_path = os.path.join(data_dir, "basic.csv")
    if not os.path.exists(basic_path):
        return {}
    df = pd.read_csv(basic_path, usecols=["ts_code", "name"])
    return dict(zip(df["ts_code"], df["name"]))


# 默认数据与输出目录：脚本所在目录下的 data/
DEFAULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="极简再平衡回测")
    parser.add_argument(
        "--weights",
        type=str,
        default="007466.OF.csv:0.7,110017.OF.csv:0.3",
        help="逗号分隔的 '文件名:权重' 列表，权重之和需为 1",
    )
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DIR, help="CSV 数据目录")
    parser.add_argument("--outdir", type=str, default=DEFAULT_DIR, help="输出目录")
    parser.add_argument("--freq", type=str, default="M", choices=["M", "Q", "Y"], help="再平衡频率：M=月，Q=季，Y=年")
    parser.add_argument("--fee", type=float, default=0.001, help="交易成本，默认 0.1%")
    return parser.parse_args()


def load_asset(path: str) -> pd.DataFrame:
    """读取单个资产的 CSV，返回 date、close 两列。"""
    df = pd.read_csv(path, parse_dates=["date"], usecols=["date", "close"])
    return df.sort_values("date").drop_duplicates("date")


def parse_weights(weight_str: str) -> dict:
    """解析 'a.csv:0.6,b.csv:0.4' 为 {文件名: 权重}。"""
    weights = {}
    for item in weight_str.split(","):
        file, w = item.strip().split(":")
        weights[file.strip()] = float(w.strip())
    return weights


def get_rebalance_dates(prices: pd.DataFrame, freq: str) -> set:
    """按频率取每个周期最后一个交易日，用于触发再平衡。"""
    period = prices.index.to_period(freq)
    return set(prices.index[~period.duplicated(keep="last")])


def backtest(prices: pd.DataFrame, target_weights: dict, rebalance_dates: set, fee: float):
    """
    极简回测主循环：遍历每个交易日，按日更新持仓市值，
    触发再平衡时扣除手续费并调仓至目标权重。
    """
    nav = pd.Series(index=prices.index, dtype=float)
    weights = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)

    # 初始化仓位：1 万元组合按目标权重买入各资产
    initial_capital = 10000.0
    positions = {asset: initial_capital * target_weights[asset] / prices.iloc[0][asset]
                 for asset in prices.columns}

    for date in prices.index:
        # 当前各资产市值
        market_values = {asset: positions[asset] * prices.loc[date, asset] for asset in prices.columns}
        total = sum(market_values.values())

        if date in rebalance_dates:
            # 计算目标市值
            target_values = {asset: total * target_weights[asset] for asset in prices.columns}
            # 成交额 ≈ 偏离市值绝对值之和 / 2（买卖各一次）
            turnover = sum(abs(market_values[asset] - target_values[asset]) for asset in prices.columns) / 2
            # 扣除交易成本
            total -= turnover * 2 * fee
            # 按目标权重重新买入
            positions = {asset: total * target_weights[asset] / prices.loc[date, asset]
                         for asset in prices.columns}
            market_values = {asset: positions[asset] * prices.loc[date, asset] for asset in prices.columns}
            total = sum(market_values.values())

        # 记录当日净值与权重
        nav.loc[date] = total
        for asset in prices.columns:
            weights.loc[date, asset] = market_values[asset] / total

    return nav, weights


def main():
    args = parse_args()

    # 解析目标权重
    target_weights = parse_weights(args.weights)

    # 合并所有资产的价格序列
    prices = None
    for file, w in target_weights.items():
        df = load_asset(os.path.join(args.data_dir, file)).rename(columns={"close": file})
        if prices is None:
            prices = df
        else:
            prices = pd.merge(prices, df, on="date", how="inner")

    if prices is None or prices.empty:
        raise ValueError("合并后的价格序列为空")

    prices = prices.set_index("date").sort_index()

    # 运行回测
    rebalance_dates = get_rebalance_dates(prices, args.freq)
    nav, weights = backtest(prices, target_weights, rebalance_dates, args.fee)
    metrics = calc_metrics(nav)

    # 输出指标
    weight_desc = ",".join(f"{k}:{v:.0%}" for k, v in target_weights.items())
    print(f"=== 再平衡回测 ({weight_desc}) ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.2%}" if k not in ("sharpe", "calmar") else f"{k}: {v:.2f}")

    # 保存结果
    os.makedirs(args.outdir, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(os.path.join(args.outdir, "rebalance_metrics.csv"),
                                   index=False, encoding="utf-8-sig")

    # 构建基金名称映射（文件名 -> 名称）
    fund_name_map = load_fund_name_map(args.data_dir)
    name_map = {file: fund_name_map.get(file.replace(".csv", ""), file) for file in target_weights.keys()}
    plot_path = plot_nav_and_weights(prices, nav, weights, args.outdir,
                                     title=f"再平衡回测 ({weight_desc})",
                                     name_map=name_map)
    print(f"图表已保存: {plot_path}")


if __name__ == "__main__":
    main()
