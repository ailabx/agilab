"""
克隆 alphalab 引擎细节，与自研引擎对比

关键差异：
- 权重只打到 98% (WeightEqually: w = 0.98 / n)
- 整数股约束 (volume = diff_value // price)
- SelectTopK buffer 机制 (X=1, strict=True)
- 手续费按买卖各扣一次，但在 order_target_percent 层面计算
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

from utils import calc_metrics


DEFAULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

DEFAULT_SYMBOLS = [
    "510300.SH.csv", "588000.SH.csv", "510880.SH.csv", "159980.SZ.csv", "159985.SZ.csv",
    "513100.SH.csv", "513520.SH.csv", "513500.SH.csv", "513030.SH.csv", "501018.SH.csv",
    "513180.SH.csv", "159915.SZ.csv", "512890.SH.csv", "511090.SH.csv", "510180.SH.csv",
]


def parse_args():
    parser = argparse.ArgumentParser(description="克隆 alphalab 引擎细节")
    parser.add_argument("--symbols", type=str, default=",".join(DEFAULT_SYMBOLS), help="标的 CSV 文件名")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DIR, help="CSV 数据目录")
    parser.add_argument("--outdir", type=str, default=DEFAULT_DIR, help="输出目录")
    parser.add_argument("--capital", type=float, default=1000000.0, help="初始资金")
    parser.add_argument("--period", type=int, default=20, help="trend_score2 周期")
    parser.add_argument("--top-k", type=int, default=4, help="Top K")
    parser.add_argument("--max-weight", type=float, default=0.25, help="单标的上限")
    parser.add_argument("--fee", type=float, default=0.0001, help="交易成本")
    parser.add_argument("--ann-factor", type=int, default=252, help="年化因子")
    parser.add_argument("--risk-free", type=float, default=0.02, help="无风险利率")
    parser.add_argument("--start-date", type=str, default="2020-01-01", help="开始日期")
    parser.add_argument("--end-date", type=str, default="", help="结束日期")
    parser.add_argument("--buffer", type=int, default=1, help="SelectTopK buffer")
    parser.add_argument("--use-98", action="store_true", default=False, help="使用 98% 仓位")
    parser.add_argument("--integer-lot", action="store_true", default=False, help="整数股约束")
    return parser.parse_args()


def load_ohlc(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").drop_duplicates("date").set_index("date")
    return df


def calc_trend_score2(prices: pd.DataFrame, period: int) -> pd.DataFrame:
    x = np.arange(period, dtype=float)
    w = np.linspace(1.0, 2.0, period) ** 2
    w_sum = w.sum()
    x_bar = (w * x).sum() / w_sum
    dx = x - x_bar
    variance_x = (w * dx ** 2).sum()

    def _score(window: np.ndarray) -> float:
        y = np.log(window)
        y_bar = (w * y).sum() / w_sum
        slope = (w * dx * (y - y_bar)).sum() / variance_x
        intercept = y_bar - slope * x_bar
        y_pred = slope * np.arange(period) + intercept
        ss_res = (w * (y - y_pred) ** 2).sum()
        ss_tot = (w * (y - y_bar) ** 2).sum()
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        r_squared = min(max(r_squared, 0.0), 1.0)
        annualized_return = np.exp(slope * 250) - 1.0
        return annualized_return * r_squared

    return pd.DataFrame(
        {col: prices[col].rolling(window=period).apply(_score, raw=True) for col in prices.columns},
        index=prices.index,
    )


def cs_zscore(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.mean(axis=1).values[:, None]) / df.std(axis=1).values[:, None]


class Portfolio:
    def __init__(self, capital, fee):
        self.cash = capital
        self.fee = fee
        self.positions = {}  # sym -> volume
        self.total_value = capital

    def get_holding_value(self):
        return sum(self.positions[sym]["close"] * self.positions[sym]["volume"] for sym in self.positions)

    def get_portfolio_value(self):
        return self.cash + self.get_holding_value()

    def get_symbol_holding_percent(self, sym):
        if sym not in self.positions:
            return 0.0
        return self.positions[sym]["close"] * self.positions[sym]["volume"] / self.get_portfolio_value()

    def update_close(self, close):
        for sym in self.positions:
            self.positions[sym]["close"] = close[sym]

    def order_target_percent(self, sym, target_percent, price):
        """alphalab 风格的 order_target_percent"""
        target_percent = max(0.0, min(1.0, target_percent))
        current_percent = self.get_symbol_holding_percent(sym)
        diff = target_percent - current_percent

        if abs(diff) < 1e-6:
            return

        diff_value = self.get_portfolio_value() * diff
        volume = abs(diff_value) // price

        if diff > 0:
            # 买入
            cost = volume * price
            commission = cost * self.fee
            if self.cash < cost + commission:
                if self.cash <= 0:
                    return
                affordable = self.cash / (1 + self.fee)
                volume = int(affordable // price)
                if volume <= 0:
                    return
                cost = volume * price
                commission = cost * self.fee
            self.cash -= cost + commission
            if sym in self.positions:
                old_cost = self.positions[sym]["volume"] * self.positions[sym]["close"]
                new_cost = volume * price
                total_volume = self.positions[sym]["volume"] + volume
                self.positions[sym]["close"] = (old_cost + new_cost) / total_volume if total_volume > 0 else price
                self.positions[sym]["volume"] = total_volume
            else:
                self.positions[sym] = {"close": price, "volume": volume}
        else:
            # 卖出
            if sym not in self.positions:
                return
            sell_volume = min(volume, self.positions[sym]["volume"])
            if sell_volume <= 0:
                return
            value = sell_volume * price
            commission = value * self.fee
            self.cash += value - commission
            self.positions[sym]["volume"] -= sell_volume
            if self.positions[sym]["volume"] <= 0:
                del self.positions[sym]


def select_top_k_with_buffer(score_row, holding, K, X, close, roc1):
    """alphalab 风格的 SelectTopK with buffer"""
    row = score_row.dropna()
    row = row[roc1 >= -0.05]
    row = row.dropna()

    if len(row) == 0:
        return []

    ranked = row.sort_values(ascending=False)
    top = ranked.head(K).index.tolist()
    buffer = ranked.iloc[K:K+X].index.tolist() if X > 0 else []

    if not buffer:
        return top

    # buffer 里只保留已持仓的
    keep = [s for s in buffer if s in holding]

    if not keep:
        return top

    # strict=True：维持 K 只，淘汰排名最末的新入选者
    # 简化处理：如果 top 里有不在 holding 的，且 keep 里有，就淘汰 top 中排名最末的非持仓
    top_set = set(top)
    non_holding_in_top = [s for s in reversed(top) if s not in holding]

    for s in non_holding_in_top:
        if len(top) <= K:
            break
        top.remove(s)

    # 如果还不够 K，从 keep 里补（但 keep 本来就是 holding，应该保留）
    return top + keep


def backtest_alphalab_clone(ohlc, score, roc1, args):
    close = ohlc["close"]
    dates = close.index
    symbols = close.columns.tolist()

    portfolio = Portfolio(args.capital, args.fee)

    nav = pd.Series(index=dates, dtype=float)
    weights = pd.DataFrame(0.0, index=dates, columns=symbols)
    hold_count = pd.Series(index=dates, dtype=int)
    trades = []

    for i, date in enumerate(dates):
        today_close = close.loc[date]
        portfolio.update_close(today_close)

        # 选股
        holding = set(portfolio.positions.keys())
        targets = select_top_k_with_buffer(score.loc[date], holding, args.top_k, args.buffer, today_close, roc1.loc[date])

        # 权重：等权，98% 或 100%
        if targets:
            if args.use_98:
                w = round(0.98 / len(targets), 3)
            else:
                w = min(1.0 / len(targets), args.max_weight)
        else:
            w = 0.0

        # 调仓
        target_weights = {sym: w for sym in targets}

        # 平掉不在目标中的持仓
        for sym in list(portfolio.positions.keys()):
            if sym not in target_weights:
                portfolio.order_target_percent(sym, 0.0, today_close[sym])
                trades.append((date, sym, "sell"))

        # 调整目标权重
        for sym in targets:
            portfolio.order_target_percent(sym, target_weights[sym], today_close[sym])
            trades.append((date, sym, "buy"))

        # 记录
        nav.loc[date] = portfolio.get_portfolio_value()
        for sym in symbols:
            if sym in portfolio.positions:
                weights.loc[date, sym] = portfolio.get_symbol_holding_percent(sym)
        hold_count.loc[date] = len(portfolio.positions)

    return nav, weights, hold_count, trades


def main():
    args = parse_args()
    start_time = time.time()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    frames = {col: [] for col in ["open", "high", "low", "close", "vol"]}
    valid_symbols = []
    for s in symbols:
        path = os.path.join(args.data_dir, s)
        if not os.path.exists(path):
            print(f"警告：文件不存在 {path}")
            continue
        df = load_ohlc(path)
        valid_symbols.append(s)
        for col in frames:
            if col in df.columns:
                frames[col].append(df[col].rename(s))
            else:
                frames[col].append(pd.Series(np.nan, index=df.index).rename(s))

    ohlc = {}
    for col in frames:
        ohlc[col] = pd.concat(frames[col], axis=1, sort=True).dropna()

    close = ohlc["close"]
    if args.start_date:
        close = close[close.index >= pd.to_datetime(args.start_date)]
    if args.end_date:
        close = close[close.index <= pd.to_datetime(args.end_date)]

    for col in ohlc:
        ohlc[col] = ohlc[col].loc[close.index]

    print(f"数据区间: {close.index[0].date()} ~ {close.index[-1].date()}")

    score = calc_trend_score2(close, args.period)
    zscore = cs_zscore(score)
    roc1 = close.pct_change(1)

    nav, weights, hold_count, trades = backtest_alphalab_clone(ohlc, zscore, roc1, args)
    metrics = calc_metrics(nav, ann_factor=args.ann_factor, risk_free=args.risk_free)

    print(f"=== 克隆 alphalab 引擎 ===")
    print(f"use_98={args.use_98}, integer_lot={args.integer_lot}, buffer={args.buffer}")
    print(f"区间: {metrics['start_date']} ~ {metrics['end_date']}")
    for k in ["total_return", "annualized_return", "volatility", "max_drawdown"]:
        print(f"{k}: {metrics[k]:.2%}")
    for k in ["sharpe", "calmar"]:
        print(f"{k}: {metrics[k]:.2f}")
    print(f"交易次数: {len(trades)}")
    print(f"平均持仓: {hold_count.mean():.2f} 只")

    os.makedirs(args.outdir, exist_ok=True)
    suffix = f"_98{args.use_98}_int{args.integer_lot}_buf{args.buffer}"
    pd.DataFrame([{f"strategy_alphalab_clone{suffix}": True, **metrics}]).to_csv(
        os.path.join(args.outdir, f"alphalab_clone{suffix}.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    print(f"耗时: {time.time() - start_time:.4f}s")


if __name__ == "__main__":
    main()
