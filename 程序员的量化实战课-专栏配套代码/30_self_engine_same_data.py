"""
用自研引擎在 alphalab 导出的数据上跑同样 JSON 配置
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

from utils import calc_metrics


DEFAULT_DIR = "e:/agilab/程序员的量化实战课-专栏配套代码/data_from_alphalab"

DEFAULT_SYMBOLS = [
    "510300.SH.csv", "588000.SH.csv", "510880.SH.csv", "159980.SZ.csv", "159985.SZ.csv",
    "513100.SH.csv", "513520.SH.csv", "513500.SH.csv", "513030.SH.csv", "501018.SH.csv",
    "513180.SH.csv", "159915.SZ.csv", "512890.SH.csv", "511090.SH.csv", "510180.SH.csv",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default=",".join(DEFAULT_SYMBOLS), help="标的 CSV 文件名")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DIR, help="CSV 数据目录")
    parser.add_argument("--capital", type=float, default=1000000.0, help="初始资金")
    parser.add_argument("--period", type=int, default=20, help="trend_score2 周期")
    parser.add_argument("--top-k", type=int, default=4, help="top-k 选股")
    parser.add_argument("--max-weight", type=float, default=0.25, help="单标的上限")
    parser.add_argument("--fee", type=float, default=0.0001, help="交易成本")
    parser.add_argument("--ann-factor", type=int, default=252, help="年化因子")
    parser.add_argument("--risk-free", type=float, default=0.02, help="无风险利率")
    parser.add_argument("--start-date", type=str, default="2020-01-01", help="开始日期")
    parser.add_argument("--end-date", type=str, default="", help="结束日期")
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


def backtest(close, score, roc1, args):
    symbols = close.columns.tolist()
    dates = close.index
    cash = args.capital
    positions = {}
    nav = pd.Series(index=dates, dtype=float)
    trade_count = 0

    for i, date in enumerate(dates):
        today_close = close.loc[date]

        # sell rules: trend_score2 >= 5 or <= 0
        for sym in list(positions.keys()):
            s = score.loc[date, sym]
            if pd.notna(s) and (s >= 5 or s <= 0):
                px = today_close[sym]
                cash += positions[sym] * px * (1 - args.fee)
                del positions[sym]
                trade_count += 1

        # buy candidates
        row = score.loc[date].dropna()
        row = row[roc1.loc[date] >= -0.05]
        row = row.dropna()

        if len(row) > 0:
            selected = row.sort_values(ascending=False).head(args.top_k)
            targets = selected.index.tolist()
        else:
            targets = []

        # 清仓不在目标中的
        for sym in list(positions.keys()):
            if sym not in targets:
                px = today_close[sym]
                cash += positions[sym] * px * (1 - args.fee)
                del positions[sym]
                trade_count += 1

        equity = cash + sum(positions[sym] * today_close[sym] for sym in positions)

        if targets:
            w = min(1.0 / len(targets), args.max_weight)
            for sym in targets:
                px = today_close[sym]
                target_amount = equity * w
                if sym in positions:
                    current_amount = positions[sym] * px
                    delta = target_amount - current_amount
                    if delta > 0:
                        cash -= delta * (1 + args.fee)
                        positions[sym] += delta / px
                    elif delta < 0:
                        cash += -delta * (1 - args.fee)
                        positions[sym] -= -delta / px
                else:
                    positions[sym] = target_amount / (1 + args.fee) / px
                    cash -= target_amount
                    trade_count += 1

        equity = cash + sum(positions[sym] * today_close[sym] for sym in positions)
        nav.loc[date] = equity

    return nav, trade_count


def main():
    args = parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    close_frames = []
    for s in symbols:
        path = os.path.join(args.data_dir, s)
        if not os.path.exists(path):
            print(f"警告：文件不存在 {path}")
            continue
        df = load_ohlc(path)
        close_frames.append(df["close"].rename(s))

    close = pd.concat(close_frames, axis=1, sort=True)
    # 对缺失数据前向填充，与 alphalab 引擎保持一致
    close = close.ffill()

    if args.start_date:
        close = close[close.index >= pd.to_datetime(args.start_date)]
    if args.end_date:
        close = close[close.index <= pd.to_datetime(args.end_date)]

    print(f"数据区间: {close.index[0].date()} ~ {close.index[-1].date()}")

    score = cs_zscore(calc_trend_score2(close, args.period))
    roc1 = close.pct_change(1)

    nav, trade_count = backtest(close, score, roc1, args)
    metrics = calc_metrics(nav, ann_factor=args.ann_factor, risk_free=args.risk_free)

    print(f"=== 自研引擎（相同 alphalab 数据） ===")
    print(f"区间: {metrics['start_date']} ~ {metrics['end_date']}")
    for k in ["total_return", "annualized_return", "volatility", "max_drawdown"]:
        print(f"{k}: {metrics[k]:.2%}")
    for k in ["sharpe", "calmar"]:
        print(f"{k}: {metrics[k]:.2f}")
    print(f"交易次数: {trade_count}")


if __name__ == "__main__":
    main()
