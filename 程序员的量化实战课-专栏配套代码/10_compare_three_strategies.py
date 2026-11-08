"""
把 ROC(20)、Slope(20)、趋势评分(加权) 三个策略的净值画在一张图里对比。

数据：
- ROC(20) 和 Slope(20) 来自 08_quad_rotation.py
- 趋势评分(加权) 来自 09_trend_score_rotation.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import backtest
from utils import calc_metrics


DEFAULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

NAME_MAP = {
    "513100.SH.csv": "纳指100",
    "512890.SH.csv": "红利低波",
    "159915.SZ.csv": "创业板",
    "518880.SH.csv": "黄金",
}


def load_asset(path: str) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["date"], usecols=["date", "close"])
    df = df.sort_values("date").drop_duplicates("date").set_index("date")
    return df["close"]


def calc_slope(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    slopes = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    for col in prices.columns:
        series = prices[col]
        slopes[col] = series.rolling(window=window).apply(
            lambda y: np.polyfit(np.arange(len(y)), y, 1)[0] / y[-1],
            raw=True,
        )
    return slopes


def calc_trend_score(prices: pd.DataFrame, period: int) -> pd.DataFrame:
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
        y_pred = slope * x + intercept
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


def generate_weights_roc(prices: pd.DataFrame, period: int) -> pd.DataFrame:
    roc = prices / prices.shift(period) - 1.0
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for i, date in enumerate(prices.index):
        if i < period:
            continue
        r = roc.loc[date]
        if r.max() >= 0:
            weights.loc[date, r.idxmax()] = 1.0
    return weights


def generate_weights_slope(prices: pd.DataFrame, period: int) -> pd.DataFrame:
    slope = calc_slope(prices, period)
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for i, date in enumerate(prices.index):
        if i < period:
            continue
        r = slope.loc[date]
        if r.max() >= 0:
            weights.loc[date, r.idxmax()] = 1.0
    return weights


def generate_weights_trend(prices: pd.DataFrame, period: int) -> pd.DataFrame:
    score = calc_trend_score(prices, period)
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for date in score.dropna().index:
        row = score.loc[date]
        if row.max() > 0:
            weights.loc[date, row.idxmax()] = 1.0
    return weights


def run_backtest(prices: pd.DataFrame, weights_target: pd.DataFrame, fee: float):
    rebalance_mask = weights_target.ne(weights_target.shift(1)).any(axis=1)
    rebalance_dates = set(weights_target[rebalance_mask].index)
    return backtest(prices, weights_target, fee, rebalance_dates=rebalance_dates)


def main():
    symbols = ["513100.SH.csv", "512890.SH.csv", "159915.SZ.csv", "518880.SH.csv"]
    prices = pd.DataFrame({s: load_asset(os.path.join(DEFAULT_DIR, s)) for s in symbols})
    prices = prices.dropna()

    weights_roc = generate_weights_roc(prices, 20)
    weights_slope = generate_weights_slope(prices, 20)
    weights_trend = generate_weights_trend(prices, 25)

    nav_roc, _ = run_backtest(prices, weights_roc, 0.0001)
    nav_slope, _ = run_backtest(prices, weights_slope, 0.0001)
    nav_trend, _ = run_backtest(prices, weights_trend, 0.0001)

    metrics_roc = calc_metrics(nav_roc)
    metrics_slope = calc_metrics(nav_slope)
    metrics_trend = calc_metrics(nav_trend)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(nav_roc.index, nav_roc / nav_roc.iloc[0], label="ROC(20)", linewidth=2)
    ax.plot(nav_slope.index, nav_slope / nav_slope.iloc[0], label="Slope(20)", linewidth=2)
    ax.plot(nav_trend.index, nav_trend / nav_trend.iloc[0], label="趋势评分(加权, period=25)", linewidth=2)
    ax.set_title("三策略净值对比：ROC(20) / Slope(20) / 趋势评分(加权)", fontsize=14)
    ax.set_xlabel("日期")
    ax.set_ylabel("归一化净值")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    outpath = os.path.join(DEFAULT_DIR, "compare_three_strategies.png")
    plt.savefig(outpath, dpi=150)
    plt.close()

    print("=== 三策略对比 ===")
    print(f"{'指标':<12} {'ROC(20)':>12} {'Slope(20)':>12} {'趋势评分(加权)':>15}")
    print(f"{'total_return':<12} {metrics_roc['total_return']:>11.2%} {metrics_slope['total_return']:>11.2%} {metrics_trend['total_return']:>15.2%}")
    print(f"{'annualized':<12} {metrics_roc['annualized_return']:>11.2%} {metrics_slope['annualized_return']:>11.2%} {metrics_trend['annualized_return']:>15.2%}")
    print(f"{'volatility':<12} {metrics_roc['volatility']:>11.2%} {metrics_slope['volatility']:>11.2%} {metrics_trend['volatility']:>15.2%}")
    print(f"{'max_dd':<12} {metrics_roc['max_drawdown']:>11.2%} {metrics_slope['max_drawdown']:>11.2%} {metrics_trend['max_drawdown']:>15.2%}")
    print(f"{'sharpe':<12} {metrics_roc['sharpe']:>11.2f} {metrics_slope['sharpe']:>11.2f} {metrics_trend['sharpe']:>15.2f}")
    print(f"{'calmar':<12} {metrics_roc['calmar']:>11.2f} {metrics_slope['calmar']:>11.2f} {metrics_trend['calmar']:>15.2f}")
    print(f"图表已保存: {outpath}")


if __name__ == "__main__":
    main()
