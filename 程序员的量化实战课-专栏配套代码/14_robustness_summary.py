"""
稳健性测试汇总图：
- 原始策略 (period=25, top_n=2)
- 滚动窗口 (window=252)
- 样本外 (2024-2026)
- 蒙特卡洛 5%/中位数/95% 净值区间
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
    "510300.SH.csv": "沪深300",
    "159915.SZ.csv": "创业板",
    "512890.SH.csv": "红利低波",
    "513100.SH.csv": "纳指100",
    "513500.SH.csv": "标普500",
    "513000.SH.csv": "日经225",
    "513520.SH.csv": "日经225-2",
    "513030.SH.csv": "德国DAX",
    "513050.SH.csv": "恒生科技",
    "159920.SZ.csv": "恒生ETF",
    "518880.SH.csv": "黄金",
    "511010.SH.csv": "国债5Y",
    "511020.SH.csv": "国债5-10Y",
}
DEFAULT_SYMBOLS = ",".join(NAME_MAP.keys())


def load_asset(path):
    df = pd.read_csv(path, parse_dates=["date"], usecols=["date", "close"])
    df = df.sort_values("date").drop_duplicates("date").set_index("date")
    return df["close"]


def calc_trend_score(prices, period):
    x = np.arange(period, dtype=float)
    w = np.linspace(1.0, 2.0, period) ** 2
    w_sum = w.sum()
    x_bar = (w * x).sum() / w_sum
    dx = x - x_bar
    variance_x = (w * dx ** 2).sum()

    def _score(window):
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


def generate_weights(prices, score, top_n):
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for date in score.dropna().index:
        row = score.loc[date]
        row = row[row > 0]
        if len(row) > 0:
            selected = row.sort_values(ascending=False).head(top_n).index
            w = 1.0 / len(selected)
            weights.loc[date, selected] = w
    return weights


def run_backtest(prices, weights, fee):
    rebalance_mask = weights.ne(weights.shift(1)).any(axis=1)
    rebalance_dates = set(weights[rebalance_mask].index)
    nav, _ = backtest(prices, weights, fee, rebalance_dates=rebalance_dates)
    return nav


def fast_backtest(prices, weights, fee):
    rets = prices.pct_change()
    w_lag = weights.shift(1).fillna(0.0)
    port_rets = (w_lag * rets).sum(axis=1)
    turnover = (weights - w_lag).abs().sum(axis=1)
    total_rets = port_rets - turnover * fee
    nav = (1 + total_rets).cumprod()
    nav.iloc[0] = 1.0
    return nav


def rolling_window_backtest(prices, period, top_n, fee, window):
    all_dates = prices.index
    score_rolling = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    for i, date in enumerate(all_dates):
        if i < period - 1:
            continue
        start_idx = max(0, i - window + 1)
        window_prices = prices.iloc[start_idx : i + 1]
        score_rolling.loc[date] = calc_trend_score(window_prices, period).iloc[-1]

    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for date in score_rolling.dropna().index:
        row = score_rolling.loc[date]
        row = row[row > 0]
        if len(row) > 0:
            selected = row.sort_values(ascending=False).head(top_n).index
            w = 1.0 / len(selected)
            weights.loc[date, selected] = w

    weights_lag = weights.shift(1).fillna(0.0)
    return fast_backtest(prices, weights_lag, fee)


def main():
    symbols = [s.strip() for s in DEFAULT_SYMBOLS.split(",") if s.strip()]
    prices = pd.DataFrame({s: load_asset(os.path.join(DEFAULT_DIR, s)) for s in symbols})
    prices = prices.dropna()

    # 原始策略
    score = calc_trend_score(prices, 25)
    weights = generate_weights(prices, score, 2)
    nav_orig = run_backtest(prices, weights, 0.0001)

    # 滚动窗口
    nav_rolling = rolling_window_backtest(prices, 25, 2, 0.0001, 252)

    # 蒙特卡洛路径
    rets = prices.pct_change().dropna()
    mc_navs = []
    for _ in range(200):
        noise_rets = rets + np.random.normal(0, 0.005, rets.shape)
        noise_prices = prices.iloc[0] * (1 + noise_rets).cumprod()
        noise_prices = pd.concat([prices.iloc[[0]], noise_prices])
        if np.random.rand() < 0.10:
            weights_mc = weights.shift(np.random.randint(1, 3)).fillna(0.0)
        else:
            weights_mc = weights
        fee_mc = 0.0001 * np.random.uniform(0.5, 2.0)
        nav_mc = fast_backtest(noise_prices, weights_mc, fee_mc)
        mc_navs.append(nav_mc)

    mc_df = pd.concat(mc_navs, axis=1)
    mc_05 = mc_df.quantile(0.05, axis=1)
    mc_50 = mc_df.quantile(0.50, axis=1)
    mc_95 = mc_df.quantile(0.95, axis=1)

    # 样本外
    prices_test = prices[prices.index > "2023-12-31"]
    score_test = calc_trend_score(prices_test, 25)
    weights_test = generate_weights(prices_test, score_test, 2)
    nav_oos = run_backtest(prices_test, weights_test, 0.0001)

    # 画图
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(nav_orig.index, nav_orig / nav_orig.iloc[0], label="原始策略 (p=25, top=2)", linewidth=2, color="#1f77b4")
    ax.plot(nav_rolling.index, nav_rolling / nav_rolling.iloc[0], label="滚动窗口 (w=252)", linewidth=2, color="#ff7f0e")
    ax.plot(nav_oos.index, nav_oos / nav_oos.iloc[0], label="样本外 (2024-2026)", linewidth=2, color="#2ca02c")
    ax.plot(mc_50.index, mc_50 / mc_50.iloc[0], label="MC 中位数", linewidth=2, color="#d62728", linestyle="--")
    ax.fill_between(
        mc_05.index,
        mc_05 / mc_05.iloc[0],
        mc_95 / mc_95.iloc[0],
        alpha=0.2,
        color="#d62728",
        label="MC 5%-95% 区间",
    )
    ax.axvline(pd.Timestamp("2023-12-31"), color="gray", linestyle="--", alpha=0.5)
    ax.set_title("稳健性测试汇总：原始策略 vs 滚动窗口 vs 样本外 vs 蒙特卡洛区间")
    ax.set_xlabel("日期")
    ax.set_ylabel("归一化净值")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    outpath = os.path.join(DEFAULT_DIR, "robustness_summary.png")
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"汇总图已保存: {outpath}")


if __name__ == "__main__":
    main()
