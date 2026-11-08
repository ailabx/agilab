"""
扩展 ETF 池子 Top-N 轮动策略

标的池：
- A股：沪深300、创业板、红利低波
- 跨境：纳指100、标普500、日本225(513000/513520)、德国DAX(513030)、恒生科技、恒生ETF
- 商品：黄金
- 债券：5年期国债、5-10年期国债

规则：
- 每天计算 trend_score(25)；
- 选评分最高的 N 只等权配置；
- 只在目标权重变化日调仓。
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

from engine import backtest
from utils import calc_metrics, plot_nav_and_weights


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


def parse_args():
    parser = argparse.ArgumentParser(description="扩展 ETF 池 Top-N 趋势评分轮动")
    parser.add_argument(
        "--symbols",
        type=str,
        default=DEFAULT_SYMBOLS,
        help="标的 CSV 文件名，逗号分隔",
    )
    parser.add_argument("--period", type=int, default=25, help="趋势评分周期")
    parser.add_argument("--top-n", type=int, default=2, help="Top N 等权")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DIR, help="CSV 数据目录")
    parser.add_argument("--outdir", type=str, default=DEFAULT_DIR, help="输出目录")
    parser.add_argument("--fee", type=float, default=0.0001, help="交易成本")
    parser.add_argument("--ann-factor", type=int, default=252, help="年化因子")
    parser.add_argument("--risk-free", type=float, default=0.02, help="无风险利率")
    return parser.parse_args()


def load_asset(path: str) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["date"], usecols=["date", "close"])
    df = df.sort_values("date").drop_duplicates("date").set_index("date")
    return df["close"]


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


def generate_weights(prices: pd.DataFrame, score: pd.DataFrame, top_n: int) -> pd.DataFrame:
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for date in score.dropna().index:
        row = score.loc[date]
        row = row[row > 0]
        if len(row) > 0:
            selected = row.sort_values(ascending=False).head(top_n).index
            w = 1.0 / len(selected)
            weights.loc[date, selected] = w
    return weights


def main():
    args = parse_args()
    start_time = time.time()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    prices = pd.DataFrame({s: load_asset(os.path.join(args.data_dir, s)) for s in symbols})
    prices = prices.dropna()

    score = calc_trend_score(prices, args.period)
    weights = generate_weights(prices, score, args.top_n)

    rebalance_mask = weights.ne(weights.shift(1)).any(axis=1)
    rebalance_dates = set(weights[rebalance_mask].index)
    nav, weights_bt = backtest(prices, weights, args.fee, rebalance_dates=rebalance_dates)
    metrics = calc_metrics(nav, ann_factor=args.ann_factor, risk_free=args.risk_free)

    elapsed = time.time() - start_time

    print(f"=== 扩展 ETF 池 Top{args.top_n} 趋势评分轮动 ===")
    print(f"区间: {metrics['start_date']} ~ {metrics['end_date']}")
    print(f"标的数: {len(symbols)}")
    for k, v in metrics.items():
        if k in ("total_return", "annualized_return", "volatility", "max_drawdown"):
            print(f"{k}: {v:.2%}")
        elif k in ("sharpe", "calmar"):
            print(f"{k}: {v:.2f}")
    print(f"运行时间: {elapsed:.4f}s")

    os.makedirs(args.outdir, exist_ok=True)
    pd.DataFrame([{"strategy": f"top{args.top_n}_trend_score", **metrics}]).to_csv(
        os.path.join(args.outdir, f"top{args.top_n}_trend_score_metrics.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    plot_path = plot_nav_and_weights(
        prices,
        nav,
        weights_bt,
        args.outdir,
        title=f"扩展 ETF 池 Top{args.top_n} 趋势评分轮动",
        name_map={s: NAME_MAP.get(s, s) for s in symbols},
        filename=f"top{args.top_n}_trend_score_rotation.png",
    )
    print(f"图表已保存: {plot_path}")


if __name__ == "__main__":
    main()
