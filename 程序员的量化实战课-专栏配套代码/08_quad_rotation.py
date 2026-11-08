"""
四资产 ROC 轮动策略：纳指100 / 红利低波 / 创业板 / 黄金

规则：
- 每天计算 4 个标的的 roc(20)；
- 选 roc(20) 最大的那一只，满仓（权重 1.0）；
- 若所有标的 roc(20) < 0，则空仓（权重 0）。

数据：Tushare 场外基金复权净值（已落盘到 data/）
  270042.OF  广发纳斯达克100指数A  -> 映射为 纳指100
  007466.OF  华泰柏瑞中证红利低波ETF联接A  -> 映射为 红利低波
  110026.OF  易方达创业板ETF联接A  -> 映射为 创业板
  000218.OF  国泰黄金ETF联接A  -> 映射为 黄金
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

from engine import backtest
from utils import calc_metrics, plot_nav_and_weights


DEFAULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# 中文名称映射（用于图表展示）
NAME_MAP = {
    "270042.OF.csv": "纳指100",
    "007466.OF.csv": "红利低波",
    "110026.OF.csv": "创业板",
    "000218.OF.csv": "黄金",
}


def parse_args():
    parser = argparse.ArgumentParser(description="四资产 ROC 轮动回测（自研）")
    parser.add_argument(
        "--symbols",
        type=str,
        default="513100.SH.csv,512890.SH.csv,159915.SZ.csv,518880.SH.csv",
        help="标的 CSV 文件名，逗号分隔",
    )
    parser.add_argument("--roc-period", type=int, default=20, help="ROC 计算周期")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DIR, help="CSV 数据目录")
    parser.add_argument("--outdir", type=str, default=DEFAULT_DIR, help="输出目录")
    parser.add_argument("--fee", type=float, default=0.0001, help="交易成本，默认万分之一")
    parser.add_argument("--ann-factor", type=int, default=252, help="年化因子")
    parser.add_argument("--risk-free", type=float, default=0.02, help="无风险利率")
    return parser.parse_args()


def load_asset(path: str) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["date"], usecols=["date", "close"])
    df = df.sort_values("date").drop_duplicates("date").set_index("date")
    return df["close"]


def calc_slope(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    计算每个标的收盘价的 n 日线性回归斜率（最小二乘法），
    并除以最新收盘价做归一化，消除不同标的价格量级差异。
    """
    x = np.arange(window)
    x_mean = x.mean()
    denom = ((x - x_mean) ** 2).sum()

    slopes = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    for col in prices.columns:
        series = prices[col]
        # 使用 rolling + apply 计算每个窗口的斜率
        slopes[col] = series.rolling(window=window).apply(
            lambda y: ((np.arange(len(y)) - x_mean) * (y - y.mean())).sum() / denom,
            raw=True,
        )
    # 归一化：slope / close，使不同价格量级的标公平可比
    return slopes / prices


def generate_weights_roc(prices: pd.DataFrame, roc_period: int) -> pd.DataFrame:
    """
    生成目标权重矩阵（ROC 版本）。
    每天取 roc(20) 最大者，权重 1.0；若全部 < 0，权重 0。
    """
    roc = prices / prices.shift(roc_period) - 1.0
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    for i, date in enumerate(prices.index):
        if i < roc_period:
            continue
        r = roc.loc[date]
        if r.max() < 0:
            weights.loc[date] = 0.0
        else:
            best = r.idxmax()
            weights.loc[date, best] = 1.0

    return weights


def generate_weights_slope(prices: pd.DataFrame, slope_period: int) -> pd.DataFrame:
    """
    生成目标权重矩阵（Slope 版本）。
    每天取 slope(close, 20) 最大者，权重 1.0；若全部 < 0，权重 0。
    """
    slope = calc_slope(prices, slope_period)
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    for i, date in enumerate(prices.index):
        if i < slope_period:
            continue
        r = slope.loc[date]
        if r.max() < 0:
            weights.loc[date] = 0.0
        else:
            best = r.idxmax()
            weights.loc[date, best] = 1.0

    return weights


def run_backtest(prices: pd.DataFrame, weights_target: pd.DataFrame, fee: float):
    """运行一次回测并返回净值与权重。"""
    rebalance_mask = weights_target.ne(weights_target.shift(1)).any(axis=1)
    rebalance_dates = set(weights_target[rebalance_mask].index)
    return backtest(prices, weights_target, fee, rebalance_dates=rebalance_dates)


def main():
    args = parse_args()
    start_time = time.time()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    prices = pd.DataFrame({s: load_asset(os.path.join(args.data_dir, s)) for s in symbols})
    prices = prices.dropna()

    # ROC 版本
    weights_roc = generate_weights_roc(prices, args.roc_period)
    nav_roc, weights_roc_bt = run_backtest(prices, weights_roc, args.fee)
    metrics_roc = calc_metrics(nav_roc, ann_factor=args.ann_factor, risk_free=args.risk_free)

    # Slope 版本
    weights_slope = generate_weights_slope(prices, args.roc_period)
    nav_slope, weights_slope_bt = run_backtest(prices, weights_slope, args.fee)
    metrics_slope = calc_metrics(nav_slope, ann_factor=args.ann_factor, risk_free=args.risk_free)

    elapsed = time.time() - start_time

    print(f"=== 四资产轮动回测对比（roc_period={args.roc_period}）===")
    print(f"区间: {metrics_roc['start_date']} ~ {metrics_roc['end_date']}")
    print(f"{'指标':<12} {'ROC(20)':>12} {'Slope(20)':>12}")
    print(f"{'total_return':<12} {metrics_roc['total_return']:>11.2%} {metrics_slope['total_return']:>11.2%}")
    print(f"{'annualized':<12} {metrics_roc['annualized_return']:>11.2%} {metrics_slope['annualized_return']:>11.2%}")
    print(f"{'volatility':<12} {metrics_roc['volatility']:>11.2%} {metrics_slope['volatility']:>11.2%}")
    print(f"{'max_dd':<12} {metrics_roc['max_drawdown']:>11.2%} {metrics_slope['max_drawdown']:>11.2%}")
    print(f"{'sharpe':<12} {metrics_roc['sharpe']:>11.2f} {metrics_slope['sharpe']:>11.2f}")
    print(f"{'calmar':<12} {metrics_roc['calmar']:>11.2f} {metrics_slope['calmar']:>11.2f}")
    print(f"运行时间: {elapsed:.4f}s")

    os.makedirs(args.outdir, exist_ok=True)
    pd.DataFrame([{"strategy": "roc", **metrics_roc}, {"strategy": "slope", **metrics_slope}]).to_csv(
        os.path.join(args.outdir, "quad_rotation_metrics.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    plot_path_roc = plot_nav_and_weights(
        prices,
        nav_roc,
        weights_roc_bt,
        args.outdir,
        title="四资产 ROC 轮动回测",
        name_map={s: NAME_MAP.get(s, s) for s in symbols},
        filename="quad_rotation.png",
    )
    plot_path_slope = plot_nav_and_weights(
        prices,
        nav_slope,
        weights_slope_bt,
        args.outdir,
        title="四资产 Slope 轮动回测",
        name_map={s: NAME_MAP.get(s, s) for s in symbols},
        filename="quad_rotation_slope.png",
    )
    print(f"图表已保存: {plot_path_roc}, {plot_path_slope}")


if __name__ == "__main__":
    main()
