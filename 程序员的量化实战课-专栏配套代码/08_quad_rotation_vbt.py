"""
四资产轮动策略（vectorbt from_orders 版）

规则：
- ROC 版本：每天计算 roc(20)，选最大者满仓，全 < 0 时空仓；
- Slope 版本：每天计算 polyfit 拟合斜率（归一化），选最大者满仓，全 < 0 时空仓。
"""
import argparse
import os
import time

import numpy as np
import pandas as pd


DEFAULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# 中文名称映射（用于图表展示）
NAME_MAP = {
    "270042.OF.csv": "纳指100",
    "007466.OF.csv": "红利低波",
    "110026.OF.csv": "创业板",
    "000218.OF.csv": "黄金",
}


def parse_args():
    parser = argparse.ArgumentParser(description="四资产轮动回测（vectorbt）")
    parser.add_argument(
        "--symbols",
        type=str,
        default="270042.OF.csv,007466.OF.csv,110026.OF.csv,000218.OF.csv",
        help="标的 CSV 文件名，逗号分隔",
    )
    parser.add_argument("--roc-period", type=int, default=20, help="ROC 计算周期")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DIR, help="CSV 数据目录")
    parser.add_argument("--outdir", type=str, default=DEFAULT_DIR, help="输出目录")
    parser.add_argument("--fee", type=float, default=0.0001, help="交易成本，默认万分之一")
    return parser.parse_args()


def load_asset(path: str) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["date"], usecols=["date", "close"])
    df = df.sort_values("date").drop_duplicates("date").set_index("date")
    return df["close"]


def generate_weights_roc(prices: pd.DataFrame, roc_period: int) -> pd.DataFrame:
    """生成目标权重矩阵（ROC 版本）。"""
    roc = prices / prices.shift(roc_period) - 1.0
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    for i, date in enumerate(prices.index):
        if i < roc_period:
            continue
        r = roc.loc[date]
        if r.max() >= 0:
            best = r.idxmax()
            weights.loc[date, best] = 1.0

    return weights


def calc_slope(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    计算每个标的收盘价的 n 日线性回归斜率（np.polyfit），
    并除以最新收盘价做归一化，消除不同标的价格量级差异。
    """
    slopes = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    for col in prices.columns:
        series = prices[col]
        slopes[col] = series.rolling(window=window).apply(
            lambda y: np.polyfit(np.arange(len(y)), y, 1)[0] / y[-1],
            raw=True,
        )
    return slopes


def generate_weights_slope(prices: pd.DataFrame, slope_period: int) -> pd.DataFrame:
    """生成目标权重矩阵（Slope 版本）。"""
    slope = calc_slope(prices, slope_period)
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    for i, date in enumerate(prices.index):
        if i < slope_period:
            continue
        r = slope.loc[date]
        if r.max() >= 0:
            best = r.idxmax()
            weights.loc[date, best] = 1.0

    return weights


def run_vbt(prices: pd.DataFrame, weights_target: pd.DataFrame, fee: float):
    import vectorbt as vbt

    vbt.settings.returns['year_freq'] = '252 days'
    vbt.settings.returns['defaults']['risk_free'] = 0.02 / 252

    pf = vbt.Portfolio.from_orders(
        close=prices,
        size=weights_target,
        size_type="targetpercent",
        direction="longonly",
        freq="1d",
        fees=fee,
        init_cash=10000.0,
        cash_sharing=True,
    )
    return pf


def main():
    args = parse_args()
    start_time = time.time()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    prices = pd.DataFrame({s: load_asset(os.path.join(args.data_dir, s)) for s in symbols})
    prices = prices.dropna()

    weights_roc = generate_weights_roc(prices, args.roc_period)
    weights_slope = generate_weights_slope(prices, args.roc_period)

    pf_roc = run_vbt(prices, weights_roc, args.fee)
    pf_slope = run_vbt(prices, weights_slope, args.fee)

    elapsed = time.time() - start_time

    print(f"=== vectorbt 四资产轮动回测对比（period={args.roc_period}）===")
    print(f"{'指标':<12} {'ROC(20)':>12} {'Slope(20)':>12}")
    print(f"{'总收益':<12} {pf_roc.total_return():>11.2%} {pf_slope.total_return():>11.2%}")
    print(f"{'年化收益':<12} {pf_roc.annualized_return():>11.2%} {pf_slope.annualized_return():>11.2%}")
    print(f"{'最大回撤':<12} {pf_roc.max_drawdown():>11.2%} {pf_slope.max_drawdown():>11.2%}")
    print(f"{'夏普':<12} {pf_roc.sharpe_ratio():>11.2f} {pf_slope.sharpe_ratio():>11.2f}")
    print(f"运行时间: {elapsed:.4f}s")

    os.makedirs(args.outdir, exist_ok=True)
    nav_roc = pf_roc.value()
    nav_slope = pf_slope.value()

    nav_roc.to_csv(os.path.join(args.outdir, "quad_rotation_vbt_nav.csv"), encoding="utf-8-sig")
    nav_slope.to_csv(os.path.join(args.outdir, "quad_rotation_slope_vbt_nav.csv"), encoding="utf-8-sig")

    print(f"净值已保存: {os.path.join(args.outdir, 'quad_rotation_vbt_nav.csv')}")
    print(f"净值已保存: {os.path.join(args.outdir, 'quad_rotation_slope_vbt_nav.csv')}")


if __name__ == "__main__":
    main()
