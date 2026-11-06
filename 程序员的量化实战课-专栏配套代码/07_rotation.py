"""
大小盘轮动：沪深300 vs 创业板
规则：每天取 roc(20) 较大者，top=1 满仓；若两者 roc(20) 均 < 0，空仓。
"""
import argparse
import os
import time

import pandas as pd

from engine import backtest
from utils import calc_metrics, plot_nav_and_weights


DEFAULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def parse_args():
    parser = argparse.ArgumentParser(description="大小盘轮动回测（自研）")
    parser.add_argument("--symbols", type=str, default="050002.OF.csv,110026.OF.csv",
                        help="标的 CSV 文件名，逗号分隔")
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


def generate_weights(prices: pd.DataFrame, roc_period: int) -> pd.DataFrame:
    """
    生成目标权重矩阵。
    每天取 roc(20) 较大者，权重 1.0；若两者均 < 0，权重 0。
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


def load_fund_name_map(data_dir: str) -> dict:
    basic_path = os.path.join(data_dir, "basic.csv")
    if not os.path.exists(basic_path):
        return {}
    df = pd.read_csv(basic_path, usecols=["ts_code", "name"])
    return dict(zip(df["ts_code"], df["name"]))


def main():
    args = parse_args()
    start_time = time.time()

    symbols = [s.strip() for s in args.symbols.split(",")]
    prices = pd.DataFrame({s: load_asset(os.path.join(args.data_dir, s)) for s in symbols})
    prices = prices.dropna()

    weights_target = generate_weights(prices, args.roc_period)

    # 只在目标权重变化的日子调仓，避免在空仓/满仓日重复扣费，与 vbt 的 order 逻辑对齐
    rebalance_mask = weights_target.ne(weights_target.shift(1)).any(axis=1)
    rebalance_dates = set(weights_target[rebalance_mask].index)
    nav, weights = backtest(prices, weights_target, args.fee, rebalance_dates=rebalance_dates)
    metrics = calc_metrics(nav, ann_factor=args.ann_factor, risk_free=args.risk_free)

    elapsed = time.time() - start_time

    print(f"=== 大小盘轮动回测 ===")
    print(f"区间: {metrics['start_date']} ~ {metrics['end_date']}")
    for k, v in metrics.items():
        if k not in ("sharpe", "calmar", "start_date", "end_date"):
            print(f"{k}: {v:.2%}")
    print(f"sharpe: {metrics['sharpe']:.2f}")
    print(f"calmar: {metrics['calmar']:.2f}")
    print(f"运行时间: {elapsed:.4f}s")

    os.makedirs(args.outdir, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(os.path.join(args.outdir, "rotation_metrics.csv"),
                                   index=False, encoding="utf-8-sig")

    fund_name_map = load_fund_name_map(args.data_dir)
    name_map = {s: fund_name_map.get(s.replace(".csv", ""), s) for s in symbols}
    plot_path = plot_nav_and_weights(prices, nav, weights, args.outdir,
                                     title="大小盘轮动回测",
                                     name_map=name_map,
                                     filename="rotation.png")
    print(f"图表已保存: {plot_path}")


if __name__ == "__main__":
    main()
