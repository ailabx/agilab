"""
极简择时回测：roc_20 > 8% 买入，< 0% 卖出，其余保持不变。
"""
import argparse
import os
import time

import pandas as pd

from engine import backtest
from utils import calc_metrics, plot_nav_and_weights


DEFAULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def parse_args():
    parser = argparse.ArgumentParser(description="极简择时回测")
    parser.add_argument("--symbol", type=str, default="110026.OF.csv", help="标的 CSV 文件名")
    parser.add_argument("--roc-period", type=int, default=20, help="ROC 计算周期")
    parser.add_argument("--buy-threshold", type=float, default=0.08, help="买入阈值")
    parser.add_argument("--sell-threshold", type=float, default=0.0, help="卖出阈值")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DIR, help="CSV 数据目录")
    parser.add_argument("--outdir", type=str, default=DEFAULT_DIR, help="输出目录")
    parser.add_argument("--fee", type=float, default=0.0001, help="交易成本，默认万分之一")
    parser.add_argument("--ann-factor", type=int, default=252, help="年化因子，默认 252 个交易日")
    parser.add_argument("--risk-free", type=float, default=0.02, help="无风险利率，默认 2%")
    return parser.parse_args()


def load_asset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"], usecols=["date", "close"])
    return df.sort_values("date").drop_duplicates("date").set_index("date")


def generate_signals(prices: pd.Series, roc_period: int, buy_threshold: float,
                     sell_threshold: float) -> pd.Series:
    """
    生成持仓信号：roc > buy_threshold 时持仓 1，< sell_threshold 时空仓 0，其余保持前一日信号。
    """
    roc = prices / prices.shift(roc_period) - 1.0
    signal = pd.Series(0.0, index=prices.index)

    for i, date in enumerate(prices.index):
        if i == 0:
            signal.iloc[i] = 0.0
            continue
        if roc.loc[date] > buy_threshold:
            signal.iloc[i] = 1.0
        elif roc.loc[date] < sell_threshold:
            signal.iloc[i] = 0.0
        else:
            signal.iloc[i] = signal.iloc[i - 1]

    return signal


def load_fund_name_map(data_dir: str) -> dict:
    """从 basic.csv 读取基金名称映射，{ts_code: name}。"""
    basic_path = os.path.join(data_dir, "basic.csv")
    if not os.path.exists(basic_path):
        return {}
    df = pd.read_csv(basic_path, usecols=["ts_code", "name"])
    return dict(zip(df["ts_code"], df["name"]))


def main():
    args = parse_args()

    start_time = time.time()

    prices = load_asset(os.path.join(args.data_dir, args.symbol))
    prices.columns = ["close"]
    close = prices["close"]

    signal = generate_signals(close, args.roc_period, args.buy_threshold, args.sell_threshold)

    # 构造目标权重矩阵：signal=1 时权重 100%，signal=0 时权重 0%
    weights_target = pd.DataFrame({args.symbol: signal}, index=close.index)

    # 运行回测
    prices_df = pd.DataFrame({args.symbol: close})
    nav, weights = backtest(prices_df, weights_target, args.fee)
    metrics = calc_metrics(nav, ann_factor=args.ann_factor, risk_free=args.risk_free)

    elapsed = time.time() - start_time

    print(f"=== 择时回测 ({args.symbol}) ===")
    print(f"区间: {metrics['start_date']} ~ {metrics['end_date']}")
    for k, v in metrics.items():
        if k not in ("sharpe", "calmar", "start_date", "end_date"):
            print(f"{k}: {v:.2%}")
    print(f"sharpe: {metrics['sharpe']:.2f}")
    print(f"calmar: {metrics['calmar']:.2f}")
    print(f"运行时间: {elapsed:.4f}s")

    os.makedirs(args.outdir, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(os.path.join(args.outdir, "timing_metrics.csv"),
                                   index=False, encoding="utf-8-sig")

    # 构建基金名称映射
    fund_name_map = load_fund_name_map(args.data_dir)
    name_map = {args.symbol: fund_name_map.get(args.symbol.replace(".csv", ""), args.symbol)}
    plot_path = plot_nav_and_weights(prices_df, nav, weights, args.outdir,
                                     title=f"择时回测 ({args.symbol})",
                                     name_map=name_map,
                                     filename="timing.png")
    print(f"图表已保存: {plot_path}")


if __name__ == "__main__":
    main()
