"""
用 vectorbt 实现创业板择时策略：
roc_20 > 8% 买入，< 0% 卖出，其余保持不变。
"""
import argparse
import os
import time

import pandas as pd
import numpy as np


DEFAULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def parse_args():
    parser = argparse.ArgumentParser(description="vectorbt 创业板择时")
    parser.add_argument("--symbol", type=str, default="110026.OF.csv", help="标的 CSV 文件名")
    parser.add_argument("--roc-period", type=int, default=20, help="ROC 计算周期")
    parser.add_argument("--buy-threshold", type=float, default=0.08, help="买入阈值")
    parser.add_argument("--sell-threshold", type=float, default=0.0, help="卖出阈值")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DIR, help="CSV 数据目录")
    parser.add_argument("--outdir", type=str, default=DEFAULT_DIR, help="输出目录")
    parser.add_argument("--fee", type=float, default=0.0001, help="交易成本，默认万分之一")
    return parser.parse_args()


def load_asset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"], usecols=["date", "close"])
    return df.sort_values("date").drop_duplicates("date").set_index("date")


def generate_signals(close: pd.Series, roc_period: int,
                     buy_threshold: float, sell_threshold: float) -> pd.Series:
    """
    生成持仓信号：roc > buy_threshold 时持仓 1，< sell_threshold 时空仓 0，其余保持前一日信号。
    """
    roc = close / close.shift(roc_period) - 1.0
    signal = pd.Series(0.0, index=close.index)

    for i, date in enumerate(close.index):
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


def main():
    args = parse_args()

    start_time = time.time()

    close = load_asset(os.path.join(args.data_dir, args.symbol))["close"]
    signal = generate_signals(close, args.roc_period, args.buy_threshold, args.sell_threshold)

    import vectorbt as vbt

    # 与我们的 calc_metrics 口径一致：按 252 交易日年化，无风险利率 2%
    vbt.settings.returns['year_freq'] = '252 days'
    vbt.settings.returns['defaults']['risk_free'] = 0.02 / 252

    # 构造价格矩阵，vbt 需要 DataFrame
    prices = pd.DataFrame({args.symbol: close})

    # 生成 vbt 信号：entries 买入，exits 卖出
    # roc > buy_threshold 当天买入，roc < sell_threshold 当天卖出
    entries = pd.DataFrame({args.symbol: signal.diff() == 1.0}, index=close.index)
    exits = pd.DataFrame({args.symbol: signal.diff() == -1.0}, index=close.index)

    pf = vbt.Portfolio.from_signals(
        close=prices,
        entries=entries,
        exits=exits,
        direction="longonly",
        freq="1d",
        fees=args.fee,
        init_cash=10000.0,
    )

    elapsed = time.time() - start_time

    # 输出指标
    print(f"=== vectorbt 择时回测 ({args.symbol}) ===")
    print(f"总收益: {pf.total_return().iloc[0]:.2%}")
    print(f"年化收益: {pf.annualized_return().iloc[0]:.2%}")
    print(f"最大回撤: {pf.max_drawdown().iloc[0]:.2%}")
    print(f"夏普: {pf.sharpe_ratio().iloc[0]:.2f}")
    print(f"运行时间: {elapsed:.4f}s")

    # 保存净值
    nav = pf.value()
    os.makedirs(args.outdir, exist_ok=True)
    nav.to_csv(os.path.join(args.outdir, "timing_vbt_nav.csv"), encoding="utf-8-sig")
    print(f"净值已保存: {os.path.join(args.outdir, 'timing_vbt_nav.csv')}")


if __name__ == "__main__":
    main()
