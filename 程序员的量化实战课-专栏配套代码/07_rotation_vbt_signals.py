"""
大小盘轮动（vectorbt from_signals 版）：沪深300 vs 创业板
规则：每天取 roc(20) 较大者，top=1 满仓；若两者 roc(20) 均 < 0，空仓。

与 07_rotation_vbt.py（from_orders targetpercent）的区别：
- 本脚本使用 vbt.Portfolio.from_signals，手动构造 entries/exits。
- 与自研 engine 使用同样的目标权重矩阵，可作为对比实现。

注意：
vectorbt 的 from_signals 在同一 bar 内同时出现 A 的 exit 和 B 的 entry 时，
即使设置 upon_long_conflict='entry'，entry 仍会被推迟到下一个 bar。
因此本版本会表现出“T+1 日买入新资产”的执行特征，
与自研 engine 的“T 日收盘满仓切换”存在差异，属于 vectorbt 的固有限制。
"""
import argparse
import os
import time

import pandas as pd
import numpy as np


DEFAULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def parse_args():
    parser = argparse.ArgumentParser(description="大小盘轮动回测（vectorbt from_signals）")
    parser.add_argument("--symbols", type=str, default="050002.OF.csv,110026.OF.csv",
                        help="标的 CSV 文件名，逗号分隔")
    parser.add_argument("--roc-period", type=int, default=20, help="ROC 计算周期")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DIR, help="CSV 数据目录")
    parser.add_argument("--outdir", type=str, default=DEFAULT_DIR, help="输出目录")
    parser.add_argument("--fee", type=float, default=0.0001, help="交易成本，默认万分之一")
    return parser.parse_args()


def load_asset(path: str) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["date"], usecols=["date", "close"])
    df = df.sort_values("date").drop_duplicates("date").set_index("date")
    return df["close"]


def generate_target_weights(prices: pd.DataFrame, roc_period: int) -> pd.DataFrame:
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
        if r.max() >= 0:
            best = r.idxmax()
            weights.loc[date, best] = 1.0

    return weights


def build_signals(prices: pd.DataFrame, weights_target: pd.DataFrame) -> tuple:
    """
    构造 from_signals 所需的 entries/exits。

    逻辑：
    - 当目标权重从 0 变为 1 时，生成该资产的 entry 信号；
    - 当目标权重从 1 变为 0 时，生成该资产的 exit 信号；
    - 同一日先 exit 旧资产、再 entry 新资产（cash_sharing=True 保证资金复用）。
    """
    entries = pd.DataFrame(False, index=prices.index, columns=prices.columns)
    exits = pd.DataFrame(False, index=prices.index, columns=prices.columns)

    prev = pd.Series(0.0, index=prices.columns)
    for date in prices.index:
        curr = weights_target.loc[date]
        for asset in prices.columns:
            if curr[asset] > 0 and prev[asset] == 0:
                entries.loc[date, asset] = True
            elif curr[asset] == 0 and prev[asset] > 0:
                exits.loc[date, asset] = True
        prev = curr.copy()

    return entries, exits


def main():
    args = parse_args()
    start_time = time.time()

    symbols = [s.strip() for s in args.symbols.split(",")]
    prices = pd.DataFrame({s: load_asset(os.path.join(args.data_dir, s)) for s in symbols})
    prices = prices.dropna()

    import vectorbt as vbt

    # 与自研 engine 口径一致
    vbt.settings.returns['year_freq'] = '252 days'
    vbt.settings.returns['defaults']['risk_free'] = 0.02 / 252

    weights_target = generate_target_weights(prices, args.roc_period)
    entries, exits = build_signals(prices, weights_target)

    # from_signals 关键参数说明：
    # - direction="longonly"：只做多，不空仓做空；exit 后变为 cash。
    # - cash_sharing=True：多个资产共享同一 pool 现金，换仓时先卖后买。
    # - size=np.inf / np.inf 在 vbt 中表示“全部可用资金买入/卖出”，
    #   但这里更稳妥的方式是不传 size，让 entry 时默认使用全部可用资金。
    # 由于 cash_sharing=True，entry 时会自动使用当前可用 cash 满仓买入目标资产。
    # 同一 bar 内 entry 与 exit 同时存在时（即换仓日），
    # 默认 ConflictMode 可能导致 entry 被忽略，从而持仓延迟一天。
    # 这里显式指定 upon_long_conflict='entry'，
    # 让 entry 优先于 exit 执行，保证 T 日收盘时满仓持有目标资产。
    pf = vbt.Portfolio.from_signals(
        close=prices,
        entries=entries,
        exits=exits,
        direction="longonly",
        freq="1d",
        fees=args.fee,
        init_cash=10000.0,
        cash_sharing=True,
        upon_long_conflict="entry",
    )

    elapsed = time.time() - start_time

    print(f"=== vectorbt (from_signals) 大小盘轮动回测 ===")
    print(f"总收益: {pf.total_return():.2%}")
    print(f"年化收益: {pf.annualized_return():.2%}")
    print(f"最大回撤: {pf.max_drawdown():.2%}")
    print(f"夏普: {pf.sharpe_ratio():.2f}")
    print(f"运行时间: {elapsed:.4f}s")

    nav = pf.value()
    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, "rotation_vbt_signals_nav.csv")
    nav.to_csv(out_path, encoding="utf-8-sig")
    print(f"净值已保存: {out_path}")


if __name__ == "__main__":
    main()
