"""
大小盘轮动（vectorbt from_orders 版）：沪深300 vs 创业板
规则：每天取 roc(20) 较大者，top=1 满仓；若两者 roc(20) 均 < 0，空仓。

注意：
from_orders + size_type='targetpercent' 在换仓日（同一 bar 内卖出旧资产、买入新资产）
会将买入操作推迟到下一个 bar，导致实际持仓比目标权重慢一天。
这是 vectorbt 的固有限制，本脚本保留作为对比实现。
"""
import argparse
import os
import time

import pandas as pd


DEFAULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def parse_args():
    parser = argparse.ArgumentParser(description="大小盘轮动回测（vectorbt）")
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


def main():
    args = parse_args()
    start_time = time.time()

    symbols = [s.strip() for s in args.symbols.split(",")]
    prices = pd.DataFrame({s: load_asset(os.path.join(args.data_dir, s)) for s in symbols})
    prices = prices.dropna()

    import vectorbt as vbt

    # 与我们的 calc_metrics 口径一致
    vbt.settings.returns['year_freq'] = '252 days'
    vbt.settings.returns['defaults']['risk_free'] = 0.02 / 252

    weights_target = generate_target_weights(prices, args.roc_period)

    # 与自研 engine 对齐：T 日收盘根据目标权重满仓买入目标资产。
    # 使用 from_orders + targetpercent 在单一资产满仓时等效于 engine，
    # 但需保证 weights 只在变化时才真正产生交易（engine 侧已做同样处理）。
    pf = vbt.Portfolio.from_orders(
        close=prices,
        size=weights_target,
        size_type="targetpercent",
        direction="longonly",
        freq="1d",
        fees=args.fee,
        init_cash=10000.0,
        cash_sharing=True,
    )

    elapsed = time.time() - start_time

    print(f"=== vectorbt 大小盘轮动回测 ===")
    print(f"总收益: {pf.total_return():.2%}")
    print(f"年化收益: {pf.annualized_return():.2%}")
    print(f"最大回撤: {pf.max_drawdown():.2%}")
    print(f"夏普: {pf.sharpe_ratio():.2f}")
    print(f"运行时间: {elapsed:.4f}s")

    nav = pf.value()
    os.makedirs(args.outdir, exist_ok=True)
    nav.to_csv(os.path.join(args.outdir, "rotation_vbt_nav.csv"), encoding="utf-8-sig")
    print(f"净值已保存: {os.path.join(args.outdir, 'rotation_vbt_nav.csv')}")


if __name__ == "__main__":
    main()
