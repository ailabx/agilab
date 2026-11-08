"""
全市场 ETF 轮动策略 v3：大类代表 ETF 轮动

每个大类只选一只：
- A股成长：创业板
- A股价值：红利低波
- 跨境：纳指100
- 商品：黄金
- 债券：国债ETF

规则：
- 每天 ROC(60) 选最强，满仓；
- 若所有大类 ROC < 0，空仓；
- 只在目标权重变化日调仓。
"""
import argparse
import os
import time

import pandas as pd

from engine import backtest
from utils import calc_metrics, plot_nav_and_weights


DEFAULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

NAME_MAP = {
    "159915.SZ.csv": "创业板",
    "512890.SH.csv": "红利低波",
    "513100.SH.csv": "纳指100",
    "518880.SH.csv": "黄金",
    "511010.SH.csv": "国债ETF",
}

DEFAULT_SYMBOLS = ",".join(NAME_MAP.keys())


def parse_args():
    parser = argparse.ArgumentParser(description="大类代表 ETF 轮动")
    parser.add_argument(
        "--symbols",
        type=str,
        default=DEFAULT_SYMBOLS,
        help="标的 CSV 文件名，逗号分隔",
    )
    parser.add_argument("--roc-period", type=int, default=60, help="ROC 计算周期")
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


def generate_weights(prices: pd.DataFrame, roc_period: int) -> pd.DataFrame:
    roc = prices / prices.shift(roc_period) - 1.0
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    for i, date in enumerate(prices.index):
        if i < roc_period:
            continue
        r = roc.loc[date]
        if r.max() >= 0:
            weights.loc[date, r.idxmax()] = 1.0

    return weights


def main():
    args = parse_args()
    start_time = time.time()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    prices = pd.DataFrame({s: load_asset(os.path.join(args.data_dir, s)) for s in symbols})
    prices = prices.dropna()

    weights = generate_weights(prices, args.roc_period)

    rebalance_mask = weights.ne(weights.shift(1)).any(axis=1)
    rebalance_dates = set(weights[rebalance_mask].index)
    nav, weights_bt = backtest(prices, weights, args.fee, rebalance_dates=rebalance_dates)
    metrics = calc_metrics(nav, ann_factor=args.ann_factor, risk_free=args.risk_free)

    elapsed = time.time() - start_time

    print(f"=== 大类代表 ETF 轮动回测（ROC{args.roc_period}）===")
    print(f"区间: {metrics['start_date']} ~ {metrics['end_date']}")
    print(f"标的数: {len(symbols)}")
    for k, v in metrics.items():
        if k in ("total_return", "annualized_return", "volatility", "max_drawdown"):
            print(f"{k}: {v:.2%}")
        elif k in ("sharpe", "calmar"):
            print(f"{k}: {v:.2f}")
    print(f"运行时间: {elapsed:.4f}s")

    os.makedirs(args.outdir, exist_ok=True)
    pd.DataFrame([{"strategy": "category_etf_roc", **metrics}]).to_csv(
        os.path.join(args.outdir, "category_etf_roc_metrics.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    plot_path = plot_nav_and_weights(
        prices,
        nav,
        weights_bt,
        args.outdir,
        title="大类代表 ETF 轮动回测（ROC60）",
        name_map={s: NAME_MAP.get(s, s) for s in symbols},
        filename="category_etf_rotation.png",
    )
    print(f"图表已保存: {plot_path}")


if __name__ == "__main__":
    main()
