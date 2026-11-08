"""
四资产 ROC 轮动策略（持有到负版本）

规则：
- 每天计算 4 个标的的 roc(20)；
- 空仓时，选 roc(20) 最大且 > 0 的那一只，满仓买入；
- 持仓时，只要当前持有标的的 roc(20) >= 0，就继续持有（不再切换）；
- 当前持有标的的 roc(20) < 0 时，卖出空仓；
- 空仓期间继续保持现金，直到再次出现 roc>0 的标的才买入。

数据：Tushare 场外基金复权净值（已落盘到 data/）
  270042.OF  纳指100
  007466.OF  红利低波
  110026.OF  创业板
  000218.OF  黄金
"""
import argparse
import os
import time

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
    parser = argparse.ArgumentParser(description="四资产 ROC 轮动回测（持有到负）")
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
    - 空仓 + 有标的 roc > 0：买入 roc 最大者；
    - 持仓 + 当前标的 roc >= 0：继续持有；
    - 当前标的 roc < 0：卖出空仓。
    """
    roc = prices / prices.shift(roc_period) - 1.0
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    current_asset = None
    for i, date in enumerate(prices.index):
        if i < roc_period:
            continue
        r = roc.loc[date]

        if current_asset is not None:
            # 持有中：若当前标的 roc < 0，卖出
            if r[current_asset] < 0:
                current_asset = None
            else:
                weights.loc[date, current_asset] = 1.0
                continue

        # 空仓：找 roc 最大者，只有 > 0 才买入
        if r.max() > 0:
            best = r.idxmax()
            current_asset = best
            weights.loc[date, best] = 1.0

    return weights


def main():
    args = parse_args()
    start_time = time.time()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    prices = pd.DataFrame({s: load_asset(os.path.join(args.data_dir, s)) for s in symbols})
    prices = prices.dropna()

    weights_target = generate_weights(prices, args.roc_period)

    # 只在目标权重变化的日子调仓，避免空仓/满仓日重复扣费
    rebalance_mask = weights_target.ne(weights_target.shift(1)).any(axis=1)
    rebalance_dates = set(weights_target[rebalance_mask].index)
    nav, weights = backtest(prices, weights_target, args.fee, rebalance_dates=rebalance_dates)
    metrics = calc_metrics(nav, ann_factor=args.ann_factor, risk_free=args.risk_free)

    elapsed = time.time() - start_time

    print(f"=== 四资产 ROC 轮动回测（持有到负）===")
    print(f"区间: {metrics['start_date']} ~ {metrics['end_date']}")
    print(f"标的: 纳指100 / 红利低波 / 创业板 / 黄金")
    for k, v in metrics.items():
        if k in ("total_return", "annualized_return", "volatility", "max_drawdown"):
            print(f"{k}: {v:.2%}")
        elif k in ("sharpe", "calmar"):
            print(f"{k}: {v:.2f}")
    print(f"运行时间: {elapsed:.4f}s")

    os.makedirs(args.outdir, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(
        os.path.join(args.outdir, "quad_rotation_hold_metrics.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    plot_path = plot_nav_and_weights(
        prices,
        nav,
        weights,
        args.outdir,
        title="四资产 ROC 轮动回测（持有到负）",
        name_map={s: NAME_MAP.get(s, s) for s in symbols},
        filename="quad_rotation_hold.png",
    )
    print(f"图表已保存: {plot_path}")


if __name__ == "__main__":
    main()
