"""
趋势评分轮动策略：纳指100 / 红利低波 / 创业板 / 黄金

趋势评分 = 年化收益率 × R平方（对收盘价取 log 后做滚动加权线性回归）：
- 斜率 slope：对数价格的回归斜率，年化收益率 = exp(slope * 250) - 1；
- R平方：拟合优度，衡量趋势的"稳定程度"，clip 到 [0, 1]；
- 综合评分 = annualized_return × r_squared。

两个版本：
- trend_score  ：等权回归（近期与远期权重相同）；
- trend_score2 ：加权回归（权重线性递增 1→2，再平方，近期权重更高，与"五福"口径一致）。

轮动规则：
- 每天取评分最高的标的满仓（权重 1.0）；
- 若所有标的评分 <= 0，则空仓；
- 只在目标权重变化的日子调仓，避免重复扣费。

对照基准：ROC(20) 轮动（与 08_quad_rotation.py 的 ROC 版本一致）。

数据：Tushare 场外基金复权净值（已落盘到 data/）
  270042.OF  广发纳斯达克100ETF联接(QDII)-A -> 纳指100
  007466.OF  华泰柏瑞中证红利低波动ETF联接-A -> 红利低波
  110026.OF  易方达创业板ETF联接-A          -> 创业板
  000218.OF  国泰黄金ETF联接-A              -> 黄金
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
    "513100.SH.csv": "纳指100",
    "512890.SH.csv": "红利低波",
    "159915.SZ.csv": "创业板",
    "518880.SH.csv": "黄金",
}


def parse_args():
    parser = argparse.ArgumentParser(description="趋势评分轮动回测（自研）")
    parser.add_argument(
        "--symbols",
        type=str,
        default="513100.SH.csv,512890.SH.csv,159915.SZ.csv,518880.SH.csv",
        help="标的 CSV 文件名，逗号分隔",
    )
    parser.add_argument("--period", type=int, default=25, help="趋势评分回归窗口")
    parser.add_argument("--roc-period", type=int, default=20, help="ROC 基准策略周期")
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


def calc_trend_score(prices: pd.DataFrame, period: int, weighted: bool) -> pd.DataFrame:
    """
    对每个标的在 period 日滚动窗口内做 log 价格的加权线性回归：
    综合评分 = 年化收益率 × R平方。
    weighted=True 时权重线性递增（1→2，平方后使用），近期权重更高。
    """
    x = np.arange(period, dtype=float)
    w = np.linspace(1.0, 2.0, period) ** 2 if weighted else np.ones(period)
    w_sum = w.sum()
    x_bar = (w * x).sum() / w_sum
    dx = x - x_bar
    variance_x = (w * dx ** 2).sum()

    def _score(window: np.ndarray) -> float:
        y = np.log(window)
        y_bar = (w * y).sum() / w_sum
        slope = (w * dx * (y - y_bar)).sum() / variance_x
        intercept = y_bar - slope * x_bar

        y_pred = slope * x + intercept
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


def generate_weights_trend(prices: pd.DataFrame, score: pd.DataFrame) -> pd.DataFrame:
    """
    生成目标权重矩阵（趋势评分版本）。
    每天取评分最高者，权重 1.0；若全部 <= 0，权重 0（空仓）。
    """
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    for date in score.dropna().index:
        row = score.loc[date]
        if row.max() > 0:
            weights.loc[date, row.idxmax()] = 1.0

    return weights


def generate_weights_roc(prices: pd.DataFrame, roc_period: int) -> pd.DataFrame:
    """生成目标权重矩阵（ROC 基准版本）。"""
    roc = prices / prices.shift(roc_period) - 1.0
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    for i, date in enumerate(prices.index):
        if i < roc_period:
            continue
        r = roc.loc[date]
        if r.max() >= 0:
            weights.loc[date, r.idxmax()] = 1.0

    return weights


def run_backtest(prices: pd.DataFrame, weights_target: pd.DataFrame, fee: float):
    """只在目标权重变化的日子调仓，返回净值与权重。"""
    rebalance_mask = weights_target.ne(weights_target.shift(1)).any(axis=1)
    rebalance_dates = set(weights_target[rebalance_mask].index)
    return backtest(prices, weights_target, fee, rebalance_dates=rebalance_dates)


def print_compare(metrics_list: list, args) -> None:
    """打印各版本的指标对比。"""
    names = ["评分(等权)", "评分(加权)", "ROC基准"]
    print(f"=== 趋势评分轮动回测（period={args.period}，roc_period={args.roc_period}）===")
    print(f"区间: {metrics_list[0]['start_date']} ~ {metrics_list[0]['end_date']}")
    print(f"{'指标':<12} {names[0]:>12} {names[1]:>12} {names[2]:>12}")
    for key, label in (
        ("total_return", "total_return"),
        ("annualized_return", "annualized"),
        ("volatility", "volatility"),
        ("max_drawdown", "max_dd"),
        ("sharpe", "sharpe"),
        ("calmar", "calmar"),
    ):
        fmt = ".2f" if key in ("sharpe", "calmar") else ".2%"
        print(f"{label:<12} "
              f"{metrics_list[0][key]:>12{fmt}} "
              f"{metrics_list[1][key]:>12{fmt}} "
              f"{metrics_list[2][key]:>12{fmt}}")


def main():
    args = parse_args()
    start_time = time.time()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    prices = pd.DataFrame({s: load_asset(os.path.join(args.data_dir, s)) for s in symbols})
    prices = prices.dropna()

    # 趋势评分：等权 / 加权 两个版本
    results = []
    navs = {}
    for weighted, tag in ((False, "trend_score"), (True, "trend_score2")):
        score = calc_trend_score(prices, args.period, weighted)
        weights = generate_weights_trend(prices, score)
        nav, weights_bt = run_backtest(prices, weights, args.fee)
        metrics = calc_metrics(nav, ann_factor=args.ann_factor, risk_free=args.risk_free)
        results.append((tag, metrics, nav, weights_bt))

    # ROC 基准
    weights_roc = generate_weights_roc(prices, args.roc_period)
    nav_roc, _ = run_backtest(prices, weights_roc, args.fee)
    metrics_roc = calc_metrics(nav_roc, ann_factor=args.ann_factor, risk_free=args.risk_free)

    elapsed = time.time() - start_time
    print_compare([results[0][1], results[1][1], metrics_roc], args)
    print(f"运行时间: {elapsed:.4f}s")

    os.makedirs(args.outdir, exist_ok=True)
    pd.DataFrame([
        {"strategy": tag, **metrics} for tag, metrics, _, _ in results
    ] + [{"strategy": "roc", **metrics_roc}]).to_csv(
        os.path.join(args.outdir, "trend_score_metrics.csv"),
        index=False, encoding="utf-8-sig",
    )

    name_map = {s: NAME_MAP.get(s, s) for s in symbols}
    filenames = {"trend_score": "trend_score_rotation.png", "trend_score2": "trend_score_rotation_weighted.png"}
    for tag, _, nav, weights_bt in results:
        nav.to_csv(os.path.join(args.outdir, f"{tag}_nav.csv"), encoding="utf-8-sig")
        plot_path = plot_nav_and_weights(
            prices, nav, weights_bt, args.outdir,
            title=f"趋势评分轮动回测（{'加权' if tag.endswith('2') else '等权'}，period={args.period}）",
            name_map=name_map,
            filename=filenames[tag],
        )
        print(f"图表已保存: {plot_path}")


if __name__ == "__main__":
    main()
