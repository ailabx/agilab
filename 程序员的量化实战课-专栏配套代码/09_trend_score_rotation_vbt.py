"""
趋势评分轮动策略（vectorbt from_orders 版）：纳指100 / 红利低波 / 创业板 / 黄金

与 09_trend_score_rotation.py 对应，使用 vectorbt 的 from_orders + targetpercent 实现，
用于和自研 engine 做对照。

规则：
- 趋势评分 = 年化收益率 × R平方（对收盘价取 log 后做滚动加权线性回归）；
- 每天取评分最高的标的满仓（权重 1.0）；
- 若所有标的评分 <= 0，则空仓（权重 0）。
"""
import argparse
import os
import time

import numpy as np
import pandas as pd


DEFAULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

NAME_MAP = {
    "513100.SH.csv": "纳指100",
    "512890.SH.csv": "红利低波",
    "159915.SZ.csv": "创业板",
    "518880.SH.csv": "黄金",
}


def parse_args():
    parser = argparse.ArgumentParser(description="趋势评分轮动回测（vectorbt）")
    parser.add_argument(
        "--symbols",
        type=str,
        default="513100.SH.csv,512890.SH.csv,159915.SZ.csv,518880.SH.csv",
        help="标的 CSV 文件名，逗号分隔",
    )
    parser.add_argument("--period", type=int, default=25, help="趋势评分回归窗口")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DIR, help="CSV 数据目录")
    parser.add_argument("--outdir", type=str, default=DEFAULT_DIR, help="输出目录")
    parser.add_argument("--fee", type=float, default=0.0001, help="交易成本，默认万分之一")
    return parser.parse_args()


def load_asset(path: str) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["date"], usecols=["date", "close"])
    df = df.sort_values("date").drop_duplicates("date").set_index("date")
    return df["close"]


def calc_trend_score(prices: pd.DataFrame, period: int, weighted: bool) -> pd.DataFrame:
    """计算趋势评分：年化收益率 × R平方。"""
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
    """生成目标权重矩阵：评分最高者满仓，全部 <= 0 时空仓。"""
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for date in score.dropna().index:
        row = score.loc[date]
        if row.max() > 0:
            weights.loc[date, row.idxmax()] = 1.0
    return weights


def run_vbt(prices: pd.DataFrame, weights_target: pd.DataFrame, fee: float):
    import vectorbt as vbt

    vbt.settings.returns["year_freq"] = "252 days"
    vbt.settings.returns["defaults"]["risk_free"] = 0.02 / 252

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

    results = {}
    for weighted, tag in ((False, "trend_score"), (True, "trend_score2")):
        score = calc_trend_score(prices, args.period, weighted)
        weights = generate_weights_trend(prices, score)
        pf = run_vbt(prices, weights, args.fee)
        results[tag] = pf

    elapsed = time.time() - start_time

    print(f"=== vectorbt 趋势评分轮动回测（period={args.period}）===")
    print(f"{'指标':<12} {'评分(等权)':>12} {'评分(加权)':>12}")
    print(f"{'总收益':<12} {results['trend_score'].total_return():>11.2%} {results['trend_score2'].total_return():>11.2%}")
    print(f"{'年化收益':<12} {results['trend_score'].annualized_return():>11.2%} {results['trend_score2'].annualized_return():>11.2%}")
    print(f"{'最大回撤':<12} {results['trend_score'].max_drawdown():>11.2%} {results['trend_score2'].max_drawdown():>11.2%}")
    print(f"{'夏普':<12} {results['trend_score'].sharpe_ratio():>11.2f} {results['trend_score2'].sharpe_ratio():>11.2f}")
    print(f"运行时间: {elapsed:.4f}s")

    os.makedirs(args.outdir, exist_ok=True)
    for tag, pf in results.items():
        nav = pf.value()
        nav.to_csv(os.path.join(args.outdir, f"{tag}_vbt_nav.csv"), encoding="utf-8-sig")
        print(f"净值已保存: {args.outdir}/{tag}_vbt_nav.csv")


if __name__ == "__main__":
    main()
