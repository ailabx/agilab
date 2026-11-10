"""
ETF 趋势轮动：市场过滤阈值 + 国债替代比例网格测试

用法：
    python 21_market_filter_grid.py --top-n=2 --period=25
"""
import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import calc_metrics


DEFAULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

NAME_MAP = {
    "510300.SH.csv": "沪深300",
    "159915.SZ.csv": "创业板",
    "512890.SH.csv": "红利低波",
    "513100.SH.csv": "纳指100",
    "513500.SH.csv": "标普500",
    "513000.SH.csv": "日经225",
    "513520.SH.csv": "日经225-2",
    "513030.SH.csv": "德国DAX",
    "513050.SH.csv": "恒生科技",
    "159920.SZ.csv": "恒生ETF",
    "518880.SH.csv": "黄金",
    "511010.SH.csv": "国债5Y",
    "511020.SH.csv": "国债5-10Y",
}

DEFAULT_SYMBOLS = ",".join(NAME_MAP.keys())
CASH_SUBSTITUTE = "511020.SH.csv"


def parse_args():
    parser = argparse.ArgumentParser(description="ETF 趋势轮动：市场过滤网格测试")
    parser.add_argument("--symbols", type=str, default=DEFAULT_SYMBOLS, help="标的 CSV 文件名，逗号分隔")
    parser.add_argument("--period", type=int, default=25, help="趋势评分周期")
    parser.add_argument("--top-n", type=int, default=2, help="Top N 等权")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DIR, help="CSV 数据目录")
    parser.add_argument("--outdir", type=str, default=DEFAULT_DIR, help="输出目录")
    parser.add_argument("--fee", type=float, default=0.0001, help="交易成本")
    parser.add_argument("--ann-factor", type=int, default=252, help="年化因子")
    parser.add_argument("--risk-free", type=float, default=0.02, help="无风险利率")
    parser.add_argument("--atr-period", type=int, default=14, help="ATR 计算周期")
    parser.add_argument("--stop-atr", type=float, default=3.0, help="入场止损 ATR 倍数")
    parser.add_argument("--risk-per-trade", type=float, default=0.02, help="单笔风险占权益比例")
    parser.add_argument("--max-gross-exposure", type=float, default=0.80, help="总仓位上限")
    parser.add_argument("--max-single-weight", type=float, default=0.30, help="单票最大权重")
    parser.add_argument("--fuse-dd", type=float, default=0.15, help="熔断回撤阈值")
    parser.add_argument("--fuse-cooldown", type=int, default=20, help="熔断冷却交易日")
    parser.add_argument("--rebalance-freq", type=int, default=1, help="调仓频率（交易日）")
    parser.add_argument("--score-thresholds", type=str, default="0.0,0.02,0.05,0.10,0.15,0.20", help="trend_score 阈值网格")
    parser.add_argument("--substitute-ratios", type=str, default="0.0,0.5,0.8,1.0", help="空仓时国债替代比例网格")
    return parser.parse_args()


def load_ohlc(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").drop_duplicates("date").set_index("date")
    return df


def calc_trend_score(prices: pd.DataFrame, period: int) -> pd.DataFrame:
    x = np.arange(period, dtype=float)
    w = np.linspace(1.0, 2.0, period) ** 2
    w_sum = w.sum()
    x_bar = (w * x).sum() / w_sum
    dx = x - x_bar
    variance_x = (w * dx ** 2).sum()

    def _score(window: np.ndarray) -> float:
        y = np.log(window)
        y_bar = (w * y).sum() / w_sum
        slope = (w * dx * (y - y_bar)).sum() / variance_x
        intercept = y_bar - slope * x_bar
        y_pred = slope * np.arange(period) + intercept
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


def calc_atr(ohlc: dict, period: int) -> pd.DataFrame:
    high = ohlc["high"]
    low = ohlc["low"]
    close = ohlc["close"]
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=0)
    tr = tr.groupby(level=0).max()
    atr = tr.rolling(window=period).mean()
    return atr


def backtest_with_filter(ohlc: dict, score: pd.DataFrame, args, score_threshold: float, substitute_ratio: float):
    close = ohlc["close"]
    dates = close.index
    symbols = close.columns.tolist()
    atr = calc_atr(ohlc, args.atr_period)

    initial_capital = 10000.0
    cash = initial_capital
    equity_peak = initial_capital
    in_fuse = False
    fuse_end = None
    recovery_level = 1.0

    positions = {}
    nav = pd.Series(index=dates, dtype=float)

    for i, date in enumerate(dates):
        today_close = close.loc[date]
        today_atr = atr.loc[date]

        # 1. 止损
        for sym in list(positions.keys()):
            pos = positions[sym]
            px = today_close[sym]
            if px <= pos["stop"]:
                cash += pos["shares"] * px * (1 - args.fee)
                del positions[sym]

        # 2. 权益
        equity = cash + sum(positions[sym]["shares"] * today_close[sym] for sym in positions)
        equity_peak = max(equity_peak, equity)

        # 3. 熔断
        dd = 1.0 - equity / equity_peak
        if in_fuse:
            if date >= fuse_end:
                in_fuse = False
                recovery_level = 0.5
        else:
            if dd >= args.fuse_dd:
                in_fuse = True
                fuse_end = date + pd.Timedelta(days=args.fuse_cooldown)
                recovery_level = 0.0

        # 4. 调仓
        is_rebalance = (i % args.rebalance_freq == 0)

        if is_rebalance and not in_fuse:
            row = score.loc[date]
            row = row.dropna()
            row = row[row > 0]
            if len(row) > 0:
                selected = row.sort_values(ascending=False).head(args.top_n)
                targets = selected[selected >= score_threshold].index.tolist()
            else:
                targets = []

            raw_amounts = {}
            if targets:
                for sym in targets:
                    px = today_close[sym]
                    atr_val = today_atr[sym]
                    if pd.isna(atr_val) or atr_val <= 0:
                        continue
                    stop_dist = args.stop_atr * atr_val
                    risk_amount = equity * args.risk_per_trade
                    amount = risk_amount / (stop_dist / px)
                    amount = min(amount, equity * args.max_single_weight)
                    raw_amounts[sym] = amount
            else:
                if substitute_ratio > 0 and CASH_SUBSTITUTE in symbols:
                    raw_amounts[CASH_SUBSTITUTE] = equity * args.max_gross_exposure * substitute_ratio

            target_weights = pd.Series(0.0, index=symbols)
            if raw_amounts:
                total_raw = sum(raw_amounts.values())
                max_exposure = equity * args.max_gross_exposure * recovery_level
                scale = min(1.0, max_exposure / total_raw)
                for sym, amount in raw_amounts.items():
                    target_weights[sym] = amount * scale / equity

            # 清仓不在目标中的持仓
            for sym in list(positions.keys()):
                if sym not in target_weights or target_weights[sym] <= 0:
                    cash += positions[sym]["shares"] * today_close[sym] * (1 - args.fee)
                    del positions[sym]

            # 调整持仓
            for sym in target_weights[target_weights > 0].index:
                px = today_close[sym]
                target_amount = target_weights[sym] * equity
                if sym in positions:
                    pos = positions[sym]
                    current_amount = pos["shares"] * px
                    delta = target_amount - current_amount
                    if delta > 0:
                        cash -= delta * (1 + args.fee)
                        pos["shares"] += delta / px
                    elif delta < 0:
                        cash += -delta * (1 - args.fee)
                        pos["shares"] -= -delta / px
                    pos["highest"] = max(pos["highest"], px)
                else:
                    shares = target_amount / px * (1 - args.fee)
                    cash -= target_amount
                    positions[sym] = {
                        "entry": px,
                        "stop": px - args.stop_atr * today_atr[sym],
                        "highest": px,
                        "shares": shares,
                    }

        # 5. 恢复仓位
        if not in_fuse and recovery_level < 1.0 and equity >= equity_peak * 0.99:
            recovery_level = min(1.0, recovery_level + 0.25)

        # 6. 记录净值
        equity = cash + sum(positions[sym]["shares"] * today_close[sym] for sym in positions)
        nav.loc[date] = equity

    return nav


def main():
    args = parse_args()
    start_time = time.time()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    frames = {col: [] for col in ["open", "high", "low", "close", "vol"]}
    for s in symbols:
        path = os.path.join(args.data_dir, s)
        df = load_ohlc(path)
        for col in frames:
            frames[col].append(df[col].rename(s))

    ohlc = {}
    for col in frames:
        ohlc[col] = pd.concat(frames[col], axis=1, sort=True).dropna()

    close = ohlc["close"]
    print(f"数据区间: {close.index[0].date()} ~ {close.index[-1].date()}")

    score = calc_trend_score(close, args.period)

    score_thresholds = [float(x) for x in args.score_thresholds.split(",")]
    substitute_ratios = [float(x) for x in args.substitute_ratios.split(",")]

    records = []
    for st in score_thresholds:
        for sr in substitute_ratios:
            print(f"\n=== score_threshold={st}, substitute_ratio={sr} ===")
            nav = backtest_with_filter(ohlc, score, args, st, sr)
            metrics = calc_metrics(nav, ann_factor=args.ann_factor, risk_free=args.risk_free)
            print(f"年化: {metrics['annualized_return']:.2%}, 回撤: {metrics['max_drawdown']:.2%}, 夏普: {metrics['sharpe']:.2f}")
            records.append({
                "score_threshold": st,
                "substitute_ratio": sr,
                **metrics,
            })

    df = pd.DataFrame(records)
    df = df[["score_threshold", "substitute_ratio", "annualized_return", "max_drawdown", "sharpe", "calmar", "volatility"]]

    os.makedirs(args.outdir, exist_ok=True)
    df.to_csv(os.path.join(args.outdir, "market_filter_grid.csv"), index=False, encoding="utf-8-sig")

    print("\n=== 汇总（按夏普排序）===")
    print(df.sort_values("sharpe", ascending=False).head(10).to_string(index=False))

    # 热力图：score_threshold x substitute_ratio，颜色为夏普
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, metric in zip(axes.flat, ["annualized_return", "sharpe", "max_drawdown", "calmar"]):
        pivot = df.pivot_table(values=metric, index="score_threshold", columns="substitute_ratio", aggfunc="mean")
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{x:.0%}" for x in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{x:.2f}" for x in pivot.index])
        ax.set_xlabel("国债替代比例")
        ax.set_ylabel("score_threshold")
        ax.set_title(metric)
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plot_path = os.path.join(args.outdir, "market_filter_grid.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\n图表已保存: {plot_path}")
    print(f"CSV 已保存: {os.path.join(args.outdir, 'market_filter_grid.csv')}")
    print(f"运行时间: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()
