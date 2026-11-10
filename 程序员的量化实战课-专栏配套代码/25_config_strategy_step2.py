"""
配置化 ETF 策略：优化二 —— ATR 初始止损 + 国债替代空仓

用法：
    python 25_config_strategy_step2.py --top-k=4 --period=20 --max-weight=0.25 --stop-atr=3.0 --score-threshold=0.05
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

DEFAULT_SYMBOLS = [
    "510300.SH.csv", "588000.SH.csv", "510880.SH.csv", "159980.SZ.csv", "159985.SZ.csv",
    "513100.SH.csv", "513520.SH.csv", "513500.SH.csv", "513030.SH.csv", "501018.SH.csv",
    "513180.SH.csv", "159915.SZ.csv", "512890.SH.csv", "511090.SH.csv", "510180.SH.csv",
]

CASH_SUBSTITUTE = "511090.SH.csv"  # 国债10Y


def parse_args():
    parser = argparse.ArgumentParser(description="配置化 ETF 策略：ATR 初始止损 + 国债替代")
    parser.add_argument("--symbols", type=str, default=",".join(DEFAULT_SYMBOLS), help="标的 CSV 文件名")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DIR, help="CSV 数据目录")
    parser.add_argument("--outdir", type=str, default=DEFAULT_DIR, help="输出目录")
    parser.add_argument("--capital", type=float, default=1000000.0, help="初始资金")
    parser.add_argument("--period", type=int, default=20, help="trend_score2 周期")
    parser.add_argument("--top-k", type=int, default=4, help="Top K")
    parser.add_argument("--max-weight", type=float, default=0.25, help="单标的上限")
    parser.add_argument("--fee", type=float, default=0.0001, help="交易成本")
    parser.add_argument("--ann-factor", type=int, default=252, help="年化因子")
    parser.add_argument("--risk-free", type=float, default=0.02, help="无风险利率")
    parser.add_argument("--start-date", type=str, default="2020-01-01", help="开始日期")
    parser.add_argument("--end-date", type=str, default="", help="结束日期")
    parser.add_argument("--atr-period", type=int, default=14, help="ATR 周期")
    parser.add_argument("--stop-atr", type=float, default=3.0, help="止损 ATR 倍数")
    parser.add_argument("--score-threshold", type=float, default=0.05, help="入选最低 trend_score2 阈值")
    parser.add_argument("--substitute-ratio", type=float, default=1.0, help="空仓国债替代比例")
    return parser.parse_args()


def load_ohlc(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").drop_duplicates("date").set_index("date")
    return df


def calc_trend_score2(prices: pd.DataFrame, period: int) -> pd.DataFrame:
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


def cs_zscore(df: pd.DataFrame) -> pd.DataFrame:
    mean = df.mean(axis=1).values[:, None]
    std = df.std(axis=1).values[:, None]
    return (df - mean) / std


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


def backtest(ohlc: dict, score: pd.DataFrame, roc1: pd.DataFrame, atr: pd.DataFrame, args):
    close = ohlc["close"]
    dates = close.index
    symbols = close.columns.tolist()

    cash = args.capital
    positions = {}

    nav = pd.Series(index=dates, dtype=float)
    weights = pd.DataFrame(0.0, index=dates, columns=symbols)
    hold_count = pd.Series(index=dates, dtype=int)
    active_equity = pd.Series(index=dates, dtype=float)
    stop_records = []

    for i, date in enumerate(dates):
        today_close = close.loc[date]
        today_atr = atr.loc[date]

        # 1. 初始止损检查
        for sym in list(positions.keys()):
            px = today_close[sym]
            if px <= positions[sym]["stop"]:
                cash += positions[sym]["shares"] * px * (1 - args.fee)
                stop_records.append((date, sym, "stop"))
                del positions[sym]

        # 2. sell rules: trend_score2 >= 5 或 <= 0
        for sym in list(positions.keys()):
            s = score.loc[date, sym]
            if pd.notna(s) and (s >= 5 or s <= 0):
                px = today_close[sym]
                cash += positions[sym]["shares"] * px * (1 - args.fee)
                del positions[sym]

        # 3. 选股
        row = score.loc[date]
        row = row.dropna()
        buy_mask = roc1.loc[date] >= -0.05
        row = row[buy_mask]
        row = row.dropna()

        if len(row) > 0:
            selected = row.sort_values(ascending=False).head(args.top_k)
            targets = selected[selected >= args.score_threshold].index.tolist()
        else:
            targets = []

        # 4. 调仓
        # 清仓不在目标中的持仓
        for sym in list(positions.keys()):
            if sym not in targets:
                px = today_close[sym]
                cash += positions[sym]["shares"] * px * (1 - args.fee)
                del positions[sym]

        # 计算权益
        equity = cash + sum(positions[sym]["shares"] * today_close[sym] for sym in positions)

        # 目标等权
        if targets:
            w = 1.0 / len(targets)
            w = min(w, args.max_weight)
            for sym in targets:
                px = today_close[sym]
                target_amount = equity * w
                if sym in positions:
                    current_amount = positions[sym]["shares"] * px
                    delta = target_amount - current_amount
                    if delta > 0:
                        cash -= delta * (1 + args.fee)
                        positions[sym]["shares"] += delta / px
                    elif delta < 0:
                        cash += -delta * (1 - args.fee)
                        positions[sym]["shares"] -= -delta / px
                else:
                    amount = target_amount / (1 + args.fee)
                    cash -= amount
                    atr_val = today_atr[sym]
                    stop = px - args.stop_atr * atr_val if pd.notna(atr_val) and atr_val > 0 else px * 0.9
                    positions[sym] = {
                        "entry": px,
                        "stop": stop,
                        "shares": amount / px,
                    }
        else:
            # 空仓替代：买入国债
            if args.substitute_ratio > 0 and CASH_SUBSTITUTE in symbols:
                px = today_close[CASH_SUBSTITUTE]
                target_amount = equity * args.substitute_ratio
                if CASH_SUBSTITUTE in positions:
                    current_amount = positions[CASH_SUBSTITUTE]["shares"] * px
                    delta = target_amount - current_amount
                    if delta > 0:
                        cash -= delta * (1 + args.fee)
                        positions[CASH_SUBSTITUTE]["shares"] += delta / px
                    elif delta < 0:
                        cash += -delta * (1 - args.fee)
                        positions[CASH_SUBSTITUTE]["shares"] -= -delta / px
                else:
                    amount = target_amount / (1 + args.fee)
                    cash -= amount
                    positions[CASH_SUBSTITUTE] = {
                        "entry": px,
                        "stop": px * 0.9,
                        "shares": amount / px,
                    }

        # 记录
        equity = cash + sum(positions[sym]["shares"] * today_close[sym] for sym in positions)
        nav.loc[date] = equity
        for sym in symbols:
            if sym in positions:
                weights.loc[date, sym] = positions[sym]["shares"] * today_close[sym] / equity
        hold_count.loc[date] = len(positions)
        active_equity.loc[date] = len([s for s in targets if s != CASH_SUBSTITUTE])

    return nav, weights, hold_count, active_equity, stop_records


def main():
    args = parse_args()
    start_time = time.time()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    frames = {col: [] for col in ["open", "high", "low", "close", "vol"]}
    for s in symbols:
        path = os.path.join(args.data_dir, s)
        if not os.path.exists(path):
            print(f"警告：文件不存在 {path}")
            continue
        df = load_ohlc(path)
        for col in frames:
            if col in df.columns:
                frames[col].append(df[col].rename(s))
            else:
                frames[col].append(pd.Series(np.nan, index=df.index).rename(s))

    ohlc = {}
    for col in frames:
        ohlc[col] = pd.concat(frames[col], axis=1, sort=True).dropna()

    close = ohlc["close"]
    if args.start_date:
        close = close[close.index >= pd.to_datetime(args.start_date)]
    if args.end_date:
        close = close[close.index <= pd.to_datetime(args.end_date)]

    for col in ohlc:
        ohlc[col] = ohlc[col].loc[close.index]

    print(f"数据区间: {close.index[0].date()} ~ {close.index[-1].date()}")

    score = calc_trend_score2(close, args.period)
    zscore = cs_zscore(score)
    roc1 = close.pct_change(1)
    atr = calc_atr(ohlc, args.atr_period)

    nav, weights, hold_count, active_equity, stops = backtest(ohlc, zscore, roc1, atr, args)
    metrics = calc_metrics(nav, ann_factor=args.ann_factor, risk_free=args.risk_free)

    print(f"=== 配置化 ETF 策略：ATR 初始止损 + 国债替代 ===")
    print(f"区间: {metrics['start_date']} ~ {metrics['end_date']}")
    for k in ["total_return", "annualized_return", "volatility", "max_drawdown"]:
        print(f"{k}: {metrics[k]:.2%}")
    for k in ["sharpe", "calmar"]:
        print(f"{k}: {metrics[k]:.2f}")
    print(f"止损次数: {len(stops)}")
    print(f"平均持仓: {hold_count.mean():.2f} 只")
    print(f"平均主动权益仓位: {active_equity.mean():.2f} 只")

    os.makedirs(args.outdir, exist_ok=True)
    pd.DataFrame([{"strategy": "config_step2", **metrics}]).to_csv(
        os.path.join(args.outdir, "config_step2_metrics.csv"), index=False, encoding="utf-8-sig"
    )

    fig, axes = plt.subplots(4, 1, figsize=(14, 14), gridspec_kw={"height_ratios": [2, 1, 1, 1]})
    ax1 = axes[0]
    ax1.plot(nav.index, nav / nav.iloc[0], label="组合净值", linewidth=2)
    ax1.set_title("配置化 ETF 策略：ATR 初始止损 + 国债替代")
    ax1.set_ylabel("归一化净值")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = axes[1]
    ax2.stackplot(weights.index, *[weights[col] for col in weights.columns], labels=weights.columns, alpha=0.7)
    ax2.set_ylabel("权重")
    ax2.set_ylim(0, 1)
    ax2.legend(loc="upper left", fontsize=6, ncol=3)
    ax2.grid(True, linestyle="--", alpha=0.5)

    ax3 = axes[2]
    ax3.plot(hold_count.index, hold_count, drawstyle="steps-post", linewidth=1)
    ax3.set_ylabel("持仓数量")
    ax3.grid(True, linestyle="--", alpha=0.5)

    ax4 = axes[3]
    ax4.plot(active_equity.index, active_equity, drawstyle="steps-post", linewidth=1, color="C2")
    ax4.set_ylabel("主动权益仓位数")
    ax4.set_ylim(-0.1, args.top_k + 0.5)
    ax4.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plot_path = os.path.join(args.outdir, "config_step2.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"图表已保存: {plot_path}")
    print(f"运行时间: {time.time() - start_time:.4f}s")


if __name__ == "__main__":
    main()
