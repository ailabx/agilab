"""
ETF 趋势轮动：止损后再买入确认条件网格测试

用法：
    python 19_confirmation_grid.py --top-n=2 --period=25
"""
import argparse
import os
import time
from collections import defaultdict

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


MODES = ["none", "ma5_20", "ma3_10", "hh5", "score_pos2", "bb_mid"]


def parse_args():
    parser = argparse.ArgumentParser(description="ETF 趋势轮动：止损后再买入确认条件网格测试")
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


def build_confirm_signals(close: pd.DataFrame, score: pd.DataFrame, mode: str):
    """
    返回 DataFrame，每个日期/标的是否满足再买入确认条件。
    """
    if mode == "none":
        return pd.DataFrame(True, index=close.index, columns=close.columns)

    confirm = pd.DataFrame(False, index=close.index, columns=close.columns)

    if mode == "ma5_20":
        ma5 = close.rolling(window=5).mean()
        ma20 = close.rolling(window=20).mean()
        confirm = (close > ma5) & (ma5 > ma20)
    elif mode == "ma3_10":
        ma3 = close.rolling(window=3).mean()
        ma10 = close.rolling(window=10).mean()
        confirm = (close > ma3) & (ma3 > ma10)
    elif mode == "hh5":
        hh5 = close.rolling(window=5).max()
        confirm = close >= hh5
    elif mode == "score_pos2":
        pos1 = score > 0
        pos2 = pos1 & pos1.shift(1)
        confirm = pos2
    elif mode == "bb_mid":
        ma20 = close.rolling(window=20).mean()
        confirm = close > ma20

    return confirm


def backtest_with_mode(ohlc: dict, score: pd.DataFrame, confirm: pd.DataFrame, args):
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
    weights = pd.DataFrame(0.0, index=dates, columns=symbols)
    total_exposure = pd.Series(index=dates, dtype=float)
    cash_ratio = pd.Series(index=dates, dtype=float)
    hold_count = pd.Series(index=dates, dtype=int)
    fuse_records = []
    stop_records = []
    waiting_confirmation = {sym: False for sym in symbols}

    for i, date in enumerate(dates):
        today_close = close.loc[date]
        today_atr = atr.loc[date]
        today_confirm = confirm.loc[date]

        # 1. 仅初始止损检查
        for sym in list(positions.keys()):
            pos = positions[sym]
            px = today_close[sym]

            if px <= pos["stop"]:
                cash += pos["shares"] * px * (1 - args.fee)
                stop_records.append((date, sym, "stop"))
                waiting_confirmation[sym] = True
                del positions[sym]

        # 2. 计算当前权益
        equity = cash + sum(positions[sym]["shares"] * today_close[sym] for sym in positions)
        equity_peak = max(equity_peak, equity)

        # 3. 熔断判断
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
                fuse_records.append((date, dd))

        # 4. 调仓日处理目标权重
        is_rebalance = (i % args.rebalance_freq == 0)

        if is_rebalance and not in_fuse:
            row = score.loc[date]
            row = row.dropna()
            row = row[row > 0]
            if len(row) > 0:
                targets = row.sort_values(ascending=False).head(args.top_n).index.tolist()
            else:
                targets = []

            raw_amounts = {}
            for sym in targets:
                # 止损后需满足确认条件才能重新买入
                if waiting_confirmation[sym] and not today_confirm[sym]:
                    continue
                px = today_close[sym]
                atr_val = today_atr[sym]
                if pd.isna(atr_val) or atr_val <= 0:
                    continue
                stop_dist = args.stop_atr * atr_val
                risk_amount = equity * args.risk_per_trade
                amount = risk_amount / (stop_dist / px)
                amount = min(amount, equity * args.max_single_weight)
                raw_amounts[sym] = amount

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

            # 调整目标持仓
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
                    if waiting_confirmation[sym]:
                        waiting_confirmation[sym] = False

        # 5. 恢复仓位水平
        if not in_fuse and recovery_level < 1.0 and equity >= equity_peak * 0.99:
            recovery_level = min(1.0, recovery_level + 0.25)

        # 6. 记录
        nav.loc[date] = equity
        te = 0.0
        for sym in symbols:
            if sym in positions:
                w = positions[sym]["shares"] * today_close[sym] / equity
                weights.loc[date, sym] = w
                te += w
            else:
                weights.loc[date, sym] = 0.0
        total_exposure.loc[date] = te
        cash_ratio.loc[date] = cash / equity
        hold_count.loc[date] = len(positions)

    return {
        "nav": nav,
        "weights": weights,
        "total_exposure": total_exposure,
        "cash_ratio": cash_ratio,
        "hold_count": hold_count,
        "fuse_records": fuse_records,
        "stop_records": stop_records,
    }


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

    records = []
    for mode in MODES:
        print(f"\n=== 运行模式: {mode} ===")
        confirm = build_confirm_signals(close, score, mode)
        result = backtest_with_mode(ohlc, score, confirm, args)
        nav = result["nav"]
        metrics = calc_metrics(nav, ann_factor=args.ann_factor, risk_free=args.risk_free)
        metrics["mode"] = mode
        metrics["fuse_count"] = len(result["fuse_records"])
        metrics["stop_count"] = len(result["stop_records"])
        metrics["avg_exposure"] = result["total_exposure"].mean()
        metrics["avg_hold"] = result["hold_count"].mean()
        records.append(metrics)

        print(f"年化: {metrics['annualized_return']:.2%}, 回撤: {metrics['max_drawdown']:.2%}, 夏普: {metrics['sharpe']:.2f}, 止损: {metrics['stop_count']}")

    df = pd.DataFrame(records)
    df = df[["mode", "annualized_return", "max_drawdown", "sharpe", "calmar", "volatility", "stop_count", "avg_exposure", "avg_hold"]]
    print("\n=== 汇总对比 ===")
    print(df.to_string(index=False))

    os.makedirs(args.outdir, exist_ok=True)
    df.to_csv(os.path.join(args.outdir, "confirmation_grid.csv"), index=False, encoding="utf-8-sig")

    # 画图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    x = np.arange(len(df))
    width = 0.35

    ax = axes[0, 0]
    ax.bar(x - width/2, df["annualized_return"], width, label="annualized_return")
    ax.set_xticks(x)
    ax.set_xticklabels(df["mode"], rotation=45)
    ax.set_ylabel("年化收益")
    ax.set_title("年化收益对比")
    ax.grid(True, linestyle="--", alpha=0.5)

    ax = axes[0, 1]
    ax.bar(x - width/2, df["max_drawdown"], width, label="max_drawdown", color="C1")
    ax.set_xticks(x)
    ax.set_xticklabels(df["mode"], rotation=45)
    ax.set_ylabel("最大回撤")
    ax.set_title("最大回撤对比")
    ax.grid(True, linestyle="--", alpha=0.5)

    ax = axes[1, 0]
    ax.bar(x - width/2, df["sharpe"], width, label="sharpe", color="C2")
    ax.set_xticks(x)
    ax.set_xticklabels(df["mode"], rotation=45)
    ax.set_ylabel("夏普")
    ax.set_title("夏普对比")
    ax.grid(True, linestyle="--", alpha=0.5)

    ax = axes[1, 1]
    ax.bar(x - width/2, df["calmar"], width, label="calmar", color="C3")
    ax.set_xticks(x)
    ax.set_xticklabels(df["mode"], rotation=45)
    ax.set_ylabel("Calmar")
    ax.set_title("Calmar 对比")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plot_path = os.path.join(args.outdir, "confirmation_grid.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\n图表已保存: {plot_path}")
    print(f"CSV 已保存: {os.path.join(args.outdir, 'confirmation_grid.csv')}")
    print(f"运行时间: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()
