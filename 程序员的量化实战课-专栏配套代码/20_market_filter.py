"""
ETF 趋势轮动：组合层面市场择时过滤
- 当被选中的 top_n 标的，trend_score <= 0 的单独剔除，不买入
- 空仓部分买入国债 ETF（511020.SH.csv) 替代

用法：
    python 20_market_filter.py --top-n=2 --period=25
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
CASH_SUBSTITUTE = "511020.SH.csv"  # 国债5-10Y


def parse_args():
    parser = argparse.ArgumentParser(description="ETF 趋势轮动：市场择时过滤 + 国债替代")
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
    # 过滤参数
    parser.add_argument("--score-threshold", type=float, default=0.0, help="入选的最低 trend_score 阈值")
    parser.add_argument("--use-substitute", action="store_true", default=True, help="空仓时是否买入国债替代")
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


def backtest_with_market_filter(ohlc: dict, score: pd.DataFrame, args):
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
    active_count = pd.Series(index=dates, dtype=int)
    fuse_records = []
    stop_records = []

    for i, date in enumerate(dates):
        today_close = close.loc[date]
        today_atr = atr.loc[date]

        # 1. 仅初始止损检查
        for sym in list(positions.keys()):
            pos = positions[sym]
            px = today_close[sym]

            if px <= pos["stop"]:
                cash += pos["shares"] * px * (1 - args.fee)
                stop_records.append((date, sym, "stop"))
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
                selected = row.sort_values(ascending=False).head(args.top_n)
                # 过滤：只保留 trend_score >= 阈值的
                targets = selected[selected >= args.score_threshold].index.tolist()
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
                # 空仓替代
                if args.use_substitute and CASH_SUBSTITUTE in symbols:
                    raw_amounts[CASH_SUBSTITUTE] = equity * args.max_gross_exposure

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

            active_count.loc[date] = len([s for s in targets if s != CASH_SUBSTITUTE])
        else:
            active_count.loc[date] = np.nan

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
        "active_count": active_count,
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
    result = backtest_with_market_filter(ohlc, score, args)
    nav = result["nav"]
    weights = result["weights"]
    total_exposure = result["total_exposure"]
    cash_ratio = result["cash_ratio"]
    hold_count = result["hold_count"]
    active_count = result["active_count"]
    stop_records = result["stop_records"]

    metrics = calc_metrics(nav, ann_factor=args.ann_factor, risk_free=args.risk_free)

    print(f"=== ETF 趋势轮动：trend_score > {args.score_threshold} 过滤 + 国债替代 ===")
    print(f"区间: {metrics['start_date']} ~ {metrics['end_date']}")
    for k in ["total_return", "annualized_return", "volatility", "max_drawdown"]:
        print(f"{k}: {metrics[k]:.2%}")
    for k in ["sharpe", "calmar"]:
        print(f"{k}: {metrics[k]:.2f}")
    print(f"熔断次数: {len(result['fuse_records'])}")
    print(f"初始止损触发次数: {len(stop_records)}")

    print(f"\n=== 仓位结构统计 ===")
    print(f"平均总仓位: {total_exposure.mean():.2%}")
    print(f"总仓位最大值: {total_exposure.max():.2%}")
    print(f"平均现金比例: {cash_ratio.mean():.2%}")
    print(f"平均持仓数量: {hold_count.mean():.2f} 只")
    print(f"平均主动权益仓位: {active_count.mean():.2f} 只")

    os.makedirs(args.outdir, exist_ok=True)
    suffix = "market_filter"
    pd.DataFrame([{"strategy": suffix, **metrics}]).to_csv(
        os.path.join(args.outdir, f"{suffix}_metrics.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    fig, axes = plt.subplots(5, 1, figsize=(14, 18), gridspec_kw={"height_ratios": [2, 1, 1, 1, 1]})

    ax1 = axes[0]
    ax1.plot(nav.index, nav / nav.iloc[0], label="组合净值", linewidth=2, color="C0")
    ax1.set_title(f"ETF 趋势轮动：trend_score > {args.score_threshold} 过滤 + 国债替代")
    ax1.set_ylabel("归一化净值")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = axes[1]
    ax2.plot(total_exposure.index, total_exposure, label="总仓位", linewidth=1)
    ax2.plot(cash_ratio.index, cash_ratio, label="现金比例", linewidth=1)
    ax2.axhline(0.8, color="red", linestyle="--", alpha=0.5, label="仓位上限 80%")
    ax2.set_ylabel("比例")
    ax2.set_ylim(0, 1)
    ax2.legend(loc="upper left")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.set_title("总仓位与现金比例")

    ax3 = axes[2]
    ax3.plot(hold_count.index, hold_count, drawstyle="steps-post", linewidth=1)
    ax3.set_ylabel("持仓数量")
    ax3.set_ylim(0, hold_count.max() + 1)
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.set_title("每日持仓数量")

    ax4 = axes[3]
    ax4.stackplot(weights.index, *[weights[col] for col in weights.columns], labels=[NAME_MAP.get(c, c) for c in weights.columns], alpha=0.7)
    ax4.set_ylabel("权重")
    ax4.set_ylim(0, 1)
    ax4.legend(loc="upper left", fontsize=8, ncol=2)
    ax4.grid(True, linestyle="--", alpha=0.5)
    ax4.set_title("各标的权重堆叠")

    ax5 = axes[4]
    ax5.plot(active_count.index, active_count, drawstyle="steps-post", linewidth=1, color="C2")
    ax5.set_ylabel("主动权益仓位数")
    ax5.set_ylim(-0.1, args.top_n + 0.5)
    ax5.grid(True, linestyle="--", alpha=0.5)
    ax5.set_title("每日满足 trend_score>0 的入选标的数")

    plt.tight_layout()
    plot_path = os.path.join(args.outdir, f"{suffix}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"图表已保存: {plot_path}")
    print(f"运行时间: {time.time() - start_time:.4f}s")


if __name__ == "__main__":
    main()
