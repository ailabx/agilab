"""
配置化 ETF 策略综合参数网格

用法：
    python 28_config_strategy_grid.py --top-k=4 --period=20 --max-weight=0.25
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

from utils import calc_metrics


DEFAULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

DEFAULT_SYMBOLS = [
    "510300.SH.csv", "588000.SH.csv", "510880.SH.csv", "159980.SZ.csv", "159985.SZ.csv",
    "513100.SH.csv", "513520.SH.csv", "513500.SH.csv", "513030.SH.csv", "501018.SH.csv",
    "513180.SH.csv", "159915.SZ.csv", "512890.SH.csv", "511090.SH.csv", "510180.SH.csv",
]

CASH_SUBSTITUTE = "511090.SH.csv"


def parse_args():
    parser = argparse.ArgumentParser(description="配置化 ETF 策略综合参数网格")
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
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)


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


def run_backtest(ohlc, score, roc1, atr, top_k, max_weight, stop_atr, trailing_atr, rebalance_freq, score_threshold, substitute_ratio, fee):
    close = ohlc["close"]
    dates = close.index
    symbols = close.columns.tolist()

    cash = 1000000.0
    positions = {}
    stop_records = []

    nav = pd.Series(index=dates, dtype=float)

    for i, date in enumerate(dates):
        today_close = close.loc[date]
        today_atr = atr.loc[date]

        # 初始止损
        if stop_atr > 0:
            for sym in list(positions.keys()):
                px = today_close[sym]
                if px <= positions[sym]["stop"]:
                    cash += positions[sym]["shares"] * px * (1 - fee)
                    stop_records.append((date, sym, "stop"))
                    del positions[sym]

        # 跟踪止损
        if trailing_atr > 0:
            for sym in list(positions.keys()):
                if sym == CASH_SUBSTITUTE:
                    continue
                px = today_close[sym]
                pos = positions[sym]
                pos["highest"] = max(pos["highest"], px)
                trailing_stop = pos["highest"] - trailing_atr * today_atr[sym]
                if px <= trailing_stop:
                    cash += pos["shares"] * px * (1 - fee)
                    stop_records.append((date, sym, "trailing"))
                    del positions[sym]

        is_rebalance = (i % rebalance_freq == 0)

        if is_rebalance:
            row = score.loc[date]
            row = row.dropna()
            buy_mask = roc1.loc[date] >= -0.05
            row = row[buy_mask]
            row = row.dropna()

            if len(row) > 0:
                selected = row.sort_values(ascending=False).head(top_k)
                targets = selected[selected >= score_threshold].index.tolist()
            else:
                targets = []

            for sym in list(positions.keys()):
                if sym not in targets:
                    px = today_close[sym]
                    cash += positions[sym]["shares"] * px * (1 - fee)
                    del positions[sym]

            equity = cash + sum(positions[sym]["shares"] * today_close[sym] for sym in positions)

            if targets:
                w = min(1.0 / len(targets), max_weight)
                for sym in targets:
                    px = today_close[sym]
                    target_amount = equity * w
                    if sym in positions:
                        current_amount = positions[sym]["shares"] * px
                        delta = target_amount - current_amount
                        if delta > 0:
                            cash -= delta * (1 + fee)
                            positions[sym]["shares"] += delta / px
                        elif delta < 0:
                            cash += -delta * (1 - fee)
                            positions[sym]["shares"] -= -delta / px
                    else:
                        amount = target_amount / (1 + fee)
                        cash -= amount
                        atr_val = today_atr[sym]
                        stop = px - stop_atr * atr_val if stop_atr > 0 and pd.notna(atr_val) and atr_val > 0 else px * 0.9
                        positions[sym] = {
                            "entry": px,
                            "stop": stop,
                            "highest": px,
                            "shares": amount / px,
                        }
            else:
                if substitute_ratio > 0 and CASH_SUBSTITUTE in symbols:
                    px = today_close[CASH_SUBSTITUTE]
                    target_amount = equity * substitute_ratio
                    if CASH_SUBSTITUTE in positions:
                        current_amount = positions[CASH_SUBSTITUTE]["shares"] * px
                        delta = target_amount - current_amount
                        if delta > 0:
                            cash -= delta * (1 + fee)
                            positions[CASH_SUBSTITUTE]["shares"] += delta / px
                        elif delta < 0:
                            cash += -delta * (1 - fee)
                            positions[CASH_SUBSTITUTE]["shares"] -= -delta / px
                    else:
                        amount = target_amount / (1 + fee)
                        cash -= amount
                        positions[CASH_SUBSTITUTE] = {
                            "entry": px,
                            "stop": px * 0.9,
                            "shares": amount / px,
                        }
        else:
            # 非调仓日只更新净值
            pass

        equity = cash + sum(positions[sym]["shares"] * today_close[sym] for sym in positions)
        nav.loc[date] = equity

    return nav, stop_records


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

    score = cs_zscore(calc_trend_score2(close, args.period))
    roc1 = close.pct_change(1)
    atr = calc_atr(ohlc, args.atr_period)

    param_grid = {
        "stop_atr": [0.0, 2.0, 3.0, 4.0],
        "trailing_atr": [0.0, 1.0, 2.0],
        "rebalance_freq": [1, 5, 10, 20],
        "score_threshold": [0.0, 0.05, 0.1, 0.2],
        "substitute_ratio": [0.0, 1.0],
    }

    results = []
    total = (len(param_grid["stop_atr"]) * len(param_grid["trailing_atr"]) *
             len(param_grid["rebalance_freq"]) * len(param_grid["score_threshold"]) *
             len(param_grid["substitute_ratio"]))
    print(f"总组合数: {total}")

    count = 0
    for stop_atr in param_grid["stop_atr"]:
        for trailing_atr in param_grid["trailing_atr"]:
            for rebalance_freq in param_grid["rebalance_freq"]:
                for score_threshold in param_grid["score_threshold"]:
                    for substitute_ratio in param_grid["substitute_ratio"]:
                        count += 1
                        nav, stops = run_backtest(
                            ohlc, score, roc1, atr,
                            top_k=args.top_k,
                            max_weight=args.max_weight,
                            stop_atr=stop_atr,
                            trailing_atr=trailing_atr,
                            rebalance_freq=rebalance_freq,
                            score_threshold=score_threshold,
                            substitute_ratio=substitute_ratio,
                            fee=args.fee,
                        )
                        metrics = calc_metrics(nav, ann_factor=args.ann_factor, risk_free=args.risk_free)
                        results.append({
                            "stop_atr": stop_atr,
                            "trailing_atr": trailing_atr,
                            "rebalance_freq": rebalance_freq,
                            "score_threshold": score_threshold,
                            "substitute_ratio": substitute_ratio,
                            "annualized_return": metrics["annualized_return"],
                            "volatility": metrics["volatility"],
                            "max_drawdown": metrics["max_drawdown"],
                            "sharpe": metrics["sharpe"],
                            "calmar": metrics["calmar"],
                            "total_return": metrics["total_return"],
                            "stop_count": len(stops),
                        })
                        print(f"[{count}/{total}] stop={stop_atr} trail={trailing_atr} freq={rebalance_freq} th={score_threshold} sub={substitute_ratio} -> 年化={metrics['annualized_return']:.2%} 回撤={metrics['max_drawdown']:.2%} 夏普={metrics['sharpe']:.2f}")

    results_df = pd.DataFrame(results)
    os.makedirs(args.outdir, exist_ok=True)
    results_df.to_csv(os.path.join(args.outdir, "config_strategy_grid.csv"), index=False, encoding="utf-8-sig")

    # Top 10 夏普
    top_sharpe = results_df.sort_values("sharpe", ascending=False).head(10)
    print("\n=== Top 10 夏普 ===")
    print(top_sharpe.to_string(index=False))

    # Top 10 年化
    top_return = results_df.sort_values("annualized_return", ascending=False).head(10)
    print("\n=== Top 10 年化 ===")
    print(top_return.to_string(index=False))

    # Top 10 Calmar
    top_calmar = results_df.sort_values("calmar", ascending=False).head(10)
    print("\n=== Top 10 Calmar ===")
    print(top_calmar.to_string(index=False))

    print(f"\n网格结果已保存到 {args.outdir}/config_strategy_grid.csv")
    print(f"耗时: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()
