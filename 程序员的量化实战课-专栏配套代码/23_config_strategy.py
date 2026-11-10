"""
按给定配置实现 ETF 策略基础版

配置语义：
- 每日运行
- 按 cs_zscore(trend_score2(close,20)) 选 top4
- 等权，max_weight=0.25
- buy: roc(close,1) >= -0.05
- sell: trend_score2 >= 5 或 <= 0
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

# 用户指定的 15 只 ETF
DEFAULT_SYMBOLS = [
    "510300.SH.csv",  # 沪深300
    "588000.SH.csv",  # 科创50
    "510880.SH.csv",  # 红利ETF
    "159980.SZ.csv",  # 能源化工
    "159985.SZ.csv",  # 豆粕
    "513100.SH.csv",  # 纳指100
    "513520.SH.csv",  # 日经225
    "513500.SH.csv",  # 标普500
    "513030.SH.csv",  # 德国DAX
    "501018.SH.csv",  # 南方原油
    "513180.SH.csv",  # 恒指科技
    "159915.SZ.csv",  # 创业板
    "512890.SH.csv",  # 红利低波
    "511090.SH.csv",  # 国债10Y
    "510180.SH.csv",  # 上证180
]

CASH_SUBSTITUTE = "511090.SH.csv"  # 国债10Y


def parse_args():
    parser = argparse.ArgumentParser(description="配置化 ETF 策略基础版")
    parser.add_argument("--symbols", type=str, default=",".join(DEFAULT_SYMBOLS), help="标的 CSV 文件名")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DIR, help="CSV 数据目录")
    parser.add_argument("--outdir", type=str, default=DEFAULT_DIR, help="输出目录")
    parser.add_argument("--capital", type=float, default=1000000.0, help="初始资金")
    parser.add_argument("--period", type=int, default=20, help="trend_score2 周期")
    parser.add_argument("--top-k", type=int, default=4, help="top-k 选股")
    parser.add_argument("--max-weight", type=float, default=0.25, help="单标的上限")
    parser.add_argument("--fee", type=float, default=0.0001, help="交易成本")
    parser.add_argument("--ann-factor", type=int, default=252, help="年化因子")
    parser.add_argument("--risk-free", type=float, default=0.02, help="无风险利率")
    parser.add_argument("--start-date", type=str, default="2020-01-01", help="开始日期")
    parser.add_argument("--end-date", type=str, default="", help="结束日期")
    return parser.parse_args()


def load_ohlc(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").drop_duplicates("date").set_index("date")
    # 兼容只有 close 列的文件
    if "close" not in df.columns:
        raise ValueError(f"{path} 缺少 close 列")
    return df


def calc_trend_score2(prices: pd.DataFrame, period: int) -> pd.DataFrame:
    """
    trend_score2: 对 log 价格做线性回归，score = 年化收益 * R^2
    """
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
    """横截面 zscore"""
    return (df - df.mean(axis=1).values[:, None]) / df.std(axis=1).values[:, None]


def backtest_config_strategy(ohlc: dict, score: pd.DataFrame, roc1: pd.DataFrame, args):
    close = ohlc["close"]
    dates = close.index
    symbols = close.columns.tolist()

    initial_capital = args.capital
    cash = initial_capital
    positions = {}  # sym -> shares

    nav = pd.Series(index=dates, dtype=float)
    weights = pd.DataFrame(0.0, index=dates, columns=symbols)
    hold_count = pd.Series(index=dates, dtype=int)
    trade_records = []

    for i, date in enumerate(dates):
        today_close = close.loc[date]

        # 生成买入候选
        row = score.loc[date]
        row = row.dropna()
        # buy rule: roc1 >= -0.05
        buy_mask = roc1.loc[date] >= -0.05
        row = row[buy_mask]
        row = row.dropna()

        # 选 top-k
        if len(row) > 0:
            selected = row.sort_values(ascending=False).head(args.top_k)
            targets = selected.index.tolist()
        else:
            targets = []

        # sell rule: 检查持仓，如果 trend_score2 >= 5 或 <= 0 卖出
        for sym in list(positions.keys()):
            s = score.loc[date, sym]
            if pd.notna(s) and (s >= 5 or s <= 0):
                px = today_close[sym]
                cash += positions[sym] * px * (1 - args.fee)
                trade_records.append((date, sym, "sell", px))
                del positions[sym]

        # 调仓到目标权重（等权，上限 max_weight）
        # 先清空不在目标中的持仓
        for sym in list(positions.keys()):
            if sym not in targets:
                px = today_close[sym]
                cash += positions[sym] * px * (1 - args.fee)
                trade_records.append((date, sym, "sell", px))
                del positions[sym]

        # 计算当前总权益
        equity = cash + sum(positions[sym] * today_close[sym] for sym in positions)

        # 新买入目标
        if targets:
            w = 1.0 / len(targets)
            w = min(w, args.max_weight)
            for sym in targets:
                px = today_close[sym]
                target_amount = equity * w
                if sym in positions:
                    current_amount = positions[sym] * px
                    delta_amount = target_amount - current_amount
                    if delta_amount > 0:
                        cash -= delta_amount * (1 + args.fee)
                        positions[sym] += delta_amount / px
                    elif delta_amount < 0:
                        cash += -delta_amount * (1 - args.fee)
                        positions[sym] -= -delta_amount / px
                else:
                    cash -= target_amount * (1 + args.fee)
                    positions[sym] = target_amount / px
                    trade_records.append((date, sym, "buy", px))

        # 记录
        equity = cash + sum(positions[sym] * today_close[sym] for sym in positions)
        nav.loc[date] = equity
        for sym in symbols:
            if sym in positions:
                weights.loc[date, sym] = positions[sym] * today_close[sym] / equity
        hold_count.loc[date] = len(positions)

    return nav, weights, hold_count, trade_records


def main():
    args = parse_args()
    start_time = time.time()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    frames = {col: [] for col in ["open", "high", "low", "close", "vol"]}
    valid_symbols = []
    for s in symbols:
        path = os.path.join(args.data_dir, s)
        if not os.path.exists(path):
            print(f"警告：文件不存在 {path}")
            continue
        df = load_ohlc(path)
        valid_symbols.append(s)
        for col in frames:
            if col in df.columns:
                frames[col].append(df[col].rename(s))
            else:
                # 如果缺少某列，用 close 或 nan 填充
                if col == "close" and "close" in df.columns:
                    frames[col].append(df["close"].rename(s))
                else:
                    frames[col].append(pd.Series(np.nan, index=df.index).rename(s))

    ohlc = {}
    for col in frames:
        ohlc[col] = pd.concat(frames[col], axis=1, sort=True).dropna()

    close = ohlc["close"]

    # 日期过滤
    if args.start_date:
        close = close[close.index >= pd.to_datetime(args.start_date)]
    if args.end_date:
        close = close[close.index <= pd.to_datetime(args.end_date)]

    # 重新构造 ohlc
    for col in ohlc:
        ohlc[col] = ohlc[col].loc[close.index]

    print(f"数据区间: {close.index[0].date()} ~ {close.index[-1].date()}")

    score = calc_trend_score2(close, args.period)
    roc1 = close.pct_change(1)

    # 横截面 zscore
    zscore = cs_zscore(score)

    nav, weights, hold_count, trades = backtest_config_strategy(ohlc, zscore, roc1, args)
    metrics = calc_metrics(nav, ann_factor=args.ann_factor, risk_free=args.risk_free)

    print(f"=== 配置化 ETF 策略基础版 ===")
    print(f"区间: {metrics['start_date']} ~ {metrics['end_date']}")
    for k in ["total_return", "annualized_return", "volatility", "max_drawdown"]:
        print(f"{k}: {metrics[k]:.2%}")
    for k in ["sharpe", "calmar"]:
        print(f"{k}: {metrics[k]:.2f}")
    print(f"交易次数: {len(trades)}")
    print(f"平均持仓数: {hold_count.mean():.2f}")

    os.makedirs(args.outdir, exist_ok=True)
    pd.DataFrame([{"strategy": "config_base", **metrics}]).to_csv(
        os.path.join(args.outdir, "config_base_metrics.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    # 画图
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={"height_ratios": [2, 1, 1]})

    ax1 = axes[0]
    ax1.plot(nav.index, nav / nav.iloc[0], label="组合净值", linewidth=2)
    ax1.set_title("配置化 ETF 策略基础版")
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

    plt.tight_layout()
    plot_path = os.path.join(args.outdir, "config_base.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"图表已保存: {plot_path}")
    print(f"运行时间: {time.time() - start_time:.4f}s")


if __name__ == "__main__":
    main()
