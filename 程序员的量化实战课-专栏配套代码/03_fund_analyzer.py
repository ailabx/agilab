#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基金业绩分析脚本。

支持读取由 01/02 号脚本生成的 <symbol>.csv（字段：date, close），
计算年化收益、最大回撤、回撤区间、滚动收益、波动率、夏普、卡玛等指标，
并用 matplotlib 绘制归一化后的时间序列对比图。

用法示例:
    python 03_fund_analyzer.py --symbols='050002.OF,270042.OF,110026.OF,000218.OF' --data-dir=E:/agilab/data/tushare --outdir=E:/agilab/data
    python 03_fund_analyzer.py --symbols='E:/agilab/data/tushare/050002.OF.csv,E:/agilab/data/tushare/270042.OF.csv' --outdir=E:/agilab/data
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")  # 无界面后端
import matplotlib.pyplot as plt

# 设置中文字体（优先使用系统常见中文字体）
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "PingFang SC", "WenQuanYi Micro Hei"]
plt.rcParams["axes.unicode_minus"] = False


# 默认无风险利率（年化）
DEFAULT_RISK_FREE_RATE = 0.02

# 默认数据目录：当前脚本目录下的 data/
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# 默认输出目录：当前脚本目录下的 data/
DEFAULT_OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# 默认基础信息文件
DEFAULT_BASIC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "basic.csv")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="基金业绩分析：收益、风险、回撤、对比图"
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default="050002.OF,270042.OF",
        help="基金代码或 CSV 路径列表，逗号分隔。默认: 050002.OF,270042.OF (沪深300, 纳指100)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help=f"CSV 数据目录。默认: {DEFAULT_DATA_DIR}",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=DEFAULT_OUTDIR,
        help=f"输出目录，默认: {DEFAULT_OUTDIR}",
    )
    parser.add_argument(
        "--risk-free",
        type=float,
        default=DEFAULT_RISK_FREE_RATE,
        help=f"无风险利率（年化），默认 {DEFAULT_RISK_FREE_RATE}",
    )
    parser.add_argument(
        "--plot-title",
        type=str,
        default="基金净值走势对比（归一化）",
        help="图表标题",
    )
    parser.add_argument(
        "--basic",
        type=str,
        default=DEFAULT_BASIC_PATH,
        help=f"基金基础信息 CSV 路径，用于显示基金名称。默认: {DEFAULT_BASIC_PATH}",
    )
    return parser.parse_args()


def load_data(symbol: str, data_dir: str = None) -> pd.DataFrame:
    """
    根据 symbol 加载 CSV 数据。
    symbol 可以是文件路径，也可以是基金代码。
    """
    if os.path.exists(symbol):
        path = symbol
    elif data_dir and os.path.exists(os.path.join(data_dir, f"{symbol}.csv")):
        path = os.path.join(data_dir, f"{symbol}.csv")
    elif os.path.exists(f"{symbol}.csv"):
        path = f"{symbol}.csv"
    else:
        raise FileNotFoundError(f"找不到 {symbol} 对应的数据文件")

    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["name"] = symbol
    return df


def annualized_return(prices: pd.Series, periods_per_year: int = 250) -> float:
    """根据价格序列计算年化收益率。"""
    if len(prices) < 2:
        return np.nan
    total_return = prices.iloc[-1] / prices.iloc[0] - 1
    n = len(prices)
    years = n / periods_per_year
    return (1 + total_return) ** (1 / years) - 1


def period_return(prices: pd.Series, start_date=None, end_date=None) -> float:
    """计算指定区间内的收益率。"""
    if start_date:
        prices = prices[prices.index >= start_date]
    if end_date:
        prices = prices[prices.index <= end_date]
    if len(prices) < 2:
        return np.nan
    return prices.iloc[-1] / prices.iloc[0] - 1


def max_drawdown(prices: pd.Series) -> tuple:
    """
    计算最大回撤及回撤区间。
    返回：(最大回撤率, 回撤开始日期, 回撤结束日期, 回撤恢复日期)
    """
    rolling_max = prices.cummax()
    drawdown = (prices - rolling_max) / rolling_max
    max_dd = drawdown.min()
    end_idx = drawdown.idxmin()
    start_idx = prices[:end_idx].idxmax()
    # 恢复日期：回撤结束后净值重新创新高的第一个日期
    recovery_candidates = prices[prices.index > end_idx]
    recovery_idx = None
    if not recovery_candidates.empty:
        for idx, val in recovery_candidates.items():
            if val >= prices.loc[start_idx]:
                recovery_idx = idx
                break
    return max_dd, start_idx, end_idx, recovery_idx


def rolling_annualized_return(prices: pd.Series, years: int, periods_per_year: int = 250) -> float:
    """计算 N 年年化收益。"""
    if len(prices) < periods_per_year * years:
        return np.nan
    sub = prices.iloc[-periods_per_year * years :]
    return (sub.iloc[-1] / sub.iloc[0]) ** (1 / years) - 1


def calculate_metrics(df: pd.DataFrame, risk_free: float = DEFAULT_RISK_FREE_RATE, periods_per_year: int = 250) -> dict:
    """计算单只基金的各项指标。"""
    df = df.sort_values("date").copy()
    df = df.set_index("date")
    prices = df["close"].squeeze()

    # 基础收益指标
    total = prices.iloc[-1] / prices.iloc[0] - 1
    ann = annualized_return(prices, periods_per_year)

    # 滚动收益
    ytd = period_return(prices, start_date=pd.Timestamp.now().replace(month=1, day=1))
    one_year = rolling_annualized_return(prices, years=1, periods_per_year=periods_per_year)
    three_year = rolling_annualized_return(prices, years=3, periods_per_year=periods_per_year)
    five_year = rolling_annualized_return(prices, years=5, periods_per_year=periods_per_year)

    # 风险指标
    returns = prices.pct_change().dropna()
    volatility = returns.std() * np.sqrt(periods_per_year)

    # 夏普比率
    excess_return = ann - risk_free
    sharpe = excess_return / volatility if volatility != 0 else np.nan

    # 最大回撤与区间
    dd, dd_start, dd_end, dd_recover = max_drawdown(prices)

    # 卡玛比率
    calmar = ann / abs(dd) if dd != 0 else np.nan

    return {
        "总收益": total,
        "历史至今年化": ann,
        "今年至今": ytd,
        "一年年化": one_year,
        "三年年化": three_year,
        "五年年化": five_year,
        "波动率": volatility,
        "夏普比率": sharpe,
        "最大回撤": dd,
        "回撤开始": dd_start,
        "回撤结束": dd_end,
        "回撤恢复": dd_recover,
        "卡玛比率": calmar,
    }


def normalize_prices(df: pd.DataFrame) -> pd.DataFrame:
    """将价格序列按各自最大日期对齐并归一化。"""
    df = df.sort_values("date").copy()
    df["normalized"] = df["close"] / df["close"].iloc[0]
    return df


def plot_time_series(dataframes: dict, outdir: str, fund_names: dict = None, title: str = "基金净值走势对比（归一化）") -> str:
    """
    绘制归一化后的时间序列对比图。
    所有基金以它们共同的起始日期为起点对齐并归一化（都从共同起点=1 开始画）。
    dataframes: {symbol: DataFrame(date, close)}
    """
    # 找到所有基金的最大日期，取最小值作为共同截止日期
    max_dates = [df["date"].max() for df in dataframes.values()]
    common_end = min(max_dates)
    # 找到所有基金的最小日期，取最大值作为共同起始日期
    min_dates = [df["date"].min() for df in dataframes.values()]
    common_start = max(min_dates)

    plt.figure(figsize=(12, 7))
    for symbol, df in dataframes.items():
        # 截取共同区间
        df = df[(df["date"] >= common_start) & (df["date"] <= common_end)].copy()
        df = normalize_prices(df)
        label = get_display_name(symbol, fund_names) if fund_names else symbol
        plt.plot(df["date"], df["normalized"], label=label, linewidth=1.5)

    plt.title(title, fontsize=14)
    plt.xlabel("日期", fontsize=12)
    plt.ylabel("归一化净值（起点=1）", fontsize=12)
    plt.legend(loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "fund_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def load_fund_names(basic_path: str) -> dict:
    """加载基金基础信息，返回 {ts_code: name} 映射。"""
    if not os.path.exists(basic_path):
        return {}
    try:
        df = pd.read_csv(basic_path, usecols=["ts_code", "name"], dtype={"ts_code": str, "name": str})
        return dict(zip(df["ts_code"], df["name"]))
    except Exception:
        return {}


def get_display_name(symbol: str, fund_names: dict) -> str:
    """根据 symbol 获取显示名称。"""
    # 标准化 symbol 为 .OF 后缀
    key = symbol if symbol.upper().endswith(".OF") else f"{symbol}.OF"
    return fund_names.get(key, symbol)


def print_metrics_table(metrics: dict, fund_names: dict = None) -> None:
    """打印指标表格。"""
    df = pd.DataFrame(metrics).T
    # 如果有基金名称，增加一列显示
    if fund_names:
        df.insert(0, "名称", [get_display_name(idx, fund_names) for idx in df.index])
    formatters = {
        "总收益": "{:.2%}",
        "历史至今年化": "{:.2%}",
        "今年至今": "{:.2%}",
        "一年年化": "{:.2%}",
        "三年年化": "{:.2%}",
        "五年年化": "{:.2%}",
        "波动率": "{:.2%}",
        "夏普比率": "{:.2f}",
        "最大回撤": "{:.2%}",
        "卡玛比率": "{:.2f}",
    }
    for col, fmt in formatters.items():
        if col in df.columns:
            df[col] = df[col].apply(lambda x: fmt.format(x) if pd.notna(x) else "-")
    print(df.to_string())


def main() -> int:
    args = parse_args()
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    data_dir = args.data_dir
    outdir = args.outdir
    risk_free = args.risk_free

    dataframes = {}
    metrics = {}

    for symbol in symbols:
        try:
            df = load_data(symbol, data_dir)
            if df.empty or len(df) < 2:
                print(f"{symbol} 数据不足，跳过")
                continue
            dataframes[symbol] = df
            metrics[symbol] = calculate_metrics(df, risk_free=risk_free)
            print(f"已加载: {symbol}，共 {len(df)} 条记录")
        except Exception as exc:
            print(f"加载 {symbol} 失败: {exc}")
            continue

    if not dataframes:
        print("没有可用的基金数据")
        return 1

    fund_names = load_fund_names(args.basic)

    print("\n=== 基金业绩指标 ===")
    print_metrics_table(metrics, fund_names)

    # 保存指标到 CSV
    metrics_df = pd.DataFrame(metrics).T
    metrics_df.insert(0, "名称", [get_display_name(idx, fund_names) for idx in metrics_df.index])
    os.makedirs(outdir, exist_ok=True)
    metrics_path = os.path.join(outdir, "fund_metrics.csv")
    metrics_df.to_csv(metrics_path, encoding="utf-8-sig")
    print(f"\n指标已保存: {metrics_path}")

    # 绘图（图例显示名称）
    plot_path = plot_time_series(dataframes, outdir, fund_names=fund_names, title=args.plot_title)
    print(f"对比图已保存: {plot_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
