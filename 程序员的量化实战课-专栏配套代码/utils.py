"""
通用工具：收益指标 + 图表绘制
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


def calc_metrics(nav: pd.Series, risk_free: float = 0.02, ann_factor: int = 252) -> dict:
    """
    输入净值序列，输出常用业绩指标。

    年化收益采用复利 CAGR 口径，按交易日年化：
        ann_ret = (end / start) ^ (ann_factor / n) - 1
    其中 n 为实际观测到的收益期数。默认 ann_factor=252，对应日频交易日。
    """
    rets = nav.pct_change().dropna()
    total_ret = nav.iloc[-1] / nav.iloc[0] - 1.0
    n = len(rets)
    ann_ret = (nav.iloc[-1] / nav.iloc[0]) ** (ann_factor / n) - 1.0 if n > 0 else 0.0
    vol = rets.std(ddof=1) * (ann_factor ** 0.5)
    sharpe = (ann_ret - risk_free) / vol if vol > 0 else 0.0
    max_dd = ((nav - nav.cummax()) / nav.cummax()).min()
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0.0
    return {
        "start_date": nav.index[0].strftime("%Y-%m-%d"),
        "end_date": nav.index[-1].strftime("%Y-%m-%d"),
        "total_return": total_ret,
        "annualized_return": ann_ret,
        "volatility": vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "calmar": calmar,
    }


def plot_nav_and_weights(prices: pd.DataFrame, nav: pd.Series,
                         weights: pd.DataFrame, outdir: str, title: str,
                         name_map: dict = None, filename: str = "fund_comparison.png") -> str:
    """绘制净值曲线和权重变化。name_map 用于把文件名映射为基金名称。"""
    os.makedirs(outdir, exist_ok=True)
    name_map = name_map or {}

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [2, 1]})

    ax1 = axes[0]
    ax1.plot(nav.index, nav / nav.iloc[0], label="组合净值", linewidth=2)
    for col in prices.columns:
        label = name_map.get(col, col)
        ax1.plot(prices.index, prices[col] / prices[col].iloc[0],
                 label=f"{label}（归一化）", linewidth=1, alpha=0.7)
    ax1.set_title(title, fontsize=14)
    ax1.set_xlabel("日期")
    ax1.set_ylabel("归一化净值")
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = axes[1]
    for col in weights.columns:
        label = name_map.get(col, col)
        ax2.plot(weights.index, weights[col], label=label, linewidth=1.5)
    ax2.set_title("权重变化")
    ax2.set_xlabel("日期")
    ax2.set_ylabel("权重")
    ax2.set_ylim(0, 1)
    ax2.legend(loc="upper left")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    path = os.path.join(outdir, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    return path
