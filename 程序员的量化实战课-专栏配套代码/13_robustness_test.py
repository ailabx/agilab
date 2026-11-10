"""
12_topn_rotation.py 的稳健性测试套件

包含：
1. 滚动窗口回测（Walk-forward / Rolling Window）
2. 参数鲁棒性测试（period × top_n × fee 网格）
3. 蒙特卡洛扰动测试
4. 样本外/前向验证

用法：
    python 13_robustness_test.py --mode=rolling
    python 13_robustness_test.py --mode=parameter
    python 13_robustness_test.py --mode=monte_carlo
    python 13_robustness_test.py --mode=out_of_sample

默认使用 12_topn_rotation.py 的扩展 ETF 池。
"""
import argparse
import os
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from engine import backtest
from utils import calc_metrics

warnings.filterwarnings("ignore")

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


def parse_args():
    parser = argparse.ArgumentParser(description="TopN ETF 轮动稳健性测试套件")
    parser.add_argument(
        "--mode",
        type=str,
        default="rolling",
        choices=["rolling", "parameter", "monte_carlo", "out_of_sample"],
        help="测试模式：rolling / parameter / monte_carlo / out_of_sample",
    )
    parser.add_argument("--symbols", type=str, default=DEFAULT_SYMBOLS, help="标的 CSV 文件名，逗号分隔")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DIR, help="CSV 数据目录")
    parser.add_argument("--outdir", type=str, default=DEFAULT_DIR, help="输出目录")
    parser.add_argument("--period", type=int, default=25, help="趋势评分周期")
    parser.add_argument("--top-n", type=int, default=2, help="Top N 等权")
    parser.add_argument("--fee", type=float, default=0.0001, help="交易成本")
    parser.add_argument("--ann-factor", type=int, default=252, help="年化因子")
    parser.add_argument("--risk-free", type=float, default=0.02, help="无风险利率")
    # 滚动窗口参数
    parser.add_argument("--window", type=int, default=252, help="滚动窗口长度（交易日）")
    # 参数网格
    parser.add_argument("--period-range", type=str, default="15,20,25,30,40,60,90,120", help="period 网格")
    parser.add_argument("--top-n-range", type=str, default="1,2,3,5", help="top_n 网格")
    parser.add_argument("--fee-range", type=str, default="0.0001,0.0005,0.001", help="fee 网格")
    # 蒙特卡洛
    parser.add_argument("--mc-sims", type=int, default=1000, help="蒙特卡洛模拟次数")
    parser.add_argument("--noise-scale", type=float, default=0.005, help="收益率噪声标准差")
    # 样本外
    parser.add_argument("--train-end", type=str, default="2023-12-31", help="训练期结束日期")
    return parser.parse_args()


def load_asset(path: str) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["date"], usecols=["date", "close"])
    df = df.sort_values("date").drop_duplicates("date").set_index("date")
    return df["close"]


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


def generate_weights(prices: pd.DataFrame, score: pd.DataFrame, top_n: int) -> pd.DataFrame:
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for date in score.dropna().index:
        row = score.loc[date]
        row = row[row > 0]
        if len(row) > 0:
            selected = row.sort_values(ascending=False).head(top_n).index
            w = 1.0 / len(selected)
            weights.loc[date, selected] = w
    return weights


def run_backtest(prices: pd.DataFrame, weights: pd.DataFrame, fee: float) -> tuple:
    rebalance_mask = weights.ne(weights.shift(1)).any(axis=1)
    rebalance_dates = set(weights[rebalance_mask].index)
    nav, weights_bt = backtest(prices, weights, fee, rebalance_dates=rebalance_dates)
    return nav, weights_bt


def fast_backtest(prices: pd.DataFrame, weights: pd.DataFrame, fee: float) -> pd.Series:
    """
    向量化快速回测，用于蒙特卡洛等需要大量模拟的场景。
    假设每日按目标权重 w_t 收盘调仓，组合收益近似为：
        R_t = sum(w_{t-1} * r_t) - fee * turnover_t
    其中 turnover_t = sum(|w_t - w_{t-1}|)。
    与 engine.backtest 结果高度一致，但速度大幅提升。
    """
    rets = prices.pct_change()
    w_lag = weights.shift(1).fillna(0.0)
    port_rets = (w_lag * rets).sum(axis=1)
    turnover = (weights - w_lag).abs().sum(axis=1)
    total_rets = port_rets - turnover * fee
    nav = (1 + total_rets).cumprod()
    nav.iloc[0] = 1.0  # 首日无收益
    return nav


# ---------------------------------------------------------------------------
# 1. 滚动窗口回测
# ---------------------------------------------------------------------------
def rolling_window_backtest(prices: pd.DataFrame, period: int, top_n: int, fee: float, window: int):
    """
    严格滚动窗口：在每个交易日，只用过去 window 个交易日的数据计算 trend_score。
    这里实现为一个滚动窗口训练/测试：
    - 用过去 window 天的 prices 计算 trend_score
    - 生成当日目标权重
    - 第二天按该权重持有
    """
    all_dates = prices.index

    # 先全量计算 score，但只使用到当日为止的 window 天数据
    # 严格版本：每天重新用过去 window 天训练
    score_full = calc_trend_score(prices, period)

    # 严格滚动：每个日期只用 [date-window+1, date] 的数据
    score_rolling = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)

    # 为了效率，使用滚动窗口 apply，但 trend_score 是自定义的，所以用最朴素的方式
    for i, date in enumerate(all_dates):
        if i < period - 1:
            continue
        start_idx = max(0, i - window + 1)
        window_prices = prices.iloc[start_idx : i + 1]
        score_rolling.loc[date] = calc_trend_score(window_prices, period).iloc[-1]

    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for date in score_rolling.dropna().index:
        row = score_rolling.loc[date]
        row = row[row > 0]
        if len(row) > 0:
            selected = row.sort_values(ascending=False).head(top_n).index
            w = 1.0 / len(selected)
            weights.loc[date, selected] = w

    # 回测时，信号生成日为 T，实际调仓日为 T+1（实盘不可能当天收盘前知道当天收盘）
    # 当前 engine 用 T 日权重 T 日收盘调仓，存在未来函数风险
    # 严格版本：weights 后移一天
    weights_lag = weights.shift(1).fillna(0.0)

    nav, weights_bt = run_backtest(prices, weights_lag, fee)
    return nav, weights_bt, score_rolling


def mode_rolling(args):
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    prices = pd.DataFrame({s: load_asset(os.path.join(args.data_dir, s)) for s in symbols})
    prices = prices.dropna()

    print(f"=== 滚动窗口回测（window={args.window}, period={args.period}, top_n={args.top_n}, fee={args.fee}) ===")

    nav, weights_bt, score = rolling_window_backtest(prices, args.period, args.top_n, args.fee, args.window)
    metrics = calc_metrics(nav, ann_factor=args.ann_factor, risk_free=args.risk_free)

    print(f"区间: {metrics['start_date']} ~ {metrics['end_date']}")
    for k in ["total_return", "annualized_return", "volatility", "max_drawdown"]:
        print(f"{k}: {metrics[k]:.2%}")
    for k in ["sharpe", "calmar"]:
        print(f"{k}: {metrics[k]:.2f}")

    os.makedirs(args.outdir, exist_ok=True)
    out = {
        "strategy": f"rolling_window_top{args.top_n}_p{args.period}_w{args.window}",
        **metrics,
    }
    pd.DataFrame([out]).to_csv(
        os.path.join(args.outdir, f"rolling_window_metrics.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    # 画图
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(nav.index, nav / nav.iloc[0], label="Rolling Window", linewidth=2)
    ax.set_title(f"滚动窗口回测 (window={args.window}, period={args.period}, top_n={args.top_n})")
    ax.set_xlabel("日期")
    ax.set_ylabel("归一化净值")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    path = os.path.join(args.outdir, "rolling_window_nav.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"图表已保存: {path}")


# ---------------------------------------------------------------------------
# 2. 参数鲁棒性测试
# ---------------------------------------------------------------------------
def mode_parameter(args):
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    prices = pd.DataFrame({s: load_asset(os.path.join(args.data_dir, s)) for s in symbols})
    prices = prices.dropna()

    period_range = [int(x) for x in args.period_range.split(",")]
    top_n_range = [int(x) for x in args.top_n_range.split(",")]
    fee_range = [float(x) for x in args.fee_range.split(",")]

    records = []
    print(f"=== 参数鲁棒性测试（period={period_range}, top_n={top_n_range}, fee={fee_range}) ===")

    for period in period_range:
        score = calc_trend_score(prices, period)
        for top_n in top_n_range:
            weights = generate_weights(prices, score, top_n)
            for fee in fee_range:
                nav, _ = run_backtest(prices, weights, fee)
                metrics = calc_metrics(nav, ann_factor=args.ann_factor, risk_free=args.risk_free)
                records.append({
                    "period": period,
                    "top_n": top_n,
                    "fee": fee,
                    **metrics,
                })
                print(f"period={period:3d}, top_n={top_n}, fee={fee:.4f}: ann={metrics['annualized_return']:6.2%}, sharpe={metrics['sharpe']:.2f}, mdd={metrics['max_drawdown']:6.2%}")

    df = pd.DataFrame(records)
    os.makedirs(args.outdir, exist_ok=True)
    df.to_csv(os.path.join(args.outdir, "parameter_robustness.csv"), index=False, encoding="utf-8-sig")

    # 画热力图：period × top_n，颜色为夏普
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics_to_plot = ["annualized_return", "sharpe", "max_drawdown", "calmar"]
    for ax, metric in zip(axes.flat, metrics_to_plot):
        # 取 fee=0.0001 的子集
        sub = df[df["fee"] == fee_range[0]].pivot_table(
            values=metric, index="top_n", columns="period", aggfunc="mean"
        )
        im = ax.imshow(sub.values, aspect="auto", cmap="RdYlGn")
        ax.set_xticks(range(len(sub.columns)))
        ax.set_xticklabels(sub.columns)
        ax.set_yticks(range(len(sub.index)))
        ax.set_yticklabels(sub.index)
        ax.set_xlabel("period")
        ax.set_ylabel("top_n")
        ax.set_title(metric)
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    path = os.path.join(args.outdir, "parameter_robustness_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"热力图已保存: {path}")


# ---------------------------------------------------------------------------
# 3. 蒙特卡洛扰动测试
# ---------------------------------------------------------------------------
def mode_monte_carlo(args):
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    prices = pd.DataFrame({s: load_asset(os.path.join(args.data_dir, s)) for s in symbols})
    prices = prices.dropna()

    print(f"=== 蒙特卡洛扰动测试（sims={args.mc_sims}, noise_scale={args.noise_scale}) ===")

    base_score = calc_trend_score(prices, args.period)
    base_weights = generate_weights(prices, base_score, args.top_n)

    base_nav, _ = run_backtest(prices, base_weights, args.fee)
    base_metrics = calc_metrics(base_nav, ann_factor=args.ann_factor, risk_free=args.risk_free)
    print(f"基准: ann={base_metrics['annualized_return']:.2%}, sharpe={base_metrics['sharpe']:.2f}, mdd={base_metrics['max_drawdown']:.2%}")

    sim_results = []
    rets = prices.pct_change().dropna()

    # 预计算原始权重，避免每次循环重复
    base_weights_for_lag = base_weights.copy()

    for sim in range(args.mc_sims):
        if (sim + 1) % 50 == 0:
            print(f"  MC progress: {sim + 1}/{args.mc_sims}")

        # 扰动 1: 价格序列加噪声
        noise_rets = rets + np.random.normal(0, args.noise_scale, rets.shape)
        noise_prices = prices.iloc[0] * (1 + noise_rets).cumprod()
        noise_prices = pd.concat([prices.iloc[[0]], noise_prices])

        # 扰动 2: 直接使用原始权重但作用于噪声价格（模拟信号不变、环境变化）
        weights = base_weights_for_lag.copy()

        # 扰动 3: 随机延迟信号（10% 概率延迟 1-2 天）
        if np.random.rand() < 0.10:
            weights = weights.shift(np.random.randint(1, 3)).fillna(0.0)

        # 扰动 4: 交易成本扰动
        fee = args.fee * np.random.uniform(0.5, 2.0)

        nav = fast_backtest(noise_prices, weights, fee)
        metrics = calc_metrics(nav, ann_factor=args.ann_factor, risk_free=args.risk_free)
        sim_results.append(metrics)

    sim_df = pd.DataFrame(sim_results)
    os.makedirs(args.outdir, exist_ok=True)
    sim_df.to_csv(os.path.join(args.outdir, "monte_carlo_results.csv"), index=False, encoding="utf-8-sig")

    print(f"MC 结果 (n={args.mc_sims}):")
    for metric in ["annualized_return", "sharpe", "max_drawdown", "total_return"]:
        vals = sim_df[metric]
        print(f"{metric}: 5%={vals.quantile(0.05):.2%}, median={vals.median():.2%}, 95%={vals.quantile(0.95):.2%}")

    # 画图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, metric in zip(axes.flat, ["annualized_return", "sharpe", "max_drawdown", "total_return"]):
        ax.hist(sim_df[metric], bins=50, edgecolor="black", alpha=0.7)
        ax.axvline(base_metrics[metric], color="red", linestyle="--", label="Baseline")
        ax.set_title(metric)
        ax.set_xlabel(metric)
        ax.set_ylabel("频数")
        ax.legend()
    plt.tight_layout()
    path = os.path.join(args.outdir, "monte_carlo_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"分布图已保存: {path}")


# ---------------------------------------------------------------------------
# 4. 样本外/前向验证
# ---------------------------------------------------------------------------
def mode_out_of_sample(args):
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    prices = pd.DataFrame({s: load_asset(os.path.join(args.data_dir, s)) for s in symbols})
    prices = prices.dropna()

    train_prices = prices[prices.index <= args.train_end]
    test_prices = prices[prices.index > args.train_end]

    print(f"=== 样本外验证：训练期 {train_prices.index[0].date()} ~ {args.train_end}，测试期 {test_prices.index[0].date()} ~ {prices.index[-1].date()} ===")

    # 训练期网格搜索
    period_range = [int(x) for x in args.period_range.split(",")]
    top_n_range = [int(x) for x in args.top_n_range.split(",")]

    best = None
    best_score = -np.inf
    for period in period_range:
        score = calc_trend_score(train_prices, period)
        for top_n in top_n_range:
            weights = generate_weights(train_prices, score, top_n)
            nav, _ = run_backtest(train_prices, weights, args.fee)
            metrics = calc_metrics(nav, ann_factor=args.ann_factor, risk_free=args.risk_free)
            # 以夏普为优化目标
            if metrics["sharpe"] > best_score:
                best_score = metrics["sharpe"]
                best = (period, top_n)

    print(f"训练期最优参数: period={best[0]}, top_n={best[1]}, 训练夏普={best_score:.2f}")

    # 测试期只用最优参数
    score = calc_trend_score(test_prices, best[0])
    weights = generate_weights(test_prices, score, best[1])
    nav, _ = run_backtest(test_prices, weights, args.fee)
    metrics = calc_metrics(nav, ann_factor=args.ann_factor, risk_free=args.risk_free)

    print(f"测试期指标:")
    print(f"区间: {metrics['start_date']} ~ {metrics['end_date']}")
    for k in ["total_return", "annualized_return", "volatility", "max_drawdown"]:
        print(f"{k}: {metrics[k]:.2%}")
    for k in ["sharpe", "calmar"]:
        print(f"{k}: {metrics[k]:.2f}")

    os.makedirs(args.outdir, exist_ok=True)
    pd.DataFrame([{"period": best[0], "top_n": best[1], **metrics}]).to_csv(
        os.path.join(args.outdir, "out_of_sample_metrics.csv"),
        index=False,
        encoding="utf-8-sig",
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    start_time = time.time()

    if args.mode == "rolling":
        mode_rolling(args)
    elif args.mode == "parameter":
        mode_parameter(args)
    elif args.mode == "monte_carlo":
        mode_monte_carlo(args)
    elif args.mode == "out_of_sample":
        mode_out_of_sample(args)

    print(f"总运行时间: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()
