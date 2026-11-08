"""
全市场 ETF 轮动策略 v2：大类轮动 + Top3 等权 + 趋势过滤

标的池：
- A 股宽基：沪深300、中证500、中证1000、科创50、创业板、上证50
- A 股行业：红利低波、证券、医药、半导体、新能源、军工、地产、银行
- 跨境：纳指100、标普500、恒生科技、恒生ETF
- 商品：黄金
- 债券：国债ETF

规则：
1. 先按大类分池，每个大类内部 ROC(60) 选最强的一只；
2. 大类之间再 ROC(60) 排序，选 top 3 等权配置；
3. 若沪深300 收盘价 < 60 日均线，且所有大类 ROC < 0，则空仓；
4. 只在目标权重变化日调仓。
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

from engine import backtest
from utils import calc_metrics, plot_nav_and_weights


DEFAULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# 大类分组
CATEGORY_MAP = {
    "510300.SH.csv": "A股宽基",
    "510500.SH.csv": "A股宽基",
    "512100.SH.csv": "A股宽基",
    "588000.SH.csv": "A股宽基",
    "159915.SZ.csv": "A股宽基",
    "510050.SH.csv": "A股宽基",
    "512890.SH.csv": "A股行业",
    "515880.SH.csv": "A股行业",
    "512010.SH.csv": "A股行业",
    "512480.SH.csv": "A股行业",
    "515030.SH.csv": "A股行业",
    "512660.SH.csv": "A股行业",
    "512200.SH.csv": "A股行业",
    "512800.SH.csv": "A股行业",
    "513100.SH.csv": "跨境",
    "513500.SH.csv": "跨境",
    "513050.SH.csv": "跨境",
    "159920.SZ.csv": "跨境",
    "518880.SH.csv": "商品",
    "511010.SH.csv": "债券",
}

NAME_MAP = {
    "510300.SH.csv": "沪深300",
    "510500.SH.csv": "中证500",
    "512100.SH.csv": "中证1000",
    "588000.SH.csv": "科创50",
    "159915.SZ.csv": "创业板",
    "510050.SH.csv": "上证50",
    "512890.SH.csv": "红利低波",
    "515880.SH.csv": "证券",
    "512010.SH.csv": "医药",
    "512480.SH.csv": "半导体",
    "515030.SH.csv": "新能源",
    "512660.SH.csv": "军工",
    "512200.SH.csv": "地产",
    "512800.SH.csv": "银行",
    "513100.SH.csv": "纳指100",
    "513500.SH.csv": "标普500",
    "513050.SH.csv": "恒生科技",
    "159920.SZ.csv": "恒生ETF",
    "518880.SH.csv": "黄金",
    "511010.SH.csv": "国债ETF",
}

DEFAULT_SYMBOLS = ",".join(NAME_MAP.keys())


def parse_args():
    parser = argparse.ArgumentParser(description="全市场 ETF 轮动 v2")
    parser.add_argument(
        "--symbols",
        type=str,
        default=DEFAULT_SYMBOLS,
        help="标的 CSV 文件名，逗号分隔",
    )
    parser.add_argument("--roc-period", type=int, default=60, help="ROC 计算周期")
    parser.add_argument("--top-n", type=int, default=3, help="每次持有大类数量")
    parser.add_argument("--trend-symbol", type=str, default="510300.SH.csv", help="趋势过滤标的")
    parser.add_argument("--trend-period", type=int, default=60, help="趋势过滤均线周期")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DIR, help="CSV 数据目录")
    parser.add_argument("--outdir", type=str, default=DEFAULT_DIR, help="输出目录")
    parser.add_argument("--fee", type=float, default=0.0001, help="交易成本")
    parser.add_argument("--ann-factor", type=int, default=252, help="年化因子")
    parser.add_argument("--risk-free", type=float, default=0.02, help="无风险利率")
    return parser.parse_args()


def load_asset(path: str) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["date"], usecols=["date", "close"])
    df = df.sort_values("date").drop_duplicates("date").set_index("date")
    return df["close"]


def generate_weights(prices: pd.DataFrame, roc_period: int, top_n: int,
                     trend_ma: pd.Series = None) -> pd.DataFrame:
    """
    大类内部选最强，再选 top 3 大类等权。
    """
    roc = prices / prices.shift(roc_period) - 1.0
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    for i, date in enumerate(prices.index):
        if i < roc_period:
            continue

        # 趋势过滤：若趋势标的价格 < 均线，且当日所有标的 ROC < 0，空仓
        if trend_ma is not None and not np.isnan(trend_ma.loc[date]):
            trend_symbol = trend_ma.name
            # 这里 trend_ma 是价格序列的均线，单独处理
        else:
            trend_symbol = None

    return weights


def main():
    args = parse_args()
    start_time = time.time()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    prices = pd.DataFrame({s: load_asset(os.path.join(args.data_dir, s)) for s in symbols})
    prices = prices.dropna()

    # 计算大类内部最强
    roc = prices / prices.shift(args.roc_period) - 1.0

    # 趋势过滤序列
    trend_prices = prices[args.trend_symbol]
    trend_ma = trend_prices.rolling(window=args.trend_period).mean()

    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    # 按大类分组
    categories = {}
    for s in symbols:
        cat = CATEGORY_MAP.get(s, "其他")
        categories.setdefault(cat, []).append(s)

    for i, date in enumerate(prices.index):
        if i < args.roc_period:
            continue

        # 趋势过滤：沪深300 收盘价 < 60 日均线，且所有标的 ROC < 0，空仓
        if trend_prices.loc[date] < trend_ma.loc[date] and (roc.loc[date] < 0).all():
            continue

        # 每个大类选 ROC 最强的
        category_best = {}
        for cat, cols in categories.items():
            best = roc.loc[date, cols].idxmax()
            category_best[cat] = best

        # 大类 ROC 排序，选 top_n
        cat_roc = pd.Series({cat: roc.loc[date, best] for cat, best in category_best.items()})
        cat_roc = cat_roc[cat_roc >= 0]
        if cat_roc.empty:
            continue

        top_cats = cat_roc.sort_values(ascending=False).head(args.top_n).index
        selected = [category_best[cat] for cat in top_cats]
        w = 1.0 / len(selected)
        for s in selected:
            weights.loc[date, s] = w

    rebalance_mask = weights.ne(weights.shift(1)).any(axis=1)
    rebalance_dates = set(weights[rebalance_mask].index)
    nav, weights_bt = backtest(prices, weights, args.fee, rebalance_dates=rebalance_dates)
    metrics = calc_metrics(nav, ann_factor=args.ann_factor, risk_free=args.risk_free)

    elapsed = time.time() - start_time

    print(f"=== 全市场 ETF 轮动 v2（大类轮动 + Top{args.top_n} + 趋势过滤）===")
    print(f"区间: {metrics['start_date']} ~ {metrics['end_date']}")
    print(f"标的数: {len(symbols)}")
    for k, v in metrics.items():
        if k in ("total_return", "annualized_return", "volatility", "max_drawdown"):
            print(f"{k}: {v:.2%}")
        elif k in ("sharpe", "calmar"):
            print(f"{k}: {v:.2f}")
    print(f"运行时间: {elapsed:.4f}s")

    os.makedirs(args.outdir, exist_ok=True)
    pd.DataFrame([{"strategy": "full_etf_v2", **metrics}]).to_csv(
        os.path.join(args.outdir, "full_etf_v2_metrics.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    plot_path = plot_nav_and_weights(
        prices,
        nav,
        weights_bt,
        args.outdir,
        title="全市场 ETF 轮动 v2（大类轮动 + Top3 + 趋势过滤）",
        name_map={s: NAME_MAP.get(s, s) for s in symbols},
        filename="full_etf_rotation_v2.png",
    )
    print(f"图表已保存: {plot_path}")


if __name__ == "__main__":
    main()
