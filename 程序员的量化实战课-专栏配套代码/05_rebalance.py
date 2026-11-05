"""
极简再平衡回测：支持 N 个资产按目标权重周期性再平衡。

示例：
    python 05_rebalance.py --weights='050002.OF.csv:0.6,000218.OF.csv:0.4' --freq=M
"""
import argparse
import os

import pandas as pd

from engine import backtest
from utils import calc_metrics, plot_nav_and_weights


DEFAULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="极简再平衡回测")
    parser.add_argument(
        "--weights",
        type=str,
        default="007466.OF.csv:0.7,110017.OF.csv:0.3",
        help="逗号分隔的 '文件名:权重' 列表，权重之和需为 1",
    )
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DIR, help="CSV 数据目录")
    parser.add_argument("--outdir", type=str, default=DEFAULT_DIR, help="输出目录")
    parser.add_argument("--freq", type=str, default="M", choices=["M", "Q", "Y"], help="再平衡频率：M=月，Q=季，Y=年")
    parser.add_argument("--fee", type=float, default=0.0001, help="交易成本，默认万分之一")
    return parser.parse_args()


def load_asset(path: str) -> pd.DataFrame:
    """读取单个资产的 CSV，返回 date、close 两列。"""
    df = pd.read_csv(path, parse_dates=["date"], usecols=["date", "close"])
    return df.sort_values("date").drop_duplicates("date")


def parse_weights(weight_str: str) -> dict:
    """解析 'a.csv:0.6,b.csv:0.4' 为 {文件名: 权重}。"""
    weights = {}
    for item in weight_str.split(","):
        file, w = item.strip().split(":")
        weights[file.strip()] = float(w.strip())
    return weights


def get_rebalance_dates(prices: pd.DataFrame, freq: str) -> set:
    """按频率取每个周期最后一个交易日，用于触发再平衡。"""
    period = prices.index.to_period(freq)
    return set(prices.index[~period.duplicated(keep="last")])


def load_fund_name_map(data_dir: str) -> dict:
    """从 basic.csv 读取基金名称映射，{ts_code: name}。"""
    basic_path = os.path.join(data_dir, "basic.csv")
    if not os.path.exists(basic_path):
        return {}
    df = pd.read_csv(basic_path, usecols=["ts_code", "name"])
    return dict(zip(df["ts_code"], df["name"]))


def main():
    args = parse_args()

    target_weights = parse_weights(args.weights)

    # 合并所有资产的价格序列
    prices = None
    for file, w in target_weights.items():
        df = load_asset(os.path.join(args.data_dir, file)).rename(columns={"close": file})
        if prices is None:
            prices = df
        else:
            prices = pd.merge(prices, df, on="date", how="inner")

    if prices is None or prices.empty:
        raise ValueError("合并后的价格序列为空")

    prices = prices.set_index("date").sort_index()

    # 构建目标权重矩阵：再平衡日设为 target_weights，其余日期保持前一日权重
    rebalance_dates = get_rebalance_dates(prices, args.freq)
    weights_target = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    current_w = None
    for date in prices.index:
        if date in rebalance_dates:
            current_w = target_weights
        # 如果是第一天且非再平衡日，用目标权重初始化
        if current_w is None:
            current_w = target_weights
        for asset in prices.columns:
            weights_target.loc[date, asset] = current_w[asset]

    # 运行回测
    nav, weights = backtest(prices, weights_target, args.fee)
    metrics = calc_metrics(nav)

    # 输出指标
    weight_desc = ",".join(f"{k}:{v:.0%}" for k, v in target_weights.items())
    print(f"=== 再平衡回测 ({weight_desc}) ===")
    print(f"区间: {metrics['start_date']} ~ {metrics['end_date']}")
    for k, v in metrics.items():
        if k not in ("sharpe", "calmar", "start_date", "end_date"):
            print(f"{k}: {v:.2%}")
    print(f"sharpe: {metrics['sharpe']:.2f}")
    print(f"calmar: {metrics['calmar']:.2f}")

    # 保存结果
    os.makedirs(args.outdir, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(os.path.join(args.outdir, "rebalance_metrics.csv"),
                                   index=False, encoding="utf-8-sig")

    # 构建基金名称映射
    fund_name_map = load_fund_name_map(args.data_dir)
    name_map = {file: fund_name_map.get(file.replace(".csv", ""), file) for file in target_weights.keys()}
    plot_path = plot_nav_and_weights(prices, nav, weights, args.outdir,
                                     title=f"再平衡回测 ({weight_desc})",
                                     name_map=name_map,
                                     filename="rebalance.png")
    print(f"图表已保存: {plot_path}")


if __name__ == "__main__":
    main()
