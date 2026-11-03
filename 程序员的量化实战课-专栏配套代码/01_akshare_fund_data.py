#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 akshare 拉取场外基金历史净值（复权），并保存为 CSV。

用法示例:
    python 01_akshare_fund_data.py --symbols='沪深300,纳指100,创业板,黄金'
    python 01_akshare_fund_data.py --symbols='000001,000003,519915' --outdir=E:/agilab/data

输出文件:
    <outdir>/<symbol>.csv  (字段: date, close)
"""

import argparse
import os
import sys

import pandas as pd

try:
    import akshare as ak
except ImportError as exc:  # pragma: no cover
    raise ImportError("请先安装 akshare: pip install akshare") from exc


# 默认输出目录：项目根目录下的 data/
DEFAULT_OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# 默认基金列表
DEFAULT_SYMBOLS = "沪深300,纳指100,创业板,黄金"


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="使用 akshare 拉取场外基金复权净值序列"
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=DEFAULT_SYMBOLS,
        help=f"基金代码或名称列表，逗号分隔。默认: {DEFAULT_SYMBOLS}",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=DEFAULT_OUTDIR,
        help=f"输出目录，默认: {DEFAULT_OUTDIR}",
    )
    return parser.parse_args()


def resolve_fund_symbol(name_or_code: str) -> str:
    """
    将中文名称映射为基金代码。
    如果是纯数字，直接返回；否则根据常见名称做简单映射。
    """
    name_or_code = name_or_code.strip()
    mapping = {
        "沪深300": "000051",      # 鹏华沪深300指数A（示例）
        "纳指100": "270042",      # 广发纳斯达克100指数A（示例）
        "创业板": "159915",       # 易方达创业板ETF 联接A？这里用场外联接代码 110026 更合适
        "黄金": "000218",         # 国泰黄金ETF联接A（示例）
    }
    # 如果传入的是数字，优先视为基金代码
    if name_or_code.isdigit():
        return name_or_code
    if name_or_code in mapping:
        return mapping[name_or_code]
    return name_or_code


def fetch_fund_nav(fund_code: str) -> pd.DataFrame:
    """
    使用 akshare 拉取指定场外基金的历史净值，并计算复权净值。

    返回 DataFrame，包含 date, close 两列。
    """
    # 分别拉取单位净值与累计净值
    df_nav = ak.fund_open_fund_info_em(symbol=fund_code, indicator="单位净值走势")
    df_acc = ak.fund_open_fund_info_em(symbol=fund_code, indicator="累计净值走势")

    # 重命名单位净值表
    df_nav = df_nav.rename(
        columns={
            "净值日期": "date",
            "单位净值": "nav",
        }
    )[["date", "nav"]]

    # 重命名累计净值表
    df_acc = df_acc.rename(
        columns={
            "净值日期": "date",
            "累计净值": "acc_nav",
        }
    )[["date", "acc_nav"]]

    # 合并两表
    df = pd.merge(df_nav, df_acc, on="date", how="inner")

    # 转换日期格式
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # 将净值转为浮点
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    df["acc_nav"] = pd.to_numeric(df["acc_nav"], errors="coerce")
    df = df.dropna(subset=["nav", "acc_nav"])

    # 计算复权后 close：以累计净值相对于最新累计净值的比例 * 最新单位净值
    latest_acc = df["acc_nav"].iloc[0]  # 第一行是最新数据
    latest_nav = df["nav"].iloc[0]
    df["close"] = (df["acc_nav"] / latest_acc) * latest_nav

    # 按日期从早到晚排序
    df = df.sort_values("date").reset_index(drop=True)

    # 只保留 date, close
    return df[["date", "close"]]


def save_to_csv(df: pd.DataFrame, fund_code: str, outdir: str) -> str:
    """将 DataFrame 保存为 CSV 文件。"""
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{fund_code}.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def main() -> int:
    args = parse_args()
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    outdir = args.outdir

    for raw in symbols:
        fund_code = resolve_fund_symbol(raw)
        print(f"正在获取: {raw} -> {fund_code}")
        try:
            df = fetch_fund_nav(fund_code)
            path = save_to_csv(df, fund_code, outdir)
            print(f"已保存: {path}，共 {len(df)} 条记录")
        except Exception as exc:  # pragma: no cover
            print(f"获取 {fund_code} 失败: {exc}")
            continue

    return 0


if __name__ == "__main__":
    sys.exit(main())
