#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 akshare 拉取场内 ETF 日线行情（后复权），并保存为 CSV。

与 01/02 的区别：01/02 拉的是场外基金净值（.OF），本脚本拉的是场内 ETF
交易所行情（.SH/.SZ），是真实可交易的标的价格。

用法示例:
    python 10_akshare_etf_data.py --symbols='513100.SH,512890.SH,159915.SZ,518880.SH'

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

# 默认 ETF 列表：纳指100 / 红利低波 / 创业板 / 黄金
DEFAULT_SYMBOLS = "513100.SH,512890.SH,159915.SZ,518880.SH"


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="使用 akshare 拉取场内 ETF 日线行情（后复权）"
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=DEFAULT_SYMBOLS,
        help=f"ETF 代码列表（带 .SH/.SZ 后缀），逗号分隔。默认: {DEFAULT_SYMBOLS}",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=DEFAULT_OUTDIR,
        help=f"输出目录，默认: {DEFAULT_OUTDIR}",
    )
    return parser.parse_args()


def fetch_etf_daily(symbol: str) -> pd.DataFrame:
    """
    使用 akshare 拉取指定场内 ETF 的日线行情（后复权收盘）。

    symbol 形如 513100.SH；akshare 接口只需要 6 位数字代码。
    返回 DataFrame，包含 date, close 两列。
    """
    code = symbol.split(".")[0]
    df = ak.fund_etf_hist_em(symbol=code, period="daily", adjust="hfq")
    if df is None or df.empty:
        raise ValueError(f"未获取到 {symbol} 的行情数据")

    df = df.rename(columns={"日期": "date", "收盘": "close"})[["date", "close"]]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date", "close"])
    return df.sort_values("date").reset_index(drop=True)


def save_to_csv(df: pd.DataFrame, symbol: str, outdir: str) -> str:
    """将 DataFrame 保存为 CSV 文件。"""
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{symbol}.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def main() -> int:
    args = parse_args()
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    for symbol in symbols:
        print(f"正在获取: {symbol}")
        try:
            df = fetch_etf_daily(symbol)
            path = save_to_csv(df, symbol, args.outdir)
            print(f"已保存: {path}，共 {len(df)} 条记录")
        except Exception as exc:  # pragma: no cover
            print(f"获取 {symbol} 失败: {exc}")
            continue

    return 0


if __name__ == "__main__":
    sys.exit(main())
