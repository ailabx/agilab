#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 Tushare 拉取场内 ETF 日行情（复权），并保存为 CSV。

用法示例:
    python 02_tushare_etf_data.py --symbols='159915.SZ,512890.SH,518880.SH,513100.SH'

输出文件:
    <outdir>/<symbol>.csv  (字段: date, close)
"""

import argparse
import os
import sys

import pandas as pd

try:
    import tushare as ts
except ImportError as exc:  # pragma: no cover
    raise ImportError("请先安装 tushare: pip install tushare") from exc


DEFAULT_OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

DEFAULT_SYMBOLS = "159915.SZ,512890.SH,518880.SH,513100.SH"


# 中文名称映射
NAME_MAP = {
    "159915.SZ": "创业板",
    "512890.SH": "红利低波",
    "518880.SH": "黄金",
    "513100.SH": "纳指100",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 Tushare 拉取场内 ETF 日行情")
    parser.add_argument(
        "--symbols",
        type=str,
        default=DEFAULT_SYMBOLS,
        help=f"ETF 代码列表，逗号分隔。默认: {DEFAULT_SYMBOLS}",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=DEFAULT_OUTDIR,
        help=f"输出目录，默认: {DEFAULT_OUTDIR}",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Tushare Pro token；默认从 ~/.agilab/config.toml 读取 TS_TOKEN",
    )
    return parser.parse_args()


def load_token_from_config() -> str:
    """从项目配置文件读取 Tushare token。"""
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from config import config
        return config.TS_TOKEN or ""
    except Exception:
        return ""


def fetch_etf_daily(fund_code: str, pro) -> pd.DataFrame:
    """
    使用 Tushare fund_daily + fund_adj 接口拉取指定场内 ETF 的日行情，
    并使用 fund_adj 复权因子做后复权。

    后复权公式：close_adj = close * adj_factor / latest_adj_factor
    其中 latest_adj_factor 取最新交易日的复权因子。
    """
    df = pro.fund_daily(ts_code=fund_code)

    if df is None or df.empty:
        raise ValueError(f"未获取到 {fund_code} 的行情数据")

    # 转换日期
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["trade_date"])
    df = df.sort_values("trade_date").reset_index(drop=True)

    # 拉取复权因子并做后复权：close_adj = close * adj_factor
    df_factor = pro.fund_adj(ts_code=fund_code)
    if df_factor is None or df_factor.empty:
        print(f"  警告: {fund_code} 未获取到 fund_adj 复权因子，使用原始 close")
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    else:
        df_factor["trade_date"] = pd.to_datetime(
            df_factor["trade_date"], format="%Y%m%d", errors="coerce"
        )
        df_factor = df_factor.dropna(subset=["trade_date"])
        df_factor["adj_factor"] = pd.to_numeric(df_factor["adj_factor"], errors="coerce")
        df = df.merge(
            df_factor[["trade_date", "adj_factor"]], on="trade_date", how="left"
        )
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce") * df["adj_factor"]

    df = df.rename(columns={"trade_date": "date"})
    df = df[["date", "open", "high", "low", "close", "vol"]]
    return df


def save_to_csv(df: pd.DataFrame, fund_code: str, outdir: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{fund_code}.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def main() -> int:
    args = parse_args()
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    outdir = args.outdir

    token = args.token or load_token_from_config()
    if not token:
        print("错误：未提供 Tushare Pro token。请通过 --token 传入或在 ~/.agilab/config.toml 中配置 TS_TOKEN")
        return 1

    pro = ts.pro_api(token)

    for raw in symbols:
        fund_code = raw
        print(f"正在获取: {NAME_MAP.get(fund_code, fund_code)} -> {fund_code}")
        try:
            df = fetch_etf_daily(fund_code, pro)
            path = save_to_csv(df, fund_code, outdir)
            print(f"已保存: {path}，共 {len(df)} 条记录")
        except Exception as exc:
            print(f"获取 {fund_code} 失败: {exc}")
            continue

    return 0


if __name__ == "__main__":
    sys.exit(main())
