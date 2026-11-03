#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 Tushare 拉取场外基金基础信息，保存为 data/basic.csv。

用法示例:
    python 04_tushare_fund_basic.py --outdir=E:/agilab/data
"""

import argparse
import os
import sys

import pandas as pd

try:
    import tushare as ts
except ImportError as exc:  # pragma: no cover
    raise ImportError("请先安装 tushare: pip install tushare") from exc


# 默认输出目录：当前脚本目录下的 data/
DEFAULT_OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="使用 Tushare 拉取场外基金基础信息"
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


def fetch_fund_basic(pro) -> pd.DataFrame:
    """使用 Tushare 拉取场外基金基础信息，分页获取全部数据。"""
    df_list = []
    offset = 0
    limit = 10000
    while True:
        df = pro.fund_basic(market="O", offset=str(offset), limit=str(limit))
        if df is None or df.empty:
            break
        df_list.append(df)
        if len(df) < limit:
            break
        offset += limit

    if not df_list:
        raise ValueError("未获取到基金基础信息")

    return pd.concat(df_list, ignore_index=True)


def save_to_csv(df: pd.DataFrame, outdir: str) -> str:
    """将 DataFrame 保存为 CSV 文件。"""
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "basic.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def main() -> int:
    args = parse_args()
    outdir = args.outdir

    token = args.token or load_token_from_config()
    if not token:
        print("错误：未提供 Tushare Pro token。请通过 --token 传入或在 ~/.agilab/config.toml 中配置 TS_TOKEN")
        return 1

    pro = ts.pro_api(token)

    print("正在获取场外基金基础信息...")
    df = fetch_fund_basic(pro)
    path = save_to_csv(df, outdir)
    print(f"已保存: {path}，共 {len(df)} 条记录")

    return 0


if __name__ == "__main__":
    sys.exit(main())
