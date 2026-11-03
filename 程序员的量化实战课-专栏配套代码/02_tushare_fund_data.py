#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 Tushare 拉取场外基金历史净值（复权），并保存为 CSV。

用法示例:
    python 02_tushare_fund_data.py --symbols='050002.OF,270042.OF,110026.OF,000218.OF'
    python 02_tushare_fund_data.py --symbols='000001,000003,519915' --outdir=E:/agilab/data

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


# 默认输出目录：项目根目录下的 data/
DEFAULT_OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# 默认基金列表（Tushare 格式代码，用 .OF 后缀）
DEFAULT_SYMBOLS = "050002.OF,270042.OF,110026.OF,000218.OF"


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="使用 Tushare 拉取场外基金复权净值序列"
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
        # config.py 位于项目根目录
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from config import config
        return config.TS_TOKEN or ""
    except Exception:
        return ""


def resolve_fund_symbol(name_or_code: str) -> str:
    """
    将用户输入的基金代码规范化成 Tushare 格式（XXXXXX.OF）。
    纯数字补全 .OF 后缀；已经是 .OF 格式直接返回；其他原样返回。
    """
    name_or_code = name_or_code.strip()
    if name_or_code.isdigit():
        return f"{name_or_code}.OF"
    if name_or_code.upper().endswith(".OF"):
        return name_or_code
    return name_or_code


def fetch_fund_nav(fund_code: str, pro) -> pd.DataFrame:
    """
    使用 Tushare 拉取指定场外基金的历史净值，并计算复权净值。

    返回 DataFrame，包含 date, close 两列。
    """
    # Tushare 接口：fund_nav（场外基金净值）
    # 返回字段包含 ts_code, ann_date, nav, acc_nav 等
    df = pro.fund_nav(ts_code=fund_code)

    if df is None or df.empty:
        raise ValueError(f"未获取到 {fund_code} 的净值数据")

    # update_flag 字段说明（Tushare 实际数据观察）：
    # '0' 通常是日常净值记录；'1' 通常是月/季末填充的汇总记录（ann_date 和 nav_date 不一致）。
    # 策略：同一天如果同时存在 update_flag='1' 和 '0'，优先保留 '1'；
    #      否则保留 '0'。即按日期去重，优先取 update_flag == '1' 的数据。
    if "update_flag" in df.columns:
        df["update_flag"] = df["update_flag"].fillna("0").astype(str)
        # 按日期排序，update_flag='1' 排前面
        df = df.sort_values("ann_date", ascending=True)
        df = df.sort_values("update_flag", ascending=True, kind="stable")
        # 按 ann_date 去重，保留第一条（即 update_flag='1' 优先）
        df = df.drop_duplicates(subset=["ann_date"], keep="first").copy()

    if df.empty:
        raise ValueError(f"{fund_code} 数据为空")

    # 字段说明：
    # ann_date: 公告日期；nav_date: 净值日期；unit_nav: 单位净值；accum_nav: 累计净值；adj_nav: 复权净值
    df = df.rename(
        columns={
            "nav_date": "date",
            "unit_nav": "nav",
            "accum_nav": "acc_nav",
            "adj_nav": "adj_nav",
        }
    )

    # 保留有效列
    df = df[["date", "nav", "acc_nav", "adj_nav"]].copy()

    # 转换日期格式
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # 将净值转为浮点
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    df["acc_nav"] = pd.to_numeric(df["acc_nav"], errors="coerce")
    df["adj_nav"] = pd.to_numeric(df["adj_nav"], errors="coerce")
    df = df.dropna(subset=["nav", "acc_nav", "adj_nav"])

    # 按日期从早到晚排序
    df = df.sort_values("date").reset_index(drop=True)

    # 如果 Tushare 已经提供复权净值 adj_nav，直接用它作为 close
    df["close"] = df["adj_nav"]

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

    token = args.token or load_token_from_config()
    if not token:
        print("错误：未提供 Tushare Pro token。请通过 --token 传入或在 ~/.agilab/config.toml 中配置 TS_TOKEN")
        return 1

    pro = ts.pro_api(token)

    for raw in symbols:
        fund_code = resolve_fund_symbol(raw)
        print(f"正在获取: {raw} -> {fund_code}")
        try:
            df = fetch_fund_nav(fund_code, pro)
            path = save_to_csv(df, fund_code, outdir)
            print(f"已保存: {path}，共 {len(df)} 条记录")
        except Exception as exc:  # pragma: no cover
            print(f"获取 {fund_code} 失败: {exc}")
            continue

    return 0


if __name__ == "__main__":
    sys.exit(main())
