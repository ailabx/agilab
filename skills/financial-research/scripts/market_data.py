#!/usr/bin/env python3
"""market_data.py —— 获取股票/ETF/期货/可转债/场外基金的原始行情数据。

用法：
    python market_data.py 600519 --market stock --days 120
    python market_data.py 510300 --market etf --days 120
    python market_data.py RB --market futures --days 120

输出 JSON：
{
  "meta": {"symbol": "600519", "market": "stock", "days": 120},
  "data": [
    {"trade_date": "20240101", "open": 1500.0, "high": 1520.0,
     "low": 1490.0, "close": 1510.0, "vol": 10000, ...},
    ...
  ],
  "latest": {"close": 1510.0, "open": 1500.0, "high": 1520.0,
             "low": 1490.0, "vol": 10000, "trade_date": "20240101"}
}

本脚本只获取原始数据，不做计算、不调用 LLM。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

# 项目根目录：脚本在 skills/financial-research/scripts/，向上 3 层
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from config import config
import tushare as ts


def _get_api():
    return ts.pro_api(config.TS_TOKEN)


def _normalize_symbol(symbol: str, market: str) -> str:
    """标准化标的代码。"""
    symbol = symbol.strip().upper()
    if market == "stock":
        if "." in symbol:
            return symbol
        # 简单按规则加交易所后缀
        if symbol.startswith(("60", "68", "90")):
            return f"{symbol}.SH"
        if symbol.startswith(("00", "30", "12")):
            return f"{symbol}.SZ"
        if symbol.startswith(("43", "83", "87", "88")):
            return f"{symbol}.BJ"
        return f"{symbol}.SH"
    if market == "etf":
        if "." in symbol:
            return symbol
        return f"{symbol}.SH" if symbol.startswith(("51", "58")) else f"{symbol}.SZ"
    if market == "futures":
        return symbol
    if market == "convertible":
        if "." in symbol:
            return symbol
        return f"{symbol}.SH" if symbol.startswith("11") else f"{symbol}.SZ"
    if market == "fund":
        if "." in symbol:
            return symbol
        return f"{symbol}.OF"
    return symbol


def _get_stock_daily(symbol: str, days: int) -> pd.DataFrame:
    api = _get_api()
    ts_code = _normalize_symbol(symbol, "stock")
    end = pd.Timestamp.now()
    start = end - pd.Timedelta(days=days + 60)
    df = ts.pro_bar(ts_code=ts_code, api=api, start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d"))
    return df


def _get_etf_daily(symbol: str, days: int) -> pd.DataFrame:
    api = _get_api()
    ts_code = _normalize_symbol(symbol, "etf")
    end = pd.Timestamp.now()
    start = end - pd.Timedelta(days=days + 60)
    df = api.fund_daily(ts_code=ts_code, start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d"))
    return df


def _get_futures_daily(symbol: str, days: int) -> pd.DataFrame:
    api = _get_api()
    end = pd.Timestamp.now()
    start = end - pd.Timedelta(days=days + 60)
    # 尝试按品种代码获取主力合约映射
    try:
        mapping = api.fut_mapping(ts_code=f"{symbol.upper()}.SHF", start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d"))
        if not mapping.empty:
            ts_code = mapping.iloc[-1]["mapping_code"]
        else:
            ts_code = f"{symbol.upper()}.SHF"
    except Exception:
        ts_code = f"{symbol.upper()}.SHF"
    df = api.fut_daily(ts_code=ts_code, start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d"))
    return df


def _get_convertible_daily(symbol: str, days: int) -> pd.DataFrame:
    api = _get_api()
    ts_code = _normalize_symbol(symbol, "convertible")
    end = pd.Timestamp.now()
    start = end - pd.Timedelta(days=days + 60)
    df = api.cb_daily(ts_code=ts_code, start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d"))
    return df


def _get_fund_daily(symbol: str, days: int) -> pd.DataFrame:
    api = _get_api()
    ts_code = _normalize_symbol(symbol, "fund")
    end = pd.Timestamp.now()
    start = end - pd.Timedelta(days=days + 60)
    df = api.fund_nav(ts_code=ts_code, start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d"))
    # 统一列名：nav_date -> trade_date，unit_nav -> close
    if not df.empty:
        df = df.rename(columns={"nav_date": "trade_date", "unit_nav": "close"})
        df["open"] = df["close"]
        df["high"] = df["close"]
        df["low"] = df["close"]
        df["vol"] = 0
        df["amount"] = 0
    return df


def get_market_data(symbol: str, market: str, days: int) -> dict:
    handlers = {
        "stock": _get_stock_daily,
        "etf": _get_etf_daily,
        "futures": _get_futures_daily,
        "convertible": _get_convertible_daily,
        "fund": _get_fund_daily,
    }
    handler = handlers.get(market)
    if not handler:
        raise ValueError(f"不支持的品类: {market}")

    normalized = _normalize_symbol(symbol, market)

    df = handler(symbol, days)
    if df is None or df.empty:
        return {
            "meta": {"symbol": symbol, "ts_code": normalized, "market": market, "days": days},
            "data": [],
            "latest": {},
        }

    # 统一列名
    df = df.rename(columns={
        "ts_code": "symbol",
        "trade_date": "trade_date",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "vol": "vol",
        "amount": "amount",
    })
    df = df.sort_values("trade_date").reset_index(drop=True)
    latest = df.iloc[-1].to_dict()

    return {
        "meta": {"symbol": symbol, "ts_code": normalized, "market": market, "days": days},
        "data": df.to_dict("records"),
        "latest": latest,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="获取金融标的原始行情数据")
    ap.add_argument("symbol", help="标的代码或名称")
    ap.add_argument("--market", default="stock", choices=["stock", "etf", "futures", "convertible", "fund"])
    ap.add_argument("--days", type=int, default=120)
    args = ap.parse_args()

    try:
        result = get_market_data(args.symbol, args.market, args.days)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    print(json.dumps(result, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
