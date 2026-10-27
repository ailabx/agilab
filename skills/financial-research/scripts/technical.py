#!/usr/bin/env python3
"""technical.py —— 基于行情数据计算技术指标。

输入：market_data.py 的输出 JSON（通过 stdin 或 --file）
输出 JSON：
{
  "trend": {"ma5": ..., "ma20": ..., "ma60": ..., "state": "up"},
  "momentum": {"rsi": ..., "macd": ..., "macd_signal": ..., "macd_hist": ...},
  "volatility": {"boll_upper": ..., "boll_mid": ..., "boll_lower": ..., "atr": ...},
  "volume": {"vol_ratio": ..., "state": "increase"}
}

纯计算，不调 LLM。
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd


def _load_input(args) -> dict:
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            return json.load(f)
    if args.stdin or not sys.stdin.isatty():
        return json.load(sys.stdin)
    raise ValueError("请通过 --file 或 stdin 提供输入 JSON")


def _ma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=window, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
    rs = gain / (loss + 1e-10)
    return 100 - 100 / (1 + rs)


def _macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist


def _bollinger(series: pd.Series, window=20):
    ma = _ma(series, window)
    std = series.rolling(window=window, min_periods=1).std()
    upper = ma + 2 * std
    lower = ma - 2 * std
    return upper, ma, lower


def _atr(df: pd.DataFrame, window=14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=window, min_periods=1).mean()


def analyze_technical(data: dict) -> dict:
    records = data.get("data", [])
    if not records:
        return {"error": "无行情数据"}

    df = pd.DataFrame(records)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["high"] = pd.to_numeric(df["high"], errors="coerce")
    df["low"] = pd.to_numeric(df["low"], errors="coerce")
    df["open"] = pd.to_numeric(df["open"], errors="coerce")
    df["vol"] = pd.to_numeric(df["vol"], errors="coerce")

    latest = df.iloc[-1]

    # 趋势
    ma5 = _ma(df["close"], 5).iloc[-1]
    ma20 = _ma(df["close"], 20).iloc[-1]
    ma60 = _ma(df["close"], 60).iloc[-1]
    trend_state = "up" if latest["close"] > ma20 > ma60 else "down" if latest["close"] < ma20 < ma60 else "sideways"

    # 动量
    rsi = _rsi(df["close"], 14).iloc[-1]
    macd, macd_signal, macd_hist = _macd(df["close"])

    # 波动率
    upper, mid, lower = _bollinger(df["close"])
    atr = _atr(df).iloc[-1]

    # 成交量
    vol_ratio = latest["vol"] / _ma(df["vol"], 20).iloc[-1] if "vol" in df.columns else None
    vol_state = "increase" if vol_ratio and vol_ratio > 1.2 else "decrease" if vol_ratio and vol_ratio < 0.8 else "normal"

    return {
        "trend": {
            "ma5": round(ma5, 4),
            "ma20": round(ma20, 4),
            "ma60": round(ma60, 4),
            "state": trend_state,
        },
        "momentum": {
            "rsi": round(rsi, 2),
            "macd": round(macd.iloc[-1], 4),
            "macd_signal": round(macd_signal.iloc[-1], 4),
            "macd_hist": round(macd_hist.iloc[-1], 4),
        },
        "volatility": {
            "boll_upper": round(upper.iloc[-1], 4),
            "boll_mid": round(mid.iloc[-1], 4),
            "boll_lower": round(lower.iloc[-1], 4),
            "atr": round(atr, 4),
        },
        "volume": {
            "vol_ratio": round(vol_ratio, 2) if vol_ratio is not None else None,
            "state": vol_state,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="基于行情数据计算技术指标")
    ap.add_argument("--file", help="输入 JSON 文件路径")
    ap.add_argument("--stdin", action="store_true", help="从 stdin 读取输入 JSON")
    args = ap.parse_args()

    try:
        data = _load_input(args)
        result = analyze_technical(data)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    print(json.dumps(result, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
