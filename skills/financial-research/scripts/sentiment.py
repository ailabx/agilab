#!/usr/bin/env python3
"""sentiment.py —— 基于行情数据计算情绪/资金指标。

输入：market_data.py 的输出 JSON
输出 JSON：价格位置、量能、趋势强度等情绪指标

纯计算，不调 LLM。
"""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd


def _load_input(args) -> dict:
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            return json.load(f)
    if args.stdin or not sys.stdin.isatty():
        return json.load(sys.stdin)
    raise ValueError("请通过 --file 或 stdin 提供输入 JSON")


def analyze_sentiment(data: dict) -> dict:
    records = data.get("data", [])
    if not records:
        return {"error": "无行情数据"}

    df = pd.DataFrame(records)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["high"] = pd.to_numeric(df["high"], errors="coerce")
    df["low"] = pd.to_numeric(df["low"], errors="coerce")
    df["vol"] = pd.to_numeric(df["vol"], errors="coerce")

    latest = df.iloc[-1]
    close = latest["close"]
    high_max = df["high"].max()
    low_min = df["low"].min()

    # 价格位置 0-1
    price_position = (close - low_min) / (high_max - low_min) if high_max != low_min else 0.5

    # 量能
    vol_ma20 = df["vol"].rolling(window=20, min_periods=1).mean().iloc[-1]
    vol_ratio = latest["vol"] / vol_ma20 if vol_ma20 else None

    # 趋势强度：最近 5 日斜率
    recent = df["close"].tail(5)
    slope = (recent.iloc[-1] - recent.iloc[0]) / 5 if len(recent) >= 2 else 0

    # 情绪状态
    if price_position > 0.8 and vol_ratio and vol_ratio > 1.2:
        state = "overbought"
    elif price_position < 0.2 and vol_ratio and vol_ratio > 1.2:
        state = "oversold"
    elif slope > 0:
        state = "bullish"
    elif slope < 0:
        state = "bearish"
    else:
        state = "neutral"

    return {
        "price_position": round(price_position, 4),
        "vol_ratio": round(vol_ratio, 2) if vol_ratio is not None else None,
        "slope": round(slope, 4),
        "trend_strength": state,
        "period_high": round(high_max, 4),
        "period_low": round(low_min, 4),
        "period_days": len(df),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="基于行情数据计算情绪指标")
    ap.add_argument("--file", help="输入 JSON 文件路径")
    ap.add_argument("--stdin", action="store_true", help="从 stdin 读取输入 JSON")
    args = ap.parse_args()

    try:
        data = _load_input(args)
        result = analyze_sentiment(data)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    print(json.dumps(result, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
