#!/usr/bin/env python3
"""main.py —— 金融研究 Skill 的统一入口。

把 market_data / technical / fundamental / sentiment 串起来，
输出标准 JSON 数据包。

用法：
    python main.py 600519 --market stock --days 120
    python main.py 510300 --market etf --days 60
    python main.py RB --market futures --days 120

输出：结构化 JSON，包含 market / technical / fundamental / sentiment。

纯计算/编排，不调 LLM。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SKILL_DIR = Path(__file__).resolve().parent


def _run_script(name: str, *args, input_json: dict | None = None) -> dict:
    """调用子脚本，返回 JSON。"""
    script = SKILL_DIR / f"{name}.py"
    cmd = [sys.executable, str(script), *args]
    kwargs = {"capture_output": True, "text": True}
    if input_json is not None:
        kwargs["input"] = json.dumps(input_json, ensure_ascii=False, default=str)
    result = subprocess.run(cmd, **kwargs)  # noqa: S603
    if result.returncode != 0:
        raise RuntimeError(f"{name} 失败: {result.stderr}")
    return json.loads(result.stdout)


def _overall_score(technical: dict, fundamental: dict, macro: dict, sentiment: dict) -> dict:
    """汇总技术面、基本面、宏观、情绪四个维度的打分，生成最终信号。"""
    tech_state = technical.get("trend", {}).get("state", "sideways")
    tech_score = {"up": 0.6, "down": -0.6, "sideways": 0.0}.get(tech_state, 0.0)

    macro_score = macro.get("score", 0.0)

    sentiment_state = sentiment.get("trend_strength", "neutral")
    sentiment_score_map = {"bullish": 0.6, "overbought": 0.2, "bearish": -0.6, "oversold": -0.2, "neutral": 0.0}
    sentiment_score = sentiment_score_map.get(sentiment_state, 0.0)

    # 基本面/宏观使用同一维度，避免重复
    total = tech_score * 0.35 + macro_score * 0.35 + sentiment_score * 0.30
    total = max(-1.0, min(1.0, total))

    if total >= 0.4:
        signal = "buy"
    elif total <= -0.4:
        signal = "sell"
    else:
        signal = "hold"

    return {
        "score": round(total, 2),
        "signal": signal,
        "weights": {
            "technical": round(tech_score, 2),
            "macro": round(macro_score, 2),
            "sentiment": round(sentiment_score, 2),
        },
    }


def research(symbol: str, market: str, days: int, focus: str) -> dict:
    # 1. 数据采集
    market_data = _run_script("market_data", symbol, "--market", market, "--days", str(days))

    # 2. 技术指标
    technical = _run_script("technical", "--stdin", input_json=market_data)

    # 3. 基本面（传入最新收盘价，便于可转债计算转股溢价）
    fundamental = _run_script("fundamental", "--stdin", input_json=market_data)

    # 4. 宏观/基本面规则判断（需要基本面数据）
    macro_input = {
        "meta": market_data.get("meta", {}),
        "fundamental": fundamental,
    }
    macro = _run_script("macro", "--stdin", input_json=macro_input)

    # 5. 情绪指标
    sentiment = _run_script("sentiment", "--stdin", input_json=market_data)

    # 6. 汇总评分
    overall = _overall_score(technical, fundamental, macro, sentiment)

    result = {
        "meta": market_data.get("meta", {}),
        "market": market_data,
        "technical": technical,
        "fundamental": fundamental,
        "macro": macro,
        "sentiment": sentiment,
        "overall": overall,
    }
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="金融研究 Skill 统一入口")
    ap.add_argument("symbol", help="标的代码或名称")
    ap.add_argument("--market", default="stock", choices=["stock", "etf", "futures", "convertible", "fund"])
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--focus", default="all", choices=["all", "technical", "fundamental"])
    args = ap.parse_args()

    try:
        result = research(args.symbol, args.market, args.days, args.focus)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    print(json.dumps(result, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
