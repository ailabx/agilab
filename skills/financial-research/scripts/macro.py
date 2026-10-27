#!/usr/bin/env python3
"""macro.py —— 按品类做宏观与基本面规则判断。

输入：包含 meta 和 fundamental 的 JSON
输出 JSON：结构化的宏观/基本面判断结论

纯规则计算，不调 LLM。
"""

from __future__ import annotations

import argparse
import json
import sys


# ---------------------------------------------------------------------------
# 通用规则工具
# ---------------------------------------------------------------------------

def _load_input(args) -> dict:
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            return json.load(f)
    if args.stdin or not sys.stdin.isatty():
        return json.load(sys.stdin)
    raise ValueError("请通过 --file 或 stdin 提供输入 JSON")


def _signal_from_score(score: float) -> str:
    if score >= 0.5:
        return "bullish"
    if score <= -0.5:
        return "bearish"
    return "neutral"


def _clamp(score: float) -> float:
    return max(-1.0, min(1.0, score))


# ---------------------------------------------------------------------------
# 股票
# ---------------------------------------------------------------------------

def _analyze_stock(fund: dict) -> dict:
    pe = fund.get("pe_ttm")
    pb = fund.get("pb")
    dy = fund.get("dividend_yield")
    roe = fund.get("roe")
    netprofit_yoy = fund.get("netprofit_yoy")
    revenue_yoy = fund.get("revenue_yoy")
    debt = fund.get("debt_to_assets")

    score = 0.0
    factors = []

    # 估值
    if pe is not None and pe > 0:
        s = 1.0 if pe < 15 else (-1.0 if pe > 40 else 0.0)
        score += s * 0.25
        factors.append({"name": "PE_TTM", "value": pe, "signal": _signal_from_score(s)})

    if pb is not None and pb > 0:
        s = 1.0 if pb < 1.5 else (-1.0 if pb > 5 else 0.0)
        score += s * 0.15
        factors.append({"name": "PB", "value": pb, "signal": _signal_from_score(s)})

    if dy is not None and dy > 0:
        s = 1.0 if dy > 0.03 else (-1.0 if dy < 0.01 else 0.0)
        score += s * 0.10
        factors.append({"name": "dividend_yield", "value": dy, "signal": _signal_from_score(s)})

    # 成长性
    if netprofit_yoy is not None:
        s = 1.0 if netprofit_yoy > 20 else (-1.0 if netprofit_yoy < 0 else 0.0)
        score += s * 0.15
        factors.append({"name": "netprofit_yoy", "value": netprofit_yoy, "signal": _signal_from_score(s)})

    if revenue_yoy is not None:
        s = 1.0 if revenue_yoy > 15 else (-1.0 if revenue_yoy < 0 else 0.0)
        score += s * 0.10
        factors.append({"name": "revenue_yoy", "value": revenue_yoy, "signal": _signal_from_score(s)})

    # 盈利质量
    if roe is not None:
        s = 1.0 if roe > 15 else (-1.0 if roe < 8 else 0.0)
        score += s * 0.15
        factors.append({"name": "ROE", "value": roe, "signal": _signal_from_score(s)})

    # 杠杆
    if debt is not None:
        s = 1.0 if debt < 50 else (-1.0 if debt > 80 else 0.0)
        score += s * 0.10
        factors.append({"name": "debt_to_assets", "value": debt, "signal": _signal_from_score(s)})

    score = _clamp(score)
    return {
        "score": round(score, 2),
        "signal": _signal_from_score(score),
        "tag": "低估质优" if score > 0.3 else "高估承压" if score < -0.3 else "合理",
        "factors": factors,
    }


# ---------------------------------------------------------------------------
# ETF / 场外基金
# ---------------------------------------------------------------------------

def _analyze_fund_like(fund: dict) -> dict:
    fund_size = fund.get("fund_size")
    nav_return = fund.get("nav_return_20d")

    score = 0.0
    factors = []

    if fund_size is not None:
        s = 1.0 if fund_size > 50 else (-1.0 if fund_size < 5 else 0.0)
        score += s * 0.5
        factors.append({"name": "fund_size", "value": fund_size, "signal": _signal_from_score(s)})

    if nav_return is not None:
        s = 1.0 if nav_return > 2 else (-1.0 if nav_return < -2 else 0.0)
        score += s * 0.3
        factors.append({"name": "nav_return_20d", "value": nav_return, "signal": _signal_from_score(s)})

    score = _clamp(score)
    return {
        "score": round(score, 2),
        "signal": _signal_from_score(score),
        "tag": "规模充裕" if score > 0.3 else "规模偏小" if score < -0.3 else "规模中等",
        "factors": factors,
    }


# ---------------------------------------------------------------------------
# 期货
# ---------------------------------------------------------------------------

def _analyze_futures(fund: dict) -> dict:
    basis = fund.get("basis", {}) or {}
    warehouse = fund.get("warehouse", {}) or {}
    warehouse_ak = fund.get("warehouse_ak", {}) or {}
    main_contract = fund.get("main_contract")
    open_interest = fund.get("open_interest")
    oi_chg = fund.get("oi_chg")
    term = fund.get("term_structure", {}) or {}
    inventory = fund.get("inventory", {}) or {}
    position_rank = fund.get("position_rank", {}) or {}

    score = 0.0
    factors = []

    # 基差：升水偏多，贴水偏空
    basis_value = basis.get("basis")
    if basis_value is not None:
        s = 1.0 if basis_value > 0 else -1.0
        score += s * 0.25
        factors.append({"name": "basis", "value": basis_value, "signal": _signal_from_score(s)})

    # 仓单：低偏多，高偏空（优先 Tushare，再 akshare）
    wsr = warehouse.get("warehouse_receipt") or warehouse_ak.get("warehouse_receipt")
    if wsr is not None:
        s = 1.0 if wsr < 10000 else -1.0
        score += s * 0.15
        factors.append({"name": "warehouse_receipt", "value": wsr, "signal": _signal_from_score(s)})

    # 库存：下降偏多，上升偏空
    inv = inventory.get("inventory")
    inv_change = inventory.get("inventory_change")
    if inv is not None and inv_change is not None:
        s = 1.0 if inv_change < 0 else -1.0
        score += s * 0.15
        factors.append({"name": "inventory_change", "value": inv_change, "signal": _signal_from_score(s)})
    elif inv is not None:
        s = 1.0 if inv < 50000 else -1.0
        score += s * 0.15
        factors.append({"name": "inventory", "value": inv, "signal": _signal_from_score(s)})

    # 持仓量：放量且价格涨偏多；放量且价格跌偏空
    if open_interest is not None:
        s = 1.0 if open_interest > 50000 else -1.0
        score += s * 0.10
        factors.append({"name": "open_interest", "value": open_interest, "signal": _signal_from_score(s)})

    if oi_chg is not None:
        s = 1.0 if oi_chg > 0 else -1.0
        score += s * 0.05
        factors.append({"name": "oi_chg", "value": oi_chg, "signal": _signal_from_score(s)})

    # 期限结构：近高远低（back）偏多，近低远高（contango）偏空
    spread = term.get("spread")
    if spread is not None:
        s = 1.0 if spread > 0 else -1.0
        score += s * 0.15
        factors.append({"name": "term_spread", "value": spread, "signal": _signal_from_score(s)})

    # 持仓排名：多头占优偏多，空头占优偏空
    long_pos = position_rank.get("long_top20")
    short_pos = position_rank.get("short_top20")
    if long_pos is not None and short_pos is not None and short_pos != 0:
        ratio = long_pos / short_pos
        s = 1.0 if ratio > 1.1 else (-1.0 if ratio < 0.9 else 0.0)
        score += s * 0.10
        factors.append({"name": "position_long_short_ratio", "value": ratio, "signal": _signal_from_score(s)})

    # 虚实比：高 → 逼仓风险，偏多；低 → 投机度低，偏中性
    virtual_real = fund.get("virtual_real_ratio")
    if virtual_real is not None:
        s = 1.0 if virtual_real > 30 else (-1.0 if virtual_real < 5 else 0.0)
        score += s * 0.05
        factors.append({"name": "virtual_real_ratio", "value": virtual_real, "signal": _signal_from_score(s)})

    score = _clamp(score)
    return {
        "score": round(score, 2),
        "signal": _signal_from_score(score),
        "tag": "现货紧张" if score > 0.3 else "现货宽松" if score < -0.3 else "基差中性",
        "main_contract": main_contract,
        "factors": factors,
    }


# ---------------------------------------------------------------------------
# 可转债
# ---------------------------------------------------------------------------

def _analyze_convertible(fund: dict) -> dict:
    premium_rate = fund.get("premium_rate")
    conv_value = fund.get("conversion_value")
    bond_rating = fund.get("bond_rating")
    remaining_years = fund.get("remaining_years")
    underlying_pe = fund.get("underlying_pe")
    underlying_roe = fund.get("underlying_roe")

    score = 0.0
    factors = []

    if premium_rate is not None:
        s = 1.0 if premium_rate < 10 else (-1.0 if premium_rate > 30 else 0.0)
        score += s * 0.25
        factors.append({"name": "premium_rate", "value": premium_rate, "signal": _signal_from_score(s)})

    if conv_value is not None:
        s = 1.0 if conv_value > 100 else -1.0
        score += s * 0.20
        factors.append({"name": "conversion_value", "value": conv_value, "signal": _signal_from_score(s)})

    # 剩余期限：越短，转债时间价值越少，偏债性保护越弱
    if remaining_years is not None:
        s = 1.0 if remaining_years > 3 else (-1.0 if remaining_years < 1 else 0.0)
        score += s * 0.10
        factors.append({"name": "remaining_years", "value": remaining_years, "signal": _signal_from_score(s)})

    # 正股估值
    if underlying_pe is not None:
        s = 1.0 if underlying_pe < 20 else (-1.0 if underlying_pe > 40 else 0.0)
        score += s * 0.15
        factors.append({"name": "underlying_pe", "value": underlying_pe, "signal": _signal_from_score(s)})

    # 正股盈利能力
    if underlying_roe is not None:
        s = 1.0 if underlying_roe > 12 else (-1.0 if underlying_roe < 6 else 0.0)
        score += s * 0.15
        factors.append({"name": "underlying_roe", "value": underlying_roe, "signal": _signal_from_score(s)})

    score = _clamp(score)
    return {
        "score": round(score, 2),
        "signal": _signal_from_score(score),
        "tag": "股性强" if score > 0.3 else "债性强" if score < -0.3 else "平衡",
        "bond_rating": bond_rating,
        "factors": factors,
    }


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def analyze_macro(data: dict) -> dict:
    market = data.get("meta", {}).get("market", "stock")
    fundamental = data.get("fundamental", {}) or {}

    handlers = {
        "stock": _analyze_stock,
        "etf": _analyze_fund_like,
        "futures": _analyze_futures,
        "convertible": _analyze_convertible,
        "fund": _analyze_fund_like,
    }

    handler = handlers.get(market)
    if not handler:
        return {"error": f"不支持的品类: {market}"}

    result = handler(fundamental)
    result["market"] = market
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="宏观与基本面规则判断")
    ap.add_argument("--file", help="输入 JSON 文件路径")
    ap.add_argument("--stdin", action="store_true", help="从 stdin 读取输入 JSON")
    args = ap.parse_args()

    try:
        data = _load_input(args)
        result = analyze_macro(data)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    print(json.dumps(result, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
