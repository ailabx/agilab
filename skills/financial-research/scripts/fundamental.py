#!/usr/bin/env python3
"""fundamental.py —— 按品类获取并整理基本面/宏观数据。

输入：market_data.py 的输出 JSON
输出 JSON：按 market 分支的基本面指标

纯计算/数据获取，不调 LLM。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from config import config
import tushare as ts
import akshare as ak


def _load_input(args) -> dict:
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            return json.load(f)
    if args.stdin or not sys.stdin.isatty():
        return json.load(sys.stdin)
    raise ValueError("请通过 --file 或 stdin 提供输入 JSON")


def _get_api():
    return ts.pro_api(config.TS_TOKEN)


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------

def _safe_float(value, default=None):
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _last_non_null(df: pd.DataFrame, col: str):
    """取某列最后一个非空值。"""
    if df is None or df.empty or col not in df.columns:
        return None
    s = df[col].dropna()
    return s.iloc[-1] if not s.empty else None


# ---------------------------------------------------------------------------
# 股票
# ---------------------------------------------------------------------------

def _analyze_stock(symbol: str) -> dict:
    api = _get_api()
    result = {}
    try:
        end = pd.Timestamp.now()
        start = end - pd.Timedelta(days=90)
        start_str = start.strftime("%Y%m%d")
        end_str = end.strftime("%Y%m%d")

        # 1. 估值指标
        try:
            df = api.daily_basic(ts_code=symbol, start_date=start_str, end_date=end_str)
            if not df.empty:
                row = df.iloc[0]
                result.update({
                    "pe_ttm": _safe_float(row.get("pe_ttm")),
                    "pb": _safe_float(row.get("pb")),
                    "ps_ttm": _safe_float(row.get("ps_ttm")),
                    "dividend_yield": _safe_float(row.get("dv_ttm")),
                    "total_mv": _safe_float(row.get("total_mv")),
                })
        except Exception:
            pass

        # 2. 财务指标（最近一期财报）
        try:
            fina = api.fina_indicator(ts_code=symbol, start_date="20200101", end_date=end_str, limit=4)
            if not fina.empty:
                row = fina.iloc[0]
                result.update({
                    "roe": _safe_float(row.get("roe")),
                    "roa": _safe_float(row.get("roa")),
                    "grossprofit_margin": _safe_float(row.get("grossprofit_margin")),
                    "netprofit_yoy": _safe_float(row.get("q_netprofit_yoy") or row.get("netprofit_yoy")),
                    "revenue_yoy": _safe_float(row.get("q_sales_yoy") or row.get("revenue_yoy")),
                    "debt_to_assets": _safe_float(row.get("debt_to_assets")),
                })
        except Exception:
            pass

        # 3. 分红
        try:
            dividend = api.dividend(ts_code=symbol, start_date="20220101", end_date=end_str)
            if not dividend.empty:
                row = dividend.iloc[0]
                result.update({
                    "dividend_ps": _safe_float(row.get("dividend")),
                    "cash_div_ltm": _safe_float(row.get("cash_div_tax")),
                })
        except Exception:
            pass

        # 4. 行业信息
        try:
            stk = api.stock_basic(ts_code=symbol)
            if not stk.empty:
                row = stk.iloc[0]
                result.update({
                    "industry": row.get("industry"),
                    "sector": row.get("sector"),
                    "list_date": row.get("list_date"),
                })
        except Exception:
            pass

        return result if result else {"error": "无法获取股票基本面数据"}
    except Exception:
        return {"error": "无法获取股票基本面数据"}


# ---------------------------------------------------------------------------
# ETF / 场外基金
# ---------------------------------------------------------------------------

def _analyze_etf(symbol: str) -> dict:
    api = _get_api()
    result = {}
    try:
        # 基本信息
        try:
            df = api.fund_basic(ts_code=symbol)
            if not df.empty:
                row = df.iloc[0]
                result.update({
                    "fund_type": row.get("fund_type"),
                    "manager": row.get("mgr_nm"),
                    "tracking_index": row.get("benchmark"),
                    "found_date": row.get("found_date"),
                })
        except Exception:
            pass

        # 最新规模和份额
        try:
            end = pd.Timestamp.now()
            end_str = end.strftime("%Y%m%d")
            start_str = (end - pd.Timedelta(days=30)).strftime("%Y%m%d")
            share = api.fund_share(ts_code=symbol, start_date=start_str, end_date=end_str)
            if not share.empty:
                row = share.iloc[-1]
                result["fund_size"] = _safe_float(row.get("total_asset"))
                result["fund_share"] = _safe_float(row.get("fd_share")) if "fd_share" in row else _safe_float(row.get("share"))
            nav = api.fund_nav(ts_code=symbol, start_date=start_str, end_date=end_str)
            if not nav.empty:
                row = nav.iloc[-1]
                result["nav"] = _safe_float(row.get("unit_nav"))
                result["acc_nav"] = _safe_float(row.get("acc_nav"))
        except Exception:
            pass

        return result if result else {"error": "无法获取 ETF 基本面数据"}
    except Exception:
        return {"error": "无法获取 ETF 基本面数据"}


def _analyze_fund(symbol: str) -> dict:
    # 场外基金和 ETF 共享逻辑
    api = _get_api()
    result = {}
    try:
        try:
            df = api.fund_basic(ts_code=symbol)
            if not df.empty:
                row = df.iloc[0]
                result.update({
                    "fund_type": row.get("fund_type"),
                    "manager": row.get("mgr_nm"),
                    "benchmark": row.get("benchmark"),
                    "found_date": row.get("found_date"),
                })
        except Exception:
            pass

        try:
            end = pd.Timestamp.now()
            end_str = end.strftime("%Y%m%d")
            start_str = (end - pd.Timedelta(days=30)).strftime("%Y%m%d")
            nav = api.fund_nav(ts_code=symbol, start_date=start_str, end_date=end_str)
            if not nav.empty:
                row = nav.iloc[-1]
                result["nav"] = _safe_float(row.get("unit_nav"))
                result["acc_nav"] = _safe_float(row.get("acc_nav"))
                # 计算近 20 日净值增长
                if len(nav) >= 20:
                    latest = _safe_float(nav.iloc[-1].get("unit_nav"))
                    past = _safe_float(nav.iloc[-20].get("unit_nav"))
                    if latest and past:
                        result["nav_return_20d"] = round((latest / past - 1) * 100, 2)
        except Exception:
            pass

        return result if result else {"error": "无法获取场外基金基本面数据"}
    except Exception:
        return {"error": "无法获取场外基金基本面数据"}


# ---------------------------------------------------------------------------
# 期货
# ---------------------------------------------------------------------------

# akshare 部分接口需要中文品种名
trad_fut_symbol = {
    "A": "豆一", "B": "豆二", "M": "豆粕", "Y": "豆油", "P": "棕榈",
    "C": "玉米", "CS": "玉米淀粉", "L": "塑料", "V": "PVC", "PP": "聚丙烯",
    "J": "焦炭", "JM": "焦煤", "I": "铁矿石", "EG": "乙二醇", "EB": "苯乙烯",
    "PG": "液化石油气", "LH": "生猪", "RR": "粳米",
    "CF": "郑棉", "SR": "白糖", "TA": "PTA", "OI": "菜油", "RI": "早籼稻",
    "MA": "甲醇", "FG": "玻璃", "RM": "菜粕", "ZC": "动力煤", "UR": "尿素",
    "CJ": "红枣", "SA": "纯碱", "PK": "花生", "PF": "短纤", "PX": "对二甲苯",
    "SH": "烧碱", "SM": "锰硅", "SF": "硅铁", "AP": "苹果", "CY": "棉纱",
    "CU": "沪铜", "AL": "沪铝", "ZN": "沪锌", "PB": "沪铅", "NI": "镍",
    "SN": "锡", "AU": "沪金", "AG": "沪银", "RB": "螺纹钢", "WR": "线材",
    "HC": "热卷", "FU": "燃油", "BU": "沥青", "RU": "橡胶", "SC": "原油",
    "NR": "20号胶", "SP": "纸浆", "SS": "不锈钢", "LU": "低硫燃料油",
    "BC": "国际铜", "AO": "氧化铝", "BR": "丁二烯橡胶", "EC": "欧线",
    "AD": "铸造铝合金", "OP": "胶版印刷纸", "SI": "工业硅", "LC": "碳酸锂",
    "PS": "多晶硅",
    "IF": "沪深300", "IC": "中证500", "IM": "中证1000", "IH": "上证50",
    "T": "10年期国债", "TF": "5年期国债", "TS": "2年期国债", "TL": "30年期国债",
}


def _get_main_contract(api, symbol: str, start_date: str, end_date: str) -> str:
    """通过 fut_mapping 获取主力合约。"""
    for suffix in ["SHF", "DCE", "CZC", "INE", "GFEX", "CFFEX"]:
        try:
            mapping = api.fut_mapping(ts_code=f"{symbol.upper()}.{suffix}", start_date=start_date, end_date=end_date)
            if not mapping.empty:
                return mapping.iloc[-1].get("mapping_ts_code", f"{symbol.upper()}.{suffix}")
        except Exception:
            continue
    return f"{symbol.upper()}.SHF"


def _analyze_futures(symbol: str) -> dict:
    """期货基本面：Tushare 为主，akshare 补充库存/持仓/仓单/现货价。"""
    api = _get_api()
    end = pd.Timestamp.now()
    start = end - pd.Timedelta(days=30)
    start_str = start.strftime("%Y%m%d")
    end_str = end.strftime("%Y%m%d")
    result = {}

    try:
        # 主力合约
        contract = _get_main_contract(api, symbol, start_str, end_str)
        result["main_contract"] = contract
        contract_base = contract.split(".")[0]
        chinese_name = trad_fut_symbol.get(symbol.upper(), symbol.upper())

        # 1) 现货基差 —— Tushare
        try:
            spot_df = api.fut_spot_price(date=end_str, vars_list=[contract_base])
            if not spot_df.empty:
                row = spot_df.iloc[0]
                result["basis"] = {
                    "basis": _safe_float(row.get("basis")),
                    "spot_price": _safe_float(row.get("spot_price")),
                    "fut_price": _safe_float(row.get("fut_price")),
                }
        except Exception:
            pass

        # 2) 仓单 —— Tushare + akshare 补充
        try:
            wsr_df = api.fut_wsr(symbol=contract, start_date=start_str, end_date=end_str)
            if not wsr_df.empty:
                row = wsr_df.iloc[-1]
                result["warehouse"] = {
                    "warehouse_receipt": _safe_float(row.get("warehouse_receipt")),
                    "exchange": row.get("exchange"),
                }
        except Exception:
            pass

        # 3) 持仓量/成交量 —— Tushare
        try:
            df = api.fut_daily(ts_code=contract, start_date=start_str, end_date=end_str)
            if not df.empty:
                result["open_interest"] = _safe_float(df.iloc[-1].get("oi"))
                result["vol"] = _safe_float(df.iloc[-1].get("vol"))
                result["oi_chg"] = _safe_float(df.iloc[-1].get("oi_chg"))
        except Exception:
            pass

        # 4) 期限结构：近月 vs 远月 —— Tushare
        try:
            all_contracts = api.fut_mapping(start_date=end_str, end_date=end_str)
            if not all_contracts.empty:
                contracts = all_contracts[all_contracts["ts_code"].str.startswith(symbol.upper())]
                if not contracts.empty:
                    codes = contracts["mapping_ts_code"].dropna().unique().tolist()
                    if len(codes) >= 2:
                        near = api.fut_daily(ts_code=codes[0], start_date=start_str, end_date=end_str)
                        far = api.fut_daily(ts_code=codes[-1], start_date=start_str, end_date=end_str)
                        if not near.empty and not far.empty:
                            near_price = _safe_float(near.iloc[-1].get("close"))
                            far_price = _safe_float(far.iloc[-1].get("close"))
                            if near_price and far_price:
                                result["term_structure"] = {
                                    "near": codes[0],
                                    "far": codes[-1],
                                    "near_price": near_price,
                                    "far_price": far_price,
                                    "spread": round(near_price - far_price, 2),
                                    "spread_pct": round((near_price - far_price) / far_price * 100, 2),
                                }
        except Exception:
            pass

        # 5) 现货价格/基差 —— akshare 补充
        try:
            spot_ak = ak.futures_spot_price(start_day=start_str, end_day=end_str, vars_list=[symbol.upper()])
            if spot_ak is not None and not spot_ak.empty:
                latest = spot_ak.iloc[-1]
                result["spot"] = {
                    "spot_price_ak": _safe_float(latest.get("spot_price")),
                    "fut_price_ak": _safe_float(latest.get("futures_price")),
                    "basis_ak": _safe_float(latest.get("basis")),
                }
        except Exception:
            pass

        # 6) 仓单 —— akshare 补充（DCE/CZCE/SHFE/GFEX 依次尝试）
        for wh_func in (
            ak.futures_warehouse_receipt_dce,
            ak.futures_warehouse_receipt_czce,
            ak.futures_shfe_warehouse_receipt,
            ak.futures_gfex_warehouse_receipt,
        ):
            try:
                wh_df = wh_func(date=end_str)
                if wh_df is not None and not wh_df.empty:
                    # 按品种代码前缀匹配
                    rows = wh_df[wh_df.iloc[:, 0].astype(str).str.upper().str.startswith(symbol.upper())]
                    if not rows.empty:
                        result["warehouse_ak"] = {
                            "exchange": wh_func.__name__.split("_")[-2].upper(),
                            "warehouse_receipt": _safe_float(rows.iloc[-1].iloc[-1]),
                        }
                        break
            except Exception:
                continue

        # 7) 库存 —— akshare（需要中文品种名）
        try:
            inv_df = ak.futures_inventory_em(symbol=chinese_name)
            if inv_df is not None and not inv_df.empty:
                latest = inv_df.iloc[-1]
                result["inventory"] = {
                    "inventory": _safe_float(latest.get("库存") if "库存" in latest else latest.get("inventory")),
                    "inventory_change": _safe_float(latest.get("增减") if "增减" in latest else latest.get("change")),
                }
        except Exception:
            pass

        # 8) 持仓排名 —— akshare（仅支持 DCE/GFEX，需要中文品种名）
        for rank_func in (ak.futures_dce_position_rank, ak.futures_gfex_position_rank):
            try:
                rank_df = rank_func(symbol=chinese_name)
                if rank_df is not None and not rank_df.empty:
                    result["position_rank"] = {
                        "exchange": rank_func.__name__.split("_")[1].upper(),
                        "long_top20": _safe_float(rank_df.iloc[0].get("long_position")),
                        "short_top20": _safe_float(rank_df.iloc[0].get("short_position")),
                    }
                    break
            except Exception:
                continue

        # 9) 虚实比 = 持仓量 / 库存（均为吨/手口径，仅做近似）
        try:
            oi = result.get("open_interest")
            inv = (result.get("inventory") or {}).get("inventory")
            if oi is not None and inv is not None and inv > 0:
                # 持仓量通常以手计，库存以吨计；这里仅做相对比值，无单位含义
                result["virtual_real_ratio"] = round(oi / inv, 4)
        except Exception:
            pass

        return result if result else {"error": "无法获取期货基本面数据"}
    except Exception as e:
        return {"error": f"无法获取期货基本面数据: {e}"}


# ---------------------------------------------------------------------------
# 可转债
# ---------------------------------------------------------------------------

def _analyze_convertible(symbol: str, cb_close: float | None = None) -> dict:
    api = _get_api()
    try:
        df = api.cb_basic(ts_code=symbol)
        if df.empty:
            return {"error": "无法获取可转债基本面数据"}

        row = df.iloc[0]
        result = {
            "conv_price": _safe_float(row.get("conv_price")),
            "bond_rating": row.get("bond_rating"),
            "maturity_date": row.get("maturity_date"),
            "underlying": row.get("stk_code"),
            "list_date": row.get("list_date"),
        }

        # 剩余期限（年）
        try:
            if row.get("maturity_date"):
                maturity = pd.Timestamp(str(row.get("maturity_date")))
                result["remaining_years"] = round((maturity - pd.Timestamp.now()).days / 365.25, 2)
        except Exception:
            pass

        # 正股财务
        stock_code = row.get("stk_code")
        if stock_code:
            try:
                end = pd.Timestamp.now()
                start = end - pd.Timedelta(days=10)
                stock_df = api.daily(ts_code=stock_code, start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d"))
                if not stock_df.empty:
                    stock_close = _safe_float(stock_df.iloc[-1].get("close"))
                    result["underlying_close"] = stock_close

                # 正股估值
                basic = api.daily_basic(ts_code=stock_code, start_date=(end - pd.Timedelta(days=30)).strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d"))
                if not basic.empty:
                    br = basic.iloc[-1]
                    result["underlying_pe"] = _safe_float(br.get("pe_ttm"))
                    result["underlying_pb"] = _safe_float(br.get("pb"))

                # 正股财务
                fina = api.fina_indicator(ts_code=stock_code, start_date="20200101", end_date=end.strftime("%Y%m%d"), limit=4)
                if not fina.empty:
                    fr = fina.iloc[0]
                    result["underlying_roe"] = _safe_float(fr.get("roe"))
            except Exception:
                pass

        # 转股价值和溢价率
        try:
            conv_price = result.get("conv_price")
            if conv_price and result.get("underlying_close"):
                conv_value = 100 / conv_price * result["underlying_close"]
                result["conversion_value"] = round(conv_value, 2)
                if cb_close is not None:
                    result["cb_close"] = cb_close
                    result["premium_rate"] = round((cb_close / conv_value - 1) * 100, 2)
        except Exception:
            pass

        return result
    except Exception:
        return {"error": "无法获取可转债基本面数据"}


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def analyze_fundamental(data: dict) -> dict:
    market = data.get("meta", {}).get("market", "stock")
    symbol = data.get("meta", {}).get("ts_code") or data.get("meta", {}).get("symbol", "")
    latest = data.get("latest", {})
    close = latest.get("close")

    # 可转债传入最新收盘价
    if market == "convertible" and close is not None:
        return _analyze_convertible(symbol, float(close))

    handlers = {
        "stock": _analyze_stock,
        "etf": _analyze_etf,
        "futures": _analyze_futures,
        "convertible": _analyze_convertible,
        "fund": _analyze_fund,
    }
    handler = handlers.get(market)
    if not handler:
        return {"error": f"不支持的品类: {market}"}

    return handler(symbol)


def main() -> int:
    ap = argparse.ArgumentParser(description="获取并整理基本面数据")
    ap.add_argument("--file", help="输入 JSON 文件路径")
    ap.add_argument("--stdin", action="store_true", help="从 stdin 读取输入 JSON")
    args = ap.parse_args()

    try:
        data = _load_input(args)
        result = analyze_fundamental(data)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    print(json.dumps(result, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
