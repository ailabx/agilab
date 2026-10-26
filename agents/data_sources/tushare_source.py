"""Tushare 数据源——极简版，只保留核心数据拉取能力。

去掉 TA-CN 的 MongoDB/异步/事件循环/日志装饰器等重设施。
纯同步调用，直接返回 pandas DataFrame 或 dict。
"""

import os
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

try:
    import tushare as ts
    _TS_AVAILABLE = True
except ImportError:
    _TS_AVAILABLE = False

from config import config


class TushareSource:
    """Tushare 数据源，从 config.TS_TOKEN 读取 token。"""

    def __init__(self):
        self._api = None

    def _get_api(self):
        """初始化或复用 Tushare API Client。"""
        if not _TS_AVAILABLE:
            raise RuntimeError("tushare 未安装，请 pip install tushare")

        token = config.TS_TOKEN
        if not token:
            raise RuntimeError("TS_TOKEN 未配置，请在 ~/.agilab/config.toml 中设置")

        if self._api is None:
            self._api = ts.pro_api(token)
        return self._api

    def _normalize_ts_code(self, symbol: str) -> str:
        """将股票代码转为 tushare 格式: 600519 → 600519.SH, 000001 → 000001.SZ。"""
        symbol = symbol.strip()
        if "." in symbol:
            return symbol.upper()

        code = symbol.replace("sh", "").replace("sz", "").replace("SH", "").replace("SZ", "")

        if code.startswith(("60", "68", "90", "11", "13", "50", "51", "52", "56", "58")):
            return f"{code}.SH"
        elif code.startswith(("00", "30", "12", "15", "16", "18")):
            return f"{code}.SZ"
        elif code.startswith(("43", "83", "87", "88")):
            return f"{code}.BJ"
        return f"{code}.SH"

    def _is_etf(self, ts_code: str) -> bool:
        code = ts_code.split(".")[0]
        return code.startswith(("51", "15", "52", "56", "58", "50"))

    def _is_convertible(self, ts_code: str) -> bool:
        code = ts_code.split(".")[0]
        return code.startswith(("11", "12", "13"))

    # ==================== 行情数据 ====================

    def get_daily(self, symbol: str, days: int = 120) -> pd.DataFrame:
        """获取日K线数据（前复权）。

        参数:
            symbol: 股票代码，如 "600519" 或 "600519.SH"
            days: 获取最近多少天的数据

        返回:
            DataFrame: 列含 trade_date, open, high, low, close, vol, amount
        """
        api = self._get_api()
        ts_code = self._normalize_ts_code(symbol)

        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")

        if self._is_etf(ts_code):
            df = api.fund_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        else:
            df = ts.pro_bar(ts_code=ts_code, api=api, start_date=start_date,
                            end_date=end_date, freq="D", adj="qfq")

        if df is None or df.empty:
            return pd.DataFrame()

        df = df.sort_values("trade_date").reset_index(drop=True)
        return df

    def get_realtime(self, symbol: str) -> dict:
        """获取最新一日行情快照。"""
        df = self.get_daily(symbol, days=5)
        if df.empty:
            return {}
        row = df.iloc[-1]
        return {
            "trade_date": row.get("trade_date", ""),
            "open": float(row.get("open", 0)),
            "high": float(row.get("high", 0)),
            "low": float(row.get("low", 0)),
            "close": float(row.get("close", 0)),
            "vol": float(row.get("vol", 0)),
            "amount": float(row.get("amount", 0)),
            "pct_chg": float(row.get("pct_chg", 0)) if "pct_chg" in row else 0.0,
        }

    # ==================== 基本面数据 ====================

    def get_stock_basic(self, symbol: str) -> dict:
        """获取股票基础信息。"""
        api = self._get_api()
        ts_code = self._normalize_ts_code(symbol)

        df = api.stock_basic(ts_code=ts_code,
                             fields="ts_code,symbol,name,area,industry,market,list_date")
        if df is None or df.empty:
            return {}
        return df.iloc[0].to_dict()

    def get_daily_basic(self, symbol: str) -> dict:
        """获取最新一日基本面指标: PE/PB/换手率等。"""
        api = self._get_api()
        ts_code = self._normalize_ts_code(symbol)

        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=5)).strftime("%Y%m%d")

        df = api.daily_basic(ts_code=ts_code, start_date=start_date, end_date=end_date)
        if df is None or df.empty:
            return {}
        row = df.iloc[0]
        return {
            "trade_date": row.get("trade_date", ""),
            "pe": float(row.get("pe", 0)) if pd.notna(row.get("pe")) else None,
            "pe_ttm": float(row.get("pe_ttm", 0)) if pd.notna(row.get("pe_ttm")) else None,
            "pb": float(row.get("pb", 0)) if pd.notna(row.get("pb")) else None,
            "total_mv": float(row.get("total_mv", 0)) if pd.notna(row.get("total_mv")) else None,
            "circ_mv": float(row.get("circ_mv", 0)) if pd.notna(row.get("circ_mv")) else None,
            "turnover_rate": float(row.get("turnover_rate", 0)) if pd.notna(row.get("turnover_rate")) else None,
        }

    # ==================== ETF 专项 ====================

    def get_fund_basic(self, symbol: str) -> dict:
        """获取 ETF/基金基础信息。"""
        api = self._get_api()
        ts_code = self._normalize_ts_code(symbol)

        df = api.fund_basic(ts_code=ts_code)
        if df is None or df.empty:
            return {}
        return df.iloc[0].to_dict()

    # ==================== 可转债专项 ====================

    def get_cb_basic(self, symbol: str) -> dict:
        """获取可转债基础信息（含转股价、溢价率等）。"""
        api = self._get_api()
        ts_code = self._normalize_ts_code(symbol)

        df = api.cb_basic(ts_code=ts_code)
        if df is None or df.empty:
            return {}
        return df.iloc[0].to_dict()

    def get_cb_daily(self, symbol: str, days: int = 60) -> pd.DataFrame:
        """获取可转债日K线。"""
        api = self._get_api()
        ts_code = self._normalize_ts_code(symbol)

        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")

        df = api.cb_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        if df is None or df.empty:
            return pd.DataFrame()
        return df.sort_values("trade_date").reset_index(drop=True)

    # ==================== 新闻数据 ====================

    def get_news(self, symbol: str, limit: int = 10) -> list:
        """获取个股相关新闻。"""
        api = self._get_api()

        try:
            df = api.news(src="sina", start_date=datetime.now().strftime("%Y%m%d"),
                          end_date=datetime.now().strftime("%Y%m%d"))
            if df is None or df.empty:
                return []
            basic = self.get_stock_basic(symbol)
            name = basic.get("name", "")
            if name:
                df = df[df["title"].str.contains(name, na=False)]
            return df.head(limit).to_dict("records")
        except Exception:
            return []

    # ==================== 综合接口 ====================

    def get_all(self, symbol: str) -> dict:
        """一次性获取行情+基本面+新闻，供 data_agent 使用。"""
        result = {"symbol": symbol}

        try:
            result["kline"] = self.get_daily(symbol).to_dict("records")
            result["realtime"] = self.get_realtime(symbol)
        except Exception as e:
            result["kline_error"] = str(e)

        try:
            result["basic"] = self.get_stock_basic(symbol)
            result["daily_basic"] = self.get_daily_basic(symbol)
        except Exception as e:
            result["basic_error"] = str(e)

        try:
            result["news"] = self.get_news(symbol)
        except Exception as e:
            result["news_error"] = str(e)

        return result
