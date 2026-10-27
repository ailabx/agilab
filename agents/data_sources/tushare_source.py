"""Tushare 数据源 —— 按需加载，增量更新。

用法：
    source = TushareSource()
    
    # 获取A股日K线（自动判断全量/增量）
    df = source.get_stock_daily("600519", days=120)
    
    # 获取ETF日K线
    df = source.get_etf_daily("510300", days=60)
    
    # 获取期货日K线
    df = source.get_futures_daily("RB2501", days=60)
"""

import os
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from config import config

from . import db_manager as db

try:
    import tushare as ts
    _TS_AVAILABLE = True
except ImportError:
    _TS_AVAILABLE = False


class TushareSource:
    """Tushare 数据源，支持按需加载和增量更新。"""

    def __init__(self):
        self._api = None

    def _get_api(self):
        """初始化或复用 Tushare API。"""
        if not _TS_AVAILABLE:
            raise RuntimeError("tushare 未安装")
        if self._api is None:
            token = config.TS_TOKEN
            if not token:
                raise RuntimeError("TS_TOKEN 未配置")
            self._api = ts.pro_api(token)
        return self._api

    def _normalize_ts_code(self, symbol: str, market: str = "stock") -> str:
        """标准化代码格式。"""
        symbol = symbol.strip().upper()
        if "." in symbol:
            return symbol
        
        if market == "futures":
            # 期货代码直接返回，如 "RB2501"
            return symbol
        
        # 股票/ETF
        if symbol.startswith(("60", "68", "90")):
            return f"{symbol}.SH"
        elif symbol.startswith(("00", "30", "12")):
            return f"{symbol}.SZ"
        elif symbol.startswith(("43", "83", "87", "88")):
            return f"{symbol}.BJ"
        return f"{symbol}.SH"

    def _is_etf(self, ts_code: str) -> bool:
        """判断是否为ETF。"""
        code = ts_code.split(".")[0]
        return code.startswith(("51", "15", "52", "56", "58", "50"))

    # ==================== 日K线：按需加载 + 增量更新 ====================

    def get_stock_daily(self, symbol: str, days: int = 120) -> pd.DataFrame:
        """获取A股日K线（按需加载，增量更新）。
        
        首次：从 Tushare 全量拉取
        后续：只拉取新增数据，合并到本地 SQLite
        """
        ts_code = self._normalize_ts_code(symbol, "stock")
        
        # 1. 检查本地是否有数据
        local_df = db.get_stock_daily(ts_code, limit=1)
        
        if local_df.empty:
            # 首次加载：从 Tushare 全量拉取
            print(f"[Tushare] {ts_code} 首次加载，全量拉取...")
            df = self._fetch_stock_daily(ts_code, days)
            if not df.empty:
                db.save_stock_daily(df)
            return df
        else:
            # 已有数据：检查是否需要增量更新
            needs_update, start_date, end_date = db.needs_update(ts_code, "stock_daily")
            if needs_update:
                print(f"[Tushare] {ts_code} 增量更新 {start_date} ~ {end_date}...")
                new_df = self._fetch_stock_daily(ts_code, days=days, start_date=start_date, end_date=end_date)
                if not new_df.empty:
                    db.save_stock_daily(new_df)
            
            # 从本地读取（已包含最新数据）
            return db.get_stock_daily(ts_code, limit=days)

    def get_etf_daily(self, symbol: str, days: int = 120) -> pd.DataFrame:
        """获取ETF日K线（按需加载，增量更新）。"""
        ts_code = self._normalize_ts_code(symbol, "etf")
        
        local_df = db.get_etf_daily(ts_code, limit=1)
        
        if local_df.empty:
            print(f"[Tushare] {ts_code} 首次加载，全量拉取...")
            df = self._fetch_etf_daily(ts_code, days)
            if not df.empty:
                db.save_etf_daily(df)
            return df
        else:
            needs_update, start_date, end_date = db.needs_update(ts_code, "etf_daily")
            if needs_update:
                print(f"[Tushare] {ts_code} 增量更新 {start_date} ~ {end_date}...")
                new_df = self._fetch_etf_daily(ts_code, days=days, start_date=start_date, end_date=end_date)
                if not new_df.empty:
                    db.save_etf_daily(new_df)
            
            return db.get_etf_daily(ts_code, limit=days)

    def get_futures_daily(self, symbol: str, days: int = 120) -> pd.DataFrame:
        """获取期货日K线（按需加载，增量更新）。"""
        ts_code = self._normalize_ts_code(symbol, "futures")
        
        local_df = db.get_futures_daily(ts_code, limit=1)
        
        if local_df.empty:
            print(f"[Tushare] {ts_code} 首次加载，全量拉取...")
            df = self._fetch_futures_daily(ts_code, days)
            if not df.empty:
                db.save_futures_daily(df)
            return df
        else:
            needs_update, start_date, end_date = db.needs_update(ts_code, "futures_daily")
            if needs_update:
                print(f"[Tushare] {ts_code} 增量更新 {start_date} ~ {end_date}...")
                new_df = self._fetch_futures_daily(ts_code, days=days, start_date=start_date, end_date=end_date)
                if not new_df.empty:
                    db.save_futures_daily(new_df)
            
            return db.get_futures_daily(ts_code, limit=days)

    # ==================== 底层拉取方法 ====================

    def _fetch_stock_daily(self, ts_code: str, days: int = 120,
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None) -> pd.DataFrame:
        """从 Tushare 拉取A股日K线。"""
        api = self._get_api()
        
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
        
        try:
            df = ts.pro_bar(ts_code=ts_code, api=api, start_date=start_date,
                            end_date=end_date, freq="D", adj="qfq")
            if df is None or df.empty:
                return pd.DataFrame()
            df = df.sort_values("trade_date").reset_index(drop=True)
            return df
        except Exception as e:
            print(f"[Tushare] 拉取 {ts_code} 失败: {e}")
            return pd.DataFrame()

    def _fetch_etf_daily(self, ts_code: str, days: int = 120,
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None) -> pd.DataFrame:
        """从 Tushare 拉取ETF日K线。"""
        api = self._get_api()
        
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
        
        try:
            df = api.fund_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
            if df is None or df.empty:
                return pd.DataFrame()
            df = df.sort_values("trade_date").reset_index(drop=True)
            return df
        except Exception as e:
            print(f"[Tushare] 拉取 {ts_code} 失败: {e}")
            return pd.DataFrame()

    def _fetch_futures_daily(self, ts_code: str, days: int = 120,
                             start_date: Optional[str] = None,
                             end_date: Optional[str] = None) -> pd.DataFrame:
        """从 Tushare 拉取期货日K线。"""
        api = self._get_api()
        
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
        
        try:
            df = api.fut_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
            if df is None or df.empty:
                return pd.DataFrame()
            df = df.sort_values("trade_date").reset_index(drop=True)
            return df
        except Exception as e:
            print(f"[Tushare] 拉取 {ts_code} 失败: {e}")
            return pd.DataFrame()

    # ==================== 基础信息：全量更新 ====================

    def refresh_stock_basic(self) -> pd.DataFrame:
        """刷新A股基础信息（全量）。"""
        api = self._get_api()
        try:
            df = api.stock_basic(fields="ts_code,symbol,name,area,industry,market,list_date")
            if df is not None and not df.empty:
                db.save_stock_basic(df)
                print(f"[Tushare] A股基础信息已更新，共 {len(df)} 条")
                return df
        except Exception as e:
            print(f"[Tushare] 刷新A股基础信息失败: {e}")
        return pd.DataFrame()

    def refresh_etf_basic(self) -> pd.DataFrame:
        """刷新ETF基础信息（全量）。"""
        api = self._get_api()
        try:
            df = api.fund_basic()
            if df is not None and not df.empty:
                db.save_etf_basic(df)
                print(f"[Tushare] ETF基础信息已更新，共 {len(df)} 条")
                return df
        except Exception as e:
            print(f"[Tushare] 刷新ETF基础信息失败: {e}")
        return pd.DataFrame()

    def refresh_futures_basic(self) -> pd.DataFrame:
        """刷新期货基础信息（全量）。"""
        api = self._get_api()
        try:
            # Tushare 期货基础信息接口
            df = api.fut_basic()
            if df is not None and not df.empty:
                db.save_futures_basic(df)
                print(f"[Tushare] 期货基础信息已更新，共 {len(df)} 条")
                return df
        except Exception as e:
            print(f"[Tushare] 刷新期货基础信息失败: {e}")
        return pd.DataFrame()
