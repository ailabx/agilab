"""SQLite 数据管理器 —— 按品种按需加载，增量更新。

设计原则：
    - 用哪支加载哪支，首次全量，后续增量
    - sqlite-utils 自动建表，无需手写 schema
    - 数据表：stock_daily, etf_daily, futures_daily（按 ts_code + trade_date 联合主键）
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlite_utils import Database

from config import BASE_DIR

# 数据目录
_DATA_DIR = BASE_DIR / "data"
_DATA_DIR.mkdir(parents=True, exist_ok=True)
_DB_PATH = _DATA_DIR / "market_data.db"


def _get_db() -> Database:
    """获取数据库实例。"""
    return Database(str(_DB_PATH))


# ==================== 查询：获取某品种最新日期 ====================

def _get_latest_date(table: str, ts_code: str) -> Optional[str]:
    """获取某品种在表中的最新 trade_date。"""
    db = _get_db()
    if table not in db.table_names():
        return None
    rows = list(db[table].rows_where("ts_code = ?", [ts_code], order_by="trade_date DESC", limit=1))
    if rows:
        return rows[0]["trade_date"]
    return None


def _get_oldest_date(table: str, ts_code: str) -> Optional[str]:
    """获取某品种在表中的最早 trade_date。"""
    db = _get_db()
    if table not in db.table_names():
        return None
    rows = list(db[table].rows_where("ts_code = ?", [ts_code], order_by="trade_date ASC", limit=1))
    if rows:
        return rows[0]["trade_date"]
    return None


# ==================== 写入：增量保存日K线 ====================

def save_daily(df: pd.DataFrame, table: str) -> None:
    """增量保存日K线数据。
    
    Args:
        df: DataFrame，必须包含 ts_code 和 trade_date 列
        table: 表名，如 "stock_daily", "etf_daily", "futures_daily"
    """
    if df.empty:
        return
    db = _get_db()
    db[table].insert_all(
        df.to_dict("records"),
        pk=["ts_code", "trade_date"],
        replace=True,  # 自动去重
    )


# ==================== 查询：读取日K线 ====================

def get_daily(ts_code: str, table: str, start_date: Optional[str] = None,
              end_date: Optional[str] = None, limit: int = 252) -> pd.DataFrame:
    """读取日K线数据。
    
    Args:
        ts_code: 品种代码，如 "600519.SH"
        table: 表名
        start_date: 开始日期 "YYYYMMDD"
        end_date: 结束日期 "YYYYMMDD"
        limit: 限制条数
    
    Returns:
        DataFrame，按 trade_date 升序排列
    """
    db = _get_db()
    if table not in db.table_names():
        return pd.DataFrame()
    
    where_clauses = ["ts_code = ?"]
    params = [ts_code]
    
    if start_date:
        where_clauses.append("trade_date >= ?")
        params.append(start_date)
    if end_date:
        where_clauses.append("trade_date <= ?")
        params.append(end_date)
    
    where_sql = " AND ".join(where_clauses)
    rows = list(db[table].rows_where(where_sql, params, order_by="trade_date DESC", limit=limit))
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("trade_date").reset_index(drop=True)
    return df


# ==================== 增量更新判断 ====================

def needs_update(ts_code: str, table: str, days_back: int = 1) -> tuple[bool, Optional[str], Optional[str]]:
    """判断某品种是否需要更新。
    
    Returns:
        (是否需要更新, 开始日期, 结束日期)
        - 首次加载：返回 (True, None, None) → 全量加载
        - 已有数据：返回 (True, latest_date+1, today) → 增量
        - 数据最新：返回 (False, None, None)
    """
    latest = _get_latest_date(table, ts_code)
    today = datetime.now().strftime("%Y%m%d")
    
    if latest is None:
        # 首次加载，全量
        return True, None, None
    
    # 检查是否是最新的
    latest_dt = datetime.strptime(latest, "%Y%m%d")
    today_dt = datetime.now()
    
    # 如果最新数据就是今天或昨天（考虑交易日），不需要更新
    if (today_dt - latest_dt).days <= days_back:
        return False, None, None
    
    # 需要增量更新
    start = (latest_dt + timedelta(days=1)).strftime("%Y%m%d")
    return True, start, today


# ==================== 便捷函数 ====================

def get_stock_daily(ts_code: str, start_date: Optional[str] = None,
                    end_date: Optional[str] = None, limit: int = 252) -> pd.DataFrame:
    """获取A股日K线。"""
    return get_daily(ts_code, "stock_daily", start_date, end_date, limit)


def get_etf_daily(ts_code: str, start_date: Optional[str] = None,
                  end_date: Optional[str] = None, limit: int = 252) -> pd.DataFrame:
    """获取ETF日K线。"""
    return get_daily(ts_code, "etf_daily", start_date, end_date, limit)


def get_futures_daily(ts_code: str, start_date: Optional[str] = None,
                      end_date: Optional[str] = None, limit: int = 252) -> pd.DataFrame:
    """获取期货日K线。"""
    return get_daily(ts_code, "futures_daily", start_date, end_date, limit)


def save_stock_daily(df: pd.DataFrame) -> None:
    """保存A股日K线。"""
    save_daily(df, "stock_daily")


def save_etf_daily(df: pd.DataFrame) -> None:
    """保存ETF日K线。"""
    save_daily(df, "etf_daily")


def save_futures_daily(df: pd.DataFrame) -> None:
    """保存期货日K线。"""
    save_daily(df, "futures_daily")


# ==================== 基础信息表（全量） ====================

def save_basic(df: pd.DataFrame, table: str) -> None:
    """全量替换保存基础信息表。"""
    if df.empty:
        return
    db = _get_db()
    db[table].insert_all(df.to_dict("records"), pk="ts_code", replace=True)


def get_basic(table: str, ts_code: Optional[str] = None) -> pd.DataFrame:
    """读取基础信息。"""
    db = _get_db()
    if table not in db.table_names():
        return pd.DataFrame()
    if ts_code:
        rows = list(db[table].rows_where("ts_code = ?", [ts_code]))
        return pd.DataFrame(rows)
    return pd.DataFrame(list(db[table].rows))


def save_stock_basic(df: pd.DataFrame) -> None:
    """保存A股基础信息。"""
    save_basic(df, "stock_basic")


def save_etf_basic(df: pd.DataFrame) -> None:
    """保存ETF基础信息。"""
    save_basic(df, "etf_basic")


def save_futures_basic(df: pd.DataFrame) -> None:
    """保存期货基础信息。"""
    save_basic(df, "futures_basic")


def get_stock_basic(ts_code: Optional[str] = None) -> pd.DataFrame:
    """获取A股基础信息。"""
    return get_basic("stock_basic", ts_code)


def get_etf_basic(ts_code: Optional[str] = None) -> pd.DataFrame:
    """获取ETF基础信息。"""
    return get_basic("etf_basic", ts_code)


def get_futures_basic(ts_code: Optional[str] = None) -> pd.DataFrame:
    """获取期货基础信息。"""
    return get_basic("futures_basic", ts_code)
