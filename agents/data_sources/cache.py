"""SQLite 数据缓存——避免重复调 tushare API 浪费配额。

使用 sqlite-utils 自动建表，无需手写 schema。
缓存策略: 按 symbol + 数据类型 + 日期 做 key，当日缓存有效（同一交易日不重复拉）。
"""

import json
from datetime import datetime
from pathlib import Path

from sqlite_utils import Database

from config import BASE_DIR

# 缓存数据库路径
_DB_PATH = BASE_DIR / "data" / "cache.db"
_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# 缓存有效期（小时）——当日数据缓存到第二天
_CACHE_HOURS = 12


def _get_db() -> Database:
    """获取缓存数据库实例。"""
    return Database(str(_DB_PATH))


def _is_expired(cached_at: str) -> bool:
    """判断缓存是否过期。"""
    try:
        cached_time = datetime.fromisoformat(cached_at)
        age = (datetime.now() - cached_time).total_seconds() / 3600
        return age > _CACHE_HOURS
    except Exception:
        return True


def get_cache(symbol: str, data_type: str) -> dict | None:
    """读取缓存。

    参数:
        symbol: 品种代码，如 "600519"
        data_type: 数据类型，如 "daily" / "basic" / "daily_basic" / "news" / "all"

    返回:
        缓存的 dict，无缓存或过期返回 None
    """
    db = _get_db()
    table = db["cache"]
    try:
        row = table.get((symbol, data_type))
    except Exception:
        return None

    if _is_expired(row.get("cached_at", "")):
        return None

    try:
        return json.loads(row["data"])
    except Exception:
        return None


def set_cache(symbol: str, data_type: str, data: dict) -> None:
    """写入缓存。

    sqlite-utils 自动建表，首次插入时自动创建 cache 表。
    """
    db = _get_db()
    db["cache"].upsert(
        {
            "symbol": symbol,
            "data_type": data_type,
            "data": json.dumps(data, ensure_ascii=False, default=str),
            "cached_at": datetime.now().isoformat(),
        },
        pk=("symbol", "data_type"),  # 联合主键
    )


def clear_cache(symbol: str = None) -> int:
    """清除缓存。不传 symbol 清全部，传了只清该品种。

    返回删除的行数。
    """
    db = _get_db()
    if "cache" not in db.table_names():
        return 0
    if symbol:
        return db["cache"].delete_where("symbol = ?", [symbol])
    return db["cache"].delete_where()


def cache_stats() -> dict:
    """缓存统计信息。"""
    db = _get_db()
    if "cache" not in db.table_names():
        return {"total": 0, "symbols": 0}
    table = db["cache"]
    total = table.count
    symbols = len(list(db.query("SELECT DISTINCT symbol FROM cache")))
    return {"total": total, "symbols": symbols}


# ==================== 带缓存的 TushareSource ====================


class CachedTushareSource:
    """带 SQLite 缓存的 Tushare 数据源。

    对每个 (symbol, data_type) 组合做缓存，当日内不重复调 API。
    """

    def __init__(self):
        from .tushare_source import TushareSource
        self._source = TushareSource()

    def _cached(self, symbol: str, data_type: str, fetcher) -> dict:
        """通用缓存逻辑: 先读缓存，miss 则拉数据并写缓存。"""
        cached = get_cache(symbol, data_type)
        if cached is not None:
            return cached
        data = fetcher()
        if data:
            set_cache(symbol, data_type, data)
        return data

    def get_daily(self, symbol: str, days: int = 120):
        import pandas as pd
        cached = get_cache(symbol, "daily")
        if cached is not None:
            return pd.DataFrame(cached)
        df = self._source.get_daily(symbol, days)
        if not df.empty:
            set_cache(symbol, "daily", df.to_dict("records"))
        return df

    def get_realtime(self, symbol: str) -> dict:
        return self._cached(symbol, "realtime", lambda: self._source.get_realtime(symbol))

    def get_stock_basic(self, symbol: str) -> dict:
        return self._cached(symbol, "basic", lambda: self._source.get_stock_basic(symbol))

    def get_daily_basic(self, symbol: str) -> dict:
        return self._cached(symbol, "daily_basic", lambda: self._source.get_daily_basic(symbol))

    def get_fund_basic(self, symbol: str) -> dict:
        return self._cached(symbol, "fund_basic", lambda: self._source.get_fund_basic(symbol))

    def get_cb_basic(self, symbol: str) -> dict:
        return self._cached(symbol, "cb_basic", lambda: self._source.get_cb_basic(symbol))

    def get_cb_daily(self, symbol: str, days: int = 60):
        import pandas as pd
        cached = get_cache(symbol, "cb_daily")
        if cached is not None:
            return pd.DataFrame(cached)
        df = self._source.get_cb_daily(symbol, days)
        if not df.empty:
            set_cache(symbol, "cb_daily", df.to_dict("records"))
        return df

    def get_news(self, symbol: str, limit: int = 10) -> list:
        return self._cached(symbol, "news", lambda: self._source.get_news(symbol, limit))

    def get_all(self, symbol: str) -> dict:
        """综合接口——走缓存。"""
        return self._cached(symbol, "all", lambda: self._source.get_all(symbol))
