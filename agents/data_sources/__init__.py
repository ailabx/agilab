"""数据源入口——按 config.data_source 返回带缓存的数据源实例。"""

from .tushare_source import TushareSource
from .cache import CachedTushareSource, get_cache, set_cache, clear_cache, cache_stats


def get_data_source(data_source: str = "tushare"):
    """按配置返回数据源实例（带 SQLite 缓存）。"""
    if data_source == "tushare":
        return CachedTushareSource()
    raise ValueError(f"暂不支持数据源: {data_source}")
