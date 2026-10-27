"""数据源入口 —— 按需加载，增量更新。

用法：
    from agents.data_sources import TushareSource
    
    source = TushareSource()
    
    # 获取A股日K线（自动全量/增量）
    df = source.get_stock_daily("600519", days=120)
    
    # 获取ETF日K线
    df = source.get_etf_daily("510300", days=60)
    
    # 获取期货日K线
    df = source.get_futures_daily("RB2501", days=60)
    
    # 刷新基础信息（全量）
    source.refresh_stock_basic()
    source.refresh_etf_basic()
    source.refresh_futures_basic()
"""

from .tushare_source import TushareSource
from .db_manager import (
    # 日K线查询
    get_stock_daily,
    get_etf_daily,
    get_futures_daily,
    # 基础信息查询
    get_stock_basic,
    get_etf_basic,
    get_futures_basic,
    get_basic,
    # 日K线保存
    save_stock_daily,
    save_etf_daily,
    save_futures_daily,
    # 基础信息保存
    save_stock_basic,
    save_etf_basic,
    save_futures_basic,
    # 增量更新判断
    needs_update,
)

__all__ = [
    "TushareSource",
    "get_stock_daily",
    "get_etf_daily",
    "get_futures_daily",
    "get_stock_basic",
    "get_etf_basic",
    "get_futures_basic",
    "get_basic",
    "save_stock_daily",
    "save_etf_daily",
    "save_futures_daily",
    "save_stock_basic",
    "save_etf_basic",
    "save_futures_basic",
    "needs_update",
]


# 兼容旧接口
def get_data_source(data_source: str = "tushare"):
    """按配置返回数据源实例（兼容旧接口）。"""
    if data_source == "tushare":
        return TushareSource()
    raise ValueError(f"暂不支持数据源: {data_source}")
