"""市场配置入口——按 asset_type 返回对应配置实例。"""

from .base import AssetConfig
from .stock import StockConfig
from .etf import ETFConfig
from .futures import FuturesConfig
from .convertible import ConvertibleConfig

# 市场类型 → 配置类映射
_PROFILE_MAP = {
    "stock": StockConfig,
    "etf": ETFConfig,
    "futures": FuturesConfig,
    "convertible": ConvertibleConfig,
}


def get_config(asset_type: str = "stock") -> AssetConfig:
    """按市场类型返回配置实例。"""
    cls = _PROFILE_MAP.get(asset_type, StockConfig)
    return cls()
