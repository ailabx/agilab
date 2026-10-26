"""AgiLab 多智能体量化分析工作流。

用法:
    from agents import create_workflow
    from agents.profiles import get_config

    config = get_config("stock")
    wf = create_workflow(config)
    wf.print_response("分析 600519 贵州茅台")
"""

from .profiles.base import AssetConfig
from .profiles import get_config
from .workflow import create_workflow

__all__ = [
    "AssetConfig",
    "get_config",
    "create_workflow",
]
