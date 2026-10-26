"""资产配置基类——所有市场配置的公共字段。"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class AssetConfig:
    """市场配置基类，子类按市场覆盖默认值。"""

    asset_type: str = "stock"
    data_source: str = "tushare"
    analysts: List[str] = field(default_factory=lambda: ["tech", "fund", "news"])
    debate_rounds: int = 2
    risk_params: Dict = field(default_factory=lambda: {
        "stop_loss": 0.05,
        "max_position": 0.3,
    })
    # 每个分析师角色的提示词模板，按 config.analysts 中的 key 取用
    analyst_prompts: Dict[str, str] = field(default_factory=dict)
    # 数据采集 Agent 的提示词
    data_prompt: str = ""
    # 裁判 Agent 的提示词
    judge_prompt: str = ""
    # 风控 Agent 的提示词
    risk_prompt: str = ""
    # 报告 Agent 的提示词
    report_prompt: str = ""
    # 多头 Agent 的提示词
    bull_prompt: str = ""
    # 空头 Agent 的提示词
    bear_prompt: str = ""
