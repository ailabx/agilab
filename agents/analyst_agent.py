"""分析师 Agent——按 config.analysts 动态创建多个。

每个分析师角色 (tech/fund/news/flow/chain/sentiment/clause) 共用同一套创建逻辑，
instructions 从 config.analyst_prompts 加载，实现多市场复用。
"""

from agno.agent import Agent

from llm import get_model
from .profiles.base import AssetConfig


def create_analyst(role: str, config: AssetConfig) -> Agent:
    """创建单个分析师 Agent。

    role 取值: tech / fund / news / flow / chain / sentiment / clause
    """
    prompt = config.analyst_prompts.get(role, "")
    if not prompt:
        raise ValueError(f"分析师角色 '{role}' 无提示词配置")

    return Agent(
        model=get_model(),
        instructions=prompt,
        description=f"分析师 ({role})",
        add_datetime_to_context=True,
    )


def create_analysts(config: AssetConfig) -> list:
    """按 config.analysts 批量创建分析师 Agent 列表。"""
    return [create_analyst(role, config) for role in config.analysts]
