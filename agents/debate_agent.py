"""多空辩论 Agent——多头 vs 空头，多轮交替。

参考 FDT 六阶段辩论 + TradingAgents 多空研究员对抗。
"""

from agno.agent import Agent

from llm import get_model
from .profiles.base import AssetConfig


def create_bull_agent(config: AssetConfig) -> Agent:
    """创建多头分析员 Agent。"""
    return Agent(
        model=get_model(),
        instructions=config.bull_prompt,
        description="多头分析员",
        add_datetime_to_context=True,
    )


def create_bear_agent(config: AssetConfig) -> Agent:
    """创建空头分析员 Agent。"""
    return Agent(
        model=get_model(),
        instructions=config.bear_prompt,
        description="空头分析员",
        add_datetime_to_context=True,
    )
