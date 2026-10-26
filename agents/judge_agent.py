"""裁判 Agent——综合多空辩论，输出裁决。"""

from agno.agent import Agent

from llm import get_model
from .profiles.base import AssetConfig


def create_judge_agent(config: AssetConfig) -> Agent:
    """创建裁判 Agent。"""
    return Agent(
        model=get_model(),
        instructions=config.judge_prompt,
        description="裁判",
        add_datetime_to_context=True,
    )
