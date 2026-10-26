"""风控 Agent——审核交易参数，输出 green/yellow/red。"""

from agno.agent import Agent

from llm import get_model
from .profiles.base import AssetConfig


def create_risk_agent(config: AssetConfig) -> Agent:
    """创建风控 Agent。"""
    # 把风控参数注入提示词
    risk_info = f"\n当前风控参数: {config.risk_params}"
    instructions = config.risk_prompt + risk_info

    return Agent(
        model=get_model(),
        instructions=instructions,
        description="风控官",
        add_datetime_to_context=True,
    )
