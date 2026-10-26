"""报告 Agent——汇总全流程产出，输出交易信号和分析报告。"""

from agno.agent import Agent

from llm import get_model
from .profiles.base import AssetConfig


def create_report_agent(config: AssetConfig) -> Agent:
    """创建报告 Agent。"""
    return Agent(
        model=get_model(),
        instructions=config.report_prompt,
        description="报告汇编员",
        add_datetime_to_context=True,
    )
