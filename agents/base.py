"""Agent 工厂——统一创建 agno Agent，配置从 AssetConfig 加载。"""

from agno.agent import Agent

from llm import get_model
from .profiles.base import AssetConfig


def create_agent(role: str, config: AssetConfig, **kwargs) -> Agent:
    """按角色名从 config 取提示词，创建 agno Agent。

    role 取值: data / tech / fund / news / flow / chain / sentiment / clause
               bull / bear / judge / risk / report
    """
    # 角色 → config 字段映射
    prompt_map = {
        "data": config.data_prompt,
        "bull": config.bull_prompt,
        "bear": config.bear_prompt,
        "judge": config.judge_prompt,
        "risk": config.risk_prompt,
        "report": config.report_prompt,
    }
    # 分析师角色从 analyst_prompts 取
    instructions = prompt_map.get(role) or config.analyst_prompts.get(role, "")

    if not instructions:
        raise ValueError(f"角色 '{role}' 无提示词配置")

    return Agent(
        model=get_model(),
        instructions=instructions,
        add_datetime_to_context=True,
        **kwargs,
    )
