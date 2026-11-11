"""agents/members.py - 定义个股评估团队中的专家 Agent

提供基于 YAML 配置动态创建专家 Agent 的能力，每个 Agent 专注于单一维度。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from agno.agent import Agent

from agents.llm import get_model
from agents.tools_tushare import get_all_tools


class AgentRegistry:
    """专家 Agent 注册表，从 config.yaml 读取人格配置并创建 Agent。"""

    def __init__(self, config_path: str | Path | None = None):
        if config_path is None:
            self.config_path = Path(__file__).resolve().parent / "config.yaml"
        else:
            self.config_path = Path(config_path)
        self._config = self._load_config()

    def _load_config(self) -> dict[str, Any]:
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    @property
    def personalities(self) -> dict[str, dict[str, Any]]:
        return self._config.get("personalities", {})

    def list_personalities(self) -> list[str]:
        return list(self.personalities.keys())

    def create_agent(self, personality_key: str) -> Agent:
        """根据 personality_key 创建一个专家 Agent。"""
        if personality_key not in self.personalities:
            raise ValueError(f"未知的专家类型: {personality_key}")

        p = self.personalities[personality_key]
        name = f"{p.get('emoji', '')} {p.get('name', personality_key)}".strip()
        instructions = self._build_instructions(p)

        return Agent(
            name=name,
            role=p.get("role", ""),
            model=get_model(),
            tools=get_all_tools(),
            instructions=instructions,
            markdown=True,
            add_datetime_to_context=True,
        )

    def create_all_agents(self) -> dict[str, Agent]:
        """创建所有配置的 Agent，返回 key -> Agent 映射。"""
        return {key: self.create_agent(key) for key in self.personalities}

    def _build_instructions(self, personality: dict[str, Any]) -> list[str]:
        """将人格配置转换为 Agent instructions。"""
        instructions = [f"你是 {personality.get('name', '分析师')}。"]
        instructions.append(f"角色定位：{personality.get('role', '')}")
        instructions.extend(personality.get("instructions", []))

        output_format = personality.get("output_format")
        if output_format:
            instructions.extend(["", "输出格式要求：", output_format])

        instructions.extend(
            [
                "",
                "通用输出要求：",
                "1. 使用中文，Markdown 格式。",
                "2. 必须给出明确的投资结论（如：强烈买入/买入/持有/卖出/强烈卖出，或技术面的买入/卖出/观望信号）。",
                "3. 使用工具获取最新数据支撑观点，引用关键数据。",
                "4. 保持分析聚焦，不要泛泛而谈。",
            ]
        )
        return instructions


# 单例注册表，供模块级别快速使用
_registry = None


def get_registry() -> AgentRegistry:
    global _registry
    if _registry is None:
        _registry = AgentRegistry()
    return _registry
