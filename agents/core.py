"""agents/core.py - 多智能体个股评估核心

提供 `StockEvaluationTeam` 类，封装 Agno Team 的创建、运行与结果解析。
借鉴 TradingAgents-CN 的工作流思想，但在 Agno 内实现：

    第一层：5 位分析师并行研究（价值/成长/技术/风险/行业宏观）
    第二层：多头/空头研究员基于第一层结果展开辩论
    第三层：研究经理汇总所有观点并给出综合结论
    第四层：风险裁决者对综合结论进行风险评估与修正

不引入 LangGraph，所有流程通过 Agno Team 与多轮 Agent 调用完成。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from agno.agent import Agent
from agno.team.team import Team, TeamMode

from agents.llm import get_model
from agents.members import get_registry
from agents.tools_tushare import get_all_tools


class StockEvaluationTeam:
    """个股评估团队：协调多位专家 Agent 对同一只股票进行综合评估。"""

    def __init__(self, config_path: str | Path | None = None):
        # 默认使用本文件所在目录的 config.yaml，确保从任意工作目录启动都能找到
        if config_path is None:
            self.config_path = Path(__file__).resolve().parent / "config.yaml"
        else:
            self.config_path = Path(config_path)
        self._config = self._load_config()
        self.registry = get_registry()
        self._analyst_team = self._build_analyst_team()
        self._debate_team = self._build_debate_team()
        self._adjudicator_team = self._build_adjudicator_team()

    def _load_config(self) -> dict[str, Any]:
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    @property
    def team_config(self) -> dict[str, Any]:
        return self._config.get("team", {})

    def _create_analyst(self, key: str) -> Agent:
        """创建单个分析师 Agent。"""
        return self.registry.create_agent(key)

    def _build_analyst_team(self) -> Team:
        """构建第一层：5 位分析师并行研究。"""
        analyst_keys = [
            "value_analyst",
            "growth_analyst",
            "technical_analyst",
            "risk_analyst",
            "industry_macro_analyst",
        ]
        members = [self._create_analyst(key) for key in analyst_keys]

        return Team(
            name="第一层：分析师研究团队",
            mode=TeamMode.coordinate,
            model=get_model(),
            members=members,
            tools=get_all_tools(),
            instructions=[
                "你是多智能体 A 股个股评估的第一层协调者。",
                "协调价值、成长、技术、风险、行业与宏观五位分析师并行完成对目标股票的独立研究。",
                "汇总各位分析师的结构化输出，保留关键数据、评级和核心论据，作为后续多头/空头辩论的事实基础。",
                "输出一份结构化的‘分析师综合备忘录’，包含每位分析师的：结论、置信度、关键证据、核心风险。",
                "用中文输出，Markdown 格式。",
            ],
            markdown=True,
            show_members_responses=False,
            share_member_interactions=True,
            add_datetime_to_context=True,
            stream=True,
            stream_events=True,
        )

    def _build_debate_team(self) -> Team:
        """构建第二层：多头/空头研究员辩论。"""
        bull = self.registry.create_agent("bull_researcher")
        bear = self.registry.create_agent("bear_researcher")

        return Team(
            name="第二层：多空辩论团队",
            mode=TeamMode.coordinate,
            model=get_model(),
            members=[bull, bear],
            tools=get_all_tools(),
            instructions=[
                "你是多空辩论环节的主持人。",
                "基于第一层的分析师综合备忘录，组织多头研究员和空头研究员进行结构化辩论。",
                "要求双方基于事实和数据提出论据，明确列出各自的核心假设、催化剂和风险。",
                "输出一份结构化的‘多空辩论记录’，包含：多头核心论点、空头核心论点、双方共识、双方分歧、关键未决问题。",
                "用中文输出，Markdown 格式。",
            ],
            markdown=True,
            show_members_responses=False,
            share_member_interactions=True,
            add_datetime_to_context=True,
        )

    def _build_adjudicator_team(self) -> Team:
        """构建第三、四层：研究经理汇总 + 风险裁决者把关。"""
        manager = self.registry.create_agent("research_manager")
        judge = self.registry.create_agent("risk_judge")

        return Team(
            name="第三、四层：研究经理与风险裁决",
            mode=TeamMode.coordinate,
            model=get_model(),
            members=[manager, judge],
            tools=get_all_tools(),
            instructions=[
                "你是最终裁决环节的协调者。",
                "首先由研究经理汇总第一层分析师综合备忘录和第二层多空辩论记录，给出最终综合评级、目标价区间、建议仓位。",
                "然后由风险裁决者对研究经理的结论进行独立风险评估，识别盲区、尾部风险和估值陷阱。",
                "输出最终结构化报告，必须包含：",
                "  1. 执行摘要",
                "  2. 分析师综合备忘录摘要",
                "  3. 多空辩论核心观点",
                "  4. 研究经理综合结论（评级、目标价区间、仓位建议）",
                "  5. 风险裁决者意见（风险等级、风险修正、关键监控指标）",
                "  6. 最终投资结论与操作建议",
                "  7. 风险提示与止损参考",
                "用中文输出，Markdown 格式。",
            ],
            markdown=True,
            show_members_responses=False,
            share_member_interactions=True,
            add_datetime_to_context=True,
        )

    def evaluate(self, symbol: str, task: str | None = None) -> dict[str, Any]:
        """对指定股票进行综合评估，返回多层分析结果。"""
        prompt = task or self._default_task(symbol)

        # 第一层：分析师并行研究
        analyst_result = self._analyst_team.run(prompt, stream=False)
        analyst_memo = self._extract_content(analyst_result)

        # 第二层：多空辩论
        debate_prompt = (
            f"目标股票：{symbol}\n\n"
            "以下是一份分析师综合备忘录，请基于其中事实和数据展开多空辩论：\n\n"
            f"{analyst_memo}"
        )
        debate_result = self._debate_team.run(debate_prompt, stream=False)
        debate_record = self._extract_content(debate_result)

        # 第三、四层：研究经理汇总 + 风险裁决
        final_prompt = (
            f"目标股票：{symbol}\n\n"
            "分析师综合备忘录：\n\n"
            f"{analyst_memo}\n\n"
            "多空辩论记录：\n\n"
            f"{debate_record}\n\n"
            "请研究经理和风险管理官基于以上信息给出最终投资结论和风险评估。"
        )
        final_result = self._adjudicator_team.run(final_prompt, stream=False)

        return {
            "symbol": symbol,
            "analyst_memo": analyst_memo,
            "debate_record": debate_record,
            "final_report": self._extract_content(final_result),
            "raw": {
                "analyst": analyst_result,
                "debate": debate_result,
                "final": final_result,
            },
        }

    def evaluate_and_print(self, symbol: str, task: str | None = None) -> None:
        """对指定股票进行评估并打印最终结果。"""
        result = self.evaluate(symbol, task=task)
        print(f"# 个股评估报告：{symbol}\n")
        print(result["final_report"])

    def _default_task(self, symbol: str) -> str:
        return (
            f"请对股票 {symbol} 进行多维度个股评估。\n"
            "每位分析师从各自维度给出独立分析，最后由团队协调者输出结构化综合报告。\n"
            "综合报告必须包含：执行摘要、各位分析师结论、共识与分歧、综合评级、目标价区间、风险提示。"
        )

    @staticmethod
    def _extract_content(result: Any) -> str:
        """从 Agno 的返回结果中提取可读的文本内容。"""
        if result is None:
            return ""
        if isinstance(result, str):
            return result
        if hasattr(result, "content"):
            return result.content
        if hasattr(result, "messages"):
            messages = getattr(result, "messages", [])
            if messages:
                last = messages[-1]
                return getattr(last, "content", str(last))
        return str(result)


# 便捷函数，保持向后兼容
def evaluate_stock(symbol: str, task: str | None = None) -> dict[str, Any]:
    """便捷的顶层函数：评估一只股票。"""
    team = StockEvaluationTeam()
    return team.evaluate(symbol, task=task)


if __name__ == "__main__":
    import sys

    symbol = (sys.argv[1] if len(sys.argv) > 1 else "600519.SH").upper()
    print(f"正在评估股票: {symbol}\n")
    evaluate_stock(symbol)
