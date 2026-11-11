"""agents/workflows.py - 投资研究主工作流

入口：一个 Workflow，包含 intake -> router -> 分支。
个股评估分支的所有子步骤直接展现在顶层 Workflow，Agent OS 可见。
"""

from __future__ import annotations

import re
from typing import Any

from agno.workflow import Parallel, Router, Step, StepInput, StepOutput, Workflow

from agents.members import get_registry


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def _extract_symbol(text: str) -> str | None:
    """从用户输入中提取股票代码。"""
    m = re.search(r"(\d{6})\.((SH|SZ|BJ))", text.upper())
    if m:
        return m.group(1) + "." + m.group(2)
    m = re.search(r"(\d{6})", text)
    if m:
        code = m.group(1)
        suffix = "SH" if code.startswith("6") else "SZ"
        return f"{code}.{suffix}"
    return None


def _get_symbol(step_input: StepInput) -> str | None:
    """从 step_input 中提取 symbol。"""
    content = step_input.input
    if isinstance(content, dict):
        return content.get("symbol")
    if isinstance(content, str):
        return _extract_symbol(content)
    previous = step_input.previous_step_content or {}
    if isinstance(previous, dict):
        return previous.get("symbol")
    return None


# ---------------------------------------------------------------------------
# Step 1: intake - 理解用户意图，提取股票代码
# ---------------------------------------------------------------------------
def intake_executor(step_input: StepInput) -> StepOutput:
    user_input = str(step_input.input or "")
    symbol = _extract_symbol(user_input)

    result: dict[str, Any] = {
        "symbol": symbol,
        "user_input": user_input,
        "reply": "",
    }
    print(f"[intake] {result}")
    return StepOutput(content=result, success=True)


# ---------------------------------------------------------------------------
# 普通问答分支
# ---------------------------------------------------------------------------
def general_chat_executor(step_input: StepInput) -> StepOutput:
    previous = step_input.previous_step_content or {}
    user_input = previous.get("user_input", "") if isinstance(previous, dict) else str(step_input.input or "")
    return StepOutput(
        content={"reply": f"我是一般问答模式，当前输入：{user_input}"},
        success=True,
    )


general_chat_step = Step(
    name="general_chat",
    executor=general_chat_executor,
    description="普通问答",
)


# ---------------------------------------------------------------------------
# 个股评估分支：5位分析师并行 -> 多空辩论 -> 最终裁决
# ---------------------------------------------------------------------------
_registry = get_registry()


def _make_analyst_executor(agent, key: str):
    def _executor(step_input: StepInput) -> StepOutput:
        symbol = _get_symbol(step_input)
        print(f"[analyst_{key}] start {symbol}")
        result = agent.run(f"请对股票 {symbol} 进行独立分析，给出结构化投资结论。", stream=False)
        content = result.content if hasattr(result, "content") else str(result)
        print(f"[analyst_{key}] done")
        return StepOutput(content=content, success=True)

    _executor.__name__ = f"analyst_executor_{key}"
    return _executor


def build_analyst_steps() -> list[Step]:
    analyst_keys = [
        "value_analyst",
        "growth_analyst",
        "technical_analyst",
        "risk_analyst",
        "industry_macro_analyst",
    ]
    steps = []
    for key in analyst_keys:
        agent = _registry.create_agent(key)
        step = Step(
            name=f"analyst_{key}",
            executor=_make_analyst_executor(agent, key),
            description=f"{key} 分析",
        )
        steps.append(step)
    return steps


def debate_executor(step_input: StepInput) -> StepOutput:
    symbol = _get_symbol(step_input)
    memo = step_input.previous_step_content or ""
    print(f"[debate] {symbol}")
    return StepOutput(
        content=f"多空辩论记录（{symbol}）：\n{memo}",
        success=True,
    )


def adjudicator_executor(step_input: StepInput) -> StepOutput:
    symbol = _get_symbol(step_input)
    debate_record = step_input.previous_step_content or ""
    print(f"[adjudicator] {symbol}")
    return StepOutput(
        content=f"最终裁决（{symbol}）：\n{debate_record}",
        success=True,
    )


# 个股评估分支：嵌套列表会被转成 Steps 容器顺序执行
evaluate_stock_branch = [
    Parallel(
        *build_analyst_steps(),
        name="analyst_team",
        description="5位分析师并行研究",
    ),
    Step(name="debate", executor=debate_executor, description="多空研究员辩论"),
    Step(name="adjudicator", executor=adjudicator_executor, description="研究经理与风险裁决"),
]


# ---------------------------------------------------------------------------
# Router: 根据 intake 结果选择分支
# ---------------------------------------------------------------------------
def router_selector(step_input: StepInput) -> str:
    previous = step_input.previous_step_content or {}
    if isinstance(previous, dict) and previous.get("symbol"):
        return "evaluate_stock"
    return "general_chat"


# ---------------------------------------------------------------------------
# 顶层工作流定义
# ---------------------------------------------------------------------------
investment_workflow = Workflow(
    name="投资研究工作流",
    id="investment-research-workflow",
    description="支持普通问答和个股综合评估",
    steps=[
        Step(name="intake", executor=intake_executor, description="理解用户意图，提取股票代码"),
        Router(
            name="router",
            selector=router_selector,
            choices=[
                general_chat_step,
                evaluate_stock_branch,
            ],
        ),
    ],
)
