"""核心工作流——按 AssetConfig 动态组装 agno Workflow。

编排逻辑:
  Step(数据采集) → Parallel(分析师团队) → Loop(多空辩论) 
  → Step(裁判裁决) → Condition(风控审核) → Step(报告输出)

所有原语来自 agno.workflow: Step / Parallel / Loop / Condition / Workflow
"""

from agno.workflow import Workflow
from agno.workflow.step import Step
from agno.workflow.parallel import Parallel
from agno.workflow.loop import Loop
from agno.workflow.condition import Condition

from .profiles.base import AssetConfig
from .data_agent import create_data_agent
from .analyst_agent import create_analysts
from .debate_agent import create_bull_agent, create_bear_agent
from .judge_agent import create_judge_agent
from .risk_agent import create_risk_agent
from .report_agent import create_report_agent


def _risk_evaluator(step_input) -> bool:
    """风控条件判断: 返回 True=通过(放行), False=否决。

    后续补充: 解析风控 Agent 输出，判断 green/yellow/red。
    当前框架: 默认放行。
    """
    return True


def create_workflow(config: AssetConfig) -> Workflow:
    """按市场配置动态组装 Workflow。

    参数:
        config: AssetConfig 实例 (stock/etf/futures/convertible)

    返回:
        agno Workflow 实例, 调用 .run(input) 启动
    """
    # Phase 1: 数据采集
    data_step = Step(
        name="数据采集",
        agent=create_data_agent(config),
    )

    # Phase 2: 分析师团队 (并行, 数量由 config.analysts 决定)
    analyst_agents = create_analysts(config)
    analyst_steps = [
        Step(name=f"分析师-{role}", agent=agent)
        for role, agent in zip(config.analysts, analyst_agents)
    ]
    analyst_parallel = Parallel(
        *analyst_steps,
        name="分析师团队",
    )

    # Phase 3: 多空辩论 (Loop, 轮次由 config.debate_rounds 决定)
    bull_step = Step(name="多头立论", agent=create_bull_agent(config))
    bear_step = Step(name="空头立论", agent=create_bear_agent(config))
    debate_loop = Loop(
        steps=[bull_step, bear_step],
        name="多空辩论",
        max_iterations=config.debate_rounds,
    )

    # Phase 4: 裁判裁决
    judge_step = Step(
        name="裁判裁决",
        agent=create_judge_agent(config),
    )

    # Phase 5: 风控审核 (Condition: green→报告, red→否决)
    report_step = Step(
        name="报告输出",
        agent=create_report_agent(config),
    )
    risk_step = Step(
        name="风控审核",
        agent=create_risk_agent(config),
    )
    # 风控通过 → 报告输出; 风控否决 → 直接结束(否决信息仍传给报告)
    risk_condition = Condition(
        steps=[report_step],
        evaluator=_risk_evaluator,
        else_steps=[report_step],  # 否决也生成报告, 标注 red
        name="风控审核",
    )

    # 组装
    workflow = Workflow(
        name=f"AgiLab-{config.asset_type}",
        description=f"{config.asset_type} 市场分析工作流",
        steps=[
            data_step,           # Phase 1
            analyst_parallel,    # Phase 2
            debate_loop,         # Phase 3
            judge_step,          # Phase 4
            risk_step,           # Phase 5 (先审核)
            risk_condition,      # Phase 5 (条件路由)
        ],
    )

    return workflow
