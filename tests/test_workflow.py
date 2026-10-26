"""逐步测试——用真实 tushare 数据跑通 Workflow 全链路。

运行: .venv/Scripts/python.exe tests/test_workflow.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents import get_config
from agents.data_agent import create_data_agent, fetch_data
from agents.analyst_agent import create_analysts
from agents.debate_agent import create_bull_agent, create_bear_agent
from agents.judge_agent import create_judge_agent
from agents.risk_agent import create_risk_agent
from agents.report_agent import create_report_agent


def run_step(name, agent, prompt):
    """跑单个 Agent，打印状态。"""
    print(f"\n{'='*50}")
    print(f"[{name}] 开始...")
    start = time.time()
    try:
        result = agent.run(prompt)
        elapsed = time.time() - start
        content = ""
        if hasattr(result, "content") and result.content:
            content = str(result.content)
        print(f"[{name}] ✓ 通过 ({elapsed:.1f}s)")
        if content:
            print(f"[{name}] 输出 ({len(content)} 字符): {content[:300]}...")
        return content
    except Exception as e:
        elapsed = time.time() - start
        print(f"[{name}] ✗ 失败 ({elapsed:.1f}s): {type(e).__name__}: {e}")
        return None


def main():
    cfg = get_config("stock")
    print(f"配置: market={cfg.asset_type} | data_source={cfg.data_source}")
    print(f"分析师: {cfg.analysts} | 辩论轮次: {cfg.debate_rounds}")

    symbol = "600519"

    # Phase 1: 数据采集——先拉真实数据，再让 LLM 整理
    print(f"\n{'='*50}")
    print("[Phase1-数据采集] 拉取真实数据...")
    start = time.time()
    data_str = fetch_data(cfg, symbol)
    print(f"[Phase1-数据采集] 数据拉取完成 ({time.time()-start:.1f}s)")

    data_agent = create_data_agent(cfg)
    prompt = f"分析品种: {symbol}\n以下是系统拉取的真实数据，请整理成 JSON:\n{data_str[:3000]}"
    data_result = run_step("Phase1-数据采集", data_agent, prompt)

    # Phase 2: 分析师团队 (串行跑)
    analyst_agents = create_analysts(cfg)
    analyst_results = {}
    for role, agent in zip(cfg.analysts, analyst_agents):
        prompt = f"分析品种: {symbol}\n角色: {role}\n数据: {data_result or data_str[:2000]}"
        result = run_step(f"Phase2-{role}", agent, prompt)
        analyst_results[role] = result

    # Phase 3: 多空辩论 (2轮)
    bull_agent = create_bull_agent(cfg)
    bear_agent = create_bear_agent(cfg)
    research_data = str(analyst_results)

    bull_out = ""
    bear_out = ""
    for i in range(cfg.debate_rounds):
        bull_prompt = f"品种: {symbol}\n第{i+1}轮立论。基于研究数据构建做多论据:\n{research_data}"
        if bear_out:
            bull_prompt += f"\n\n空头上一轮论据，请反驳:\n{bear_out}"
        bull_out = run_step(f"Phase3-轮{i+1}-多头", bull_agent, bull_prompt) or ""

        bear_prompt = f"品种: {symbol}\n第{i+1}轮立论。基于研究数据构建做空论据:\n{research_data}"
        if bull_out:
            bear_prompt += f"\n\n多头上一轮论据，请反驳:\n{bull_out}"
        bear_out = run_step(f"Phase3-轮{i+1}-空头", bear_agent, bear_prompt) or ""

    # Phase 4: 裁判裁决
    judge_agent = create_judge_agent(cfg)
    judge_prompt = f"品种: {symbol}\n综合多空辩论裁决方向。\n多头论据:\n{bull_out}\n\n空头论据:\n{bear_out}\n\n研究数据:\n{research_data}"
    judge_out = run_step("Phase4-裁判裁决", judge_agent, judge_prompt)

    # Phase 5: 风控审核
    risk_agent = create_risk_agent(cfg)
    risk_prompt = f"品种: {symbol}\n审核以下交易参数:\n{judge_out}"
    risk_out = run_step("Phase5-风控审核", risk_agent, risk_prompt)

    # Phase 6: 报告输出
    report_agent = create_report_agent(cfg)
    report_prompt = f"品种: {symbol}\n汇总全流程产出。数据:\n{data_result}\n\n裁判:\n{judge_out}\n\n风控:\n{risk_out}"
    report_out = run_step("Phase6-报告输出", report_agent, report_prompt)

    # 总结
    print(f"\n{'='*50}")
    print("全流程完成!")
    print(f"最终报告: {report_out[:500] if report_out else '无'}")


if __name__ == "__main__":
    main()
