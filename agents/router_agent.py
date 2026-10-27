"""统一入口 Agent。

当用户要求分析金融标的时，调用 financial-research Skill 获取数据包，
然后基于数据包生成结构化中文报告。
"""

import json
import subprocess
import sys
from pathlib import Path

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.skills import LocalSkills, Skills
from agno.tools import Function

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from llm import get_model


_AGENT_STORAGE = SqliteDb(db_file=str(ROOT / "data" / "agent_storage.db"))

# 品类关键词映射
_MARKET_ALIASES = {
    "stock": ["股票", "A股", "个股"],
    "etf": ["ETF", "etf", "基金（场内）"],
    "futures": ["期货", "螺纹钢", "沪铜", "铁矿石", "原油", "沪金", "焦炭", "豆粕"],
    "convertible": ["可转债", "转债"],
    "fund": ["基金", "场外基金", "公募基金"],
}


def detect_market(message: str) -> str:
    """从用户消息中识别品类。"""
    message = message.lower()
    for market, aliases in _MARKET_ALIASES.items():
        for alias in aliases:
            if alias.lower() in message:
                return market
    return "stock"


def run_financial_research(symbol: str, market: str = "stock", days: int = 120) -> dict:
    """调用 financial-research Skill 获取数据包。"""
    script = ROOT / "skills" / "financial-research" / "scripts" / "main.py"
    cmd = [sys.executable, str(script), symbol, "--market", market, "--days", str(days)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603
    if result.returncode != 0:
        raise RuntimeError(f"financial-research 失败: {result.stderr}")
    return json.loads(result.stdout)


def _extract_symbol(message: str) -> str | None:
    """简单提取标的代码/名称。"""
    import re
    # 优先匹配 6 位数字代码
    m = re.search(r"\b\d{6}\b", message)
    if m:
        return m.group(0)
    # 否则取第一个非停用词
    stopwords = {"分析", "看看", "研究", "一下", "怎么", "如何", "样", "的", "是", "a股", "股票", "期货", "基金", "etf", "转债", "可转债"}
    words = re.findall(r"[\u4e00-\u9fa5A-Za-z0-9]+", message)
    for w in words:
        if w and w not in stopwords:
            return w
    return None


def create_router_agent() -> Agent:
    """创建统一入口 Agent，使用 agno 原生 Skills 机制加载本地 Skill。"""
    skills_dir = ROOT / "skills"
    all_skills = Skills(loaders=[LocalSkills(str(skills_dir))])
    skill_tools = all_skills.get_tools()

    # 为了让 Agent 能调用 run_financial_research，把它注册为 tool
    def financial_research_tool(symbol: str, market: str = "stock", days: int = 120) -> str:
        """调用 financial-research Skill 获取标的结构化数据包。

        Args:
            symbol: 标的代码或名称，如 600519、RB、510300
            market: 品类 stock/etf/futures/convertible/fund
            days: 历史数据天数，默认 120
        """
        try:
            data = run_financial_research(symbol, market, days)
            return json.dumps(data, ensure_ascii=False, default=str)
        except Exception as e:
            return f"调用失败: {e}"

    financial_tool = Function(
        name="financial_research",
        description="调用 financial-research Skill 获取标的结构化数据包，支持股票/ETF/期货/可转债/场外基金。",
        entrypoint=financial_research_tool,
    )
    skill_tools.append(financial_tool)

    return Agent(
        name="AGI实验室总分析师",
        id="unified-analyst",
        model=get_model(),
        add_history_to_context=True,
        num_history_runs=10,
        add_datetime_to_context=True,
        db=_AGENT_STORAGE,
        skills=all_skills,
        tools=skill_tools,
        instructions="""你是 AGI 实验室的统一金融分析师，可以分析股票、ETF、期货、可转债、场外基金五类资产。

## 分析流程

1. 识别用户提到的标的和品类。
2. 调用 `financial_research_tool(symbol, market, days)` 获取结构化数据包。
3. 基于数据包，按以下结构输出中文分析报告：
   - 核心结论
   - 技术面
   - 基本面
   - 情绪面
   - 多空因素
   - 策略建议与风险提示

## 品种识别

- 股票：600519、贵州茅台
- ETF：510300
- 期货：RB、螺纹钢、CU、铁矿石
- 可转债：113052
- 场外基金：000001

## 输出要求

1. 先给核心结论。
2. 必须调用 `financial_research_tool` 获取数据，所有数字必须来自工具返回，不得编造、 extrapolate 或改写字段含义。
3. 若工具返回字段为空或缺失，明确说明"数据缺失"，不要脑补。
4. 涉及多空判断时给出依据和风险点。
5. 数字保留适当精度。
6. `sentiment.period_high` / `sentiment.period_low` 是当前数据窗口（近 `days` 个交易日）内的最高/最低价，不要改写成其他周期。
""",
        markdown=True,
        telemetry=False,
    )
