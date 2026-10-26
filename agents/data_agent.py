"""数据采集 Agent——先从 tushare 拉真实数据，再交给 LLM 整理输出。

不再让 LLM 编数据，而是:
1. 调用 TushareSource 拉取 K线/基本面/新闻
2. 把原始数据塞进 prompt（K线只取最近 30 条避免 token 爆炸）
3. LLM 只负责整理成结构化 JSON
"""

import json

from agno.agent import Agent

from llm import get_model
from .profiles.base import AssetConfig
from .data_sources import get_data_source


def fetch_data(config: AssetConfig, symbol: str) -> str:
    """从数据源拉取真实数据，返回格式化字符串供 Agent 使用。

    K线只取最近 30 条，避免 JSON 太长被截断导致日期丢失。
    """
    source = get_data_source(config.data_source)
    data = source.get_all(symbol)

    # K线只保留最近 30 条
    kline = data.get("kline", [])
    if len(kline) > 30:
        data["kline"] = kline[-30:]

    return json.dumps(data, ensure_ascii=False, default=str)


def create_data_agent(config: AssetConfig) -> Agent:
    """创建数据采集 Agent。"""
    instructions = config.data_prompt + "\n\n数据将由系统自动拉取，你只需整理成 JSON 输出。"

    return Agent(
        model=get_model(),
        instructions=instructions,
        description=f"数据采集 ({config.data_source})",
        add_datetime_to_context=True,
    )
