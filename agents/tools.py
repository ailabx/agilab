"""agents/tools.py - 个股评估共享工具

封装 yfinance 实时数据、DuckDuckGo 搜索，以及基于 yfinance 的简单量化指标。
保持可复用、可测试，避免在 Agent 内部直接依赖外部 API。
"""

from __future__ import annotations

from agno.tools.yfinance import YFinanceTools
from agno.tools.duckduckgo import DuckDuckGoTools


def create_finance_tools() -> YFinanceTools:
    """创建 yfinance 金融数据工具，启用常用能力。"""
    return YFinanceTools(
        enable_stock_price=True,
        enable_analyst_recommendations=True,
        enable_company_info=True,
        enable_company_news=True,
        enable_technical_indicators=True,
        enable_key_financial_ratios=True,
        enable_income_statements=True,
        enable_stock_fundamentals=True,
        enable_historical_prices=True,
    )


def create_search_tools() -> DuckDuckGoTools:
    """创建 DuckDuckGo 搜索工具，用于获取最新市场新闻。"""
    return DuckDuckGoTools()


def get_all_tools():
    """返回所有工具列表，方便在 Agent 中批量注册。"""
    return [create_finance_tools(), create_search_tools()]
