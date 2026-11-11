"""agents/tools_tushare.py - 基于 tushare 的个股数据工具

封装为 Agno 可识别的 Toolkit，供专家 Agent 调用。
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import tushare as ts
from agno.tools import Toolkit

from config import config


class TushareStockTools(Toolkit):
    """tushare 个股数据工具集，支持日线、基本面、技术指标等查询。"""

    def __init__(self):
        super().__init__(name="tushare_stock_tools", auto_register=False)
        self._token = config.TS_TOKEN
        if not self._token:
            raise ValueError("未配置 Tushare token，请在 ~/.agilab/config.toml 中设置 TS_TOKEN")
        ts.set_token(self._token)
        self.pro = ts.pro_api()
        self._register_all_tools()

    def _to_ts_code(self, symbol: str) -> str:
        """统一转换为 tushare 的 ts_code 格式。"""
        symbol = symbol.strip().upper()
        if "." in symbol:
            return symbol
        # 简单规则：6 开头为上海，其他深圳
        if symbol.startswith("6"):
            return f"{symbol}.SH"
        return f"{symbol}.SZ"

    def _trade_date(self, days_ago: int = 0) -> str:
        """返回 N 天前的交易日（简化按自然日）。"""
        d = datetime.now() - timedelta(days=days_ago)
        return d.strftime("%Y%m%d")

    def get_daily(
        self, ts_code: str, start_date: str | None = None, end_date: str | None = None
    ) -> str:
        """获取个股日线行情数据。

        Args:
            ts_code: tushare 格式代码，如 600519.SH
            start_date: 开始日期 YYYYMMDD，默认 90 天前
            end_date: 结束日期 YYYYMMDD，默认今天
        """
        code = self._to_ts_code(ts_code)
        end_date = end_date or self._trade_date(0)
        start_date = start_date or self._trade_date(90)
        df = self.pro.daily(ts_code=code, start_date=start_date, end_date=end_date)
        if df is None or df.empty:
            return "未获取到数据"
        return df.to_json(orient="records", force_ascii=False)

    def get_daily_basic(self, ts_code: str, trade_date: str | None = None) -> str:
        """获取个股每日指标（PE、PB、换手率、市值等）。

        Args:
            ts_code: tushare 格式代码
            trade_date: 交易日 YYYYMMDD，默认最近一天
        """
        code = self._to_ts_code(ts_code)
        if not trade_date:
            trade_date = self._trade_date(1)
        df = self.pro.daily_basic(ts_code=code, trade_date=trade_date)
        if df is None or df.empty:
            return "未获取到数据"
        return df.to_json(orient="records", force_ascii=False)

    def get_fina_indicator(self, ts_code: str) -> str:
        """获取个股财务指标（ROE、毛利率、净利率、负债率等）。"""
        code = self._to_ts_code(ts_code)
        df = self.pro.fina_indicator(ts_code=code)
        if df is None or df.empty:
            return "未获取到数据"
        return df.head(8).to_json(orient="records", force_ascii=False)

    def get_income(self, ts_code: str) -> str:
        """获取利润表（最近 8 个季度）。"""
        code = self._to_ts_code(ts_code)
        df = self.pro.income(ts_code=code)
        if df is None or df.empty:
            return "未获取到数据"
        return df.head(8).to_json(orient="records", force_ascii=False)

    def get_name(self, ts_code: str) -> str:
        """获取股票名称及基础信息。"""
        code = self._to_ts_code(ts_code)
        df = self.pro.stock_basic(ts_code=code)
        if df is None or df.empty:
            return "未获取到数据"
        return df.to_json(orient="records", force_ascii=False)

    def get_industry_peers(self, ts_code: str) -> str:
        """获取同行业可比公司（按申万行业）。"""
        code = self._to_ts_code(ts_code)
        basic = self.pro.stock_basic(ts_code=code)
        if basic is None or basic.empty:
            return "未获取到基础信息"
        industry = basic.iloc[0].get("industry")
        if not industry:
            return "该股票无行业信息"
        df = self.pro.stock_basic(industry=industry)
        if df is None or df.empty:
            return "未获取到同行业公司"
        return df[["ts_code", "name", "industry"]].head(20).to_json(
            orient="records", force_ascii=False
        )

    def _register_all_tools(self) -> None:
        """手动注册所有公开工具方法。"""
        tool_methods = [
            self.get_daily,
            self.get_daily_basic,
            self.get_fina_indicator,
            self.get_income,
            self.get_name,
            self.get_industry_peers,
        ]
        for method in tool_methods:
            self.register(method)


# 兼容旧接口，让 members.py 可以继续使用
def get_all_tools():
    return [TushareStockTools()]

