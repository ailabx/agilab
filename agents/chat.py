"""agents/chat.py - 简化版 intake

先保证 workflow 跑通，不调用 LLM。
"""

from __future__ import annotations

import re
from typing import Any


def _extract_symbol(text: str) -> str | None:
    # 优先匹配 6位数字.交易所 格式
    m = re.search(r"(\d{6})\.((SH|SZ|BJ))", text.upper())
    if m:
        return m.group(1) + "." + m.group(2)
    # 再匹配 6位连续数字
    m = re.search(r"(\d{6})", text)
    if m:
        code = m.group(1)
        suffix = "SH" if code.startswith("6") else "SZ"
        return f"{code}.{suffix}"
    return None


class IntakeAgent:
    """简化版 intake，纯规则提取股票代码。"""

    def run(self, user_input: str, **kwargs) -> dict[str, Any]:
        symbol = _extract_symbol(user_input)
        return {
            "symbol": symbol,
            "user_input": user_input,
            "reply": "",
        }
