"""agents/demo.py - 轻量级单 Agent 演示，绕过团队调用，验证链路可用。

用法:
    python agents/demo.py AAPL
"""

import sys

from agents.members import get_registry


def main():
    symbol = (sys.argv[1] if len(sys.argv) > 1 else "AAPL").upper()
    registry = get_registry()
    agent = registry.create_agent("value_analyst")

    prompt = (
        f"请对 {symbol} 进行简要价值投资分析，包括护城河、财务质量、估值水平和投资结论。"
    )
    print(f"正在运行单 Agent 演示: {symbol}\n")
    response = agent.run(prompt, stream=False)
    print(response.content if hasattr(response, "content") else response)


if __name__ == "__main__":
    main()
