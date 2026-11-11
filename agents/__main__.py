"""agents/__main__.py - 命令行入口

用法:
    python -m agents <symbol> [task]
    python -m agents AAPL
    python -m agents 600519 "分析茅台的长期投资价值"
"""

import argparse
import sys

from agents.core import StockEvaluationTeam


def main():
    parser = argparse.ArgumentParser(description="多智能体个股评估")
    parser.add_argument("symbol", help="股票代码，例如 AAPL、600519.SH、00700.HK")
    parser.add_argument("--task", "-t", default=None, help="自定义分析任务描述（可选）")

    args = parser.parse_args()

    team = StockEvaluationTeam()
    try:
        team.evaluate_and_print(args.symbol, task=args.task)
    except Exception as e:
        print(f"运行失败: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
