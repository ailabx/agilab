"""测试 LLM 连通性与首 token 耗时，验证 config.toml 配置是否生效。

运行：.venv/Scripts/python.exe tests/test_llm.py
"""
import sys
import time
from pathlib import Path

# 将项目根目录加入 sys.path，使 tests 下脚本能直接导入 llm/config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agno.agent import Agent

from llm import get_model


def main():
    print("=" * 50)
    print("LLM 连通性测试")
    print("=" * 50)

    # 从 config 加载模型配置
    from config import config

    model = get_model()
    print(f"模型:   {config.MODEL}")
    print(f"接口:   {config.BASE_URL}")
    print(f"密钥:   {'已设置' if config.KEY else '未设置'}")

    # 创建 Agent
    agent = Agent(
        model=model,
        instructions="用简洁的中文回答问题。",
    )

    # 发送一条简单消息，stream 模式下逐 token 打印
    prompt = "用一句话介绍你自己。"
    print(f"\n提问: {prompt}")
    print("-" * 50)

    # stream=True 逐块输出，同时返回 metrics
    start = time.time()
    response = agent.run(prompt, stream=False)
    wall_time = time.time() - start

    # 打印回复内容
    print(f"回复: {response.content}")

    # 打印性能指标
    print("\n" + "=" * 50)
    print("性能指标")
    print("=" * 50)
    m = response.metrics
    if m:
        print(f"首 token 耗时: {m.time_to_first_token:.3f}s" if m.time_to_first_token else "首 token 耗时: N/A")
        print(f"总耗时(agent): {m.duration:.3f}s" if m.duration else f"总耗时(wall):  {wall_time:.3f}s")
        print(f"输入 tokens:  {m.input_tokens}")
        print(f"输出 tokens:  {m.output_tokens}")
        print(f"总 tokens:    {m.total_tokens}")
        print(f"费用:         ${m.cost:.6f}" if m.cost else "费用:         N/A")
    else:
        print(f"无 metrics 数据，wall 时间: {wall_time:.3f}s")

    # 判断是否成功：有 metrics 且有输出 token 说明模型真正响应了
    ok = bool(response and response.content and m and m.output_tokens > 0)
    print("\n测试通过 ✓" if ok else "\n测试失败 ✗（请检查 API 密钥与模型权限）")


if __name__ == "__main__":
    main()
