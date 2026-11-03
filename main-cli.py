"""AgiLab CLI 入口（基于 agno.os.AgentOS）。

实现通用 agno Agent 工厂 `agno_agent`，挂载 skills/ 下全部技能，
注册到 agno.os.AgentOS 并启动（Web 服务 / 流式控制台）。

运行:
    .venv/Scripts/python.exe main-cli.py            # 启动 AgentOS Web 服务（默认 127.0.0.1:7777）
    .venv/Scripts/python.exe main-cli.py --cli      # 流式控制台对话
    .venv/Scripts/python.exe main-cli.py --cli --query "分析 600519"
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.memory import MemoryManager
from agno.os import AgentOS
from agno.skills import Skills, LocalSkills
from agno.tools.coding import CodingTools
from agno.tools.reasoning import ReasoningTools
from agno.tools.file.generation import FileGenerationTools
from agno.run.agent import (
    RunContentEvent,
    ToolCallStartedEvent,
    ToolCallCompletedEvent,
    ToolCallErrorEvent,
)
from agno.tools.shell import ShellTools

from llm import get_model

ROOT = Path(__file__).resolve().parent
SKILLS_DIR = ROOT / "skills"
DB_FILE = str(ROOT / "data" / "agno.db")
OUTPUT_DIR = str(ROOT / "output")
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Windows 下 node / ffmpeg 真实路径（已在本机验证可用）
NODE_BIN = r"C:\Users\weijiabin\.workbuddy\binaries\node\versions\22.22.2-6"
FFMPEG_BIN = r"C:\Users\weijiabin\AppData\Local\Microsoft\WinGet\Links"

# 共享基础设施：模型、持久化 DB、记忆管理
db = SqliteDb(db_file=DB_FILE)
model = get_model()
memory_manager = MemoryManager(model=model, db=db)


def mount_skills(skills_dir: Path = SKILLS_DIR) -> Skills:
    """挂载 skills/ 下所有技能。

    使用 validate=False，以兼容 workbuddy-pptx 等非标准 frontmatter 的 SKILL.md；
    这样能真正把 skills/ 目录下的全部技能都加载进来。
    """
    return Skills(loaders=[LocalSkills(str(skills_dir), validate=False)])


def _env_with_toolchain() -> dict[str, str]:
    """返回包含 node/ffmpeg 的 PATH 环境变量。"""
    env = dict(os.environ)
    path_parts = [NODE_BIN, FFMPEG_BIN, env.get("PATH", "")]
    env["PATH"] = os.pathsep.join(path_parts)
    return env


def execute_command(command: str, cwd: str | None = None, timeout: int = 300) -> dict[str, Any]:
    """执行任意 shell 命令并返回结果。用于运行 npx hyperframes 等外部 CLI。

    Args:
        command: 要执行的命令字符串。
        cwd: 可选工作目录，默认为当前目录。
        timeout: 超时秒数，默认 300 秒。

    Returns:
        {"ok": bool, "returncode": int, "stdout": str, "stderr": str}
    """
    env = _env_with_toolchain()
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd or str(ROOT),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        return {
            "ok": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "returncode": -1, "stdout": "", "stderr": f"timeout after {timeout}s"}
    except Exception as e:
        return {"ok": False, "returncode": -1, "stdout": "", "stderr": str(e)}


def agno_agent(
    name: str = "AgiLab 通用助手",
    agent_id: str = "agilab-generic",
    *,
    agent_model=None,
    skills_dir: Path = SKILLS_DIR,
    instructions: str | None = None,
    tools=None,
    memory: bool = True,
    **kwargs,
) -> Agent:
    """通用的 agno Agent 工厂。

    - 自动挂载 skills/ 下全部技能
    - 默认携带 ReasoningTools + FileGenerationTools
    - 接入 MemoryManager 与 SqliteDb（会话/记忆持久化）

    返回的是标准 agno.agent.Agent，可直接注册进 agno.os.AgentOS。
    """
    skills = mount_skills(skills_dir)

    default_tools = [
        ReasoningTools(add_instructions=True),
        FileGenerationTools(
            enable_html_generation=True,
            enable_json_generation=True,
            enable_csv_generation=True,
            enable_txt_generation=True,
            save_files=True,
            output_directory=OUTPUT_DIR,
        ),


        execute_command,
    ]
    tool_list = list(tools) if tools is not None else default_tools

    default_instructions = (
        "你是 AgiLab 的通用助手。\n"
        f"你的默认工作目录是{os.getcwd()}\n"
        "npx的位置：D:/Program Files/nodejs/npx.cmd\n"
        "执行 HyperFrames 渲染时，优先使用 execute_command(command='npx --yes hyperframes@latest ...')。"
    )

    return Agent(
        name=name,
        id=agent_id,
        model=agent_model or model,
        add_history_to_context=True,
        num_history_runs=20,
        send_media_to_model=False,
        memory_manager=memory_manager if memory else None,
        enable_agentic_memory=memory,
        tools=tool_list,
        skills=skills,
        instructions=instructions or default_instructions,
        markdown=True,
        telemetry=False,
        add_datetime_to_context=True,
        stream_events=True,
        db=db,
        **kwargs,
    )


# --------------------------------------------------------------------------
# 构建默认通用 Agent，并注册到 agno.os.AgentOS
# --------------------------------------------------------------------------
generic_agent = agno_agent(name="AgiLab 通用分析师", agent_id="agilab-generic")

agent_os = AgentOS(agents=[generic_agent], db=db)
app = agent_os.get_app()


# --------------------------------------------------------------------------
# 流式控制台（可选模式，便于无 Web 环境下直接对话）
# --------------------------------------------------------------------------
def _fmt_tool_call(event) -> str | None:
    tool = event.tool
    if tool is None:
        return None
    name = tool.tool_name or "?"
    args = tool.tool_args or {}
    parts = []
    for k, v in args.items():
        sv = str(v)
        if len(sv) > 80:
            sv = sv[:77] + "..."
        parts.append(f"{k}={sv}")
    return f"{name}({', '.join(parts)})" if parts else name


def run_cli(query: str):
    t0 = time.perf_counter()
    t0_tool = None
    t0_gen = None
    ttft = None

    for event in generic_agent.run(query, stream=True):
        if isinstance(event, ToolCallStartedEvent):
            if t0_tool is None:
                t0_tool = time.perf_counter()
            s = _fmt_tool_call(event)
            if s:
                print(f"\n  [tool_call] {s}  ({time.perf_counter() - t0_tool:.1f}s)", flush=True)
            continue

        if isinstance(event, ToolCallCompletedEvent):
            s = _fmt_tool_call(event)
            if s:
                res_len = len(event.tool.result) if event.tool and event.tool.result else 0
                print(f"  [tool_done] {s} -> {res_len} chars ({time.perf_counter() - t0_tool:.1f}s)\n", flush=True)
            continue

        if isinstance(event, ToolCallErrorEvent):
            s = _fmt_tool_call(event)
            err = (event.error or "unknown error")[:80]
            print(f"  [tool_error] {s} -> {err}\n", flush=True)
            continue

        if isinstance(event, RunContentEvent):
            chunk = event.reasoning_content if (not event.content and event.reasoning_content) else event.content
            if chunk:
                if t0_gen is None:
                    t0_gen = time.perf_counter()
                    ttft = t0_gen - t0_tool if t0_tool else None
                print(chunk, end="", flush=True)

    elapsed = time.perf_counter() - t0
    tool_elapsed = (t0_gen - t0_tool) if (t0_tool and t0_gen) else None
    gen_elapsed = (time.perf_counter() - t0_gen) if t0_gen else None

    parts = []
    if tool_elapsed:
        parts.append(f"模型决策+工具执行 {tool_elapsed:.1f}s")
    if ttft:
        parts.append(f"生成首 token {ttft:.3f}s")
    if gen_elapsed:
        parts.append(f"生成总耗时 {gen_elapsed:.3f}s")
    parts.append(f"总耗时 {elapsed:.3f}s")
    print(f"\n\n─── {' · '.join(parts)} ───")


def main():
    parser = argparse.ArgumentParser(description="AgiLab CLI (agno.os.AgentOS)")
    parser.add_argument("--cli", action="store_true", help="流式控制台模式（不启动 Web 服务）")
    parser.add_argument("--query", help="CLI 模式下的一次性提问")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7777)
    args = parser.parse_args()

    # 已挂载技能名（用于日志）
    skill_names = generic_agent.skills.get_skill_names() if generic_agent.skills else []

    if args.cli:
        if args.query:
            run_cli(args.query)
        else:
            print("AgiLab 通用分析师 · 流式控制台（输入 exit/quit 退出）")
            print(f"已挂载技能: {', '.join(skill_names) or '无'}\n")
            while True:
                try:
                    q = input("你> ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if not q:
                    continue
                if q.lower() in {"exit", "quit", "q"}:
                    break
                run_cli(q)
        return

    print(f"AgiLab AgentOS 启动于 http://{args.host}:{args.port}")
    print(f"已注册 Agent: {generic_agent.name} (id={generic_agent.id})")
    print(f"已挂载技能: {', '.join(skill_names) or '无'}")
    # serve() 既接受 import 字符串也接受 FastAPI 对象；
    # 这里直接传 app 对象，规避 main-cli.py 文件名含连字符无法作为模块名的问题。
    agent_os.serve(app=app, host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
