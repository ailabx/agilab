"""AgiLab Web 服务端——FastAPI + Jinja2 模板渲染。

启动:
    .venv/Scripts/python.exe backend/server.py

访问:
    http://localhost:8765
"""

import json
import os
import queue
import threading
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from config import config
from llm import get_model
from agents.router_agent import create_router_agent

app = FastAPI(title="AGI实验室", version="0.2")

# 模板目录
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# ==================== 页面 ====================

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse(request, "chat.html", {"active_page": "chat"})


@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    return templates.TemplateResponse(request, "chat.html", {"active_page": "chat"})


def _count_all(api, func, **kwargs) -> int:
    """分页统计 Tushare 接口返回的总记录数。"""
    total = 0
    offset = 0
    page_size = 8000
    while True:
        try:
            df = func(offset=offset, limit=page_size, **kwargs)
            if df is None or df.empty:
                break
            total += len(df)
            if len(df) < page_size:
                break
            offset += page_size
        except Exception:
            break
    return total


def _load_dashboard_counts() -> dict:
    """从 Tushare 拉取各品类标的数量。"""
    import tushare as ts
    from config import config as cfg

    api = ts.pro_api(cfg.TS_TOKEN)
    cache = {"astock": 0, "etf": 0, "future": 0, "bond": 0, "fund": 0, "total": 0}

    try:
        cache["astock"] = _count_all(api, api.stock_basic, exchange="", list_status="L", fields="ts_code")
    except Exception:
        pass

    try:
        cache["etf"] = _count_all(api, api.fund_basic, market="E", fields="ts_code")
    except Exception:
        pass

    try:
        cache["fund"] = _count_all(api, api.fund_basic, market="O", fields="ts_code")
    except Exception:
        pass

    try:
        cache["future"] = _count_all(api, api.fut_basic, fields="ts_code")
    except Exception:
        pass

    try:
        cache["bond"] = _count_all(api, api.cb_basic, fields="ts_code")
    except Exception:
        pass

    cache["total"] = cache["astock"] + cache["etf"] + cache["future"] + cache["bond"] + cache["fund"]
    return cache


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    cache = _load_dashboard_counts()
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "active_page": "dashboard",
            "cache": cache,
            "stats": {"completed_tasks": 0},
            "cfg": {"MODEL": "default"},
            "recent_analyses": [],
        },
    )


@app.get("/single-analysis", response_class=HTMLResponse)
async def single_analysis_page(request: Request):
    return templates.TemplateResponse(request, "single_analysis.html", {"active_page": "single-analysis"})


@app.get("/queue", response_class=HTMLResponse)
async def queue_page(request: Request):
    return templates.TemplateResponse(request, "queue.html", {"active_page": "queue"})


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    return templates.TemplateResponse(
        request,
        "settings.html",
        {
            "active_page": "settings",
            "saved": False,
            "saved_fields": [],
            "config_path": str(Path(__file__).resolve().parent.parent / "config" / "config.py"),
            "cfg": {"MODEL": "", "BASE_URL": "", "KEY": "", "TS_TOKEN": ""},
        },
    )


# ==================== 流式问答接口 ====================

from agno.run.agent import (
    RunContentEvent,
    ToolCallStartedEvent,
    ToolCallCompletedEvent,
    ReasoningContentDeltaEvent,
    ReasoningStepEvent,
    RunCompletedEvent,
)


def _event_to_sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


def _run_agent_stream(message: str, session_id: str):
    """在后台线程运行 Agent 并 yield SSE 事件。"""
    q: queue.Queue = queue.Queue()

    def _target():
        try:
            agent = create_router_agent()
            for event in agent.run(message, stream=True, stream_events=True, session_id=session_id):
                q.put(event)
            q.put(None)
        except Exception as e:
            q.put(e)

    threading.Thread(target=_target, daemon=True).start()

    while True:
        item = q.get()
        if item is None:
            break
        if isinstance(item, Exception):
            yield _event_to_sse({"type": "error", "content": str(item)})
            break

        event_type = type(item).__name__
        payload: dict = {"type": event_type}

        if isinstance(item, RunContentEvent):
            payload["content"] = item.content
            payload["reasoning_content"] = item.reasoning_content
        elif isinstance(item, ToolCallStartedEvent):
            tool = item.tool
            payload["tool"] = tool.tool_name if tool else None
            payload["input"] = tool.tool_args if tool else None
        elif isinstance(item, ToolCallCompletedEvent):
            tool = item.tool
            payload["tool"] = tool.tool_name if tool else None
            payload["output"] = item.content
        elif isinstance(item, (ReasoningContentDeltaEvent, ReasoningStepEvent)):
            payload["content"] = getattr(item, "reasoning_content", None) or getattr(item, "content", None)
        elif isinstance(item, RunCompletedEvent):
            payload["content"] = item.content
        else:
            payload["content"] = getattr(item, "content", None)

        yield _event_to_sse({**payload, "event": getattr(item, "event", event_type)})


@app.get("/chat/stream")
async def chat_stream(message: str, session_id: str = "default"):
    """SSE 流式接口。"""
    return StreamingResponse(
        _run_agent_stream(message, session_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8765)))
    args = parser.parse_args()
    uvicorn.run(app, host="127.0.0.1", port=args.port)
