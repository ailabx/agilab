"""agents/agent_os.py - Agno Agent OS 入口

运行:
    python agents/agent_os.py

然后在 Agno Agent OS UI 中添加端点 localhost:7777。
"""

from __future__ import annotations

from agno.os import AgentOS

from agents.workflows import investment_workflow


def get_app():
    """创建并返回 AgentOS 的 FastAPI app。"""
    workflow = investment_workflow

    agent_os = AgentOS(
        name="A 股个股评估 AgentOS",
        description="基于 Agno 的多智能体 A 股个股评估系统",
        workflows=[workflow],
    )
    app = agent_os.get_app()

    # 自行添加 CORS middleware，兼容 https://os.agno.com 等在线 UI
    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )
    return app


app = get_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=7777)
