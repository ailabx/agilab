"""AgiLab Agent 入口。

目前核心入口：
    from agents.router_agent import create_router_agent
"""

from .router_agent import create_router_agent

__all__ = ["create_router_agent"]
