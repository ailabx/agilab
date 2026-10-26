"""AgiLab 入口——启动 FastAPI Web 服务。

运行: .venv/Scripts/python.exe main.py
访问: http://localhost:8765
"""

import os
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import uvicorn
from backend.server import app


def main():
    """启动 Web 服务"""
    uvicorn.run(app, host="127.0.0.1", port=8765)


if __name__ == "__main__":
    main()
