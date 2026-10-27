# AgiLab

AgiLab，AGI实验室。

## 架构

```
.
├── backend/
│   ├── server.py              # FastAPI + SSE 流式问答服务
│   └── templates/
│       └── chat.html          # 问答界面
├── agents/
│   └── router_agent.py        # 统一入口 Agent，调用 Skill 并生成报告
├── skills/
│   └── financial-research/    # 标准化 Skill：数据采集 → 指标计算 → 结构化输出
│       ├── SKILL.md
│       ├── references/
│       │   └── analysis-framework.md
│       └── scripts/
│           ├── main.py        # 编排入口
│           ├── market_data.py # 行情数据
│           ├── technical.py   # 技术指标
│           ├── fundamental.py # 基本面数据
│           └── sentiment.py   # 情绪指标
└── config.py                  # 配置（TS_TOKEN 等）
```

## 设计原则

- **Skill 只做数据**：`financial-research` Skill 只负责数据采集、指标计算、结构化输出，不调 LLM。
- **Agent 负责推理**：`router_agent.py` 调用 Skill 获取数据包，由 LLM 生成中文分析报告。
- **标准化 CLI**：所有脚本独立可运行，输出 stdout JSON。

## 启动

```bash
.venv/Scripts/python.exe backend/server.py
```

访问 http://localhost:8765

## 支持的品类

- `stock`：股票，如 `600519`
- `etf`：ETF，如 `510300`
- `futures`：期货，如 `RB`
- `convertible`：可转债，如 `113052`
- `fund`：场外基金，如 `000001`

## 测试 Skill

```bash
.venv/Scripts/python skills/financial-research/scripts/main.py 600519 --market stock --days 30
```
