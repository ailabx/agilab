"""AgiLab Web 服务端——FastAPI + Jinja2 模板渲染，后端路由驱动。

启动:
    .venv/Scripts/python.exe main.py

访问:
    http://localhost:8765
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from config import config
from agents import get_config
from agents.data_sources import cache_stats, clear_cache

app = FastAPI(title="AGI实验室", version="0.1")

# 模板目录
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# 常量
MARKETS = ["stock", "etf", "futures", "convertible"]
_SENSITIVE_KEYS = ["KEY", "TS_TOKEN"]

# Agent 角色列表
AGENT_LIST = [
    {"name": "数据采集", "icon": "bi bi-download", "desc": "从 tushare 拉取 K线/基本面/新闻，LLM 整理成 JSON。"},
    {"name": "分析师团队", "icon": "bi bi-bar-chart", "desc": "技术面 / 基本面 / 情绪面（并行），中立不下结论。"},
    {"name": "多头分析员", "icon": "bi bi-arrow-up-circle", "desc": "基于研究数据构建做多论据，反驳空头。"},
    {"name": "空头分析员", "icon": "bi bi-arrow-down-circle", "desc": "基于研究数据构建做空论据，反驳多头。"},
    {"name": "裁判", "icon": "bi bi-balance", "desc": "六维评分，输出方向+交易参数。"},
    {"name": "风控官", "icon": "bi bi-shield-check", "desc": "审核交易参数，green/yellow/red。"},
]

# 文件树
FILE_TREE = [
    {"text": "agents/", "type": "dir", "indent": 0},
    {"text": "__init__.py          # 入口", "type": "file", "indent": 16},
    {"text": "base.py              # Agent 工厂", "type": "file", "indent": 16},
    {"text": "data_agent.py        # 数据采集 (tushare)", "type": "file", "indent": 16},
    {"text": "analyst_agent.py     # 分析师", "type": "file", "indent": 16},
    {"text": "debate_agent.py      # 多头/空头", "type": "file", "indent": 16},
    {"text": "judge_agent.py       # 裁判", "type": "file", "indent": 16},
    {"text": "risk_agent.py        # 风控", "type": "file", "indent": 16},
    {"text": "report_agent.py     # 报告", "type": "file", "indent": 16},
    {"text": "workflow.py          # Workflow 组装", "type": "file", "indent": 16},
    {"text": "data_sources/", "type": "dir", "indent": 36},
    {"text": "tushare_source.py    # Tushare 拉取", "type": "file", "indent": 52},
    {"text": "cache.py             # SQLite 缓存", "type": "file", "indent": 52},
    {"text": "profiles/", "type": "dir", "indent": 36},
    {"text": "stock/etf/futures/convertible.py", "type": "file", "indent": 52},
    {"text": "config.py            # 配置管理", "type": "dir", "indent": 0},
    {"text": "llm.py               # LLM 模型", "type": "dir", "indent": 0},
    {"text": "main.py              # 入口", "type": "dir", "indent": 0},
]


def get_market_list():
    """获取市场配置列表"""
    result = []
    for m in MARKETS:
        cfg = get_config(m)
        result.append({
            "market": m, "data_source": cfg.data_source,
            "analysts": cfg.analysts, "debate_rounds": cfg.debate_rounds,
        })
    return result


def get_config_safe():
    """获取配置（敏感字段脱敏）"""
    all_cfg = config.get_all()
    for k in _SENSITIVE_KEYS:
        if all_cfg.get(k):
            all_cfg[k] = all_cfg[k][:4] + "***"
    return all_cfg


# ==================== 页面路由 ====================

@app.get("/", response_class=HTMLResponse)
async def overview(request: Request):
    return templates.TemplateResponse(request, "overview.html", {
        "active_page": "overview",
        "markets": get_market_list(), "cache": cache_stats(),
    })


# ==================== 标的分析（核心主流程）====================

import threading
import uuid
import time

# 全局任务存储——生产环境应换 Redis/DB
_tasks = {}


@app.get("/analyze", response_class=HTMLResponse)
async def analyze_form(request: Request):
    """分析输入页"""
    return templates.TemplateResponse(request, "analyze.html", {
        "active_page": "analyze",
    })


@app.post("/analyze")
async def analyze_start(market: str = Form("stock"), symbol: str = Form(...)):
    """启动后台分析，返回 JSON {task_id} 供前端轮询"""
    task_id = str(uuid.uuid4())[:8]
    _tasks[task_id] = {
        "status": "running", "symbol": symbol, "market": market,
        "phases": [], "signal": {}, "duration": 0, "error": None,
        "created": time.strftime("%Y%m%d%H%M%S"),
    }

    # 后台线程跑
    t = threading.Thread(target=_run_analysis, args=(task_id, market, symbol), daemon=True)
    t.start()

    return {"task_id": task_id, "symbol": symbol, "status": "running"}


def _run_analysis(task_id: str, market: str, symbol: str):
    """后台执行 6 步分析（在工作线程中跑）"""
    import json
    import time

    task = _tasks[task_id]
    phases = []
    signal = {}
    start = time.time()

    def _cancelled():
        """检查任务是否被取消"""
        return task.get("status") == "cancelled"

    try:
        from agents import get_config
        from agents.data_agent import create_data_agent, fetch_data
        from agents.analyst_agent import create_analysts
        from agents.debate_agent import create_bull_agent, create_bear_agent
        from agents.judge_agent import create_judge_agent
        from agents.risk_agent import create_risk_agent
        from agents.report_agent import create_report_agent

        cfg = get_config(market)

        # Phase 1: 数据采集
        if _cancelled(): return
        phases.append({"name": "Phase1-数据采集", "icon": "download", "status": "运行中", "status_color": "primary", "output": ""})
        task["phases"] = phases
        data_str = fetch_data(cfg, symbol)
        data_agent = create_data_agent(cfg)
        r1 = data_agent.run(f"分析品种: {symbol}\n数据:\n{data_str}")
        data_result = str(r1.content) if hasattr(r1, "content") and r1.content else data_str
        phases[-1].update({"status": "✓ 通过", "status_color": "success", "output": data_result})
        task["phases"] = list(phases)

        # Phase 2: 分析师团队
        if _cancelled(): return
        analyst_agents = create_analysts(cfg)
        research_data = {}
        for role, agent in zip(cfg.analysts, analyst_agents):
            if _cancelled(): return
            phases.append({"name": f"Phase2-{role}", "icon": "bar-chart", "status": "运行中", "status_color": "primary", "output": ""})
            task["phases"] = list(phases)
            r2 = agent.run(f"品种: {symbol}\n角色: {role}\n数据: {data_result[:2000]}")
            out2 = str(r2.content) if hasattr(r2, "content") and r2.content else ""
            research_data[role] = out2
            phases[-1].update({"status": "✓ 通过", "status_color": "success", "output": out2})
            task["phases"] = list(phases)

        # Phase 3: 多空辩论
        if _cancelled(): return
        bull_agent = create_bull_agent(cfg)
        bear_agent = create_bear_agent(cfg)
        research_str = json.dumps(research_data, ensure_ascii=False, default=str)[:3000]
        bull_out = ""
        bear_out = ""

        for i in range(cfg.debate_rounds):
            if _cancelled(): return
            phases.append({"name": f"Phase3-轮{i+1}-多头", "icon": "arrow-up-circle", "status": "运行中", "status_color": "primary", "output": ""})
            task["phases"] = list(phases)
            bp = f"品种: {symbol}\n第{i+1}轮立论。\n{research_str}"
            if bear_out:
                bp += f"\n空头论据:\n{bear_out[:1000]}"
            rb = bull_agent.run(bp)
            bull_out = str(rb.content) if hasattr(rb, "content") and rb.content else ""
            phases[-1].update({"status": "✓ 通过", "status_color": "success", "output": bull_out})
            task["phases"] = list(phases)

            if _cancelled(): return
            phases.append({"name": f"Phase3-轮{i+1}-空头", "icon": "arrow-down-circle", "status": "运行中", "status_color": "primary", "output": ""})
            task["phases"] = list(phases)
            sp = f"品种: {symbol}\n第{i+1}轮立论。\n{research_str}"
            if bull_out:
                sp += f"\n多头论据:\n{bull_out[:1000]}"
            rs = bear_agent.run(sp)
            bear_out = str(rs.content) if hasattr(rs, "content") and rs.content else ""
            phases[-1].update({"status": "✓ 通过", "status_color": "success", "output": bear_out})
            task["phases"] = list(phases)

        # Phase 4: 裁判
        if _cancelled(): return
        phases.append({"name": "Phase4-裁判裁决", "icon": "balance", "status": "运行中", "status_color": "primary", "output": ""})
        task["phases"] = list(phases)
        judge = create_judge_agent(cfg)
        rj = judge.run(f"品种: {symbol}\n多头:\n{bull_out[:2000]}\n空头:\n{bear_out[:2000]}")
        judge_out = str(rj.content) if hasattr(rj, "content") and rj.content else ""
        phases[-1].update({"status": "✓ 通过", "status_color": "success", "output": judge_out})
        task["phases"] = list(phases)

        # Phase 5: 风控
        if _cancelled(): return
        phases.append({"name": "Phase5-风控审核", "icon": "shield-check", "status": "运行中", "status_color": "primary", "output": ""})
        task["phases"] = list(phases)
        risk = create_risk_agent(cfg)
        rr = risk.run(f"品种: {symbol}\n审核:\n{judge_out[:2000]}")
        risk_out = str(rr.content) if hasattr(rr, "content") and rr.content else ""
        phases[-1].update({"status": "✓ 通过", "status_color": "success", "output": risk_out})
        task["phases"] = list(phases)

        # Phase 6: 报告
        phases.append({"name": "Phase6-报告输出", "icon": "file-text", "status": "运行中", "status_color": "primary", "output": ""})
        task["phases"] = list(phases)
        report = create_report_agent(cfg)
        rrep = report.run(f"品种: {symbol}\n裁判:\n{judge_out[:1000]}\n风控:\n{risk_out[:500]}")
        report_out = str(rrep.content) if hasattr(rrep, "content") and rrep.content else ""
        phases[-1].update({"status": "✓ 通过", "status_color": "success", "output": report_out})

        signal = _extract_signal(judge_out, risk_out)

    except Exception as e:
        phases.append({"name": "错误", "icon": "x-circle", "status": "✗ 失败", "status_color": "danger", "output": str(e)})
        task["error"] = str(e)

    task["status"] = "done"
    task["phases"] = phases
    task["signal"] = signal
    task["duration"] = round(time.time() - start, 1)


@app.get("/api/analyze/{task_id}")
async def analyze_status(task_id: str):
    """API: 查询任务状态（供前端 JS 轮询）"""
    task = _tasks.get(task_id)
    if not task:
        return {"status": "not_found"}
    return {
        "status": task["status"],
        "phases": task["phases"],
        "signal": task["signal"],
        "duration": task["duration"],
    }


@app.get("/api/tasks")
async def list_tasks():
    """列出所有任务（最新的在前）"""
    result = []
    for tid, t in sorted(_tasks.items(), key=lambda x: x[1].get("created", ""), reverse=True):
        result.append({
            "task_id": tid,
            "symbol": t["symbol"],
            "market": t["market"],
            "status": t["status"],
            "duration": t["duration"],
        })
    return result


@app.post("/api/tasks/{task_id}/cancel")
async def cancel_task(task_id: str):
    """取消任务（标记为 cancelled，后台线程检测后停止）"""
    task = _tasks.get(task_id)
    if not task:
        return {"error": "任务不存在"}
    if task["status"] != "running":
        return {"error": "任务不在运行中"}
    task["status"] = "cancelled"
    # 给运行中的 Phase 标记取消
    for p in task.get("phases", []):
        if p["status"] == "运行中":
            p["status"] = "已取消"
            p["status_color"] = "secondary"
    return {"status": "cancelled"}


@app.delete("/api/tasks/{task_id}")
async def delete_task(task_id: str):
    """删除任务"""
    if task_id not in _tasks:
        return {"error": "任务不存在"}
    del _tasks[task_id]
    return {"deleted": task_id}


def _extract_signal(judge_out: str, risk_out: str) -> dict:
    """从裁判和风控输出中提取交易信号"""
    import re

    def find(pattern, text):
        m = re.search(pattern, text, re.IGNORECASE)
        return m.group(1).strip() if m else ""

    direction = find(r'"direction"\s*:\s*"([^"]*)"', judge_out) or find(r'方向[：:]\s*(做多|做空|观望)', judge_out)
    confidence = find(r'"confidence"\s*:\s*([\d.]+)', judge_out)
    entry = find(r'"entry"\s*:\s*([\d.]+)', judge_out)
    stop_loss = find(r'"stop_loss"\s*:\s*([\d.]+)', judge_out)
    target = find(r'"target"\s*:\s*([\d.]+)', judge_out)
    position = find(r'"position_pct"\s*:\s*([\d.]+)', judge_out)
    risk = find(r'"signal"\s*:\s*"(green|yellow|red)"', risk_out) or find(r'(green|yellow|red)', risk_out)

    return {
        "direction": direction or "观望",
        "confidence": float(confidence) if confidence else 0,
        "entry": float(entry) if entry else 0,
        "stop_loss": float(stop_loss) if stop_loss else 0,
        "target": float(target) if target else 0,
        "position_pct": float(position) if position else 0,
        "risk": risk,
    }


@app.get("/workflow", response_class=HTMLResponse)
async def workflow(request: Request):
    return templates.TemplateResponse(request, "workflow.html", {
        "active_page": "workflow",
    })


@app.get("/agents", response_class=HTMLResponse)
async def agents(request: Request):
    return templates.TemplateResponse(request, "agents.html", {
        "active_page": "agents", "agents": AGENT_LIST,
    })


@app.get("/files", response_class=HTMLResponse)
async def files(request: Request):
    return templates.TemplateResponse(request, "files.html", {
        "active_page": "files", "file_tree": FILE_TREE,
    })


@app.get("/cache", response_class=HTMLResponse)
async def cache_page(request: Request):
    return templates.TemplateResponse(request, "cache.html", {
        "active_page": "cache", "cache": cache_stats(),
    })


@app.post("/cache/clear")
async def cache_clear_action():
    clear_cache()
    return RedirectResponse(url="/cache", status_code=303)


# ==================== 设置 ====================

@app.get("/settings", response_class=HTMLResponse)
async def settings(request: Request):
    return templates.TemplateResponse(request, "settings.html", {
        "active_page": "settings",
        "cfg": get_config_safe(), "config_path": str(config._config_file),
    })


@app.post("/settings")
async def save_settings(
    request: Request,
    MODEL: str = Form(""), BASE_URL: str = Form(""),
    KEY: str = Form(""), TS_TOKEN: str = Form(""),
):
    updates = {}
    for field, val in [("MODEL", MODEL), ("BASE_URL", BASE_URL),
                      ("KEY", KEY), ("TS_TOKEN", TS_TOKEN)]:
        if val and "***" not in val:
            updates[field] = val
    saved_count = 0
    saved_fields = []
    if updates:
        config.update(**updates)
        saved_count = len(updates)
        saved_fields = list(updates.keys())

    return templates.TemplateResponse(request, "settings.html", {
        "active_page": "settings",
        "cfg": get_config_safe(), "config_path": str(config._config_file),
        "saved": saved_count, "saved_fields": saved_fields,
    })


# ==================== API (供前端 JS 调用) ====================

@app.get("/api/markets")
async def api_markets():
    return get_market_list()


@app.get("/api/config")
async def api_config():
    return get_config_safe()


@app.get("/api/cache/stats")
async def api_cache_stats():
    return cache_stats()


@app.post("/api/cache/clear")
async def api_cache_clear(symbol: str = ""):
    count = clear_cache(symbol if symbol else None)
    return {"cleared": count}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8765)
