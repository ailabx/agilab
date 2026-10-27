---
name: financial-research
description: 对股票、ETF、期货、可转债、场外基金进行标准化金融研究，输出结构化数据与分析框架。当用户说"分析、看看、研究、诊断、怎么样"等涉及金融标的时使用。
---

# 金融研究

> 本 Skill 是一套金融研究工具与流程规范，把"数据采集 → 指标计算 → 基本面加工 → 情绪量化 → 结构化输出"固化为确定性脚本，LLM 只负责基于输出数据写洞察与报告，不编造数字。

## 职责边界

| 需求 | 交给 |
|------|------|
| 金融标的完整研究 | **本 Skill** |
| 只看行情/K 线 | market_data 脚本 |
| 只算技术指标 | technical 脚本 |
| 只写通用营销文案 | copywriting |
| 只做可视化图表 | chart-visualization |

## 输入

```bash
python scripts/main.py <symbol> --market <stock|etf|futures|convertible|fund> --days 120
```

- `symbol`：标的代码或名称，例如 `600519`、`RB`、`510300`
- `--market`：品类，默认 `stock`
- `--days`：历史数据天数，默认 120
- `--focus`：分析侧重，`all` / `technical` / `fundamental`，默认 `all`

## 输出

`main.py` stdout 输出 JSON：

```json
{
  "meta": { "symbol": "600519", "market": "stock", "days": 120 },
  "market": { "data": [...], "latest": {...} },
  "technical": { "trend": {...}, "momentum": {...}, "volatility": {...}, "volume": {...} },
  "fundamental": { "pe_ttm": ..., "pb": ..., "total_mv": ... },
  "macro": { "score": 0.35, "signal": "bullish", "tag": "低估", "factors": [...] },
  "sentiment": { "price_position": 0.72, "vol_ratio": 1.15, "trend_strength": "bullish" },
  "overall": { "score": 0.42, "signal": "buy", "weights": {...} }
}
```

## 执行步骤

### 1. 数据采集（market_data.py）

获取原始 OHLCV 行情数据，输出统一 JSON 协议。

```bash
python scripts/market_data.py 600519 --market stock --days 120
```

### 2. 技术指标计算（technical.py）

基于行情数据计算趋势、动量、波动率、成交量指标。

```bash
cat market.json | python scripts/technical.py --stdin
```

### 3. 基本面加工（fundamental.py）

按品类整理估值/基本面指标：

- 股票：PE_TTM、PB、PS_TTM、股息率、市值、ROE、ROA、毛利率、净利润同比、营收同比、资产负债率、行业、上市日期
- ETF/场外基金：类型、管理公司、跟踪指数、规模、份额、净值、近20日净值增长
- 期货：主力合约、基差、仓单、持仓量、成交量、期限结构（近月/远月价差）
- 可转债：转股价、转股价值、溢价率、评级、剩余期限、正股PE/PB/ROE

```bash
cat market.json | python scripts/fundamental.py --stdin
```

### 4. 宏观与基本面判断（macro.py）

对基本面指标做规则化判断，输出 `score` / `signal` / `tag` / `factors`：

- 股票：估值（PE/PB/股息率）+ 成长（净利润/营收同比）+ 质量（ROE/资产负债率）
- 期货：基差 + 仓单 + 持仓量 + 期限结构
- 可转债：转股溢价率 + 转股价值 + 剩余期限 + 正股估值/盈利
- ETF/基金：规模打分

```bash
cat market.json | python scripts/macro.py --stdin
```

### 5. 情绪指标计算（sentiment.py）

基于价格位置和成交量计算情绪状态。

```bash
cat market.json | python scripts/sentiment.py --stdin
```

### 6. 汇总评分（main.py 内部）

综合 `technical` / `macro` / `sentiment` 三个维度，计算 `overall.score` 与 `overall.signal`。

### 7. 结构化输出与报告框架

`main.py` 输出标准 JSON 数据包；LLM 读取 `references/analysis-framework.md` 生成中文研究报告。

## 报告结构

1. **核心结论**：一句话方向判断与关键依据。
2. **技术面**：趋势、均线、关键位、动量、量价。
3. **基本面**：估值、行业、业绩、资金面（按品类差异化）。
4. **情绪面**：价格位置、量能、情绪状态。
5. **多空因素**：利多 / 利空各 3-5 条。
6. **策略建议**：方向、目标价/关键位、止损、仓位、风险。

## 要点

- **数字来自脚本**，LLM 只做解读，不要手算。
- **多品类统一接口**：股票/ETF/期货/可转债/场外基金用同一套 CLI 和输出结构。
- **缺失数据时脚本会显式标注**，LLM 据此说明限制，不编造。
