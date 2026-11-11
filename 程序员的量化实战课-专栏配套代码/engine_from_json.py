"""
按用户 JSON 配置，从头实现 ETF 趋势轮动策略：
- RunDaily
- buy_rules: roc(close,1) >= -0.05
- sell_rules: trend_score2(close,20) >= 5 或 <= 0
- order_by: cs_zscore(trend_score2(close,20))
- order_by_topk=4, buffer=1, strict=True
- WeightEqually(max_weight=0.25), 即 round(0.98/n, 3) 且单只不超过 0.25
- Rebalance 5% 阈值 + 整数股
"""
import argparse
import os

import numpy as np
import pandas as pd

from utils import calc_metrics


DEFAULT_DIR = "data_from_alphalab"

DEFAULT_SYMBOLS = [
    "510300.SH", "588000.SH", "510880.SH", "159980.SZ", "159985.SZ",
    "513100.SH", "513520.SH", "513500.SH", "513030.SH", "501018.SH",
    "513180.SH", "159915.SZ", "512890.SH", "511090.SH", "510180.SH",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default=",".join(DEFAULT_SYMBOLS), help="标的 symbol 列表")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DIR, help="CSV 数据目录")
    parser.add_argument("--capital", type=float, default=1000000.0, help="初始资金")
    parser.add_argument("--period", type=int, default=20, help="trend_score2 周期")
    parser.add_argument("--top-k", type=int, default=4, help="top-k 选股")
    parser.add_argument("--buffer", type=int, default=1, help="buffer 位数")
    parser.add_argument("--strict", type=bool, default=True, help="strict 模式")
    parser.add_argument("--max-weight", type=float, default=0.25, help="单标的上限")
    parser.add_argument("--fee", type=float, default=0.0001, help="交易成本")
    parser.add_argument("--ann-factor", type=int, default=252, help="年化因子")
    parser.add_argument("--risk-free", type=float, default=0.02, help="无风险利率")
    parser.add_argument("--start-date", type=str, default="2020-01-01", help="开始日期")
    parser.add_argument("--end-date", type=str, default="", help="结束日期")
    return parser.parse_args()


def load_ohlc(path: str, sym: str) -> pd.DataFrame:
    """读取单个 symbol 的 CSV，返回带 symbol 列的 DataFrame。"""
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").drop_duplicates("date").set_index("date")
    df["symbol"] = sym
    return df


def calc_trend_score2(prices: pd.Series, period: int) -> pd.Series:
    """计算单只标的 trend_score2。"""
    x = np.arange(period, dtype=float)
    w = np.linspace(1.0, 2.0, period) ** 2
    w_sum = w.sum()
    x_bar = (w * x).sum() / w_sum
    dx = x - x_bar
    variance_x = (w * dx ** 2).sum()

    def _score(window: np.ndarray) -> float:
        y = np.log(window)
        y_bar = (w * y).sum() / w_sum
        slope = (w * dx * (y - y_bar)).sum() / variance_x
        intercept = y_bar - slope * x_bar
        y_pred = slope * np.arange(period) + intercept
        ss_res = (w * (y - y_pred) ** 2).sum()
        ss_tot = (w * (y - y_bar) ** 2).sum()
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        r_squared = min(max(r_squared, 0.0), 1.0)
        annualized_return = np.exp(slope * 250) - 1.0
        return annualized_return * r_squared

    valid = prices.dropna()
    if len(valid) < period:
        return pd.Series(np.nan, index=prices.index)
    rolled = valid.rolling(window=period).apply(_score, raw=True)
    return pd.Series(rolled, index=valid.index).reindex(prices.index)


def cs_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """截面 z-score，每行独立计算，忽略 NaN。"""
    mean = df.mean(axis=1)
    std = df.std(axis=1)
    return ((df.T - mean) / std).T


def compute_signals_and_score(close, period, ann_factor=252):
    """返回：
    - score: cs_zscore(trend_score2(close,20))
    - roc1: close.pct_change(1)
    - where: 每日独立过滤信号 (buy_rules / sell_rules 的 OR)
    """
    symbols = close.columns.tolist()

    # trend_score2 按列独立计算
    ts2 = pd.DataFrame({s: calc_trend_score2(close[s], period) for s in symbols}, index=close.index)

    # cs_zscore
    score = cs_zscore(ts2)

    # roc1
    roc1 = close.pct_change(1)

    # buy_rules: roc(close,1) >= -0.05 -> where = True
    # sell_rules: trend_score2 >= 5 或 <= 0 -> where = False
    # 有状态信号：buy 触发 -> 1，sell 触发 -> 0，否则 ffill
    # alphalab 的 buy/sell rules 是 OR 组合：只要任一 buy rule 满足就买入；任一 sell rule 满足就卖出
    signal = pd.DataFrame(np.nan, index=close.index, columns=symbols)
    for s in symbols:
        s_buy = (roc1[s] >= -0.05).astype(float)
        s_sell = ((ts2[s] >= 5) | (ts2[s] <= 0)).astype(float)
        sig = pd.Series(np.where(s_buy, 1.0, np.nan), index=close.index)
        sig = pd.Series(np.where(s_sell, 0.0, sig), index=close.index)
        sig = sig.ffill().fillna(0)
        signal[s] = sig

    return score, roc1, signal


def select_topk(score_row, signal_row, holding_symbols, top_k, buffer_x, strict):
    """复刻 alphalab SelectTopK：
    - score_row: 当日 score Series（可能含 NaN）
    - signal_row: 当日 signal Series (0/1)
    - holding_symbols: 当前持仓集合
    返回 target_list（保持排名顺序）。

    关键：候选池包含 signal==1 的所有 symbol，score 为 NaN 的排在最后；
         缓冲区内 score 为 NaN 的旧持仓也应保留。
    """
    # 候选池：signal == 1
    candidates = [s for s in score_row.index if signal_row[s] == 1.0]

    if len(candidates) == 0:
        return []

    # 按 score 降序排序，NaN 自动排最后
    ranked = score_row[candidates].sort_values(ascending=False, na_position='last')
    ranked_syms = ranked.index.tolist()

    # 如果 score 全为 NaN，alphalab 仍按 symbol 字母顺序取 top_k
    if ranked.isna().all():
        ranked_syms = sorted(candidates)

    top = ranked_syms[:top_k]
    buffer_zone = ranked_syms[top_k:top_k + buffer_x] if buffer_x > 0 else []

    # 缓冲区内已持仓的才保留
    keep = [s for s in buffer_zone if s in holding_symbols]

    if not keep:
        return top

    if strict:
        # 从排名最靠后的新入选者开始淘汰，给保留的旧持仓腾位置
        to_drop = set()
        for s in reversed(top):
            if len(to_drop) >= len(keep):
                break
            if s not in holding_symbols:
                to_drop.add(s)
        top = [s for s in top if s not in to_drop]

    # 合并并去重，保持 top 在前，keep 在后
    result = []
    seen = set()
    for s in top + keep:
        if s not in seen:
            result.append(s)
            seen.add(s)

    # strict 模式下最终仍应不超过 top_k 只
    if strict and len(result) > top_k:
        result = result[:top_k]

    return result


def backtest(close, close_ffill, score, signal, args):
    symbols = close.columns.tolist()
    dates = close.index
    cash = args.capital
    positions = {}  # sym -> volume
    nav = pd.Series(index=dates, dtype=float)
    trade_count = 0
    daily_records = []

    for date in dates:
        today_close = close.loc[date]
        today_close_ffill = close_ffill.loc[date]
        holding_symbols = set(positions.keys())

        # SelectTopK
        target_list = select_topk(
            score.loc[date],
            signal.loc[date],
            holding_symbols,
            args.top_k,
            args.buffer,
            args.strict,
        )

        # WeightEqually：过滤停牌的标的，不参与调仓
        active_target_list = [s for s in target_list if pd.notna(today_close_ffill[s])]
        n = len(active_target_list)
        if n > 0:
            w = round(0.98 / n, 3)
            if args.max_weight and w > args.max_weight:
                w = round(args.max_weight, 3)
            target_weights = {s: w for s in active_target_list}
        else:
            target_weights = {}

        # Rebalance 5% 阈值（市值计算和交易用 ffill 后价格，复刻 _fix_missing_bar）
        need_rebalance = False
        if holding_symbols != set(target_weights.keys()):
            need_rebalance = True
        else:
            total_value = cash + sum(positions[s] * today_close_ffill[s] for s in positions)
            for s in target_weights:
                val = positions.get(s, 0) * today_close_ffill[s]
                current_pct = val / total_value if total_value > 0 else 0
                if abs(current_pct - target_weights[s]) > 0.05:
                    need_rebalance = True
                    break

        if need_rebalance:
            # 第一步：平仓不在目标中的
            for s in list(positions.keys()):
                if s not in target_weights:
                    px = today_close_ffill[s]
                    if pd.isna(px):
                        continue
                    volume = positions[s]
                    value = volume * px
                    commission = value * args.fee
                    cash += value - commission
                    del positions[s]
                    trade_count += 1

            # 第二步：减仓超出目标 >5% 的
            total_value = cash + sum(positions[s] * today_close_ffill[s] for s in positions)
            for s in list(target_weights.keys()):
                if s not in positions:
                    continue
                px = today_close_ffill[s]
                if pd.isna(px):
                    continue
                current_value = positions[s] * px
                current_pct = current_value / total_value if total_value > 0 else 0
                if current_pct > target_weights[s] and (current_pct - target_weights[s]) > 0.05:
                    target_value = total_value * target_weights[s]
                    diff_value = current_value - target_value
                    volume = int(diff_value // px)
                    if volume <= 0:
                        continue
                    value = volume * px
                    commission = value * args.fee
                    cash += value - commission
                    positions[s] -= volume
                    if positions[s] <= 0:
                        del positions[s]

            # 第三步：买入/加仓
            total_value = cash + sum(positions[s] * today_close_ffill[s] for s in positions)
            for s in target_weights:
                px = today_close_ffill[s]
                if pd.isna(px):
                    continue
                current_value = positions.get(s, 0) * px
                current_pct = current_value / total_value if total_value > 0 else 0
                diff_pct = target_weights[s] - current_pct
                if diff_pct > 0.05 or (s not in positions):
                    target_value = total_value * target_weights[s]
                    diff_value = target_value - current_value
                    if diff_value > 0:
                        volume = int(diff_value // px)
                        if volume <= 0:
                            continue
                        value = volume * px
                        commission = value * args.fee
                        cash -= value + commission
                        if s in positions:
                            positions[s] += volume
                        else:
                            positions[s] = volume
                            trade_count += 1

        # 计算 equity
        holding_value = sum(positions[s] * today_close_ffill[s] for s in positions)
        equity = cash + holding_value
        nav.loc[date] = equity
        daily_records.append({
            'date': date,
            'cash': cash,
            'holdings_value': holding_value,
            'total_value': equity,
            'positions': positions.copy(),
        })

    daily_df = pd.DataFrame(daily_records).set_index('date')
    return nav, trade_count, daily_df


def main():
    args = parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    close_frames = []
    for s in symbols:
        path = os.path.join(args.data_dir, f"{s}.csv")
        if not os.path.exists(path):
            print(f"警告：文件不存在 {path}")
            continue
        df = load_ohlc(path, s)
        close_frames.append(df["close"].rename(s))

    close = pd.concat(close_frames, axis=1, sort=True)
    # 交易/市值计算用 ffill（复刻 alphalab 的 _fix_missing_bar），因子计算用 raw close
    close_ffill = close.ffill()

    print(f"完整数据区间: {close.index[0].date()} ~ {close.index[-1].date()}")

    score, roc1, signal = compute_signals_and_score(close, args.period)

    start_dt = pd.to_datetime(args.start_date) if args.start_date else close.index[0]
    end_dt = pd.to_datetime(args.end_date) if args.end_date else close.index[-1]
    close = close[(close.index >= start_dt) & (close.index <= end_dt)]
    score = score[(score.index >= start_dt) & (score.index <= end_dt)]
    signal = signal[(signal.index >= start_dt) & (signal.index <= end_dt)]

    print(f"回测区间: {close.index[0].date()} ~ {close.index[-1].date()}")

    nav, trade_count, daily_df = backtest(close, close_ffill, score, signal, args)
    daily_df.to_csv("data_from_alphalab/self_daily.json.csv")
    print(f"自研 daily 导出完成: {len(daily_df)} rows")

    # 导出 score / signal
    score.to_csv("data_from_alphalab/self_score.json.csv")
    signal.to_csv("data_from_alphalab/self_signal.json.csv")

    metrics = calc_metrics(nav, ann_factor=args.ann_factor, risk_free=args.risk_free)

    print(f"=== 自研引擎（JSON 配置复现） ===")
    print(f"区间: {metrics['start_date']} ~ {metrics['end_date']}")
    for k in ["total_return", "annualized_return", "volatility", "max_drawdown"]:
        print(f"{k}: {metrics[k]:.2%}")
    for k in ["sharpe", "calmar"]:
        print(f"{k}: {metrics[k]:.2f}")
    print(f"交易次数: {trade_count}")

    nav.to_csv("data_from_alphalab/self_nav.json.csv", header=['nav'])
    print(f"自研 nav 导出完成: {len(nav)} rows")


if __name__ == "__main__":
    main()
