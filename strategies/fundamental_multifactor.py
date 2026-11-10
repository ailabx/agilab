"""Simple fundamental + multi-factor strategy to test performance."""

from __future__ import annotations

import time

import pandas as pd

from alphalab.backtest import BacktestEngine, Strategy
from alphalab.factor.expression.core import FactorExpression


class FundamentalMultiFactorStrategy(Strategy):
    def __init__(self, top_n: int = 30):
        self.top_n = top_n
        self.factors = {
            "momentum_20": "close / Delay(close, 20) - 1",
            "volatility_20": "Std(Log(close / Delay(close, 1)), 20)",
            "rsi_14": "RSI(close, 14)",
            "ma_cross": "MA(close, 5) / MA(close, 20) - 1",
        }
        self._exprs = {k: FactorExpression(v) for k, v in self.factors.items()}
        self._last_month = None

    def handle_data(self, context, data):
        current_month = (context.current_date.year, context.current_date.month)
        if self._last_month == current_month:
            return
        self._last_month = current_month

        # 1. Fundamental selection: load latest fina for all symbols with data
        start = time.perf_counter()
        universe_df = self._select_by_fundamental(data)
        if universe_df.empty:
            return
        universe = universe_df["symbol"].tolist()
        t1 = time.perf_counter()

        # 2. Multi-factor timing: load candidate prices and read factor values
        data.load(universe)
        factor_scores = self._compute_factor_scores(data, universe)
        t2 = time.perf_counter()

        if factor_scores.empty:
            return

        # 3. Position sizing: equal weight on positive-score stocks
        buy_symbols = factor_scores[factor_scores["signal"] > 0].index.tolist()
        if not buy_symbols:
            return

        weight = 1.0 / len(buy_symbols)
        weights = {sym: weight for sym in buy_symbols}
        context.order_target_percent_many(weights)

        t3 = time.perf_counter()
        print(
            f"{context.current_date.date()} "
            f"fina={t1 - start:.3f}s factors={t2 - t1:.3f}s order={t3 - t2:.3f}s "
            f"universe={len(universe)} buys={len(buy_symbols)}"
        )

    def _select_by_fundamental(self, data):
        # Read fina data up to current date, take latest per symbol
        end = data.current_date.strftime("%Y%m%d")
        start = (data.current_date - pd.Timedelta(days=365)).strftime("%Y%m%d")

        df = data.get_fina(start=start, end=end, fields=[
            "symbol", "ann_date", "roe", "netprofit_yoy", "grossprofit_margin", "debt_to_assets"
        ])
        if df.empty:
            return pd.DataFrame()

        df["ann_date"] = pd.to_datetime(df["ann_date"])
        df = df.sort_values("ann_date").groupby("symbol").last().reset_index()

        # Filter
        df = df[
            (df["roe"] > 0.05) &
            (df["netprofit_yoy"] > 0.05) &
            (df["grossprofit_margin"] > 20) &
            (df["debt_to_assets"] < 70)
        ]

        # Score
        df["score"] = (
            df["roe"].rank(pct=True) * 0.4 +
            df["netprofit_yoy"].rank(pct=True) * 0.3 +
            df["grossprofit_margin"].rank(pct=True) * 0.2 -
            df["debt_to_assets"].rank(pct=True) * 0.1
        )
        return df.sort_values("score", ascending=False).head(self.top_n)

    def _compute_factor_scores(self, data, universe):
        # Keep only symbols that exist in the price panel
        available = data.available_symbols
        universe = [s for s in universe if s in available]
        if not universe:
            return pd.DataFrame()

        scores = pd.DataFrame(index=universe)
        for name, expr in self.factors.items():
            try:
                val = data.get_factor(expr)
                scores[name] = val[universe].loc[data.current_date]
            except Exception as exc:
                print(f"Factor {name} failed: {exc}")
                scores[name] = float("nan")

        scores = scores.dropna(how="all")
        if scores.empty:
            return scores

        for col in scores.columns:
            scores[col] = (scores[col] - scores[col].mean()) / scores[col].std()
        scores["signal"] = scores.mean(axis=1)
        return scores


if __name__ == "__main__":
    strategy = FundamentalMultiFactorStrategy(top_n=30)
    result = BacktestEngine(
        start="20240101",
        end="20241106",
        capital_base=1_000_000,
        strategy=strategy,
    ).run()

    print("总收益:", result.total_return)
    print("最终市值:", result.final_value)
    print(result.performance.tail())
