"""Backend API helpers for single-factor analysis."""

from __future__ import annotations

from datetime import datetime
import time

import pandas as pd

from alphalab.data.portal import DataPortal
from alphalab.data.pools import Pool
from alphalab.factor.expression import FactorExpression
from alphalab.factor.analysis import FactorAnalyser
from alphalab.factor.utils import wide_from_long


_AVAILABLE_FIELDS = ["open", "high", "low", "close", "vol", "amount"]

# In-memory cache for wide-format fields keyed by (pool, start, end, adjusted).
_FieldsCache: dict[tuple[str, str, str, str], dict[str, pd.DataFrame]] = {}


def _today() -> str:
    return datetime.now().strftime("%Y%m%d")


def _extract_fields(expression: str) -> list[str]:
    """Extract field names referenced in the expression."""
    fields = [f for f in _AVAILABLE_FIELDS if f" {f}" in f" {expression} " or expression.startswith(f)]
    # close is required for forward returns anyway
    if "close" not in fields:
        fields.append("close")
    return fields


def _load_fields(
    symbols: list[str],
    start: str,
    end: str,
    adjusted: str = "qfq",
    expression: str = "",
    pool: str = "",
) -> dict[str, pd.DataFrame]:
    cache_key = (pool, start, end, adjusted)
    if cache_key in _FieldsCache:
        cached = _FieldsCache[cache_key]
        needed = _extract_fields(expression)
        return {k: v for k, v in cached.items() if k in needed}

    portal = DataPortal()
    fields: dict[str, pd.DataFrame] = {}
    needed = _extract_fields(expression)
    for field in needed:
        df = portal.get_prices(
            symbols,
            start=start,
            end=end,
            fields=[field],
            adjusted=adjusted,
        )
        fields[field] = wide_from_long(df, value_col=field)

    _FieldsCache[cache_key] = fields.copy()
    return fields


def run_single_factor_analysis(
    expression: str,
    pool: str = "csi300",
    start: str = "20240101",
    end: str | None = None,
    horizon: int = 20,
    adjusted: str = "qfq",
) -> dict:
    """Run single-factor analysis and return JSON-serializable results."""
    t0 = time.perf_counter()
    symbols = Pool(pool).symbols(start_year=2020)
    if not symbols:
        raise ValueError(f"no symbols for pool {pool}")

    if end is None or end == "":
        end = _today()

    t1 = time.perf_counter()
    fields = _load_fields(symbols, start, end, adjusted=adjusted, expression=expression, pool=pool)
    t2 = time.perf_counter()

    expr = FactorExpression(expression)
    factor = expr.evaluate(fields)
    # Avoid lookahead bias: use t-1 factor to predict t onward.
    factor = factor.shift(1)
    t3 = time.perf_counter()

    close = fields["close"]
    forward_return = close.shift(-horizon) / close - 1.0

    analyser = FactorAnalyser(factor, forward_return, periods=horizon)
    t4 = time.perf_counter()

    ic = analyser.ic_summary("pearson")
    rank_ic = analyser.ic_summary("spearman")
    ic_series = analyser.ic_series("pearson")
    rank_ic_series = analyser.ic_series("spearman")
    qret = analyser.quantile_returns(quantiles=5)
    turnover = analyser.turnover(top_n=50)
    t5 = time.perf_counter()

    def _clean(v):
        if isinstance(v, float):
            return None if pd.isna(v) else round(v, 6)
        return v

    timings = {
        "pool": round(t1 - t0, 3),
        "load_fields": round(t2 - t1, 3),
        "factor_eval": round(t3 - t2, 3),
        "analysis_init": round(t4 - t3, 3),
        "analysis_compute": round(t5 - t4, 3),
        "total": round(time.perf_counter() - t0, 3),
    }
    print(f"[factor-analysis timing] {timings}")

    return {
        "expression": expression,
        "pool": pool,
        "start": start,
        "end": end,
        "horizon": horizon,
        "symbols": len(symbols),
        "ic": {k: _clean(v) for k, v in ic.items()},
        "rank_ic": {k: _clean(v) for k, v in rank_ic.items()},
        "ic_series": {
            "dates": ic_series.index.strftime("%Y-%m-%d").tolist(),
            "ic": [_clean(x) for x in ic_series.values.tolist()],
            "rank_ic": [_clean(x) for x in rank_ic_series.values.tolist()],
        },
        "quantile_returns": {
            "labels": [f"Q{i+1}" for i in range(len(qret))],
            "values": [_clean(x) for x in qret["mean_return"].values.tolist()],
        },
        "turnover": {"mean": _clean(float(turnover.mean())) if not turnover.empty else None},
    }
