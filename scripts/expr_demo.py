"""Demo for the mini AST factor expression engine."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from alphalab.data.portal import DataPortal
from alphalab.data.pools import Pool
from alphalab.factor.utils import wide_from_long
from alphalab.factor.expression import FactorExpression
from alphalab.factor.analysis import FactorAnalyser


def load_fields(symbols: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    portal = DataPortal()
    fields: dict[str, pd.DataFrame] = {}
    for field in ["open", "high", "low", "close", "vol", "amount"]:
        df = portal.get_prices(symbols, start=start, end=end, fields=[field], adjusted="qfq")
        fields[field] = wide_from_long(df, value_col=field)
    return fields


def main() -> None:
    symbols = Pool("csi300").symbols(start_year=2020)
    print(f"Symbols: {len(symbols)}")

    fields = load_fields(symbols, "20220101", "20241106")
    close = fields["close"]

    # Define factor expressions
    expressions = {
        "momentum_20": "close / DELAY(close, 20) - 1",
        "volatility_20": "STD(LOG(close / DELAY(close, 1)), 20)",
        "ma_ratio": "MA(close, 10) / MA(close, 30)",
        "price_volume_corr": "CORR(close, vol, 20)",
        "rank_low_volatility": "RANK(STD(close, 20))",
    }

    forward_return = close.shift(-20) / close - 1.0

    for name, expr_source in expressions.items():
        expr = FactorExpression(expr_source)
        factor = expr.evaluate(fields)
        factor = factor.shift(1)  # avoid lookahead bias

        analyser = FactorAnalyser(factor, forward_return, periods=20)
        ic = analyser.ic_summary("pearson")
        rank_ic = analyser.ic_summary("spearman")
        print(f"\n{name}: {expr_source}")
        print(f"  IC mean={ic['mean']:.4f}, IR={ic['ir']:.4f}")
        print(f"  Rank IC mean={rank_ic['mean']:.4f}, IR={rank_ic['ir']:.4f}")


if __name__ == "__main__":
    main()
