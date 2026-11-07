"""Single-factor analysis demo using existing csi300 data."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from alphalab.data.portal import DataPortal
from alphalab.data.pools import Pool
from alphalab.factor.engine import FactorEngine
from alphalab.factor.analysis import FactorAnalyser
from alphalab.factor.utils import wide_from_long


def load_fields(portal: DataPortal, symbols: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    fields_long: dict[str, pd.DataFrame] = {}
    for field in ["open", "high", "low", "close", "vol", "amount"]:
        df = portal.get_prices(symbols, start=start, end=end, fields=[field], adjusted="qfq")
        fields_long[field] = df
        print(f"Loaded {field}: {len(df)} rows")
    return {k: wide_from_long(v, value_col=k) for k, v in fields_long.items()}


def make_factors(engine: FactorEngine) -> dict[str, pd.DataFrame]:
    """Build a set of sample factors from the engine."""
    return {
        "momentum_20": engine.momentum(window=20),
        "reversal_5": engine.reversal(window=5),
        "volatility_20": engine.volatility(window=20),
        "volume_momentum_20": engine.volume_momentum(window=20),
        "ma_ratio_10_30": engine.ma_ratio(fast=10, slow=30),
        "rsi_14": engine.rsi(window=14),
    }


def forward_returns(close: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Future return from t to t+horizon, aligned at t."""
    return close.shift(-horizon) / close - 1.0


def analyse_factor(
    name: str,
    factor: pd.DataFrame,
    close: pd.DataFrame,
    horizons: list[int],
) -> pd.DataFrame:
    rows = []
    for h in horizons:
        fr = forward_returns(close, h)
        analyser = FactorAnalyser(factor, fr, periods=h)
        ic = analyser.ic_summary("pearson")
        rank_ic = analyser.ic_summary("spearman")
        rows.append({
            "factor": name,
            "horizon": h,
            "ic_mean": ic["mean"],
            "ic_std": ic["std"],
            "ic_ir": ic["ir"],
            "rank_ic_mean": rank_ic["mean"],
            "rank_ic_std": rank_ic["std"],
            "rank_ic_ir": rank_ic["ir"],
        })
    return pd.DataFrame(rows)


def main() -> None:
    pool = Pool("csi300")
    symbols = pool.symbols(start_year=2020)
    print(f"Pool csi300 symbols (2020+): {len(symbols)}")

    portal = DataPortal()
    start, end = "20220101", "20241106"

    fields_wide = load_fields(portal, symbols, start, end)
    engine = FactorEngine(fields_wide)
    close = fields_wide["close"]

    factors = make_factors(engine)
    # avoid lookahead bias: use information available at t-1 close
    factors = {k: v.shift(1) for k, v in factors.items()}

    horizons = [1, 5, 10, 20]
    reports = []
    for name, factor in factors.items():
        reports.append(analyse_factor(name, factor, close, horizons))

    report_df = pd.concat(reports, ignore_index=True)
    print("\n=== Single Factor IC Report ===")
    print(report_df.to_string(index=False))

    output_path = Path("single_factor_report.csv")
    report_df.to_csv(output_path, index=False)
    print(f"\nSaved {output_path}")


if __name__ == "__main__":
    main()
