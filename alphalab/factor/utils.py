"""Utilities for converting data formats used by factor analysis."""

from __future__ import annotations

import pandas as pd


def wide_from_long(
    df: pd.DataFrame,
    value_col: str,
    date_col: str = "trade_date",
    symbol_col: str = "symbol",
    dropna: bool = True,
) -> pd.DataFrame:
    """Convert long-format DataFrame to wide-format (date x symbol) matrix.

    Parameters
    ----------
    df:
        Long-format DataFrame with at least date, symbol, and value columns.
    value_col:
        Column to pivot into wide matrix values.
    date_col:
        Date column, will become the index.
    symbol_col:
        Symbol column, will become columns.
    dropna:
        If True, drop rows/columns that are all NaN.

    Returns
    -------
    wide : pd.DataFrame
        Date x symbol matrix.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    if value_col not in df.columns:
        raise ValueError(f"value column '{value_col}' not in DataFrame columns {list(df.columns)}")
    wide = df.pivot(index=date_col, columns=symbol_col, values=value_col)
    wide = wide.sort_index()
    if dropna:
        wide = wide.dropna(how="all", axis=0).dropna(how="all", axis=1)
    return wide


def align_series(*series: pd.Series | pd.DataFrame) -> list[pd.Series | pd.DataFrame]:
    """Align multiple pandas objects by index/columns, filling missing with NaN."""
    first = series[0]
    for s in series[1:]:
        first, s = first.align(s, join="outer")
    return [first, *series[1:]]
