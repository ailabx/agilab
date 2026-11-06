"""DataPortal: unified read-only interface over local datalake."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import duckdb
import pandas as pd


try:
    from config import DATALAKE_DIR
except ImportError:  # pragma: no cover
    from alphalab.config import DATALAKE_DIR  # type: ignore


ADJUSTMENT = Literal["raw", "qfq", "hfq"]


class DataPortal:
    """Read daily prices with optional adjustment from local datalake."""

    def __init__(self, datalake_dir: Optional[str | Path] = None) -> None:
        self._datalake = Path(datalake_dir or DATALAKE_DIR)
        self._con = duckdb.connect()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def get_prices(
        self,
        symbols: list[str],
        start: str,
        end: str,
        fields: Optional[list[str]] = None,
        adjusted: ADJUSTMENT = "raw",
    ) -> pd.DataFrame:
        """Load daily price DataFrame with optional adjustment.

        Returns a long-format DataFrame with columns:
        symbol, trade_date, open, high, low, close, pre_close, ...
        """
        fields = fields or ["open", "high", "low", "close", "pre_close", "vol", "amount"]
        if adjusted == "raw":
            return self._raw_prices(symbols, start, end, fields)
        return self._adjusted_prices(symbols, start, end, fields, mode=adjusted)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _raw_prices(
        self, symbols: list[str], start: str, end: str, fields: list[str]
    ) -> pd.DataFrame:
        daily_path = self._datalake / "daily" / "stock" / "**" / "data.parquet"
        symbol_list = ", ".join(f"'{s}'" for s in symbols)
        field_str = ", ".join(["symbol", "trade_date", *fields])
        query = f"""
            SELECT {field_str}
            FROM read_parquet('{daily_path.as_posix()}', hive_partitioning=true)
            WHERE symbol IN ({symbol_list})
              AND trade_date BETWEEN '{start}' AND '{end}'
            ORDER BY symbol, trade_date
        """
        return self._con.execute(query).fetchdf()

    def _adjusted_prices(
        self, symbols: list[str], start: str, end: str, fields: list[str], mode: str
    ) -> pd.DataFrame:
        daily_path = self._datalake / "daily" / "stock" / "**" / "data.parquet"
        adj_path = self._datalake / "adj_factor" / "stock" / "**" / "data.parquet"
        symbol_list = ", ".join(f"'{s}'" for s in symbols)
        price_fields = ["symbol", "trade_date", *fields]

        # For qfq, we divide by the last adj_factor in the requested window.
        # For hfq, we multiply by adj_factor directly.
        if mode == "hfq":
            adjusted_exprs = [f"d.{f} * a.adj_factor AS {f}" for f in fields]
        else:  # qfq
            adjusted_exprs = [
                f"d.{f} * a.adj_factor / la.latest AS {f}" for f in fields
            ]

        query = f"""
            WITH last_adj AS (
                SELECT symbol, MAX(adj_factor) AS latest
                FROM read_parquet('{adj_path.as_posix()}', hive_partitioning=true)
                WHERE symbol IN ({symbol_list})
                  AND trade_date BETWEEN '{start}' AND '{end}'
                GROUP BY symbol
            )
            SELECT
                d.symbol,
                d.trade_date,
                {", ".join(adjusted_exprs)}
            FROM read_parquet('{daily_path.as_posix()}', hive_partitioning=true) d
            LEFT JOIN read_parquet('{adj_path.as_posix()}', hive_partitioning=true) a
                ON d.symbol = a.symbol AND d.trade_date = a.trade_date
            LEFT JOIN last_adj la
                ON d.symbol = la.symbol
            WHERE d.symbol IN ({symbol_list})
              AND d.trade_date BETWEEN '{start}' AND '{end}'
            ORDER BY d.symbol, d.trade_date
        """
        return self._con.execute(query).fetchdf()
