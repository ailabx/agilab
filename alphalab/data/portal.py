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
            WITH ranked_adj AS (
                SELECT
                    a.symbol,
                    a.trade_date,
                    a.adj_factor,
                    ROW_NUMBER() OVER (PARTITION BY a.symbol, a.trade_date ORDER BY a.adj_factor DESC) AS rn
                FROM read_parquet('{adj_path.as_posix()}', hive_partitioning=true) a
                WHERE a.symbol IN ({symbol_list})
                  AND a.trade_date <= '{end}'
            ),
            dedup_adj AS (
                SELECT symbol, trade_date, adj_factor
                FROM ranked_adj
                WHERE rn = 1
            ),
            filled_adj AS (
                SELECT
                    d.symbol,
                    d.trade_date,
                    LAST_VALUE(a.adj_factor IGNORE NULLS)
                        OVER (
                            PARTITION BY d.symbol
                            ORDER BY d.trade_date
                            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                        ) AS adj_factor
                FROM read_parquet('{daily_path.as_posix()}', hive_partitioning=true) d
                LEFT JOIN dedup_adj a
                    ON d.symbol = a.symbol AND d.trade_date = a.trade_date
                WHERE d.symbol IN ({symbol_list})
                  AND d.trade_date BETWEEN '{start}' AND '{end}'
            ),
            last_adj AS (
                SELECT symbol, adj_factor AS latest
                FROM (
                    SELECT
                        symbol,
                        adj_factor,
                        ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY trade_date DESC) AS rn
                    FROM filled_adj
                )
                WHERE rn = 1
            )
            SELECT
                d.symbol,
                d.trade_date,
                {", ".join(adjusted_exprs)}
            FROM read_parquet('{daily_path.as_posix()}', hive_partitioning=true) d
            LEFT JOIN filled_adj a
                ON d.symbol = a.symbol AND d.trade_date = a.trade_date
            LEFT JOIN last_adj la
                ON d.symbol = la.symbol
            WHERE d.symbol IN ({symbol_list})
              AND d.trade_date BETWEEN '{start}' AND '{end}'
            ORDER BY d.symbol, d.trade_date
        """
        return self._con.execute(query).fetchdf()
