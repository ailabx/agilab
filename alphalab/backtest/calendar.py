"""Trading calendar derived from stored daily bars."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


try:
    from config import DATALAKE_DIR
except ImportError:  # pragma: no cover
    from alphalab.config import DATALAKE_DIR


class TradingCalendar:
    """Calendar whose trade dates are derived from stored daily bars."""

    def __init__(self, datalake_dir: str | Path | None = None) -> None:
        self._datalake = Path(datalake_dir or DATALAKE_DIR)
        self._dates = self._load_dates()

    def _load_dates(self) -> pd.DatetimeIndex:
        daily_path = self._datalake / "daily" / "stock" / "**" / "data.parquet"
        if not any(self._datalake.glob("daily/stock/year=*/data.parquet")):
            return pd.DatetimeIndex([], name="trade_date")

        import duckdb

        con = duckdb.connect()
        try:
            df = con.execute(
                f"SELECT DISTINCT trade_date FROM read_parquet('{daily_path.as_posix()}', hive_partitioning=true) ORDER BY trade_date"
            ).fetchdf()
        finally:
            con.close()

        if df.empty:
            return pd.DatetimeIndex([], name="trade_date")
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        return pd.DatetimeIndex(df["trade_date"]).sort_values()

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self._dates

    def dates_in_range(self, start: str, end: str) -> pd.DatetimeIndex:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        return self._dates[(self._dates >= start_ts) & (self._dates <= end_ts)]