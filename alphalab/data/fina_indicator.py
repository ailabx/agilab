"""Financial indicator storage backed by tushare fina_indicator.

Stored as year-partitioned parquet files:
    {DATALAKE_DIR}/fina_indicator/stock/year=YYYY/data.parquet
"""

from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path
from typing import Optional

import pandas as pd


try:
    from config import DATALAKE_DIR
except ImportError:  # pragma: no cover
    from alphalab.config import DATALAKE_DIR  # type: ignore

from alphalab.data.tushare_source import TushareSource


FINA_ROOT = Path(DATALAKE_DIR) / "fina_indicator" / "stock"
STATE_DB = FINA_ROOT / ".sync" / "state.sqlite"

_PRIMARY_KEY = ["symbol", "ann_date", "end_date"]


class _FinaState:
    """SQLite-backed sync state for (symbol, year) fina_indicator tasks."""

    def __init__(self, path: Path) -> None:
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self._path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS fina_state (
                    symbol TEXT NOT NULL,
                    year INTEGER NOT NULL,
                    start_date TEXT NOT NULL,
                    end_date TEXT NOT NULL,
                    status TEXT NOT NULL,
                    rows INTEGER DEFAULT 0,
                    updated_at TEXT,
                    PRIMARY KEY (symbol, year)
                )
                """
            )
            conn.commit()

    def get_pending(
        self, tasks: list[tuple[str, int, str, str]]
    ) -> list[tuple[str, int, str, str]]:
        with sqlite3.connect(self._path) as conn:
            cursor = conn.execute(
                "SELECT symbol, year, status FROM fina_state WHERE status = 'done'"
            )
            done = {(row[0], row[1]) for row in cursor.fetchall()}
        return [t for t in tasks if (t[0], t[1]) not in done]

    def set_status(
        self,
        symbol: str,
        year: int,
        start_date: str,
        end_date: str,
        status: str,
        rows: int = 0,
    ) -> None:
        with sqlite3.connect(self._path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO fina_state
                (symbol, year, start_date, end_date, status, rows, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
                """,
                (symbol, year, start_date, end_date, status, rows),
            )
            conn.commit()


class FinaIndicatorStore:
    """Store and update tushare fina_indicator data."""

    def __init__(self, source: Optional[TushareSource] = None) -> None:
        self._source = source or TushareSource()
        self._state = _FinaState(STATE_DB)
        FINA_ROOT.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def update(
        self,
        symbols: list[str],
        start: str,
        end: str,
        *,
        sleep: float = 0.1,
        force: bool = False,
    ) -> int:
        """Incremental update for a list of symbols over a date range.

        Returns the number of rows fetched from the source.
        """
        tasks = self._build_tasks(symbols, start, end)
        pending = tasks if force else self._state.get_pending(tasks)
        total_rows = 0
        for symbol, year, year_start, year_end in pending:
            try:
                self._state.set_status(symbol, year, year_start, year_end, "pending")
                rows = self._update_one(symbol, year, year_start, year_end)
                self._state.set_status(
                    symbol, year, year_start, year_end, "done", rows=rows
                )
                total_rows += rows
            except Exception:
                self._state.set_status(
                    symbol, year, year_start, year_end, "failed", rows=0
                )
                raise
            if sleep:
                time.sleep(sleep)
        return total_rows

    def load(
        self,
        start: str,
        end: str,
        symbols: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Load fina_indicator records from local store."""
        return self._read_years(start, end, symbols)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_tasks(
        symbols: list[str], start: str, end: str
    ) -> list[tuple[str, int, str, str]]:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        years = sorted(set(range(start_ts.year, end_ts.year + 1)))
        tasks: list[tuple[str, int, str, str]] = []
        for symbol in symbols:
            for year in years:
                year_start = max(start_ts, pd.Timestamp(f"{year}0101"))
                year_end = min(end_ts, pd.Timestamp(f"{year}1231"))
                if year_start > year_end:
                    continue
                tasks.append(
                    (symbol, year, year_start.strftime("%Y%m%d"), year_end.strftime("%Y%m%d"))
                )
        return tasks

    def _year_path(self, year: int) -> Path:
        return FINA_ROOT / f"year={year}" / "data.parquet"

    def _update_one(self, symbol: str, year: int, start_date: str, end_date: str) -> int:
        path = self._year_path(year)
        df = self._source.fetch_fina_indicator(symbol, start=start_date, end=end_date)
        if df.empty:
            return 0
        if path.exists():
            old = pd.read_parquet(path)
            merged = pd.concat([old, df], ignore_index=True)
            merged = merged.drop_duplicates(subset=_PRIMARY_KEY, keep="last")
        else:
            merged = df
        merged = merged.sort_values(_PRIMARY_KEY).reset_index(drop=True)
        self._write_atomic(merged, path)
        return len(df)

    @staticmethod
    def _write_atomic(df: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.parquet")
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)

    def _read_years(
        self,
        start: str,
        end: str,
        symbols: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        frames: list[pd.DataFrame] = []
        for year in range(start_ts.year, end_ts.year + 1):
            path = self._year_path(year)
            if not path.exists():
                continue
            df = pd.read_parquet(path)
            if symbols is not None:
                df = df[df["symbol"].isin(symbols)]
            df = df[
                (df["ann_date"] >= start) & (df["ann_date"] <= end)
            ]
            frames.append(df)
        if not frames:
            return pd.DataFrame(columns=_PRIMARY_KEY)
        return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="A-share fina_indicator updater")
    parser.add_argument("--update", action="store_true", help="fetch/merge fina_indicator")
    parser.add_argument("--symbols", nargs="+", help="list of symbols to update")
    parser.add_argument("--start", default="20100101", help="start date")
    parser.add_argument("--end", default=pd.Timestamp.now().strftime("%Y%m%d"), help="end date")
    args = parser.parse_args()

    store = FinaIndicatorStore()
    if args.update:
        if not args.symbols:
            raise ValueError("--symbols is required for --update")
        rows = store.update(args.symbols, start=args.start, end=args.end)
        print(f"Updated {rows} rows")
    else:
        df = store.load(start=args.start, end=args.end)
        print(f"Loaded {len(df)} rows")
