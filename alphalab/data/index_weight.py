"""Index constituent weight storage.

Stores per-index parquet files with columns:
    index_code, con_code, trade_date, weight
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import pandas as pd


try:
    from config import DATALAKE_DIR
except ImportError:  # pragma: no cover
    from alphalab.config import DATALAKE_DIR  # type: ignore

from alphalab.data.tushare_source import TushareSource


INDEX_WEIGHT_ROOT = Path(DATALAKE_DIR) / "index_weight"

_PRIMARY_KEY = ["index_code", "con_code", "trade_date"]


class IndexWeightStore:
    """Store and update index constituent weights."""

    def __init__(self, source: Optional[TushareSource] = None) -> None:
        self._source = source or TushareSource()
        INDEX_WEIGHT_ROOT.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def update(
        self,
        index_code: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        *,
        force: bool = False,
    ) -> int:
        """Fetch and merge index weights for a single index.

        Returns the number of rows fetched from the source.
        """
        start, end = self._resolve_range(index_code, start, end, force=force)
        rows_total = 0
        for year_start, year_end in self._yearly_ranges(start, end):
            df = self._source.fetch_index_weight(index_code, year_start, year_end)
            if not df.empty:
                self._merge(df, index_code)
                rows_total += len(df)
        return rows_total

    def load(self, index_code: str) -> pd.DataFrame:
        """Load stored weights for an index."""
        path = self._path(index_code)
        if not path.exists():
            return pd.DataFrame(columns=_PRIMARY_KEY + ["weight"])
        return pd.read_parquet(path)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _path(self, index_code: str) -> Path:
        return INDEX_WEIGHT_ROOT / f"{index_code}.parquet"

    def _resolve_range(
        self,
        index_code: str,
        start: Optional[str],
        end: Optional[str],
        *,
        force: bool,
    ) -> tuple[str, str]:
        if end is None:
            end = pd.Timestamp.now().strftime("%Y%m%d")
        if start is None and not force:
            existing = self.load(index_code)
            if not existing.empty:
                latest = existing["trade_date"].max()
                start = (pd.Timestamp(latest) - pd.offsets.MonthBegin(3)).strftime("%Y%m%d")
            else:
                start = "20050101"
        elif start is None:
            start = "20050101"
        return start, end

    def _merge(self, df: pd.DataFrame, index_code: str) -> None:
        path = self._path(index_code)
        if path.exists():
            old = pd.read_parquet(path)
            merged = pd.concat([old, df], ignore_index=True)
        else:
            merged = df
        merged = merged.drop_duplicates(subset=_PRIMARY_KEY, keep="last")
        merged = merged.sort_values(_PRIMARY_KEY).reset_index(drop=True)
        self._write_atomic(merged, path)

    @staticmethod
    def _yearly_ranges(start: str, end: str) -> list[tuple[str, str]]:
        """Split a date range into yearly chunks.

        Tushare index_weight returns only the most recent snapshots when the
        requested range is too wide, so we query year by year.
        """
        start_year = pd.Timestamp(start).year
        end_year = pd.Timestamp(end).year
        ranges: list[tuple[str, str]] = []
        for year in range(start_year, end_year + 1):
            year_start = max(start, f"{year}0101")
            year_end = min(end, f"{year}1231")
            ranges.append((year_start, year_end))
        return ranges

    @staticmethod
    def _write_atomic(df: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.parquet")
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index constituent weight updater")
    parser.add_argument(
        "--index-code",
        action="append",
        required=True,
        help="index code, e.g. 000300.SH",
    )
    parser.add_argument(
        "--start",
        help="start date (default: incremental or 20050101 with --all)",
    )
    parser.add_argument(
        "--end",
        help="end date (default: today)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="full",
        help="fetch full history from 2005-01-01",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    store = IndexWeightStore()
    for code in args.index_code:
        start = args.start if args.start else ("20050101" if args.full else None)
        end = args.end
        rows = store.update(code, start=start, end=end)
        print(f"{code}: fetched {rows} rows")
