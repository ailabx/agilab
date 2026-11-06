"""A-share trading calendar backed by tushare trade_cal."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


try:
    from config import DATALAKE_DIR
except ImportError:  # pragma: no cover
    from alphalab.config import DATALAKE_DIR  # type: ignore


CALENDAR_PATH = Path(DATALAKE_DIR) / "calendar" / "trade_cal.parquet"


class TradingCalendar:
    """Trading calendar for the A-share market (SSE)."""

    def __init__(self, source: Optional["TushareSource"] = None) -> None:
        if source is None:
            from alphalab.data.tushare_source import TushareSource

            source = TushareSource()
        self._source = source
        CALENDAR_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._df = self._load()
        self._trade_dates = self._build_trade_dates()

    # ------------------------------------------------------------------ #
    # Loading / updating
    # ------------------------------------------------------------------ #
    def _load(self) -> pd.DataFrame:
        if CALENDAR_PATH.exists():
            return pd.read_parquet(CALENDAR_PATH)
        return self.update()

    def _build_trade_dates(self) -> pd.DatetimeIndex:
        if self._df.empty:
            return pd.DatetimeIndex([])
        df = self._df[self._df["is_open"] == 1].copy()
        return pd.DatetimeIndex(pd.to_datetime(df["cal_date"].astype(str)))

    def update(self, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
        """Fetch calendar from tushare and merge with local store."""
        new = self._source.fetch_trade_cal(start=start, end=end)
        if CALENDAR_PATH.exists():
            old = pd.read_parquet(CALENDAR_PATH)
            merged = pd.concat([old, new], ignore_index=True)
            merged = merged.drop_duplicates(subset=["exchange", "cal_date"], keep="last")
        else:
            merged = new
        merged = merged.sort_values(["exchange", "cal_date"]).reset_index(drop=True)
        merged.to_parquet(CALENDAR_PATH, index=False)
        self._df = merged
        self._trade_dates = self._build_trade_dates()
        return merged

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    @property
    def trade_dates(self) -> pd.DatetimeIndex:
        return self._trade_dates

    def is_trade_date(self, date: str | pd.Timestamp) -> bool:
        ts = pd.Timestamp(date)
        return ts in self._trade_dates

    def trade_dates_in_range(
        self, start: str | pd.Timestamp, end: str | pd.Timestamp
    ) -> pd.DatetimeIndex:
        return self._trade_dates[(self._trade_dates >= start) & (self._trade_dates <= end)]

    def next_trade_date(self, date: str | pd.Timestamp) -> pd.Timestamp:
        pos = self._trade_dates.searchsorted(pd.Timestamp(date), side="right")
        if pos >= len(self._trade_dates):
            raise IndexError("no next trade date")
        return self._trade_dates[pos]

    def previous_trade_date(self, date: str | pd.Timestamp) -> pd.Timestamp:
        pos = self._trade_dates.searchsorted(pd.Timestamp(date), side="left") - 1
        if pos < 0:
            raise IndexError("no previous trade date")
        return self._trade_dates[pos]


def _default_start() -> str:
    """Default update start: 3 months ago."""
    return (pd.Timestamp.now() - pd.Timedelta(days=90)).strftime("%Y%m%d")


def _default_end() -> str:
    """Default update end: today."""
    return pd.Timestamp.now().strftime("%Y%m%d")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="A-share trading calendar updater")
    parser.add_argument(
        "--update",
        action="store_true",
        help="fetch and merge latest trade calendar from tushare",
    )
    parser.add_argument(
        "--start",
        default=_default_start(),
        help="start date for update (default: 3 months ago)",
    )
    parser.add_argument(
        "--end",
        default=_default_end(),
        help="end date for update (default: today)",
    )
    args = parser.parse_args()

    cal = TradingCalendar()
    if args.update:
        cal.update(start=args.start, end=args.end)
        print(f"Calendar updated: {len(cal.trade_dates)} trade dates")
    else:
        print("Trade dates:", len(cal.trade_dates))
        print("First 5:", cal.trade_dates[:5].tolist())
        print("Is 20240102:", cal.is_trade_date("20240102"))
        print("Range Jan 2024:", cal.trade_dates_in_range("20240101", "20240131").tolist()[:5])
        print("Next 20240101:", cal.next_trade_date("20240101"))
