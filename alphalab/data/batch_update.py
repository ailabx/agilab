"""Batch update daily bars and adj_factor for the whole A-share market."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import sqlite3


try:
    from config import DATALAKE_DIR
except ImportError:  # pragma: no cover
    from alphalab.config import DATALAKE_DIR  # type: ignore

from alphalab.data.adj_factor import AdjFactorStore
from alphalab.data.daily import DailyStore
from alphalab.data.fina_indicator import FinaIndicatorStore
from alphalab.data.stock_basic import StockBasic


BATCH_STATE_DB = Path(DATALAKE_DIR) / "daily" / "stock" / ".sync" / "batch_state.sqlite"


class _BatchState:
    def __init__(self, path: Path) -> None:
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self._path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS batch_state (
                    symbol TEXT NOT NULL,
                    year INTEGER NOT NULL,
                    table_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    rows INTEGER DEFAULT 0,
                    updated_at TEXT,
                    PRIMARY KEY (symbol, year, table_name)
                )
                """
            )
            conn.commit()

    def is_done(self, symbol: str, year: int, table_name: str) -> bool:
        with sqlite3.connect(self._path) as conn:
            cursor = conn.execute(
                "SELECT status FROM batch_state WHERE symbol=? AND year=? AND table_name=?",
                (symbol, year, table_name),
            )
            row = cursor.fetchone()
            return row is not None and row[0] == "done"

    def set_status(self, symbol: str, year: int, table_name: str, status: str, rows: int = 0) -> None:
        with sqlite3.connect(self._path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO batch_state
                (symbol, year, table_name, status, rows, updated_at)
                VALUES (?, ?, ?, ?, ?, datetime('now'))
                """,
                (symbol, year, table_name, status, rows),
            )
            conn.commit()


def _resolve_years(year: int | None) -> list[int]:
    if year is not None:
        return [year]
    current = pd.Timestamp.now().year
    return list(range(2010, current + 1))


def _symbols_from_index(index_code: str) -> list[str]:
    from alphalab.data.index_weight import IndexWeightStore

    store = IndexWeightStore()
    df = store.load(index_code)
    if df.empty:
        raise ValueError(f"no index_weight data for {index_code}")
    # Use all distinct symbols that ever appeared in this index, so historical
    # backtests can reconstruct the correct constituent set at any date.
    return [s.strip() for s in df["con_code"].dropna().unique().tolist() if s.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch update daily bars and adj_factor")
    parser.add_argument("--year", type=int, help="specific year to update (default: all years from 2010 to now)")
    parser.add_argument("--symbols", nargs="+", help="specific symbols to update (default: all from stock_basic)")
    parser.add_argument("--from-file", dest="from_file", help="file containing symbols, one per line")
    parser.add_argument("--index-code", dest="index_code", help="index code (e.g. 000300.SH) to update its historical constituents")
    parser.add_argument("--pool", help="pool name (e.g. csi300, csi500, csi1000) to update its historical constituents")
    parser.add_argument("--sleep", type=float, default=0.05, help="sleep seconds between symbols")
    parser.add_argument("--force", action="store_true", help="force re-update even if done")
    parser.add_argument("--include-fina", action="store_true", dest="include_fina", help="also update fina_indicator")
    parser.add_argument("--fina-only", action="store_true", dest="fina_only", help="only update fina_indicator")
    args = parser.parse_args()

    years = _resolve_years(args.year)
    if args.from_file:
        raw_symbols = Path(args.from_file).read_text(encoding="utf-8").splitlines()
        symbols = [s.strip() for s in raw_symbols if s.strip()]
    elif args.pool:
        from alphalab.data.pools import Pool
        pool = Pool(args.pool)
        symbols = pool.symbols()
        print(f"Pool {args.pool} ({pool.display_name}): {len(symbols)} distinct historical constituents")
    elif args.index_code:
        symbols = _symbols_from_index(args.index_code)
    else:
        symbols = [s.strip() for s in (args.symbols or StockBasic().symbols)]
    state = _BatchState(BATCH_STATE_DB)
    daily_store = DailyStore() if not args.fina_only else None
    adj_store = AdjFactorStore() if not args.fina_only else None
    fina_store = FinaIndicatorStore() if (args.include_fina or args.fina_only) else None

    total = len(symbols) * len(years)
    print(f"Updating {len(symbols)} symbols x {len(years)} years = {total} tasks")

    current_year = pd.Timestamp.now().year
    for year in years:
        start = f"{year}0101"
        end = f"{year}1231"
        # Current year is always re-fetched so the latest data is included.
        is_current_year = year == current_year
        print(f"[{year}] Starting update: {len(symbols)} symbols")
        for i, symbol in enumerate(symbols, 1):
            stores: list[tuple[str, object]] = []
            if daily_store is not None:
                stores.append(("daily", daily_store))
            if adj_store is not None:
                stores.append(("adj_factor", adj_store))
            if fina_store is not None:
                stores.append(("fina_indicator", fina_store))
            for table_name, store in stores:
                if not args.force and not is_current_year and state.is_done(symbol, year, table_name):
                    continue
                state.set_status(symbol, year, table_name, "pending")
                try:
                    rows = store.update([symbol], start=start, end=end, force=args.force)
                    state.set_status(symbol, year, table_name, "done", rows=rows)
                except Exception as exc:
                    print(f"FAILED {symbol} {year} {table_name}: {exc}")
                    state.set_status(symbol, year, table_name, "failed", rows=0)
            if i % 100 == 0:
                print(f"  [{year}] {i}/{len(symbols)} symbols processed")
            if args.sleep:
                time.sleep(args.sleep)
        print(f"[{year}] Finished update")

    print("Batch update finished.")


if __name__ == "__main__":
    main()
