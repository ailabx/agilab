"""A-share stock basic info backed by tushare stock_basic."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


try:
    from config import DATALAKE_DIR
except ImportError:  # pragma: no cover
    from alphalab.config import DATALAKE_DIR  # type: ignore

from alphalab.data.tushare_source import TushareSource


BASIC_PATH = Path(DATALAKE_DIR) / "basic" / "stock_basic.parquet"


class StockBasic:
    """Stock basic metadata (listings, delistings, industry, etc.)."""

    def __init__(self, source: Optional[TushareSource] = None) -> None:
        self._source = source or TushareSource()
        BASIC_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._df = self._load()

    # ------------------------------------------------------------------ #
    # Loading / updating
    # ------------------------------------------------------------------ #
    def _load(self) -> pd.DataFrame:
        if BASIC_PATH.exists():
            return pd.read_parquet(BASIC_PATH)
        return self.update()

    def update(self) -> pd.DataFrame:
        """Fetch full stock_basic from tushare and overwrite local store."""
        df = self._source.fetch_stock_basic()
        df.to_parquet(BASIC_PATH, index=False)
        self._df = df
        return df

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @property
    def symbols(self) -> list[str]:
        return self._df["symbol"].tolist()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="A-share stock basic updater")
    parser.add_argument(
        "--update",
        action="store_true",
        help="fetch and overwrite stock_basic from tushare",
    )
    args = parser.parse_args()

    basic = StockBasic()
    if args.update:
        basic.update()
        print(f"Stock basic updated: {len(basic.df)} stocks")
    else:
        print(f"Stock basic: {len(basic.df)} stocks")
        print(basic.df.tail())
