"""Pool definitions and helpers for progressive data builds.

A pool is a named set of symbols derived from an index's historical constituent
snapshots.  Building data by pools lets us start small (e.g. CSI300) and expand
later without re-downloading symbols already present in the datalake.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from config import DATALAKE_DIR
except ImportError:  # pragma: no cover
    from alphalab.config import DATALAKE_DIR  # type: ignore

from alphalab.data.index_weight import IndexWeightStore


POOL_MAP = {
    "csi300": {"name": "沪深300", "index_code": "000300.SH"},
    "csi500": {"name": "中证500", "index_code": "000905.SH"},
    "csi1000": {"name": "中证1000", "index_code": "000852.SH"},
    "csia500": {"name": "中证A500", "index_code": "000510.SH"},
    "sz50": {"name": "上证50", "index_code": "000016.SH"},
    "hs300": {"name": "沪深300", "index_code": "000300.SH"},
    "zz500": {"name": "中证500", "index_code": "000905.SH"},
    "zz1000": {"name": "中证1000", "index_code": "000852.SH"},
    "zza500": {"name": "中证A500", "index_code": "000510.SH"},
}


class Pool:
    """A named pool of symbols derived from index constituents."""

    def __init__(self, name: str) -> None:
        if name not in POOL_MAP:
            raise ValueError(f"Unknown pool: {name}. Supported: {list(POOL_MAP.keys())}")
        self.name = name
        self.index_code = POOL_MAP[name]["index_code"]
        self.display_name = POOL_MAP[name]["name"]

    def symbols(self, start_year: int | None = 2010) -> list[str]:
        """All distinct symbols that ever appeared in this pool.

        By default only constituents from 2010 onwards are considered,
        matching the default daily data build range.
        """
        store = IndexWeightStore()
        df = store.load(self.index_code)
        if df.empty:
            return []
        if start_year is not None:
            cutoff = f"{start_year}0101"
            df = df[df["trade_date"] >= cutoff]
        return sorted(df["con_code"].dropna().unique().tolist())

    def constituents(self, trade_date: str) -> list[str]:
        """Symbols that belong to the pool on a specific trade date."""
        store = IndexWeightStore()
        df = store.load(self.index_code)
        if df.empty:
            return []
        ts = pd.Timestamp(trade_date)
        target = df[df["trade_date"] <= ts.strftime("%Y%m%d")]
        if target.empty:
            return []
        latest = target["trade_date"].max()
        return sorted(target[target["trade_date"] == latest]["con_code"].unique().tolist())


def pool_names() -> list[str]:
    """Return supported pool names."""
    return list(POOL_MAP.keys())
