"""Extended portal helpers for dashboard queries."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

try:
    from config import DATALAKE_DIR
except ImportError:  # pragma: no cover
    from alphalab.config import DATALAKE_DIR


class DashboardPortal:
    """Read-only portal for dashboard needs."""

    def __init__(self, datalake_dir: str | Path | None = None) -> None:
        self._datalake = Path(datalake_dir or DATALAKE_DIR)

    def _con(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect()

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    def summary(self) -> dict:
        total = 0
        basic_path = self._datalake / "basic" / "stock_basic.parquet"
        if basic_path.exists():
            df = pd.read_parquet(basic_path)
            total = int(len(df))

        with_data = 0
        daily_path = self._datalake / "daily" / "stock" / "**" / "data.parquet"
        if any(self._datalake.glob("daily/stock/year=*/data.parquet")):
            con = self._con()
            try:
                df = con.execute(
                    f"SELECT COUNT(DISTINCT symbol) AS cnt FROM read_parquet('{daily_path.as_posix()}', hive_partitioning=true)"
                ).fetchdf()
                with_data = int(df["cnt"].iloc[0])
            finally:
                con.close()

        return {"total_symbols": total, "symbols_with_data": with_data}

    # ------------------------------------------------------------------ #
    # Momentum
    # ------------------------------------------------------------------ #
    def _load_names(self) -> pd.DataFrame:
        basic_path = self._datalake / "basic" / "stock_basic.parquet"
        if not basic_path.exists():
            return pd.DataFrame(columns=["symbol", "name", "industry", "area"])
        df = pd.read_parquet(basic_path, columns=["symbol", "name", "industry", "area"])
        return df

    def momentum_top(
        self,
        n: int = 30,
        lookback: int = 20,
        sort_by: str = "momentum",
        ascending: bool = True,
    ) -> pd.DataFrame:
        """Return top N symbols by lookback-day return or volatility as of the latest date."""
        daily_path = self._datalake / "daily" / "stock" / "**" / "data.parquet"

        # Determine latest date with data across the universe.
        con = self._con()
        try:
            latest_date_df = con.execute(
                f"""
                SELECT MAX(trade_date) AS latest_date
                FROM read_parquet('{daily_path.as_posix()}', hive_partitioning=true)
                """
            ).fetchdf()
        finally:
            con.close()

        if latest_date_df.empty or latest_date_df["latest_date"].isna().all():
            return pd.DataFrame()

        latest_date = latest_date_df["latest_date"].iloc[0]
        # Load a window large enough to cover the lookback (~250 trade days per year).
        end_ts = pd.Timestamp(latest_date)
        start_ts = end_ts - pd.Timedelta(days=lookback * 2 + 10)
        start_str = start_ts.strftime("%Y%m%d")
        end_str = end_ts.strftime("%Y%m%d")

        con = self._con()
        try:
            df = con.execute(
                f"""
                SELECT symbol, trade_date, close
                FROM read_parquet('{daily_path.as_posix()}', hive_partitioning=true)
                WHERE trade_date BETWEEN '{start_str}' AND '{end_str}'
                ORDER BY symbol, trade_date
                """
            ).fetchdf()
        finally:
            con.close()

        if df.empty:
            return pd.DataFrame()

        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df.sort_values(["symbol", "trade_date"])

        latest = df.groupby("symbol").tail(1)[["symbol", "trade_date", "close"]].copy()

        def _lag20(g):
            g = g.sort_values("trade_date").set_index("trade_date").resample("B").last()
            g["close_lag"] = g["close"].shift(lookback)
            g["volatility_20"] = np.log(g["close"] / g["close"].shift(1)).rolling(window=lookback, min_periods=lookback).std()
            return g

        lag = df.groupby("symbol").apply(_lag20).reset_index()
        lag = lag[lag["close_lag"].notna()][["symbol", "trade_date", "close_lag", "volatility_20"]].copy()
        latest = latest.merge(lag, on=["symbol", "trade_date"], how="inner")
        latest["momentum"] = latest["close"] / latest["close_lag"] - 1.0
        latest = latest[latest["momentum"].notna()]
        latest["trade_date"] = pd.to_datetime(latest["trade_date"]).dt.strftime("%Y-%m-%d")

        # Sort according to user selection
        latest = latest.sort_values(sort_by, ascending=ascending).head(n)

        names = self._load_names()
        if not names.empty:
            latest = latest.merge(names, on="symbol", how="left")

        latest["name"] = latest["name"].fillna("-")
        latest["industry"] = latest["industry"].fillna("-")
        latest["area"] = latest["area"].fillna("-")
        return latest[["symbol", "name", "industry", "area", "trade_date", "close", "momentum", "volatility_20"]].reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Single stock detail
    # ------------------------------------------------------------------ #
    def stock_detail(
        self,
        symbol: str,
        days: int | None = None,
        adjusted: str = "qfq",
    ) -> dict:
        from alphalab.data.portal import DataPortal

        portal = DataPortal(self._datalake)
        df = portal.get_prices([symbol], start="20100101", end="20991231", adjusted=adjusted)  # type: ignore
        if df.empty:
            return {"symbol": symbol, "error": "no data"}

        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df.sort_values("trade_date")
        chart_df = df.tail(days) if days else df

        latest = chart_df.iloc[-1]
        prev = chart_df.iloc[-2] if len(chart_df) > 1 else latest
        momentum_20 = latest["close"] / chart_df.iloc[-20]["close"] - 1.0 if len(chart_df) >= 20 else None

        return {
            "symbol": symbol,
            "latest_date": latest["trade_date"].strftime("%Y-%m-%d"),
            "latest_close": float(latest["close"]),
            "day_change": float(latest["close"] - prev["close"]),
            "day_pct": float(latest["close"] / prev["close"] - 1.0) if prev["close"] else None,
            "momentum_20": float(momentum_20) if momentum_20 is not None else None,
            "all_dates": df["trade_date"].dt.strftime("%Y-%m-%d").tolist(),
            "all_closes": df["close"].astype(float).tolist(),
            "all_volumes": df["vol"].astype(float).tolist(),
            "dates": chart_df["trade_date"].dt.strftime("%Y-%m-%d").tolist(),
            "closes": chart_df["close"].astype(float).tolist(),
            "volumes": chart_df["vol"].astype(float).tolist(),
            "fina": self._load_fina_summary(symbol),
        }

    def _load_fina_summary(self, symbol: str) -> list[dict]:
        """Load the most recent quarterly fina_indicator records."""
        fina_path = self._datalake / "fina_indicator" / "stock" / "**" / "data.parquet"
        if not any(self._datalake.glob("fina_indicator/stock/year=*/data.parquet")):
            return []
        con = self._con()
        try:
            df = con.execute(
                f"""
                SELECT *
                FROM read_parquet('{fina_path.as_posix()}', hive_partitioning=true)
                WHERE symbol = '{symbol}'
                ORDER BY ann_date DESC, end_date DESC
                LIMIT 8
                """
            ).fetchdf()
        finally:
            con.close()

        if df.empty:
            return []

        core_cols = ["ann_date", "end_date", "eps", "dt_eps", "roe", "roa", "grossprofit_margin",
                     "netprofit_margin", "debt_to_assets", "current_ratio", "quick_ratio",
                     "op_yoy", "netprofit_yoy", "or_yoy", "total_revenue_ps", "revenue_ps",
                     "bps", "cfps", "op_income", "ebitda", "ebit", "total_profit",
                     "net_profit", "profit_dedt"]
        keep = [c for c in core_cols if c in df.columns]
        df = df[keep].copy()
        df = df.sort_values("end_date", ascending=False)
        df = df.replace({float("nan"): None, float("inf"): None, float("-inf"): None})
        return df.to_dict(orient="records")
