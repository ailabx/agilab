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
        has_daily = any(self._datalake.glob("daily/stock/year=*/data.parquet"))
        if has_daily:
            con = self._con()
            try:
                df = con.execute(
                    f"SELECT COUNT(DISTINCT symbol) AS cnt FROM read_parquet('{daily_path.as_posix()}', hive_partitioning=true)"
                ).fetchdf()
                with_data = int(df["cnt"].iloc[0])
            finally:
                con.close()

        latest_date = None
        latest_count = None
        if has_daily:
            con = self._con()
            try:
                df = con.execute(
                    f"""
                    SELECT trade_date, COUNT(DISTINCT symbol) AS cnt
                    FROM read_parquet('{daily_path.as_posix()}', hive_partitioning=true)
                    GROUP BY trade_date
                    ORDER BY trade_date DESC
                    LIMIT 1
                    """
                ).fetchdf()
                if not df.empty:
                    latest_date = df["trade_date"].iloc[0]
                    latest_count = int(df["cnt"].iloc[0])
                    if isinstance(latest_date, pd.Timestamp):
                        latest_date = latest_date.strftime("%Y-%m-%d")
                    elif latest_date is not None:
                        latest_date = str(latest_date)
            finally:
                con.close()

        return {
            "total_symbols": total,
            "symbols_with_data": with_data,
            "latest_date": latest_date,
            "latest_count": latest_count,
        }

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
        sort_by: str = "ytd_return",
        ascending: bool = False,
    ) -> pd.DataFrame:
        """Return top N symbols by selected return horizon as of the latest date."""
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
        end_ts = pd.Timestamp(latest_date)
        # Load enough history for YTD + 3 months.
        start_ts = pd.Timestamp(f"{end_ts.year}0101") - pd.Timedelta(days=100)
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

        # Pivot to wide format for vectorised cross-sectional calculation.
        price_df = df.pivot(index="trade_date", columns="symbol", values="close")
        # Latest price per symbol (respecting each symbol's own last trading day).
        latest_idx = price_df.apply(lambda s: s.last_valid_index())
        latest_close = price_df.apply(lambda s: s.dropna().iloc[-1])

        # YTD reference: first trading day of the current year for each symbol.
        year_start = pd.Timestamp(f"{end_ts.year}0101")
        ytd_ref = price_df.apply(lambda s: s.loc[s.index >= year_start].dropna().iloc[0] if s.loc[s.index >= year_start].dropna().size else np.nan)

        # Use each symbol's own latest close for returns.
        ret_1w = price_df.apply(lambda s: s / s.shift(1).iloc[-1] - 1.0 if len(s.dropna()) >= 6 else np.nan)
        ret_1m = price_df.apply(lambda s: s / s.shift(5).iloc[-1] - 1.0 if len(s.dropna()) >= 22 else np.nan)
        ret_3m = price_df.apply(lambda s: s / s.shift(21).iloc[-1] - 1.0 if len(s.dropna()) >= 64 else np.nan)
        # Volatility: rolling 20-day std of log returns, take last value.
        vol_20 = price_df.apply(lambda s: np.log(s / s.shift(1)).rolling(window=20, min_periods=20).std().iloc[-1] if len(s.dropna()) >= 20 else np.nan)

        result = pd.DataFrame({
            "latest_date": latest_idx,
            "latest_close": latest_close,
            "ytd_return": latest_close / ytd_ref - 1.0,
            "return_1w": latest_close / price_df.apply(lambda s: s.dropna().iloc[-6] if len(s.dropna()) >= 6 else np.nan) - 1.0,
            "return_1m": latest_close / price_df.apply(lambda s: s.dropna().iloc[-22] if len(s.dropna()) >= 22 else np.nan) - 1.0,
            "return_3m": latest_close / price_df.apply(lambda s: s.dropna().iloc[-64] if len(s.dropna()) >= 64 else np.nan) - 1.0,
            "volatility_20": vol_20,
        }).reset_index()

        # Sort according to user selection
        if sort_by in result.columns:
            result = result.sort_values(sort_by, ascending=ascending)
        result = result.head(n)

        names = self._load_names()
        if not names.empty:
            result = result.merge(names, on="symbol", how="left")

        result["name"] = result["name"].fillna("-")
        result["industry"] = result["industry"].fillna("-")
        result["area"] = result["area"].fillna("-")
        result["latest_date"] = pd.to_datetime(result["latest_date"]).dt.strftime("%Y-%m-%d")
        return result[["symbol", "name", "industry", "area", "latest_date", "latest_close", "ytd_return", "return_1w", "return_1m", "return_3m", "volatility_20"]].reset_index(drop=True)

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

        names = self._load_names()
        info = names[names["symbol"] == symbol]
        if not info.empty:
            name = info["name"].iloc[0]
            industry = info["industry"].iloc[0]
            area = info["area"].iloc[0]
        else:
            name = industry = area = "-"

        return {
            "symbol": symbol,
            "name": name if name else "-",
            "industry": industry if industry else "-",
            "area": area if area else "-",
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
