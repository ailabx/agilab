"""Backtest data portal — in-memory wide-format price and factor store."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


try:
    from config import DATALAKE_DIR
except ImportError:  # pragma: no cover
    from alphalab.config import DATALAKE_DIR

from alphalab.data.portal import DataPortal
from alphalab.factor.expression.core import FactorExpression


class BacktestDataPortal:
    """In-memory point-in-time data portal used by the backtest engine.

    Loads the full price panel for the requested range once, then serves
    history/current/factor queries from memory. Factors are evaluated lazily
    and cached.
    """

    def __init__(
        self,
        datalake_dir: str | Path | None = None,
        adjusted: str = "qfq",
        start: str | None = None,
        end: str | None = None,
    ) -> None:
        self._datalake = Path(datalake_dir or DATALAKE_DIR)
        self._portal = DataPortal(self._datalake)
        self._adjusted = adjusted
        self._current_date: pd.Timestamp | None = None

        self._start = start
        self._end = end
        self._panel: dict[str, pd.DataFrame] | None = None
        self._dates: pd.DatetimeIndex | None = None
        self._symbols: list[str] | None = None
        self._factor_cache: dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # Date management
    # ------------------------------------------------------------------
    def set_date(self, date: str | pd.Timestamp) -> None:
        self._current_date = pd.Timestamp(date)

    @property
    def current_date(self) -> pd.Timestamp:
        if self._current_date is None:
            raise RuntimeError("BacktestDataPortal date not set")
        return self._current_date

    @property
    def available_symbols(self) -> list[str]:
        self._ensure_loaded()
        return self._panel["close"].columns.tolist()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load(self, symbols: list[str] | None = None) -> None:
        """Load full price panel into memory.

        If ``symbols`` is None, load all symbols present in ``daily``.
        Can be called multiple times to switch the loaded universe.
        """
        if symbols is None:
            symbols = self._all_symbols()

        self._symbols = list(symbols)
        start = self._start or "20100101"
        end = self._end or "20991231"

        fields = ["open", "high", "low", "close", "vol", "amount"]
        df = self._portal.get_prices(
            self._symbols,
            start=start,
            end=end,
            fields=fields,
            adjusted=self._adjusted,
        )

        if df.empty:
            self._panel = {f: pd.DataFrame() for f in fields}
            self._dates = pd.DatetimeIndex([], name="trade_date")
            return

        df["trade_date"] = pd.to_datetime(df["trade_date"])
        for f in fields:
            if f not in df.columns:
                df[f] = float("nan")

        self._panel = {
            f: df.pivot(index="trade_date", columns="symbol", values=f).sort_index()
            for f in fields
        }
        self._dates = self._panel["close"].index
        # Invalidate factor cache because symbol universe changed
        self._factor_cache.clear()

    def _all_symbols(self) -> list[str]:
        daily_path = self._datalake / "daily" / "stock" / "**" / "data.parquet"
        query = f"""
            SELECT DISTINCT symbol
            FROM read_parquet('{daily_path.as_posix()}', hive_partitioning=true)
        """
        df = self._portal._con.execute(query).fetchdf()
        return df["symbol"].tolist() if not df.empty else []

    def _ensure_loaded(self) -> None:
        if self._panel is None:
            self.load()

    # ------------------------------------------------------------------
    # Strategy helpers
    # ------------------------------------------------------------------
    def history(
        self,
        symbols: str | list[str],
        field: str,
        window: int,
    ) -> pd.DataFrame | pd.Series:
        """Return the last ``window`` trade-date values of ``field``."""
        self._ensure_loaded()
        single_symbol = isinstance(symbols, str)
        symbols = [symbols] if single_symbol else list(symbols)

        panel = self._panel.get(field, pd.DataFrame())
        if panel.empty:
            wide = pd.DataFrame(index=pd.DatetimeIndex([], name="trade_date"), columns=symbols)
        else:
            wide = panel[symbols].sort_index()

        idx = wide.index.get_indexer([self.current_date], method="ffill")[0]
        if idx < 0:
            return wide.iloc[:0]
        tail = wide.iloc[max(0, idx - window + 1) : idx + 1]
        return tail[symbols[0]] if single_symbol else tail

    def current(self, symbols: list[str], field: str = "close") -> pd.Series:
        """Latest available value of ``field`` for ``symbols`` as of current date."""
        self._ensure_loaded()
        panel = self._panel.get(field, pd.DataFrame())
        if panel.empty:
            return pd.Series(dtype=float)
        available = symbols if all(s in panel.columns for s in symbols) else [s for s in symbols if s in panel.columns]
        if not available:
            return pd.Series(dtype=float)
        row = panel[available].loc[: self.current_date].iloc[-1]
        return row

    def get_prices(
        self,
        symbols: list[str],
        start: str,
        end: str,
        fields: list[str] | None = None,
    ) -> pd.DataFrame:
        """Direct access to underlying price data (uses on-demand portal read)."""
        return self._portal.get_prices(
            symbols,
            start=start,
            end=end,
            fields=fields,
            adjusted=self._adjusted,
        )

    def get_fina(
        self,
        start: str,
        end: str,
        fields: list[str] | None = None,
    ) -> pd.DataFrame:
        """Read fina_indicator rows with ann_date in [start, end]."""
        fina_path = self._datalake / "fina_indicator" / "stock" / "**" / "data.parquet"
        field_str = ", ".join(fields or ["*"])
        query = f"""
            SELECT {field_str}
            FROM read_parquet('{fina_path.as_posix()}', hive_partitioning=true)
            WHERE ann_date >= '{start}' AND ann_date <= '{end}'
        """
        return self._portal._con.execute(query).fetchdf()

    # ------------------------------------------------------------------
    # Factor computation
    # ------------------------------------------------------------------
    def get_factor(self, expression: str) -> pd.DataFrame:
        """Evaluate a factor expression across the full in-memory panel.

        Results are cached by expression string.
        """
        self._ensure_loaded()
        if expression in self._factor_cache:
            return self._factor_cache[expression]

        expr = FactorExpression(expression)
        fields = {
            "open": self._panel["open"],
            "high": self._panel["high"],
            "low": self._panel["low"],
            "close": self._panel["close"],
            "vol": self._panel["vol"],
            "amount": self._panel["amount"],
        }
        result = expr.evaluate(fields)
        self._factor_cache[expression] = result
        return result

    def get_factor_value(self, expression: str, symbols: list[str] | None = None) -> pd.Series:
        """Return factor values for ``symbols`` as of current date."""
        df = self.get_factor(expression)
        if df.empty:
            return pd.Series(dtype=float)
        symbols = symbols or df.columns.tolist()
        available = [s for s in symbols if s in df.columns]
        return df[available].loc[: self.current_date].iloc[-1]
