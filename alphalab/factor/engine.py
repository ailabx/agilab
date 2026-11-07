"""Factor engine: build factor matrices from wide-format price data."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


class FactorExpression:
    """A callable factor expression that operates on a wide DataFrame panel."""

    def __init__(self, name: str, func: Callable[[dict[str, pd.DataFrame]], pd.DataFrame]):
        self.name = name
        self.func = func

    def __call__(self, fields: dict[str, pd.DataFrame]) -> pd.DataFrame:
        return self.func(fields)


class FactorEngine:
    """Compute factors from wide-format OHLCV panels.

    Required input fields (all date x symbol DataFrames):
        - open, high, low, close, vol, amount
    """

    def __init__(self, fields: dict[str, pd.DataFrame]):
        self.fields = fields
        self._validate_fields()

    def _validate_fields(self) -> None:
        required = {"open", "high", "low", "close", "vol", "amount"}
        missing = required - set(self.fields.keys())
        if missing:
            raise ValueError(f"Missing fields: {missing}")
        # Ensure consistent shape
        shapes = {k: v.shape for k, v in self.fields.items()}
        if len(set(shapes.values())) > 1:
            # align by outer join
            index = self.fields["close"].index
            columns = self.fields["close"].columns
            aligned: dict[str, pd.DataFrame] = {}
            for key, df in self.fields.items():
                aligned[key] = df.reindex(index=index, columns=columns)
            self.fields = aligned

    # ------------------------------------------------------------------ #
    # Built-in factor expressions
    # ------------------------------------------------------------------ #
    def momentum(self, window: int = 20) -> pd.DataFrame:
        """Return factor = close / close.shift(window) - 1."""
        close = self.fields["close"]
        return close / close.shift(window) - 1.0

    def reversal(self, window: int = 20) -> pd.DataFrame:
        """Short-term reversal: -momentum over short window."""
        return -self.momentum(window=window)

    def volume_momentum(self, window: int = 20) -> pd.DataFrame:
        """Volume moving average ratio."""
        vol = self.fields["vol"]
        ma = vol.rolling(window=window, min_periods=1).mean()
        return vol / ma

    def volatility(self, window: int = 20) -> pd.DataFrame:
        """Realized volatility (std of log returns)."""
        close = self.fields["close"]
        log_ret = np.log(close / close.shift(1))
        return log_ret.rolling(window=window, min_periods=1).std()

    def ma_ratio(self, fast: int = 10, slow: int = 30) -> pd.DataFrame:
        """Moving average ratio: MA(fast) / MA(slow)."""
        close = self.fields["close"]
        ma_fast = close.rolling(window=fast, min_periods=1).mean()
        ma_slow = close.rolling(window=slow, min_periods=1).mean()
        return ma_fast / ma_slow

    def rsi(self, window: int = 14) -> pd.DataFrame:
        """Relative Strength Index."""
        close = self.fields["close"]
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100.0 - (100.0 / (1.0 + rs))

    # ------------------------------------------------------------------ #
    # Generic helpers
    # ------------------------------------------------------------------ #
    def apply_expression(self, expr: FactorExpression) -> pd.DataFrame:
        return expr(self.fields)


def zscore_cross_sectional(factor: pd.DataFrame) -> pd.DataFrame:
    """Z-score across symbols for each date."""
    mean = factor.mean(axis=1)
    std = factor.std(axis=1)
    return factor.sub(mean, axis=0).div(std, axis=0)


def winsorize(factor: pd.DataFrame, lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    """Winsorize factor values per date."""
    low = factor.quantile(lower, axis=1)
    high = factor.quantile(upper, axis=1)
    clipped = factor.clip(low, high, axis=0)
    return clipped


def neutralize_industry(
    factor: pd.DataFrame,
    industry_map: pd.Series,
) -> pd.DataFrame:
    """Subtract industry mean for each date.

    industry_map: symbol -> industry name
    """
    df = factor.T
    df["industry"] = industry_map
    industry_mean = df.groupby("industry")[factor.index].transform("mean")
    neutral = (df[factor.index] - industry_mean).T
    return neutral
