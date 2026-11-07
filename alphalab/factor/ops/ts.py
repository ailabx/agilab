"""Time-series operators."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphalab.factor.expression.registry import register_op


@register_op
def MA(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(window=int(n), min_periods=int(n)).mean()


@register_op
def EMA(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.ewm(span=int(n), adjust=False).mean()


@register_op
def WMA(x: pd.DataFrame, n: int) -> pd.DataFrame:
    n = int(n)
    weights = np.arange(1, n + 1)
    return x.rolling(window=n).apply(lambda s: np.dot(s, weights) / weights.sum(), raw=True)


@register_op
def STD(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(window=int(n), min_periods=int(n)).std()


@register_op
def STDDEV(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return STD(x, n)


@register_op
def VAR(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(window=int(n), min_periods=int(n)).var()


@register_op
def SUM(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(window=int(n), min_periods=int(n)).sum()


@register_op
def PRODUCT(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(window=int(n), min_periods=int(n)).apply(lambda s: np.prod(s), raw=True)


@register_op
def MAX(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(window=int(n), min_periods=int(n)).max()


@register_op
def MIN(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(window=int(n), min_periods=int(n)).min()


@register_op
def MEDIAN(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(window=int(n), min_periods=int(n)).median()


@register_op
def DELAY(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.shift(int(n))


@register_op
def REF(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.shift(int(n))


@register_op
def DIFF(x: pd.DataFrame, n: int = 1) -> pd.DataFrame:
    return x.diff(int(n))


@register_op
def DELTA(x: pd.DataFrame, n: int = 1) -> pd.DataFrame:
    return x.diff(int(n))


@register_op
def ROC(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x / x.shift(int(n)) - 1.0


@register_op
def PCT_CHANGE(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x / x.shift(int(n)) - 1.0


@register_op
def RETURNS(x: pd.DataFrame, n: int = 1) -> pd.DataFrame:
    return x / x.shift(int(n)) - 1.0


@register_op
def TS_MAX(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(window=int(n), min_periods=int(n)).max()


@register_op
def TS_MIN(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(window=int(n), min_periods=int(n)).min()


@register_op
def TS_MEAN(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return MA(x, n)


@register_op
def TS_MEDIAN(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return MEDIAN(x, n)


@register_op
def TS_ARGMAX(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(window=int(n)).apply(lambda s: np.argmax(s), raw=True)


@register_op
def TS_ARGMIN(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(window=int(n)).apply(lambda s: np.argmin(s), raw=True)


@register_op
def TS_RANK(x: pd.DataFrame, n: int) -> pd.DataFrame:
    def _rank_window(s):
        ranked = pd.Series(s).rank(pct=True)
        return ranked.iloc[-1]
    return x.rolling(window=int(n)).apply(_rank_window, raw=True)


@register_op
def TS_ZSCORE(x: pd.DataFrame, n: int) -> pd.DataFrame:
    mean = x.rolling(window=int(n)).mean()
    std = x.rolling(window=int(n)).std().replace(0, np.nan)
    return (x - mean) / std


@register_op
def TS_SKEW(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(window=int(n)).skew()


@register_op
def TS_KURT(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(window=int(n)).kurt()


@register_op
def TS_MAD(x: pd.DataFrame, n: int) -> pd.DataFrame:
    def _mad(s):
        m = np.median(s)
        return np.median(np.abs(s - m))
    return x.rolling(window=int(n)).apply(_mad, raw=True)


@register_op
def TS_REGRESSION(x: pd.DataFrame, y: pd.DataFrame, n: int) -> pd.DataFrame:
    """Rolling regression slope of x on y."""
    n = int(n)
    mean_x = x.rolling(window=n).mean()
    mean_y = y.rolling(window=n).mean()
    cov = ((x - mean_x) * (y - mean_y)).rolling(window=n).mean()
    var_x = ((x - mean_x) ** 2).rolling(window=n).mean()
    return cov / var_x.replace(0, np.nan)


@register_op
def CORR(x: pd.DataFrame, y: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(window=int(n)).corr(y)


@register_op
def COV(x: pd.DataFrame, y: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(window=int(n)).cov(y)


@register_op
def COVARIANCE(x: pd.DataFrame, y: pd.DataFrame, n: int) -> pd.DataFrame:
    return COV(x, y, n)


@register_op
def CORRELATION(x: pd.DataFrame, y: pd.DataFrame, n: int) -> pd.DataFrame:
    return CORR(x, y, n)
