"""Technical indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphalab.factor.expression.registry import register_op


@register_op
def RSI(x: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    delta = x.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=int(n)).mean()
    avg_loss = loss.rolling(window=int(n)).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


@register_op
def MACD(x: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = x.ewm(span=int(fast), adjust=False).mean()
    ema_slow = x.ewm(span=int(slow), adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=int(signal), adjust=False).mean()
    return dif - dea


@register_op
def MACD_DIF(x: pd.DataFrame, fast: int = 12, slow: int = 26) -> pd.DataFrame:
    ema_fast = x.ewm(span=int(fast), adjust=False).mean()
    ema_slow = x.ewm(span=int(slow), adjust=False).mean()
    return ema_fast - ema_slow


@register_op
def MACD_DEA(x: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = x.ewm(span=int(fast), adjust=False).mean()
    ema_slow = x.ewm(span=int(slow), adjust=False).mean()
    dif = ema_fast - ema_slow
    return dif.ewm(span=int(signal), adjust=False).mean()


@register_op
def BOLL_UPPER(x: pd.DataFrame, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    ma = x.rolling(window=int(n)).mean()
    std = x.rolling(window=int(n)).std()
    return ma + float(k) * std


@register_op
def BOLL_LOWER(x: pd.DataFrame, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    ma = x.rolling(window=int(n)).mean()
    std = x.rolling(window=int(n)).std()
    return ma - float(k) * std


@register_op
def BOLL_MID(x: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    return x.rolling(window=int(n)).mean()


@register_op
def ATR(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=0).groupby(level=0).max()
    return tr.rolling(window=int(n)).mean()


@register_op
def CCI(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    tp = (high + low + close) / 3.0
    ma = tp.rolling(window=int(n)).mean()
    md = tp.rolling(window=int(n)).apply(lambda s: np.mean(np.abs(s - np.mean(s))), raw=True)
    return (tp - ma) / (0.015 * md)


@register_op
def WR(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    hh = high.rolling(window=int(n)).max()
    ll = low.rolling(window=int(n)).min()
    return (hh - close) / (hh - ll) * -100.0


@register_op
def BIAS(x: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    ma = x.rolling(window=int(n)).mean()
    return (x - ma) / ma


@register_op
def OBV(close: pd.DataFrame, vol: pd.DataFrame) -> pd.DataFrame:
    diff = close.diff()
    obv = vol.where(diff > 0, -vol).where(diff != 0, 0)
    return obv.cumsum()


@register_op
def MFI(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, vol: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    tp = (high + low + close) / 3.0
    gain = tp.where(tp > tp.shift(1), 0)
    loss = tp.where(tp < tp.shift(1), 0)
    mf_gain = gain * vol
    mf_loss = loss * vol
    total_gain = mf_gain.rolling(window=int(n)).sum()
    total_loss = mf_loss.rolling(window=int(n)).sum()
    return 100.0 - (100.0 / (1.0 + total_gain / total_loss.replace(0, np.nan)))


@register_op
def KDJ_K(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, n: int = 9) -> pd.DataFrame:
    rsv = (close - low.rolling(window=int(n)).min()) / (high.rolling(window=int(n)).max() - low.rolling(window=int(n)).min())
    return rsv.ewm(com=2, adjust=False).mean()


@register_op
def KDJ_D(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, n: int = 9) -> pd.DataFrame:
    k = KDJ_K(high, low, close, n)
    return k.ewm(com=2, adjust=False).mean()


@register_op
def KDJ_J(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, n: int = 9) -> pd.DataFrame:
    k = KDJ_K(high, low, close, n)
    d = KDJ_D(high, low, close, n)
    return 3.0 * k - 2.0 * d
