"""Element-wise arithmetic operators."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphalab.factor.expression.registry import register_op


@register_op
def ABS(x: pd.DataFrame) -> pd.DataFrame:
    return x.abs()


@register_op
def LOG(x: pd.DataFrame) -> pd.DataFrame:
    return np.log(x.where(x > 0))


@register_op
def LOGABS(x: pd.DataFrame) -> pd.DataFrame:
    return np.log(x.abs().where(x.abs() > 0))


@register_op
def SIGN(x: pd.DataFrame) -> pd.DataFrame:
    return np.sign(x)


@register_op
def POWER(x: pd.DataFrame, n: float) -> pd.DataFrame:
    return x ** float(n)


@register_op
def SIGNEDPOWER(x: pd.DataFrame, n: float) -> pd.DataFrame:
    return np.sign(x) * x.abs() ** float(n)


@register_op
def SQRT(x: pd.DataFrame) -> pd.DataFrame:
    return np.sqrt(x.where(x >= 0))


@register_op
def EXP(x: pd.DataFrame) -> pd.DataFrame:
    return np.exp(x)


@register_op
def RD(x: pd.DataFrame, n: int = 2) -> pd.DataFrame:
    return x.round(int(n))


@register_op
def AS_FLOAT(x: pd.DataFrame) -> pd.DataFrame:
    return x.astype(float)
