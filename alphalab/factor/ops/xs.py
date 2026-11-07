"""Cross-sectional operators."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphalab.factor.expression.registry import register_op


@register_op
def RANK(x: pd.DataFrame) -> pd.DataFrame:
    return x.rank(axis=1, pct=True, method="average", na_option="keep")


@register_op
def ZSCORE(x: pd.DataFrame) -> pd.DataFrame:
    mean = x.mean(axis=1)
    std = x.std(axis=1, ddof=0).replace(0, np.nan)
    return x.sub(mean, axis=0).div(std, axis=0)


@register_op
def SCALE(x: pd.DataFrame) -> pd.DataFrame:
    minimum = x.min(axis=1)
    maximum = x.max(axis=1)
    spread = (maximum - minimum).replace(0, np.nan)
    return x.sub(minimum, axis=0).div(spread, axis=0).mul(2.0).sub(1.0)


@register_op
def WINSORIZE(x: pd.DataFrame, lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    low = x.quantile(lower, axis=1)
    high = x.quantile(upper, axis=1)
    return x.clip(low, high, axis=0)
