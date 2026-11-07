"""Logical / conditional operators."""

from __future__ import annotations

import pandas as pd

from alphalab.factor.expression.registry import register_op


@register_op
def IF(condition: pd.DataFrame, then_branch: pd.DataFrame, else_branch: pd.DataFrame) -> pd.DataFrame:
    return then_branch.where(condition.astype(bool), else_branch)


@register_op
def IFF(condition: pd.DataFrame, then_branch: pd.DataFrame, else_branch: pd.DataFrame) -> pd.DataFrame:
    return IF(condition, then_branch, else_branch)


@register_op
def EVERY(condition: pd.DataFrame, n: int) -> pd.DataFrame:
    return condition.rolling(window=int(n)).min()


@register_op
def EXIST(condition: pd.DataFrame, n: int) -> pd.DataFrame:
    return condition.rolling(window=int(n)).max()


@register_op
def COUNT(condition: pd.DataFrame, n: int) -> pd.DataFrame:
    return condition.rolling(window=int(n)).sum()
