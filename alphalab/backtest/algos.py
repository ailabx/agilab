"""Composable strategy building blocks inspired by bt/algos.

A strategy is built as a list of Algo objects that are executed in order.
Each Algo receives an :class:`AlgoTarget` and returns ``True`` to continue
or ``False`` to stop the chain for the current date.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd


if TYPE_CHECKING:
    from .context import Context
    from .data_portal import BacktestDataPortal


class AlgoTarget:
    """Bridge between the bt-style Algo chain and Alphalab's engine.

    ``AlgoTarget`` exposes a bt-like API while delegating to the Alphalab
    :class:`Context` and :class:`BacktestDataPortal`.
    """

    def __init__(self, context: "Context", data: "BacktestDataPortal") -> None:
        self.context = context
        self.data = data
        self.selected: list[str] = []
        self.weights: dict[str, float] = {}
        self._state: dict[str, Any] = {}

    @property
    def now(self) -> pd.Timestamp:
        return self.context.current_date

    @property
    def cash(self) -> float:
        return self.context.portfolio.cash

    @property
    def positions(self) -> dict:
        return self.context.portfolio.positions

    @property
    def value(self) -> float:
        return self.context.portfolio_value()

    def get_state(self, key: str, default: Any = None) -> Any:
        return self._state.get(key, default)

    def set_state(self, key: str, value: Any) -> None:
        self._state[key] = value

    def order_target_percent_many(self, weights: dict[str, float]) -> None:
        self.context.order_target_percent_many(weights)


class Algo:
    """Base class for a strategy building block."""

    def __call__(self, target: AlgoTarget) -> bool:
        return True


class SelectAll(Algo):
    """Select all symbols with valid price data on the current date."""

    def __call__(self, target: AlgoTarget) -> bool:
        symbols = target.data.available_symbols
        prices = target.data.current(symbols, field="close")
        target.selected = prices.dropna().index.tolist()
        return len(target.selected) > 0


class SelectWhere(Algo):
    """Select symbols where a boolean signal is True on the current date.

    ``signal`` can be a pandas Series indexed by (date, symbol) or a
    DataFrame (date x symbol). True values select the corresponding symbols.
    """

    def __init__(self, signal: pd.Series | pd.DataFrame) -> None:
        self.signal = signal

    def __call__(self, target: AlgoTarget) -> bool:
        date = target.now
        if isinstance(self.signal, pd.DataFrame):
            if date not in self.signal.index:
                return False
            row = self.signal.loc[date]
            selected = row[row].index.tolist()
        else:
            if date not in self.signal.index.get_level_values(0):
                return False
            selected = self.signal.loc[date][lambda s: s].index.tolist()
        target.selected = selected
        return len(target.selected) > 0


class SelectMomentum(Algo):
    """Select top ``n`` symbols by lookback-day momentum."""

    def __init__(self, lookback: int = 20, top_n: int = 30, ascending: bool = False) -> None:
        self.lookback = lookback
        self.top_n = top_n
        self.ascending = ascending

    def __call__(self, target: AlgoTarget) -> bool:
        close = target.data.get_factor("close")
        if close.empty:
            return False
        momentum = close / close.shift(self.lookback) - 1.0
        date = target.now
        if date not in momentum.index:
            return False
        selected = momentum.loc[date].dropna().sort_values(ascending=self.ascending).head(self.top_n).index.tolist()
        target.selected = selected
        return len(target.selected) > 0


class WeighEqually(Algo):
    """Assign equal weights to the currently selected symbols."""

    def __call__(self, target: AlgoTarget) -> bool:
        selected = getattr(target, "selected", [])
        if not selected:
            return False
        target.weights = {s: 1.0 / len(selected) for s in selected}
        return True


class WeighTarget(Algo):
    """Assign weights from a pre-computed dict or DataFrame.

    If a DataFrame is supplied, the row for the current date is used.
    Only symbols in ``target.selected`` receive weights if ``selected_only``
    is True (default False).
    """

    def __init__(self, weights: dict[str, float] | pd.DataFrame, selected_only: bool = False) -> None:
        self.weights = weights
        self.selected_only = selected_only

    def __call__(self, target: AlgoTarget) -> bool:
        date = target.now
        if isinstance(self.weights, pd.DataFrame):
            if date not in self.weights.index:
                return False
            weights = self.weights.loc[date].to_dict()
        else:
            weights = self.weights.copy()

        if self.selected_only:
            selected = set(target.selected)
            weights = {s: w for s, w in weights.items() if s in selected}

        if not weights:
            return False
        target.weights = weights
        return True


class Rebalance(Algo):
    """Execute target-percent orders for the current target weights."""

    def __call__(self, target: AlgoTarget) -> bool:
        weights = getattr(target, "weights", {})
        if not weights:
            return False
        target.order_target_percent_many(weights)
        return True


class RunMonthly(Algo):
    """Only continue on the first trading day of each month."""

    def __init__(self) -> None:
        self._last_month = None

    def __call__(self, target: AlgoTarget) -> bool:
        date = target.now
        current_month = (date.year, date.month)
        if self._last_month == current_month:
            return False
        self._last_month = current_month
        return True


class RunQuarterly(Algo):
    """Only continue on the first trading day of each quarter."""

    def __call__(self, target: AlgoTarget) -> bool:
        state = target.get_state("run_quarterly_last")
        current = (target.now.year, target.now.quarter)
        if state == current:
            return False
        target.set_state("run_quarterly_last", current)
        return True


class PrintWeights(Algo):
    """Debug Algo that prints current target weights."""

    def __call__(self, target: AlgoTarget) -> bool:
        print(f"[{target.now.date()}] selected={target.selected} weights={target.weights}")
        return True
