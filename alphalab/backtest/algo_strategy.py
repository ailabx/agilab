"""Adapter to run bt-style Algo chains inside Alphalab's backtest engine."""

from __future__ import annotations

from .algos import Algo, AlgoTarget
from .strategy import Strategy


class AlgoStrategy(Strategy):
    """Runs a list of Algo objects as a single strategy.

    Example::

        strategy = AlgoStrategy([
            RunMonthly(),
            SelectAll(),
            WeighEqually(),
            Rebalance(),
        ])
    """

    def __init__(self, algos: list[Algo]) -> None:
        self.algos = algos

    def handle_data(self, context, data) -> None:
        target = AlgoTarget(context, data)
        for algo in self.algos:
            if not algo(target):
                break
