"""Alphalab backtest engine."""

from __future__ import annotations

from .broker import ChinaFeeModel, SimulationBroker
from .calendar import TradingCalendar
from .context import Context
from .data_portal import BacktestDataPortal
from .engine import BacktestEngine
from .portfolio import Portfolio
from .result import BacktestResult
from .algo_strategy import AlgoStrategy
from .algos import (
    Algo,
    AlgoTarget,
    PrintWeights,
    Rebalance,
    RunMonthly,
    RunQuarterly,
    SelectAll,
    SelectMomentum,
    SelectWhere,
    WeighEqually,
    WeighTarget,
)
from .strategy import Strategy

__all__ = [
    "Algo",
    "AlgoStrategy",
    "AlgoTarget",
    "BacktestEngine",
    "BacktestDataPortal",
    "BacktestResult",
    "ChinaFeeModel",
    "Context",
    "Portfolio",
    "PrintWeights",
    "Rebalance",
    "RunMonthly",
    "RunQuarterly",
    "SelectAll",
    "SelectMomentum",
    "SelectWhere",
    "SimulationBroker",
    "Strategy",
    "TradingCalendar",
    "WeighEqually",
    "WeighTarget",
]


def run_algorithm(
    *,
    start: str,
    end: str,
    strategy: Strategy,
    capital_base: float = 1_000_000.0,
    execution: str = "close",
    adjusted: str = "qfq",
    datalake_dir: str | None = None,
) -> BacktestResult:
    """Convenience entry point."""
    engine = BacktestEngine(
        start=start,
        end=end,
        capital_base=capital_base,
        strategy=strategy,
        execution=execution,
        adjusted=adjusted,
        datalake_dir=datalake_dir,
    )
    return engine.run()