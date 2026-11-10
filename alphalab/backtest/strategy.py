"""Strategy interface for the backtest engine."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .context import Context
    from .data_portal import BacktestDataPortal


class Strategy:
    """Base class for user strategies."""

    def initialize(self, context: "Context") -> None:
        """Called once before the first bar."""
        pass

    def handle_data(self, context: "Context", data: "BacktestDataPortal") -> None:
        """Called on every trade date."""
        raise NotImplementedError