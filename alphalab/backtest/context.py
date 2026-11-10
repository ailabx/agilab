"""Execution context exposed to strategies."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .broker import SimulationBroker
    from .data_portal import BacktestDataPortal
    from .portfolio import Portfolio


class Context:
    """Mutable namespace passed to strategy callbacks."""

    def __init__(self, portfolio: "Portfolio", broker: "SimulationBroker", data: "BacktestDataPortal") -> None:
        self.portfolio = portfolio
        self.broker = broker
        self.data = data
        self.current_date = None
        self.capital_base = portfolio.cash

    @property
    def cash(self) -> float:
        return self.portfolio.cash

    def order(self, symbol: str, amount: float) -> None:
        """Submit a quantity order (positive=buy, negative=sell)."""
        self.broker.submit_order(symbol, amount)

    def order_value(self, symbol: str, value: float) -> None:
        """Buy/sell ``value`` CNY worth of ``symbol``."""
        prices = self.data.current([symbol], field="close")
        price = prices.get(symbol)
        if price is None or price <= 0:
            return
        amount = value / price
        self.broker.submit_order(symbol, amount)

    def order_percent(self, symbol: str, percent: float) -> None:
        """Allocate ``percent`` of current portfolio value to ``symbol``."""
        target_value = self.portfolio_value() * percent
        self.order_value(symbol, target_value)

    def order_target(self, symbol: str, target_amount: float) -> None:
        """Rebalance position in ``symbol`` to ``target_amount`` shares."""
        self.broker.submit_target(symbol, target_amount)

    def order_target_value(self, symbol: str, target_value: float) -> None:
        """Rebalance position in ``symbol`` to ``target_value`` CNY."""
        prices = self.data.current([symbol], field="close")
        price = prices.get(symbol)
        if price is None or price <= 0:
            return
        self.broker.submit_target(symbol, target_value / price)

    def order_target_percent(self, symbol: str, target_percent: float) -> None:
        """Rebalance position in ``symbol`` to ``target_percent`` of portfolio."""
        target_value = self.portfolio_value() * target_percent
        self.order_target_value(symbol, target_value)

    def order_target_percent_many(self, weights: dict[str, float]) -> None:
        """Batch rebalance many symbols to target weights.

        Reads prices once and submits all target orders together.
        """
        symbols = list(weights.keys())
        all_symbols = list(set(symbols + list(self.portfolio.positions.keys())))
        prices = self.data.current(all_symbols, field="close")
        portfolio_value = self.portfolio.portfolio_value(prices.to_dict())

        for symbol, target_percent in weights.items():
            price = prices.get(symbol)
            if price is None or price <= 0:
                continue
            target_value = portfolio_value * target_percent
            self.broker.submit_target(symbol, target_value / price)

    def portfolio_value(self) -> float:
        prices = self.data.current(list(self.portfolio.positions.keys()), field="close")
        return self.portfolio.portfolio_value(prices.to_dict())