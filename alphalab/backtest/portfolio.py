"""Portfolio tracking for the backtest engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import pandas as pd


@dataclass
class Position:
    symbol: str
    amount: float = 0.0
    cost_basis: float = 0.0


@dataclass
class Portfolio:
    cash: float = 1_000_000.0
    positions: Dict[str, Position] = field(default_factory=dict)

    def __post_init__(self):
        if not self.positions:
            self.positions = {}

    def positions_value(self, prices: dict[str, float] | None = None) -> float:
        if not prices:
            return 0.0
        total = 0.0
        for sym, pos in self.positions.items():
            price = prices.get(sym, 0.0)
            if pd.notna(price) and price > 0:
                total += pos.amount * price
        return total

    def amount(self, symbol: str) -> float:
        return self.positions.get(symbol, Position(symbol)).amount

    def apply_transaction(self, symbol: str, amount: float, price: float, fees: float) -> None:
        cash_flow = -amount * price
        self.cash += cash_flow - fees
        pos = self.positions.setdefault(symbol, Position(symbol))
        if amount > 0:
            total_cost = pos.amount * pos.cost_basis + amount * price + fees
            pos.amount += amount
            pos.cost_basis = total_cost / pos.amount if pos.amount else 0.0
        else:
            pos.amount += amount
            if abs(pos.amount) < 1e-9:
                del self.positions[symbol]

    def mark_to_market(self, prices: dict[str, float]) -> float:
        return self.cash + self.positions_value(prices)

    def portfolio_value(self, prices: dict[str, float]) -> float:
        return self.mark_to_market(prices)