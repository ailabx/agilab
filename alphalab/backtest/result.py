"""Backtest result container."""

from __future__ import annotations

import pandas as pd

from .portfolio import Portfolio


class BacktestResult:
    def __init__(self, records: list[dict], transactions: list[dict], portfolio: Portfolio) -> None:
        self.records = records
        self.transactions = transactions
        self.portfolio = portfolio
        self.performance = pd.DataFrame(records).set_index("date") if records else pd.DataFrame()

    @property
    def total_return(self) -> float:
        if self.performance.empty:
            return 0.0
        return self.performance["period_return"].iloc[-1]

    @property
    def final_value(self) -> float:
        if self.performance.empty:
            return self.portfolio.cash
        return self.performance["portfolio_value"].iloc[-1]