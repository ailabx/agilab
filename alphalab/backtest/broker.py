"""Simulation broker for the backtest engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd

from .portfolio import Portfolio


@dataclass
class FeeBreakdown:
    commission: float = 0.0
    stamp_tax: float = 0.0
    transfer_fee: float = 0.0

    @property
    def total(self) -> float:
        return self.commission + self.stamp_tax + self.transfer_fee


class ChinaFeeModel:
    """Simplified A-share fee model."""

    def __init__(self, commission_rate: float = 0.0003, min_commission: float = 5.0) -> None:
        self.commission_rate = commission_rate
        self.min_commission = min_commission

    def calculate(self, value: float, is_sell: bool) -> FeeBreakdown:
        commission = max(self.min_commission, value * self.commission_rate)
        stamp_tax = value * 0.001 if is_sell else 0.0
        transfer_fee = value * 0.00002
        return FeeBreakdown(commission=commission, stamp_tax=stamp_tax, transfer_fee=transfer_fee)


class Order:
    def __init__(self, symbol: str, amount: float, target: bool = False):
        self.symbol = symbol
        self.amount = amount  # positive=buy, negative=sell
        self.target = target


class SimulationBroker:
    """Collects orders, then matches them against daily bars."""

    def __init__(self, portfolio: Portfolio, fee_model: ChinaFeeModel | None = None, execution: str = "close") -> None:
        self.portfolio = portfolio
        self.fee_model = fee_model or ChinaFeeModel()
        self.execution = execution
        self.open_orders: list[Order] = []
        self.transactions: list[dict] = []

    def submit_order(self, symbol: str, amount: float, target: bool = False) -> None:
        self.open_orders.append(Order(symbol, amount, target))

    def submit_target(self, symbol: str, target_amount: float) -> None:
        self.open_orders.append(Order(symbol, target_amount, target=True))

    def match_orders(self, date: pd.Timestamp, prices: dict[str, float]) -> None:
        # First pass: execute sells to free cash, then buys.
        sells, buys = [], []
        for order in self.open_orders:
            price = prices.get(order.symbol)
            if price is None or pd.isna(price) or price <= 0:
                continue

            current = self.portfolio.amount(order.symbol)
            if order.target:
                amount = order.amount - current
            else:
                amount = order.amount

            if abs(amount) < 1:
                continue

            if amount < 0:
                sells.append((order.symbol, amount, current))
            else:
                buys.append((order.symbol, amount))

        # Execute sells first
        for symbol, amount, current in sells:
            lots = int(-amount // 100)
            amount = -lots * 100
            if amount == 0 or current + amount < 0:
                continue
            self._execute_order(date, symbol, amount, prices[symbol])

        # Then execute buys, but respect available cash
        available_cash = self.portfolio.cash
        for symbol, amount in buys:
            lots = int(amount // 100)
            amount = lots * 100
            if amount <= 0:
                continue
            cost = amount * prices[symbol]
            estimated_fees = self.fee_model.calculate(cost, is_sell=False).total
            if cost + estimated_fees > available_cash:
                continue
            self._execute_order(date, symbol, amount, prices[symbol])
            available_cash = self.portfolio.cash

        self.open_orders.clear()

    def _execute_order(self, date: pd.Timestamp, symbol: str, amount: float, price: float) -> None:
        value = abs(amount) * price
        is_sell = amount < 0
        fees = self.fee_model.calculate(value, is_sell=is_sell)
        self.portfolio.apply_transaction(symbol, amount, price, fees.total)
        self.transactions.append({
            "date": date,
            "symbol": symbol,
            "amount": amount,
            "price": price,
            "fees": fees.total,
        })