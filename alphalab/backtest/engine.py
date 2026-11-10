"""Event-driven backtest engine."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


try:
    from config import DATALAKE_DIR
except ImportError:  # pragma: no cover
    from alphalab.config import DATALAKE_DIR

from .broker import ChinaFeeModel, SimulationBroker
from .calendar import TradingCalendar
from .context import Context
from .data_portal import BacktestDataPortal
from .portfolio import Portfolio
from .result import BacktestResult
from .strategy import Strategy


class BacktestEngine:
    """Minimal event-driven backtest engine."""

    def __init__(
        self,
        start: str,
        end: str,
        capital_base: float = 1_000_000.0,
        strategy: Strategy | None = None,
        execution: str = "close",
        adjusted: str = "qfq",
        datalake_dir: str | Path | None = None,
    ) -> None:
        self.start = start
        self.end = end
        self.capital_base = capital_base
        self.strategy = strategy
        self.execution = execution
        self.adjusted = adjusted
        self._datalake = Path(datalake_dir or DATALAKE_DIR)

        self.calendar = TradingCalendar(self._datalake)
        self.data = BacktestDataPortal(self._datalake, adjusted=adjusted, start=start, end=end)
        self.portfolio = Portfolio(cash=capital_base)
        self.broker = SimulationBroker(self.portfolio, execution=execution)
        self.context = Context(self.portfolio, self.broker, self.data)
        self.records: list[dict] = []

    def run(self) -> BacktestResult:
        dates = self.calendar.dates_in_range(self.start, self.end)

        if self.strategy is not None:
            self.strategy.initialize(self.context)

        previous_value = self.capital_base

        for date in dates:
            self.data.set_date(date)
            self.context.current_date = date

            # 1. Match orders from previous day at execution point.
            symbols = list(self.portfolio.positions.keys())
            if self.broker.open_orders:
                symbols += [o.symbol for o in self.broker.open_orders]
            prices = self._get_prices(date, symbols)
            self.broker.match_orders(date, prices)

            # 2. Let strategy react to the date.
            if self.strategy is not None:
                self.strategy.handle_data(self.context, self.data)

            # 3. End-of-day valuation.
            prices = self._get_prices(date, list(self.portfolio.positions.keys()))
            value = self.portfolio.portfolio_value(prices)
            daily_return = value / previous_value - 1.0 if previous_value else 0.0
            self.records.append({
                "date": date,
                "cash": self.portfolio.cash,
                "positions_value": self.portfolio.positions_value(prices),
                "portfolio_value": value,
                "returns": daily_return,
                "period_return": value / self.capital_base - 1.0,
            })
            previous_value = value

        return BacktestResult(self.records, self.broker.transactions, self.portfolio)

    def _get_prices(self, date: pd.Timestamp, symbols: list[str]) -> dict[str, float]:
        if not symbols:
            return {}
        # On-demand read for broker/valuation when in-memory panel is not loaded yet.
        df = self.data.get_prices(symbols, start=date.strftime("%Y%m%d"), end=date.strftime("%Y%m%d"), fields=["close"])
        if df.empty:
            return {}
        return df.set_index("symbol")["close"].to_dict()