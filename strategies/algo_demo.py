"""Demo of bt-style Algo strategy."""

from alphalab.backtest import BacktestEngine, AlgoStrategy
from alphalab.backtest.algos import (
    RunMonthly,
    SelectAll,
    WeighEqually,
    Rebalance,
    PrintWeights,
)


if __name__ == "__main__":
    strategy = AlgoStrategy([
        RunMonthly(),
        SelectAll(),
        WeighEqually(),
        Rebalance(),
    ])

    result = BacktestEngine(
        start="20240101",
        end="20241106",
        capital_base=1_000_000,
        strategy=strategy,
    ).run()

    print("总收益:", result.total_return)
    print("最终市值:", result.final_value)
    print(result.performance.tail())
