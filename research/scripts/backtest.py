import numpy as np
import pandas as pd
import vectorbt as vbt


def run_sample_backtest() -> vbt.Portfolio:
    price = pd.Series(
        np.cumsum(np.random.randn(1000) * 0.1) + 100,
        index=pd.date_range("2024-01-01", periods=1000, freq="h"),
        name="price",
    )

    fast_ma = vbt.MA.run(price, window=10)
    slow_ma = vbt.MA.run(price, window=50)

    entries = fast_ma.ma_above(slow_ma, crossover=True)
    exits = fast_ma.ma_below(slow_ma, crossover=True)

    portfolio = vbt.Portfolio.from_signals(price, entries, exits, init_cash=10000)
    return portfolio


if __name__ == "__main__":
    pf = run_sample_backtest()
    print(pf.stats())
