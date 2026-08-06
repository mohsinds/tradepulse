"""Offline unit tests for the research parameter-sweep tooling."""

from __future__ import annotations

import asyncio
import unittest
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import vectorbt as vbt

from api.adapters import Bar, ContractType, Instrument
from api.config import load_all
from research.costs import (
    CFDSwapCost,
    FuturesRollCost,
    NoCost,
    PerpetualFundingCost,
    cost_overlay_for,
)
from research.data import _bars_to_df, fetch_bars, make_synthetic_bars
from research.param_sweep import (
    _best_params_for_signal,
    sweep_underlying,
    write_underlying_shared_config,
)
from research.strategy import SIGNAL_FUNCS


START = datetime(2024, 1, 1)


def _instrument(
    symbol: str = "MGC",
    venue: str = "ibkr",
    contract_type: ContractType = ContractType.FUTURE,
) -> Instrument:
    return Instrument(
        symbol=symbol,
        venue=venue,
        contract_type=contract_type,
        exchange="COMEX",
        underlying="GOLD",
        multiplier=Decimal("1"),
        currency="USD",
    )


def _make_bars(n: int = 10) -> list[Bar]:
    bars = []
    price = 100.0
    for i in range(n):
        open_ = price
        price += 0.5
        bars.append(
            Bar(
                timestamp=START + timedelta(hours=i),
                open=Decimal(str(open_)),
                high=Decimal(str(open_ + 1.0)),
                low=Decimal(str(open_ - 0.5)),
                close=Decimal(str(price)),
                volume=Decimal("1000"),
            )
        )
    return bars


class FakeAdapter:
    """Stand-in venue adapter that never touches the network."""

    def __init__(self, bars: list[Bar] | None = None):
        self.bars = bars if bars is not None else _make_bars()
        self.connected = False

    async def connect(self) -> None:
        self.connected = True

    async def disconnect(self) -> None:
        self.connected = False

    async def get_historical_bars(
        self,
        instrument: Instrument,
        start: datetime,
        end: datetime,
        timeframe: str,
    ) -> list[Bar]:
        return [b for b in self.bars if start <= b.timestamp <= end]


class TestData(unittest.IsolatedAsyncioTestCase):
    async def test_fetch_bars_with_fake_adapter(self):
        inst = _instrument()
        adapter = FakeAdapter()
        df = await fetch_bars(
            inst, {}, START, START + timedelta(days=1), "1h", _adapter=adapter
        )
        self.assertEqual(list(df.columns), ["open", "high", "low", "close", "volume"])
        self.assertEqual(len(df), 10)
        self.assertFalse(adapter.connected)

    def test_make_synthetic_bars(self):
        df = make_synthetic_bars(50, seed=42)
        self.assertEqual(len(df), 50)
        self.assertIn("close", df.columns)
        self.assertTrue((df["high"] >= df["low"]).all())


class TestCosts(unittest.IsolatedAsyncioTestCase):
    def _holding_portfolio(self) -> vbt.Portfolio:
        price = pd.Series(
            100 + np.cumsum(np.random.RandomState(42).normal(0.02, 0.5, size=60)),
            index=pd.date_range(START, periods=60, freq="h"),
        )
        entries = pd.Series(False, index=price.index)
        exits = pd.Series(False, index=price.index)
        entries.iloc[10] = True
        entries.iloc[35] = True
        exits.iloc[25] = True
        exits.iloc[50] = True
        return vbt.Portfolio.from_signals(
            price, entries, exits, init_cash=10000.0, freq="1h"
        )

    def test_futures_roll_cost_reduces_returns(self):
        pf = self._holding_portfolio()
        overlay = FuturesRollCost(_instrument(), roll_cost_pct=0.0002, rolls_per_year=12)
        adj = overlay.apply_cost(pf)
        self.assertLess(adj.sum(), pf.returns().fillna(0).sum())

    def test_cfd_swap_cost_reduces_returns(self):
        pf = self._holding_portfolio()
        overlay = CFDSwapCost(_instrument(), annual_swap_rate=0.03)
        adj = overlay.apply_cost(pf)
        self.assertLess(adj.sum(), pf.returns().fillna(0).sum())

    def test_perpetual_funding_cost_reduces_returns(self):
        pf = self._holding_portfolio()
        overlay = PerpetualFundingCost(
            _instrument(symbol="XAUUSDT.P", contract_type=ContractType.PERPETUAL),
            funding_rate=0.001,
        )
        adj = overlay.apply_cost(pf)
        self.assertLess(adj.sum(), pf.returns().fillna(0).sum())

    def test_no_cost_is_neutral(self):
        pf = self._holding_portfolio()
        overlay = NoCost(_instrument())
        adj = overlay.apply_cost(pf)
        pd.testing.assert_series_equal(adj, pf.returns().fillna(0.0), check_names=False)

    def test_cost_dispatcher(self):
        future = _instrument(contract_type=ContractType.FUTURE)
        perp = _instrument(symbol="XAUUSDT.P", contract_type=ContractType.PERPETUAL)
        spot = _instrument(symbol="XAUUSD", contract_type=ContractType.SPOT)
        self.assertIsInstance(cost_overlay_for(future), FuturesRollCost)
        self.assertIsInstance(cost_overlay_for(perp), PerpetualFundingCost)
        self.assertIsInstance(cost_overlay_for(spot), NoCost)


class TestParamSweep(unittest.IsolatedAsyncioTestCase):
    def _trending_df(self, n: int = 300) -> pd.DataFrame:
        """Strong upward drift so a faster trend MA should win."""
        return make_synthetic_bars(n, trend_drift=0.08, volatility=0.5, seed=123)

    def test_best_params_distinct_for_trend(self):
        from research import param_sweep

        original_grid = param_sweep.SIGNAL_PARAMS.copy()
        self.addCleanup(setattr, param_sweep, "SIGNAL_PARAMS", original_grid)

        # Small, deliberately uneven grid: the fast 5/40 combo should dominate
        # on a strongly trending series, while 30/100 should lag.
        param_sweep.SIGNAL_PARAMS = {
            "trend": [
                {"fast": 5, "slow": 40, "atr_period": 10},
                {"fast": 30, "slow": 100, "atr_period": 14},
            ]
        }

        inst = _instrument()
        df = self._trending_df(300)
        data_map = [(inst, df)]

        params, metrics = _best_params_for_signal("trend", data_map, train_frac=0.7)
        self.assertEqual(params["fast"], 5)
        self.assertEqual(params["slow"], 40)
        self.assertIn("composite", metrics)

    async def test_sweep_underlying_dry_run(self):
        from research import param_sweep

        original_grid = param_sweep.SIGNAL_PARAMS.copy()
        self.addCleanup(setattr, param_sweep, "SIGNAL_PARAMS", original_grid)

        param_sweep.SIGNAL_PARAMS = {
            "trend": [
                {"fast": 5, "slow": 40, "atr_period": 10},
                {"fast": 30, "slow": 100, "atr_period": 14},
            ]
        }

        inst = _instrument()
        result = await sweep_underlying(
            "GOLD",
            [inst],
            {},
            START,
            START + timedelta(days=30),
            "1h",
            train_frac=0.7,
            dry_run=True,
            synthetic_bars=300,
        )
        self.assertIn("trend", result)
        self.assertIn("test_metrics", result["trend"])


class TestSharedConfigWriter(unittest.TestCase):
    def tearDown(self):
        path = Path("api/config/signals/GOLD_shared.yaml")
        if path.exists():
            path.unlink()

    def test_write_and_load_merged_config(self):
        best_params = {
            "trend": {
                "fast": 7,
                "slow": 40,
                "atr_period": 10,
                "test_metrics": {"composite": 0.5},
            },
            "momentum": {
                "lookback": 12,
                "threshold": 0.04,
                "test_metrics": {"composite": 0.3},
            },
        }
        written = write_underlying_shared_config("GOLD", best_params)
        self.assertTrue(written.exists())

        config = load_all()
        for symbol in ("MGC", "XAUUSD", "XAUUSDT.P"):
            with self.subTest(symbol=symbol):
                inst_cfg = config["instruments"][symbol]
                trend = inst_cfg["signals"]["trend"]
                self.assertEqual(trend["weight"], 0.25)
                self.assertEqual(trend["fast"], 7)


if __name__ == "__main__":
    unittest.main()
