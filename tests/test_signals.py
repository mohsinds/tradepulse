"""Deterministic, offline tests for the signal calculators.

No network, no LLM, no adapters — synthetic bars only.
"""

from __future__ import annotations

import math
import unittest
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List

import pandas as pd

from api.adapters.base import Bar, ContractType, Instrument
from api.signals import (
    SIGNAL_REGISTRY,
    BreakoutSignal,
    FundingRateSignal,
    MeanReversionSignal,
    MomentumSignal,
    Signal,
    TrendSignal,
    VolatilitySignal,
    VolumeSignal,
    build_signals,
)

START = datetime(2024, 1, 1)

SPOT = Instrument(symbol="BTC/USD", venue="ccxt", contract_type=ContractType.SPOT)
PERP = Instrument(
    symbol="XAUUSDT.P", venue="ccxt", contract_type=ContractType.PERPETUAL
)
FUTURE = Instrument(symbol="MGC", venue="ibkr", contract_type=ContractType.FUTURE)


def make_bar(index: int, close: float, spread: float = 1.0, volume: float = 1000.0,
             open_: float = None, high: float = None, low: float = None) -> Bar:
    open_price = close if open_ is None else open_
    high_price = max(open_price, close) + spread / 2.0 if high is None else high
    low_price = min(open_price, close) - spread / 2.0 if low is None else low
    return Bar(
        timestamp=START + timedelta(hours=index),
        open=Decimal(str(round(open_price, 6))),
        high=Decimal(str(round(high_price, 6))),
        low=Decimal(str(round(low_price, 6))),
        close=Decimal(str(round(close, 6))),
        volume=Decimal(str(round(volume, 6))),
    )


def trending_bars(n: int = 200, start: float = 100.0, step: float = 1.0) -> List[Bar]:
    """Steady uptrend with a small constant range."""
    bars = []
    price = start
    for i in range(n):
        prev = price
        price = start + step * i
        bars.append(make_bar(i, price, spread=step, volume=1000.0, open_=prev))
    return bars


def downtrending_bars(n: int = 200, start: float = 300.0, step: float = 1.0) -> List[Bar]:
    bars = []
    price = start
    for i in range(n):
        prev = price
        price = start - step * i
        bars.append(make_bar(i, price, spread=step, volume=1000.0, open_=prev))
    return bars


def ranging_bars(n: int = 200, mid: float = 100.0, amplitude: float = 2.0,
                 period: int = 20, volume: float = 1000.0) -> List[Bar]:
    """Deterministic sine oscillation around ``mid``."""
    bars = []
    prev = mid
    for i in range(n):
        close = mid + amplitude * math.sin(2.0 * math.pi * i / period)
        bars.append(make_bar(i, close, spread=amplitude / 2.0, volume=volume,
                             open_=prev))
        prev = close
    return bars


class TrendSignalTests(unittest.TestCase):
    def test_uptrend_is_long(self):
        sig = TrendSignal({"weight": 0.25, "fast": 20, "slow": 50, "atr_period": 14})
        out = sig.generate(SPOT, trending_bars())
        self.assertEqual(out["name"], "trend")
        self.assertEqual(out["direction"], "long")
        self.assertGreater(out["score"], 0.0)
        self.assertEqual(out["weight"], 0.25)
        self.assertGreater(out["metrics"]["fast_ema"], out["metrics"]["slow_ema"])
        self.assertGreater(out["metrics"]["slope_atr_per_bar"], 0.0)

    def test_downtrend_is_short(self):
        sig = TrendSignal({"weight": 0.25, "fast": 20, "slow": 50, "atr_period": 14})
        out = sig.generate(SPOT, downtrending_bars())
        self.assertEqual(out["direction"], "short")
        self.assertLess(out["score"], 0.0)

    def test_range_is_flat(self):
        sig = TrendSignal({"weight": 0.25, "fast": 20, "slow": 50, "atr_period": 14})
        out = sig.generate(SPOT, ranging_bars(n=300))
        self.assertLess(abs(out["score"]), 0.5)

    def test_insufficient_data(self):
        sig = TrendSignal({"weight": 0.25, "fast": 20, "slow": 50, "atr_period": 14})
        out = sig.generate(SPOT, trending_bars(n=10))
        self.assertEqual(out["direction"], "flat")
        self.assertEqual(out["score"], 0.0)
        self.assertFalse(out["ok"])
        self.assertEqual(out["reason"], "insufficient_data")


class MomentumSignalTests(unittest.TestCase):
    cfg = {"weight": 0.2, "lookback": 20, "threshold": 0.05}

    def test_strong_up_move(self):
        out = MomentumSignal(self.cfg).generate(SPOT, trending_bars(n=100))
        self.assertEqual(out["name"], "momentum")
        self.assertEqual(out["direction"], "long")
        self.assertGreater(out["metrics"]["roc"], 0.0)

    def test_strong_down_move(self):
        out = MomentumSignal(self.cfg).generate(SPOT, downtrending_bars(n=100))
        self.assertEqual(out["direction"], "short")
        self.assertLess(out["score"], 0.0)

    def test_flat_series(self):
        bars = [make_bar(i, 100.0) for i in range(60)]
        out = MomentumSignal(self.cfg).generate(SPOT, bars)
        self.assertEqual(out["direction"], "flat")
        self.assertAlmostEqual(out["score"], 0.0)

    def test_score_is_clamped(self):
        out = MomentumSignal(self.cfg).generate(SPOT, trending_bars(n=100, step=5.0))
        self.assertLessEqual(out["score"], 1.0)
        self.assertGreaterEqual(out["score"], -1.0)

    def test_insufficient_data(self):
        out = MomentumSignal(self.cfg).generate(SPOT, trending_bars(n=5))
        self.assertFalse(out["ok"])
        self.assertEqual(out["direction"], "flat")


class MeanReversionSignalTests(unittest.TestCase):
    cfg = {
        "weight": 0.15,
        "rsi_period": 14,
        "oversold": 30,
        "overbought": 70,
        "zscore_period": 20,
    }

    def test_overbought_is_short(self):
        out = MeanReversionSignal(self.cfg).generate(SPOT, trending_bars(n=80))
        self.assertEqual(out["name"], "meanrev")
        self.assertGreater(out["metrics"]["rsi"], 70)
        self.assertEqual(out["direction"], "short")
        self.assertLess(out["score"], 0.0)

    def test_oversold_is_long(self):
        out = MeanReversionSignal(self.cfg).generate(SPOT, downtrending_bars(n=80))
        self.assertLess(out["metrics"]["rsi"], 30)
        self.assertEqual(out["direction"], "long")
        self.assertGreater(out["score"], 0.0)

    def test_zscore_reported(self):
        out = MeanReversionSignal(self.cfg).generate(SPOT, ranging_bars(n=120))
        self.assertIn("zscore", out["metrics"])
        self.assertTrue(math.isfinite(out["metrics"]["zscore"]))

    def test_insufficient_data(self):
        out = MeanReversionSignal(self.cfg).generate(SPOT, trending_bars(n=8))
        self.assertFalse(out["ok"])
        self.assertEqual(out["score"], 0.0)


class BreakoutSignalTests(unittest.TestCase):
    cfg = {
        "weight": 0.15,
        "channel_period": 20,
        "atr_period": 14,
        "atr_buffer": 0.25,
    }

    def test_upside_breakout(self):
        bars = ranging_bars(n=60, mid=100.0, amplitude=2.0)
        bars.append(make_bar(60, 130.0, spread=1.0, open_=102.0))
        out = BreakoutSignal(self.cfg).generate(SPOT, bars)
        self.assertEqual(out["name"], "breakout")
        self.assertEqual(out["metrics"]["breakout"], "up")
        self.assertEqual(out["direction"], "long")
        self.assertGreater(out["score"], 0.5)

    def test_downside_breakout(self):
        bars = ranging_bars(n=60, mid=100.0, amplitude=2.0)
        bars.append(make_bar(60, 70.0, spread=1.0, open_=98.0))
        out = BreakoutSignal(self.cfg).generate(SPOT, bars)
        self.assertEqual(out["metrics"]["breakout"], "down")
        self.assertEqual(out["direction"], "short")
        self.assertLess(out["score"], -0.5)

    def test_inside_channel_is_no_breakout(self):
        bars = ranging_bars(n=60, mid=100.0, amplitude=2.0)
        bars.append(make_bar(60, 100.0, spread=0.5, open_=100.0))
        out = BreakoutSignal(self.cfg).generate(SPOT, bars)
        self.assertEqual(out["metrics"]["breakout"], "none")
        self.assertEqual(out["direction"], "flat")

    def test_insufficient_data(self):
        out = BreakoutSignal(self.cfg).generate(SPOT, ranging_bars(n=5))
        self.assertFalse(out["ok"])


class VolatilitySignalTests(unittest.TestCase):
    cfg = {
        "weight": 0.1,
        "atr_period": 14,
        "regime_period": 100,
        "squeeze_percentile": 0.25,
    }

    def test_squeeze_detected(self):
        # Wide-range history followed by a long stretch of very tight bars.
        bars = ranging_bars(n=120, mid=100.0, amplitude=10.0)
        for i in range(60):
            bars.append(make_bar(120 + i, 100.0, spread=0.05, open_=100.0))
        out = VolatilitySignal(self.cfg).generate(SPOT, bars)
        self.assertEqual(out["name"], "volatility")
        self.assertTrue(out["metrics"]["squeeze"])
        self.assertEqual(out["metrics"]["regime"], "squeeze")
        self.assertEqual(out["direction"], "flat")
        self.assertGreater(out["score"], 0.0)

    def test_expansion_detected(self):
        bars = [make_bar(i, 100.0, spread=0.05, open_=100.0) for i in range(120)]
        for i in range(20):
            bars.append(make_bar(120 + i, 100.0 + i, spread=15.0, open_=100.0))
        out = VolatilitySignal(self.cfg).generate(SPOT, bars)
        self.assertTrue(out["metrics"]["expansion"])
        self.assertEqual(out["metrics"]["regime"], "expansion")
        self.assertLess(out["score"], 0.0)

    def test_atr_positive(self):
        out = VolatilitySignal(self.cfg).generate(SPOT, ranging_bars(n=150))
        self.assertGreater(out["metrics"]["atr"], 0.0)
        self.assertGreaterEqual(out["metrics"]["atr_percentile"], 0.0)
        self.assertLessEqual(out["metrics"]["atr_percentile"], 1.0)

    def test_insufficient_data(self):
        out = VolatilitySignal(self.cfg).generate(SPOT, ranging_bars(n=5))
        self.assertFalse(out["ok"])


class VolumeSignalTests(unittest.TestCase):
    cfg = {
        "weight": 0.15,
        "lookback": 20,
        "climax_sigma": 2.0,
        "absorption_ratio": 0.35,
        "spread_percentile": 0.7,
    }

    def test_high_volume_wide_up_bar_is_long(self):
        bars = ranging_bars(n=60, mid=100.0, amplitude=1.0, volume=1000.0)
        bars.append(
            Bar(
                timestamp=START + timedelta(hours=60),
                open=Decimal("100"),
                high=Decimal("110"),
                low=Decimal("99.5"),
                close=Decimal("109.8"),
                volume=Decimal("10000"),
            )
        )
        out = VolumeSignal(self.cfg).generate(SPOT, bars)
        self.assertEqual(out["name"], "volume")
        self.assertTrue(out["metrics"]["climax"])
        self.assertEqual(out["metrics"]["pattern"], "buying_climax")
        self.assertEqual(out["direction"], "long")

    def test_high_volume_wide_down_bar_is_short(self):
        bars = ranging_bars(n=60, mid=100.0, amplitude=1.0, volume=1000.0)
        bars.append(
            Bar(
                timestamp=START + timedelta(hours=60),
                open=Decimal("100"),
                high=Decimal("100.5"),
                low=Decimal("90"),
                close=Decimal("90.2"),
                volume=Decimal("10000"),
            )
        )
        out = VolumeSignal(self.cfg).generate(SPOT, bars)
        self.assertTrue(out["metrics"]["climax"])
        self.assertEqual(out["metrics"]["pattern"], "selling_climax")
        self.assertEqual(out["direction"], "short")

    def test_absorption_fades_the_pushing_side(self):
        # Big volume, almost no range, bar closed up -> demand absorbed.
        bars = ranging_bars(n=60, mid=100.0, amplitude=4.0, volume=1000.0)
        bars.append(
            Bar(
                timestamp=START + timedelta(hours=60),
                open=Decimal("100.00"),
                high=Decimal("100.06"),
                low=Decimal("99.98"),
                close=Decimal("100.05"),
                volume=Decimal("20000"),
            )
        )
        out = VolumeSignal(self.cfg).generate(SPOT, bars)
        self.assertTrue(out["metrics"]["absorption"])
        self.assertEqual(out["metrics"]["pattern"], "absorption")
        self.assertEqual(out["direction"], "short")

    def test_quiet_bar_is_flat(self):
        bars = ranging_bars(n=60, mid=100.0, amplitude=2.0, volume=1000.0)
        bars.append(
            Bar(
                timestamp=START + timedelta(hours=60),
                open=Decimal("100.0"),
                high=Decimal("100.2"),
                low=Decimal("99.8"),
                close=Decimal("100.0"),
                volume=Decimal("50"),
            )
        )
        out = VolumeSignal(self.cfg).generate(SPOT, bars)
        self.assertEqual(out["direction"], "flat")

    def test_insufficient_data(self):
        out = VolumeSignal(self.cfg).generate(SPOT, ranging_bars(n=5))
        self.assertFalse(out["ok"])


class FundingRateSignalTests(unittest.TestCase):
    cfg = {
        "weight": 0.2,
        "neutral_band": 0.0001,
        "extreme": 0.0005,
        "periods_per_day": 3,
        "unfavorable_penalty": 0.5,
    }

    def _payload(self, rate) -> Dict[str, Any]:
        return {
            "bars": ranging_bars(n=60),
            "quote": {"symbol": "XAUUSDT.P", "funding_rate": rate},
        }

    def test_positive_funding_is_bearish(self):
        out = FundingRateSignal(self.cfg).generate(PERP, self._payload(Decimal("0.0006")))
        self.assertEqual(out["name"], "funding_rate")
        self.assertTrue(out["active"])
        self.assertEqual(out["direction"], "short")
        self.assertLess(out["score"], 0.0)
        self.assertEqual(out["metrics"]["pays"], "long")
        self.assertEqual(out["metrics"]["receives"], "short")
        self.assertTrue(out["metrics"]["is_extreme"])
        self.assertAlmostEqual(out["annualized_rate"], 0.0006 * 3 * 365)

    def test_negative_funding_is_bullish(self):
        out = FundingRateSignal(self.cfg).generate(PERP, self._payload(-0.0006))
        self.assertEqual(out["direction"], "long")
        self.assertGreater(out["score"], 0.0)
        self.assertEqual(out["metrics"]["pays"], "short")
        self.assertLess(out["annualized_rate"], 0.0)

    def test_within_neutral_band_is_flat(self):
        out = FundingRateSignal(self.cfg).generate(PERP, self._payload(0.00005))
        self.assertEqual(out["direction"], "flat")
        self.assertAlmostEqual(out["score"], 0.0)
        self.assertTrue(out["metrics"]["neutral"])

    def test_spot_instrument_is_inactive(self):
        out = FundingRateSignal(self.cfg).generate(SPOT, self._payload(0.0006))
        self.assertFalse(out["active"])
        self.assertEqual(out["direction"], "flat")
        self.assertEqual(out["score"], 0.0)
        self.assertFalse(out["metrics"]["applicable"])
        self.assertEqual(out["reason"], "not_perpetual")

    def test_future_instrument_is_inactive(self):
        out = FundingRateSignal(self.cfg).generate(FUTURE, self._payload(0.0006))
        self.assertFalse(out["active"])

    def test_missing_funding_rate_degrades(self):
        out = FundingRateSignal(self.cfg).generate(PERP, {"bars": ranging_bars(n=60)})
        self.assertFalse(out["ok"])
        self.assertEqual(out["reason"], "missing_funding_rate")
        self.assertEqual(out["score"], 0.0)

    def test_top_level_funding_rate_accepted(self):
        out = FundingRateSignal(self.cfg).generate(
            PERP, {"bars": ranging_bars(n=60), "funding_rate": 0.0006}
        )
        self.assertTrue(out["ok"])
        self.assertEqual(out["direction"], "short")


class DataShapeTests(unittest.TestCase):
    cfg = {"weight": 0.2, "lookback": 20, "threshold": 0.05}

    def test_dataframe_input(self):
        bars = trending_bars(n=100)
        frame = pd.DataFrame(
            {
                "open": [float(b.open) for b in bars],
                "high": [float(b.high) for b in bars],
                "low": [float(b.low) for b in bars],
                "close": [float(b.close) for b in bars],
                "volume": [float(b.volume) for b in bars],
            }
        )
        from_list = MomentumSignal(self.cfg).generate(SPOT, bars)
        from_frame = MomentumSignal(self.cfg).generate(SPOT, frame)
        self.assertAlmostEqual(from_list["score"], from_frame["score"])

    def test_dict_with_bars(self):
        bars = trending_bars(n=100)
        out = MomentumSignal(self.cfg).generate(SPOT, {"bars": bars, "quote": {}})
        self.assertEqual(out["direction"], "long")

    def test_empty_and_none(self):
        for payload in (None, [], {}, pd.DataFrame()):
            out = MomentumSignal(self.cfg).generate(SPOT, payload)
            self.assertFalse(out["ok"])
            self.assertEqual(out["direction"], "flat")

    def test_dict_bar_input(self):
        bars = [
            {"open": 100 + i, "high": 101 + i, "low": 99 + i, "close": 100.5 + i,
             "volume": 10}
            for i in range(50)
        ]
        out = MomentumSignal(self.cfg).generate(SPOT, bars)
        self.assertEqual(out["direction"], "long")


class ContractTests(unittest.TestCase):
    """Every calculator must honour the common return-dict contract."""

    all_cfg: Dict[str, Dict[str, Any]] = {
        "trend": {"weight": 0.25, "fast": 20, "slow": 50, "atr_period": 14},
        "momentum": {"weight": 0.2, "lookback": 20, "threshold": 0.05},
        "meanrev": {
            "weight": 0.15,
            "rsi_period": 14,
            "oversold": 30,
            "overbought": 70,
            "zscore_period": 20,
        },
        "breakout": {
            "weight": 0.15,
            "channel_period": 20,
            "atr_period": 14,
            "atr_buffer": 0.25,
        },
        "volatility": {
            "weight": 0.1,
            "atr_period": 14,
            "regime_period": 100,
            "squeeze_percentile": 0.25,
        },
        "volume": {
            "weight": 0.15,
            "lookback": 20,
            "climax_sigma": 2.0,
            "absorption_ratio": 0.35,
            "spread_percentile": 0.7,
        },
        "funding_rate": {
            "weight": 0.2,
            "neutral_band": 0.0001,
            "extreme": 0.0005,
            "periods_per_day": 3,
        },
    }

    def test_contract_on_all_signals(self):
        payload = {
            "bars": trending_bars(n=200),
            "quote": {"funding_rate": 0.0003},
        }
        for name, cls in SIGNAL_REGISTRY.items():
            with self.subTest(signal=name):
                sig = cls(self.all_cfg[name])  # type: ignore[call-arg]
                out = sig.generate(PERP, payload)
                self.assertEqual(out["name"], name)
                self.assertIsInstance(out["score"], float)
                self.assertGreaterEqual(out["score"], -1.0)
                self.assertLessEqual(out["score"], 1.0)
                self.assertIn(out["direction"], ("long", "short", "flat"))
                self.assertEqual(out["weight"], self.all_cfg[name]["weight"])
                self.assertIsInstance(out["metrics"], dict)

    def test_contract_on_insufficient_data(self):
        for name, cls in SIGNAL_REGISTRY.items():
            with self.subTest(signal=name):
                sig = cls(self.all_cfg[name])  # type: ignore[call-arg]
                out = sig.generate(PERP, [])
                self.assertEqual(out["name"], name)
                self.assertEqual(out["score"], 0.0)
                self.assertEqual(out["direction"], "flat")
                self.assertIsInstance(out["metrics"], dict)

    def test_defaults_when_config_missing(self):
        out = TrendSignal().generate(SPOT, trending_bars())
        self.assertEqual(out["direction"], "long")
        self.assertEqual(out["weight"], 0.25)


class RegistryTests(unittest.TestCase):
    def test_registry_contents(self):
        self.assertEqual(
            set(SIGNAL_REGISTRY),
            {
                "trend",
                "momentum",
                "meanrev",
                "breakout",
                "volatility",
                "volume",
                "funding_rate",
            },
        )
        for name, cls in SIGNAL_REGISTRY.items():
            self.assertTrue(issubclass(cls, Signal))
            self.assertEqual(cls.name, name)

    def test_build_signals_from_merged_block(self):
        merged = {
            "trend": {"weight": 0.3, "fast": 10, "slow": 30, "atr_period": 14},
            "volume": {"weight": 0.15, "lookback": 20},
        }
        signals = build_signals(merged)
        self.assertEqual({s.name for s in signals}, {"trend", "volume"})
        trend = next(s for s in signals if s.name == "trend")
        self.assertIsInstance(trend, TrendSignal)
        self.assertEqual(trend.config["fast"], 10)
        self.assertEqual(trend.weight, 0.3)

    def test_build_signals_from_instrument_config(self):
        cfg = {
            "venue": "ccxt",
            "contract_type": "perpetual",
            "signals": {
                "trend": {"weight": 0.25},
                "funding_rate": {"weight": 0.2},
                "not_a_signal": {"weight": 1.0},
            },
        }
        signals = build_signals(cfg)
        self.assertEqual({s.name for s in signals}, {"trend", "funding_rate"})

    def test_build_signals_empty(self):
        self.assertEqual(build_signals({}), [])
        self.assertEqual(build_signals({"signals": {}}), [])

    def test_build_signals_null_block_uses_defaults(self):
        signals = build_signals({"momentum": None})
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].config["lookback"], 20)

    def test_all_generated_signals_run(self):
        merged = {name: {"weight": 0.1} for name in SIGNAL_REGISTRY}
        payload = {"bars": trending_bars(n=200), "quote": {"funding_rate": 0.0}}
        for sig in build_signals(merged):
            out = sig.generate(PERP, payload)
            self.assertEqual(out["weight"], 0.1)
            self.assertIn("metrics", out)


if __name__ == "__main__":
    unittest.main()
