"""End-to-end seam test: real signal calculators -> real ranker graph, mocked LLM."""

from __future__ import annotations

import asyncio
import unittest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch

from api.adapters.base import Bar, ContractType, Instrument
from api.config import load_all
from api.signals import build_signals


def _bars(n: int, drift: float, start_price: float = 2000.0) -> list[Bar]:
    bars = []
    price = start_price
    base = datetime(2025, 1, 1)
    for i in range(n):
        price += drift
        bars.append(
            Bar(
                timestamp=base + timedelta(hours=i),
                open=Decimal(str(round(price - drift, 4))),
                high=Decimal(str(round(price + 1.0, 4))),
                low=Decimal(str(round(price - 1.0, 4))),
                close=Decimal(str(round(price, 4))),
                volume=Decimal(str(1000 + (i % 7) * 25)),
            )
        )
    return bars


class TestSignalsRankerIntegration(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = load_all()
        self.instruments = self.cfg["instruments"]

    def test_configs_load_with_merged_signal_weights(self) -> None:
        for symbol in ("MGC", "XAUUSD", "XAUUSDT.P", "BTC/USD"):
            self.assertIn(symbol, self.instruments)
            blocks = self.instruments[symbol]["signals"]
            self.assertIn("trend", blocks)
            self.assertIn("weight", blocks["trend"])
        # funding_rate only merged into the perpetual
        self.assertIn("funding_rate", self.instruments["XAUUSDT.P"]["signals"])
        self.assertNotIn("funding_rate", self.instruments["BTC/USD"]["signals"])

    def _compute(self, symbol: str, instrument: Instrument, data) -> list[dict]:
        merged = self.instruments[symbol]["signals"]
        results = []
        for sig in build_signals(merged):
            results.append(sig.generate(instrument, data))
        return results

    def test_end_to_end_ranking_with_mocked_llm(self) -> None:
        from api.agents.ranker import RankerAgent

        perp = Instrument(
            symbol="XAUUSDT.P",
            venue="ccxt",
            contract_type=ContractType.PERPETUAL,
            exchange="bybit",
            underlying="GOLD",
            currency="USDT",
        )
        spot = Instrument(
            symbol="BTC/USD",
            venue="ccxt",
            contract_type=ContractType.SPOT,
            exchange="bybit",
            underlying="BTC",
            currency="USD",
        )

        # uptrending perp with strongly positive (longs pay) funding
        perp_data = {"bars": _bars(160, 1.5), "quote": {"funding_rate": 0.0006}}
        spot_data = {"bars": _bars(160, 2.0, start_price=60000.0)}

        signal_results = {
            "XAUUSDT.P": self._compute("XAUUSDT.P", perp, perp_data),
            "BTC/USD": self._compute("BTC/USD", spot, spot_data),
        }

        names = {r["name"] for r in signal_results["XAUUSDT.P"]}
        self.assertEqual(
            names,
            {"trend", "momentum", "meanrev", "breakout", "volatility", "volume", "funding_rate"},
        )

        instruments = {
            "XAUUSDT.P": {**self.instruments["XAUUSDT.P"], "contract_type": ContractType.PERPETUAL},
            "BTC/USD": {**self.instruments["BTC/USD"], "contract_type": ContractType.SPOT},
        }

        fake_model = MagicMock()
        fake_model.ainvoke = MagicMock(
            side_effect=lambda *a, **k: asyncio.sleep(0, result=MagicMock(content="Mocked rationale."))
        )

        agent = RankerAgent()
        with patch("api.agents.ranker.get_chat_model_for_config", return_value=fake_model), patch(
            "api.agents.ranker.get_chat_model", return_value=fake_model
        ):
            out = asyncio.run(
                agent.run({"instruments": instruments, "signal_results": signal_results})
            )

        self.assertTrue(out["advisory_only"])
        self.assertFalse(out.get("execution_enabled", False))
        self.assertEqual(len(out["ideas"]), 2)

        by_symbol = {i["instrument"]: i for i in out["ideas"]}
        perp_idea = by_symbol["XAUUSDT.P"]
        spot_idea = by_symbol["BTC/USD"]

        # every idea carries a rationale and per-signal contributions
        for idea in out["ideas"]:
            self.assertTrue(idea["rationale"])
            self.assertIn("trend", idea["contributions"])
            self.assertIn(idea["rank"], (1, 2))

        # perp long into positive funding must be down-weighted
        self.assertEqual(perp_idea["direction"], "long")
        self.assertIsNotNone(perp_idea["funding"])
        self.assertTrue(perp_idea["funding"]["unfavorable"])
        self.assertLess(abs(perp_idea["adjusted_score"]), abs(perp_idea["raw_score"]))

        # spot instrument is untouched by the funding node
        self.assertIsNone(spot_idea["funding"])
        self.assertAlmostEqual(spot_idea["adjusted_score"], spot_idea["raw_score"], places=9)

    def test_llm_failure_falls_back_to_template_rationale(self) -> None:
        from api.agents.ranker import RankerAgent

        spot = Instrument(
            symbol="BTC/USD",
            venue="ccxt",
            contract_type=ContractType.SPOT,
            exchange="bybit",
            underlying="BTC",
            currency="USD",
        )
        signal_results = {
            "BTC/USD": self._compute("BTC/USD", spot, {"bars": _bars(160, 2.0, 60000.0)})
        }
        instruments = {"BTC/USD": {**self.instruments["BTC/USD"], "contract_type": ContractType.SPOT}}

        boom = MagicMock()
        boom.ainvoke = MagicMock(side_effect=RuntimeError("provider down"))

        agent = RankerAgent()
        with patch("api.agents.ranker.get_chat_model_for_config", return_value=boom), patch(
            "api.agents.ranker.get_chat_model", return_value=boom
        ):
            out = asyncio.run(
                agent.run({"instruments": instruments, "signal_results": signal_results})
            )

        self.assertEqual(len(out["ideas"]), 1)
        self.assertTrue(out["ideas"][0]["rationale"])


if __name__ == "__main__":
    unittest.main()
