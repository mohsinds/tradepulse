from __future__ import annotations

import unittest
from decimal import Decimal
from unittest.mock import MagicMock, patch

from api.adapters.base import BrokerAdapter, Instrument, OrderRequest
from api.execution import PaperExecutionEngine

ACCOUNT_VALUE = Decimal("100000")


class FakeBroker(BrokerAdapter):
    """In-memory broker adapter for unit tests."""

    def __init__(self, name: str, config: dict, positions=None):
        self.name = name
        self.config = config
        self.connected = False
        self.positions = positions or []
        self.order_id = "order-123"

    async def connect(self) -> None:
        self.connected = True

    async def place_order(self, order: OrderRequest) -> str:
        return self.order_id

    async def cancel_order(self, order_id: str) -> bool:
        return True

    async def get_positions(self) -> list:
        return self.positions

    async def get_account(self) -> dict:
        return {"broker": self.name}

    async def get_order_status(self, order_id: str) -> dict:
        return {"order_id": order_id, "status": "filled"}


class TestPaperExecutionEngine(unittest.IsolatedAsyncioTestCase):
    def _config(self) -> dict:
        return {
            "instruments": {
                "MGC": {
                    "venue": "ibkr",
                    "contract_type": "future",
                    "underlying": "GOLD",
                    "exchange": "COMEX",
                    "currency": "USD",
                    "multiplier": "10",
                    "paper_quantity": "0.01",
                    "last_price": "2500",
                }
            },
            "underlyings": {
                "GOLD": {"max_combined_position_pct": 0.05},
            },
        }

    def _engine(self, adapter, **kwargs):
        defaults = {
            "adapters": {"ibkr": adapter},
            "account_value": ACCOUNT_VALUE,
            "underlying_configs": {"GOLD": {"max_combined_position_pct": 0.05}},
        }
        defaults.update(kwargs)
        return PaperExecutionEngine(**defaults)

    async def test_risk_ok_and_sandbox_calls_place_order(self) -> None:
        broker = FakeBroker("ibkr", {"sandbox": True})
        engine = self._engine(broker)

        with patch("api.execution.engine.load_all", return_value=self._config()):
            result = await engine.execute_idea(
                {
                    "instrument": "MGC",
                    "direction": "long",
                    "quantity": "0.01",
                    "price": "2500",
                }
            )

        self.assertEqual(result["status"], "filled")
        self.assertEqual(result["order_id"], "order-123")
        self.assertIn("risk", result)
        self.assertTrue(broker.connected)

    async def test_risk_rejected_when_cap_exceeded(self) -> None:
        broker = FakeBroker("ibkr", {"sandbox": False})
        engine = self._engine(broker)

        with patch("api.execution.engine.load_all", return_value=self._config()):
            result = await engine.execute_idea(
                {
                    "instrument": "MGC",
                    "direction": "long",
                    "quantity": "10",
                    "price": "2500",
                }
            )

        self.assertEqual(result["status"], "risk_rejected")
        self.assertIn("risk", result)
        self.assertIn("exceeds max", result["risk"]["reason"])

    async def test_paper_mode_simulates_order(self) -> None:
        broker = FakeBroker("ibkr", {"sandbox": False})
        engine = self._engine(broker, paper=True)

        with patch("api.execution.engine.load_all", return_value=self._config()):
            result = await engine.execute_idea(
                {
                    "instrument": "MGC",
                    "direction": "long",
                    "quantity": "0.01",
                    "price": "2500",
                }
            )

        self.assertEqual(result["status"], "paper_filled")
        self.assertTrue(result["order_id"].startswith("paper-"))

    async def test_aggregates_positions_from_multiple_adapters(self) -> None:
        ibkr_broker = FakeBroker(
            "ibkr",
            {"sandbox": False},
            positions=[
                {
                    "symbol": "MGC",
                    "quantity": Decimal("0.01"),
                    "last": Decimal("2500"),
                    "multiplier": "10",
                }
            ],
        )
        ibkr_broker.connected = True
        ccxt_broker = FakeBroker(
            "ccxt",
            {"sandbox": False},
            positions=[
                {
                    "symbol": "XAUUSDT.P",
                    "quantity": Decimal("0.1"),
                    "last": Decimal("2500"),
                    "multiplier": "1",
                }
            ],
        )
        ccxt_broker.connected = True

        config = {
            "instruments": {
                "MGC": {"underlying": "GOLD"},
                "XAUUSDT.P": {"underlying": "GOLD"},
            },
            "underlyings": {"GOLD": {"max_combined_position_pct": 0.05}},
        }

        engine = PaperExecutionEngine(
            adapters={"ibkr": ibkr_broker, "ccxt": ccxt_broker},
            account_value=ACCOUNT_VALUE,
            underlying_configs={"GOLD": {"max_combined_position_pct": 0.05}},
        )

        with patch("api.execution.engine.load_all", return_value=config):
            positions = await engine._positions_by_symbol()

        self.assertIn("MGC", positions)
        self.assertIn("XAUUSDT.P", positions)
        for pos_list in positions.values():
            for pos in pos_list:
                self.assertEqual(pos.get("underlying"), "GOLD")

    async def test_route_direction_and_quantity(self) -> None:
        instrument = Instrument(
            symbol="MGC",
            venue="ibkr",
            contract_type="future",
            exchange="COMEX",
            underlying="GOLD",
            multiplier=Decimal("10"),
            currency="USD",
            meta={"paper_quantity": Decimal("0.02"), "last_price": Decimal("2500")},
        )
        engine = PaperExecutionEngine({}, ACCOUNT_VALUE, {})

        order = await engine.route(
            instrument, {"direction": "short", "quantity": "0.05", "price": "2600"}
        )

        self.assertEqual(order.side, "sell")
        self.assertEqual(order.quantity, Decimal("0.05"))
        self.assertEqual(order.price, Decimal("2600"))
