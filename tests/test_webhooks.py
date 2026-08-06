"""TradingView webhook handler and route tests."""

from __future__ import annotations

import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from api.webhooks.tradingview import TradingViewWebhookHandler


class TestTradingViewWebhookHandler(unittest.IsolatedAsyncioTestCase):
    async def test_buy_alert_normalizes_to_long_signal(self) -> None:
        handler = TradingViewWebhookHandler(secret="s3cr3t", signal_weight=0.25)
        payload = {
            "ticker": "XAUUSDT.P",
            "action": "buy",
            "price": 2500.5,
            "interval": "60",
            "secret": "s3cr3t",
        }
        result = await handler.handle(payload)
        self.assertEqual(result["name"], "tradingview")
        self.assertEqual(result["direction"], "long")
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["weight"], 0.25)
        self.assertEqual(result["metrics"]["ticker"], "XAUUSDT.P")
        self.assertEqual(result["metrics"]["price"], 2500.5)
        self.assertTrue(result["ok"])

    async def test_sell_alert_normalizes_to_short_signal(self) -> None:
        handler = TradingViewWebhookHandler(secret="s3cr3t")
        payload = {
            "ticker": "MGC",
            "action": "sell",
            "price": 2500,
            "secret": "s3cr3t",
        }
        result = await handler.handle(payload)
        self.assertEqual(result["direction"], "short")
        self.assertEqual(result["score"], -1.0)

    async def test_invalid_secret_raises_permission_error(self) -> None:
        handler = TradingViewWebhookHandler(secret="s3cr3t")
        payload = {"ticker": "XAUUSD", "action": "buy", "secret": "wrong"}
        with self.assertRaises(PermissionError):
            await handler.handle(payload)

    async def test_perp_ticker_mapping(self) -> None:
        handler = TradingViewWebhookHandler(secret="")
        result = await handler.handle(
            {"ticker": "XAUUSDTPERP", "action": "buy", "price": 2500}
        )
        self.assertEqual(result["metrics"]["ticker"], "XAUUSDT.P")


class TestTradingViewRoute(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["TRADINGVIEW_WEBHOOK_SECRET"] = "webhook-secret"

    def tearDown(self) -> None:
        os.environ.pop("TRADINGVIEW_WEBHOOK_SECRET", None)

    def test_post_tradingview_returns_advisory_only_ranking(self) -> None:
        from fastapi.testclient import TestClient

        from api.main import app

        fake_ranker = MagicMock()
        fake_ranker.run = AsyncMock(
            return_value={
                "ideas": [
                    {
                        "instrument": "XAUUSDT.P",
                        "direction": "long",
                        "raw_score": 0.25,
                        "adjusted_score": 0.25,
                        "rank": 1,
                        "rationale": "TradingView long aligned with config weight.",
                        "contributions": {"tradingview": 0.25},
                        "funding": None,
                    }
                ],
                "advisory_only": True,
                "execution_enabled": False,
            }
        )

        with patch("api.agents.RankerAgent", new=MagicMock(return_value=fake_ranker)):
            client = TestClient(app)
            response = client.post(
                "/webhooks/tradingview",
                json={
                    "ticker": "XAUUSDT.P",
                    "action": "buy",
                    "price": 2500,
                    "secret": "webhook-secret",
                },
            )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["advisory_only"])
        self.assertFalse(data["execution_enabled"])
        self.assertEqual(data["source_signal"]["name"], "tradingview")
        self.assertEqual(data["source_signal"]["direction"], "long")
        self.assertEqual(data["ranking"]["ideas"][0]["instrument"], "XAUUSDT.P")

    def test_post_tradingview_rejects_bad_secret(self) -> None:
        from fastapi.testclient import TestClient

        from api.main import app

        client = TestClient(app)
        response = client.post(
            "/webhooks/tradingview",
            json={
                "ticker": "XAUUSDT.P",
                "action": "buy",
                "price": 2500,
                "secret": "wrong",
            },
        )
        self.assertEqual(response.status_code, 401)

    def test_post_tradingview_rejects_unknown_ticker(self) -> None:
        from fastapi.testclient import TestClient

        from api.main import app

        with patch.dict(os.environ, {"TRADINGVIEW_WEBHOOK_SECRET": ""}):
            client = TestClient(app)
            response = client.post(
                "/webhooks/tradingview",
                json={
                    "ticker": "UNKNOWN",
                    "action": "buy",
                    "price": 100,
                },
            )
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
