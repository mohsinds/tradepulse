import sys
import types
import unittest
from datetime import datetime, timezone
from decimal import Decimal
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, patch

from api.adapters.base import Bar, ContractType, Instrument, OrderRequest
from api.adapters.ccxt_venue import CCXTVenue


def _make_exchange():
    ex = MagicMock()
    ex.load_markets = AsyncMock(
        return_value={
            "XAUUSDT:USDT": {
                "symbol": "XAUUSDT:USDT",
                "type": "swap",
                "base": "XAU",
                "quote": "USDT",
            },
            "BTC/USDT": {
                "symbol": "BTC/USDT",
                "type": "spot",
                "base": "BTC",
                "quote": "USDT",
            },
        }
    )
    ex.fetch_ohlcv = AsyncMock(
        return_value=[
            [1609459200000, 100.0, 110.0, 90.0, 105.0, 1000.0],
            [1609462800000, 105.0, 115.0, 95.0, 110.0, 2000.0],
        ]
    )
    ex.fetch_ticker = AsyncMock(
        return_value={
            "timestamp": 1609459200000,
            "bid": 100.0,
            "ask": 101.0,
            "last": 100.5,
        }
    )
    ex.fetchFundingRate = AsyncMock(
        return_value={
            "fundingRate": "0.0001",
            "fundingTimestamp": 1609459200000,
        }
    )
    ex.create_order = AsyncMock(return_value={"id": "order-123"})
    ex.cancel_order = AsyncMock()
    ex.fetch_positions = AsyncMock(
        return_value=[
            {
                "symbol": "XAUUSDT:USDT",
                "contracts": Decimal("1.5"),
                "notional": Decimal("150.0"),
                "unrealizedPnl": Decimal("5.0"),
                "side": "long",
            }
        ]
    )
    ex.fetch_balance = AsyncMock(
        return_value={
            "free": {"USDT": 1000.0},
            "total": {"USDT": 1500.0},
            "used": {"USDT": 500.0},
        }
    )
    ex.fetch_order = AsyncMock(return_value={"id": "order-123", "status": "closed"})
    ex.close = AsyncMock()
    return ex


class TestCCXTVenue(IsolatedAsyncioTestCase):
    def setUp(self):
        self.config = {
            "exchange_id": "bybit",
            "api_key": "test-key",
            "api_secret": "test-secret",
        }
        ccxt_mod = types.ModuleType("ccxt")
        ccxt_async = types.ModuleType("ccxt.async_support")
        self.exchange = _make_exchange()
        ccxt_async.bybit = MagicMock(return_value=self.exchange)

        self.modules_patch = patch.dict(
            sys.modules,
            {
                "ccxt": ccxt_mod,
                "ccxt.async_support": ccxt_async,
            },
        )
        self.modules_patch.start()
        self.addCleanup(self.modules_patch.stop)

    async def _connect(self):
        venue = CCXTVenue(self.config)
        await venue.connect()
        return venue

    async def test_connect(self):
        venue = await self._connect()
        self.assertTrue(venue.connected)
        self.assertIs(venue._exchange, self.exchange)
        self.exchange.load_markets.assert_awaited_once()

    async def test_disconnect(self):
        venue = await self._connect()
        await venue.disconnect()
        self.assertFalse(venue.connected)
        self.assertIsNone(venue._exchange)
        self.exchange.close.assert_awaited_once()

    async def test_get_historical_bars(self):
        venue = await self._connect()
        instrument = Instrument(
            symbol="XAUUSDT.P",
            venue="ccxt",
            contract_type=ContractType.PERPETUAL,
        )
        start = datetime(2021, 1, 1, 0, 0, tzinfo=timezone.utc)
        end = datetime(2021, 1, 1, 1, 0, tzinfo=timezone.utc)
        bars = await venue.get_historical_bars(instrument, start, end, "1h")
        self.assertEqual(len(bars), 2)
        self.assertIsInstance(bars[0], Bar)
        self.assertEqual(bars[0].close, Decimal("105"))
        self.exchange.fetch_ohlcv.assert_awaited_with(
            "XAUUSDT:USDT",
            "1h",
            since=int(start.timestamp() * 1000),
            limit=1000,
        )

    async def test_subscribe_quotes_perpetual(self):
        venue = await self._connect()
        instrument = Instrument(
            symbol="XAUUSDT.P",
            venue="ccxt",
            contract_type=ContractType.PERPETUAL,
        )
        quotes = []
        async for quote in venue.subscribe_quotes(instrument):
            quotes.append(quote)
            break
        self.assertEqual(len(quotes), 1)
        self.assertIn("funding_rate", quotes[0])
        self.assertEqual(quotes[0]["funding_rate"], Decimal("0.0001"))
        self.exchange.fetch_ticker.assert_awaited_with("XAUUSDT:USDT")
        self.exchange.fetchFundingRate.assert_awaited_with("XAUUSDT:USDT")

    async def test_subscribe_quotes_spot(self):
        venue = await self._connect()
        instrument = Instrument(
            symbol="BTC/USD",
            venue="ccxt",
            contract_type=ContractType.SPOT,
        )
        quotes = []
        async for quote in venue.subscribe_quotes(instrument):
            quotes.append(quote)
            break
        self.assertEqual(len(quotes), 1)
        self.assertNotIn("funding_rate", quotes[0])
        self.exchange.fetch_ticker.assert_awaited_with("BTC/USDT")
        self.exchange.fetchFundingRate.assert_not_awaited()

    async def test_place_order(self):
        venue = await self._connect()
        instrument = Instrument(
            symbol="XAUUSDT.P",
            venue="ccxt",
            contract_type=ContractType.PERPETUAL,
        )
        order = OrderRequest(
            instrument=instrument,
            side="buy",
            quantity=Decimal("0.5"),
            order_type="limit",
            price=Decimal("100.0"),
            time_in_force="gtc",
        )
        order_id = await venue.place_order(order)
        self.assertEqual(order_id, "order-123")
        self.exchange.create_order.assert_awaited_with(
            "XAUUSDT:USDT",
            "limit",
            "buy",
            0.5,
            100.0,
        )

    async def test_get_positions(self):
        venue = await self._connect()
        positions = await venue.get_positions()
        self.assertEqual(len(positions), 1)
        self.assertEqual(positions[0]["symbol"], "XAUUSDT:USDT")
        self.exchange.fetch_positions.assert_awaited_once()

    async def test_get_account(self):
        venue = await self._connect()
        account = await venue.get_account()
        self.assertIn("free", account)
        self.assertEqual(account["free"]["USDT"], 1000.0)
        self.exchange.fetch_balance.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
