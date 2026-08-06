import sys
import unittest
from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

from api.adapters import ContractType, Instrument, OrderRequest
from api.adapters.ibkr_venue import IBKRVenue


def _make_mock_module():
    """Build a fake ``ib_async`` module for unit testing."""
    mod = MagicMock(name="ib_async")
    shared_contract = MagicMock()
    mod.Contract = MagicMock(return_value=shared_contract)
    mod.Future = MagicMock(return_value=shared_contract)
    mod.Stock = MagicMock(return_value=shared_contract)
    mod.CFD = MagicMock(return_value=shared_contract)
    mod.Option = MagicMock(return_value=shared_contract)
    mod.LimitOrder = MagicMock(return_value=MagicMock())
    mod.MarketOrder = MagicMock(return_value=MagicMock())
    mod.StopOrder = MagicMock(return_value=MagicMock())
    mod.StopLimitOrder = MagicMock(return_value=MagicMock())
    mod.Order = MagicMock(return_value=MagicMock())

    ib = MagicMock()
    ib.connectAsync = AsyncMock()
    ib.disconnect = MagicMock()
    ib.qualifyContractsAsync = AsyncMock(return_value=[shared_contract])
    ib.reqMktData = MagicMock()
    ib.placeOrder = MagicMock()
    ib.cancelOrder = MagicMock()
    ib.positions = MagicMock(return_value=[])
    ib.accountSummary = MagicMock(return_value=[])
    ib.trades = MagicMock(return_value=[])

    fake_bar = MagicMock()
    fake_bar.date = datetime(2026, 4, 19, 12, 0, 0)
    fake_bar.open = 2300.0
    fake_bar.high = 2310.0
    fake_bar.low = 2290.0
    fake_bar.close = 2305.0
    fake_bar.volume = 100
    ib.reqHistoricalDataAsync = AsyncMock(return_value=[fake_bar])

    async def _ticker_stream():
        ticker = MagicMock()
        ticker.contract = shared_contract
        ticker.bid = 2000.0
        ticker.ask = 2001.0
        ticker.last = 2000.5
        ticker.close = 1999.0
        ticker.volume = 50
        yield [ticker]

    ib.pendingTickersEvent = _ticker_stream()
    mod.IB = MagicMock(return_value=ib)
    return mod, ib


class TestIBKRVenue(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.mock_module, self.mock_ib = _make_mock_module()

    def _patch(self):
        return patch.dict(sys.modules, {"ib_async": self.mock_module})

    async def test_connect_and_disconnect(self):
        with self._patch():
            venue = IBKRVenue(
                {"host": "127.0.0.1", "port": 7497, "client_id": 10}
            )
            await venue.connect()
            self.mock_module.IB.assert_called_once()
            self.mock_ib.connectAsync.assert_awaited_once_with(
                "127.0.0.1", 7497, 10, 4
            )
            self.assertTrue(venue.connected)

            await venue.disconnect()
            self.mock_ib.disconnect.assert_called_once()
            self.assertFalse(venue.connected)

    async def test_get_historical_bars_xauusd(self):
        with self._patch():
            venue = IBKRVenue({})
            await venue.connect()
            inst = Instrument(
                symbol="XAUUSD",
                venue="ibkr",
                contract_type=ContractType.SPOT,
                currency="USD",
            )
            start = datetime(2026, 4, 19, 0, 0, 0)
            end = datetime(2026, 4, 20, 0, 0, 0)
            bars = await venue.get_historical_bars(inst, start, end, "1h")
            self.mock_ib.reqHistoricalDataAsync.assert_awaited_once()
            self.mock_module.Contract.assert_any_call(
                secType="CMDTY",
                symbol="XAUUSD",
                exchange="SMART",
                currency="USD",
            )
            self.assertEqual(len(bars), 1)
            self.assertEqual(bars[0].close, Decimal("2305.0"))

    async def test_get_historical_bars_mgc_no_rollover(self):
        with self._patch():
            venue = IBKRVenue({})
            await venue.connect()
            inst = Instrument(
                symbol="MGC",
                venue="ibkr",
                contract_type=ContractType.FUTURE,
                exchange="COMEX",
            )
            start = datetime(2026, 4, 1, 0, 0, 0)
            end = datetime(2026, 4, 20, 0, 0, 0)
            await venue.get_historical_bars(inst, start, end, "1h")
            self.mock_module.Contract.assert_any_call(
                secType="FUT",
                symbol="MGC",
                exchange="COMEX",
                currency="USD",
                lastTradeDateOrContractMonth="202604",
            )

    async def test_get_historical_bars_mgc_rollover(self):
        with self._patch():
            venue = IBKRVenue({})
            await venue.connect()
            inst = Instrument(
                symbol="MGC",
                venue="ibkr",
                contract_type=ContractType.FUTURE,
                exchange="COMEX",
            )
            start = datetime(2026, 4, 1, 0, 0, 0)
            end = datetime(2026, 4, 25, 0, 0, 0)
            await venue.get_historical_bars(inst, start, end, "1h")
            self.mock_module.Contract.assert_any_call(
                secType="FUT",
                symbol="MGC",
                exchange="COMEX",
                currency="USD",
                lastTradeDateOrContractMonth="202605",
            )

    async def test_place_order(self):
        trade = MagicMock()
        trade.order.orderId = 123
        self.mock_ib.placeOrder.return_value = trade

        with self._patch():
            venue = IBKRVenue({})
            await venue.connect()
            inst = Instrument(
                symbol="XAUUSD",
                venue="ibkr",
                contract_type=ContractType.SPOT,
            )
            order = OrderRequest(
                instrument=inst,
                side="buy",
                quantity=Decimal("10"),
                order_type="limit",
                price=Decimal("2000.5"),
                time_in_force="gtc",
            )
            order_id = await venue.place_order(order)
            self.assertEqual(order_id, "123")
            self.mock_module.LimitOrder.assert_called_once()
            args, kwargs = self.mock_module.LimitOrder.call_args
            self.assertEqual(args[0], "BUY")
            self.assertEqual(args[1], 10.0)
            self.assertEqual(args[2], 2000.5)
            self.assertEqual(kwargs["tif"], "GTC")
            self.mock_ib.placeOrder.assert_called_once()

    async def test_subscribe_quotes(self):
        with self._patch():
            venue = IBKRVenue({})
            await venue.connect()
            inst = Instrument(
                symbol="XAUUSD",
                venue="ibkr",
                contract_type=ContractType.SPOT,
            )
            quotes = []
            async for q in venue.subscribe_quotes(inst):
                quotes.append(q)
                break
            self.mock_ib.reqMktData.assert_called_once()
            self.assertEqual(len(quotes), 1)
            self.assertEqual(quotes[0]["symbol"], "XAUUSD")

    async def test_get_positions(self):
        pos = MagicMock()
        pos.contract.symbol = "MGC"
        pos.contract.currency = "USD"
        pos.position = 2
        pos.avgCost = 1800.0
        self.mock_ib.positions.return_value = [pos]

        with self._patch():
            venue = IBKRVenue({})
            await venue.connect()
            positions = await venue.get_positions()
            self.mock_ib.positions.assert_called_once()
            self.assertEqual(positions[0]["symbol"], "MGC")
            self.assertEqual(positions[0]["position"], 2.0)
            self.assertEqual(positions[0]["avg_cost"], 1800.0)

    async def test_get_account(self):
        av = MagicMock()
        av.tag = "CashBalance"
        av.value = "12345.67"
        self.mock_ib.accountSummary.return_value = [av]

        with self._patch():
            venue = IBKRVenue({})
            await venue.connect()
            account = await venue.get_account()
            self.mock_ib.accountSummary.assert_called_once()
            self.assertEqual(account["CashBalance"], "12345.67")


if __name__ == "__main__":
    unittest.main()
