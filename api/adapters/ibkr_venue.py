from __future__ import annotations

import calendar
from datetime import datetime
from decimal import Decimal
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from .base import Bar, BrokerAdapter, DataProvider, Instrument, OrderRequest


class IBKRVenue(DataProvider, BrokerAdapter):
    """Interactive Brokers venue adapter for multiple contract types."""

    name = "ibkr"

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self._ib: Optional[Any] = None
        self._connected = False
        self._roll_days = int(self.config.get("roll_days", 5))

    @property
    def connected(self) -> bool:
        return self._connected

    def supports(self, instrument: Instrument) -> bool:
        return instrument.venue.lower() == self.name

    async def connect(self) -> None:
        import ib_async

        self._ib = ib_async.IB()
        await self._ib.connectAsync(
            self.config.get("host", "127.0.0.1"),
            self.config.get("port", 7497),
            self.config.get("client_id", 1),
            self.config.get("timeout", 4),
        )
        self._connected = True

    async def disconnect(self) -> None:
        if self._ib is not None:
            self._ib.disconnect()
            self._ib = None
        self._connected = False

    def _mgc_front_month(self, as_of: datetime) -> str:
        """Return the active MGC contract month, rolling early near expiry."""
        year, month = as_of.year, as_of.month
        last_day = calendar.monthrange(year, month)[1]
        expiry = datetime(year, month, last_day)
        if (expiry - as_of).days <= self._roll_days:
            month += 1
            if month > 12:
                month = 1
                year += 1
        return f"{year}{month:02d}"

    async def _resolve_contract(
        self, instrument: Instrument, as_of: Optional[datetime] = None
    ) -> Any:
        import ib_async

        as_of = as_of or datetime.now()
        symbol = instrument.symbol.upper()
        exchange = instrument.exchange or self.config.get("exchange", "SMART")
        currency = instrument.currency or "USD"
        ctype = instrument.contract_type

        if symbol == "XAUUSD":
            contract = ib_async.Contract(
                secType="CMDTY",
                symbol="XAUUSD",
                exchange=exchange,
                currency=currency,
            )
        elif ctype.value == "spot":
            contract = ib_async.Stock(symbol, exchange, currency)
        elif ctype.value == "future":
            if symbol == "MGC":
                month = self._mgc_front_month(as_of)
                contract = ib_async.Contract(
                    secType="FUT",
                    symbol="MGC",
                    exchange="COMEX",
                    currency="USD",
                    lastTradeDateOrContractMonth=month,
                )
            else:
                expiry = instrument.meta.get("expiry")
                if expiry:
                    contract = ib_async.Future(
                        symbol=symbol,
                        lastTradeDateOrContractMonth=expiry,
                        exchange=exchange,
                        currency=currency,
                    )
                else:
                    contract = ib_async.Contract(
                        secType="FUT",
                        symbol=symbol,
                        exchange=exchange,
                        currency=currency,
                    )
        elif ctype.value == "perpetual":
            contract = ib_async.CFD(symbol, exchange, currency)
        elif ctype.value == "cfd":
            contract = ib_async.CFD(symbol, exchange, currency)
        elif ctype.value == "option":
            meta = instrument.meta
            contract = ib_async.Option(
                symbol=symbol,
                lastTradeDateOrContractMonth=meta.get("expiry"),
                strike=float(meta.get("strike", 0)),
                right=meta.get("right", "C"),
                exchange=exchange,
                multiplier=float(instrument.multiplier),
                currency=currency,
            )
        else:
            contract = ib_async.Contract(
                secType="STK",
                symbol=symbol,
                exchange=exchange,
                currency=currency,
            )

        qualified = await self._ib.qualifyContractsAsync(contract)
        if not qualified:
            raise ValueError(f"Could not qualify contract for {instrument}")
        return qualified[0]

    async def get_historical_bars(
        self,
        instrument: Instrument,
        start: datetime,
        end: datetime,
        timeframe: str,
    ) -> List[Bar]:
        if not self._connected or self._ib is None:
            raise ConnectionError("IBKR venue is not connected")

        contract = await self._resolve_contract(instrument, as_of=end)

        duration_days = max(1, (end - start).days)
        duration_str = f"{duration_days} D"

        bar_size = self._timeframe_to_bar_size(timeframe)
        what_to_show = "MIDPOINT" if instrument.symbol.upper() == "XAUUSD" else "TRADES"

        raw_bars = await self._ib.reqHistoricalDataAsync(
            contract,
            endDateTime=end.strftime("%Y%m%d %H:%M:%S"),
            durationStr=duration_str,
            barSizeSetting=bar_size,
            whatToShow=what_to_show,
            useRTH=False,
            formatDate=1,
        )

        result: List[Bar] = []
        for b in raw_bars or []:
            ts = b.date if isinstance(b.date, datetime) else datetime.strptime(
                str(b.date), "%Y%m%d %H:%M:%S"
            )
            if start <= ts <= end:
                result.append(
                    Bar(
                        timestamp=ts,
                        open=Decimal(str(b.open)),
                        high=Decimal(str(b.high)),
                        low=Decimal(str(b.low)),
                        close=Decimal(str(b.close)),
                        volume=Decimal(str(b.volume or 0)),
                    )
                )
        return result

    def _timeframe_to_bar_size(self, timeframe: str) -> str:
        mapping = {
            "1m": "1 min",
            "5m": "5 mins",
            "15m": "15 mins",
            "30m": "30 mins",
            "1h": "1 hour",
            "4h": "4 hours",
            "1d": "1 day",
        }
        return mapping.get(timeframe.lower(), timeframe)

    async def subscribe_quotes(
        self, instrument: Instrument
    ) -> AsyncIterator[Dict[str, Any]]:
        if not self._connected or self._ib is None:
            raise ConnectionError("IBKR venue is not connected")

        contract = await self._resolve_contract(instrument)
        self._ib.reqMktData(contract, "", False, False)

        async for tickers in self._ib.pendingTickersEvent:
            for t in tickers:
                if t.contract is contract or getattr(t.contract, "conId", None) == getattr(
                    contract, "conId", None
                ):
                    yield {
                        "symbol": instrument.symbol,
                        "bid": t.bid,
                        "ask": t.ask,
                        "last": t.last,
                        "close": t.close,
                        "volume": t.volume,
                    }

    async def get_instruments(self) -> List[Instrument]:
        return []

    async def place_order(self, order: OrderRequest) -> str:
        if not self._connected or self._ib is None:
            raise ConnectionError("IBKR venue is not connected")

        contract = await self._resolve_contract(order.instrument)
        ib_order = self._make_ib_order(order)
        trade = self._ib.placeOrder(contract, ib_order)
        return str(trade.order.orderId)

    def _make_ib_order(self, order: OrderRequest) -> Any:
        import ib_async

        action = order.side.upper()
        total = float(order.quantity)
        tif = self._map_tif(order.time_in_force)
        order_type = (order.order_type or "market").lower()

        if order_type == "limit":
            return ib_async.LimitOrder(
                action, total, float(order.price), tif=tif
            )
        if order_type == "stop":
            return ib_async.StopOrder(action, total, float(order.price), tif=tif)
        if order_type == "stop_limit":
            return ib_async.StopLimitOrder(
                action,
                total,
                float(order.price),
                float(order.price),
                tif=tif,
            )
        return ib_async.MarketOrder(action, total, tif=tif)

    def _map_tif(self, time_in_force: str) -> str:
        mapping = {
            "day": "DAY",
            "gtc": "GTC",
            "ioc": "IOC",
            "fok": "FOK",
        }
        return mapping.get(time_in_force.lower(), time_in_force.upper())

    async def cancel_order(self, order_id: str) -> bool:
        if not self._connected or self._ib is None:
            raise ConnectionError("IBKR venue is not connected")

        import ib_async

        self._ib.cancelOrder(ib_async.Order(orderId=int(order_id)))
        return True

    async def get_positions(self) -> List[Dict[str, Any]]:
        if not self._connected or self._ib is None:
            raise ConnectionError("IBKR venue is not connected")

        positions = self._ib.positions() or []
        return [
            {
                "symbol": p.contract.symbol,
                "position": float(p.position),
                "avg_cost": float(p.avgCost),
                "currency": p.contract.currency,
            }
            for p in positions
        ]

    async def get_account(self) -> Dict[str, Any]:
        if not self._connected or self._ib is None:
            raise ConnectionError("IBKR venue is not connected")

        summary = self._ib.accountSummary() or []
        return {item.tag: item.value for item in summary}

    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        if not self._connected or self._ib is None:
            raise ConnectionError("IBKR venue is not connected")

        for trade in self._ib.trades() or []:
            if str(getattr(trade.order, "orderId", None)) == str(order_id):
                return {
                    "order_id": str(order_id),
                    "status": getattr(trade.orderStatus, "status", ""),
                    "filled": getattr(trade.orderStatus, "filled", 0),
                    "remaining": getattr(trade.orderStatus, "remaining", 0),
                }
        return {"order_id": str(order_id), "status": "not_found"}
