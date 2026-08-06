from __future__ import annotations

import asyncio
from datetime import datetime
from decimal import Decimal
from typing import Any, AsyncIterator, Dict, List, Optional

from .base import (
    Bar,
    BrokerAdapter,
    ContractType,
    DataProvider,
    Instrument,
    OrderRequest,
)


class CCXTVenue(DataProvider, BrokerAdapter):
    """CCXT venue adapter for multiple contract types across exchanges."""

    name = "ccxt"

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self._connected = False
        self._exchange = None
        self._exchange_id = config.get("exchange_id", "bybit")

    @property
    def connected(self) -> bool:
        return self._connected

    def supports(self, instrument: Instrument) -> bool:
        return instrument.venue.lower() == self.name

    async def connect(self) -> None:
        import ccxt.async_support as ccxt

        exchange_cls = getattr(ccxt, self._exchange_id)
        self._exchange = exchange_cls(
            {
                "apiKey": self.config.get("api_key"),
                "secret": self.config.get("api_secret"),
                "password": self.config.get("password"),
                "sandbox": self.config.get("sandbox", False),
                "enableRateLimit": True,
            }
        )
        await self._exchange.load_markets()
        self._connected = True

    async def disconnect(self) -> None:
        if self._exchange is not None:
            try:
                await self._exchange.close()
            except Exception:
                pass
            self._exchange = None
        self._connected = False

    def _ccxt_symbol(self, instrument: Instrument) -> str:
        """Map adapter symbols to CCXT market symbols."""
        if self._exchange_id == "bybit":
            if (
                instrument.symbol == "XAUUSDT.P"
                and instrument.contract_type == ContractType.PERPETUAL
            ):
                return "XAUUSDT:USDT"
            if (
                instrument.symbol == "BTC/USD"
                and instrument.contract_type == ContractType.SPOT
            ):
                return "BTC/USDT"
        return instrument.symbol

    def _timeframe_to_ms(self, timeframe: str) -> int:
        if timeframe.endswith("m"):
            return int(timeframe[:-1]) * 60_000
        if timeframe.endswith("h"):
            return int(timeframe[:-1]) * 60 * 60_000
        if timeframe.endswith("d"):
            return int(timeframe[:-1]) * 24 * 60 * 60_000
        if timeframe.endswith("w"):
            return int(timeframe[:-1]) * 7 * 24 * 60 * 60_000
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    async def get_historical_bars(
        self,
        instrument: Instrument,
        start: datetime,
        end: datetime,
        timeframe: str,
    ) -> List[Bar]:
        symbol = self._ccxt_symbol(instrument)
        tf_ms = self._timeframe_to_ms(timeframe)
        start_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        since = start_ms
        limit = 1000
        bars: List[Bar] = []

        while since < end_ms:
            ohlcv = await self._exchange.fetch_ohlcv(
                symbol, timeframe, since=since, limit=limit
            )
            if not ohlcv:
                break

            for row in ohlcv:
                ts_ms, open_, high, low, close, volume = row
                if ts_ms > end_ms:
                    break
                bars.append(
                    Bar(
                        timestamp=datetime.fromtimestamp(ts_ms / 1000),
                        open=Decimal(str(open_)),
                        high=Decimal(str(high)),
                        low=Decimal(str(low)),
                        close=Decimal(str(close)),
                        volume=Decimal(str(volume)),
                    )
                )

            since += limit * tf_ms
            if len(ohlcv) < limit:
                break

        return bars

    async def subscribe_quotes(
        self, instrument: Instrument
    ) -> AsyncIterator[Dict[str, Any]]:
        symbol = self._ccxt_symbol(instrument)
        is_perp = instrument.contract_type == ContractType.PERPETUAL

        while self.connected:
            ticker = await self._exchange.fetch_ticker(symbol)
            quote: Dict[str, Any] = {
                "symbol": symbol,
                "timestamp": ticker.get("timestamp"),
                "bid": (
                    Decimal(str(ticker["bid"]))
                    if ticker.get("bid") is not None
                    else None
                ),
                "ask": (
                    Decimal(str(ticker["ask"]))
                    if ticker.get("ask") is not None
                    else None
                ),
                "last": (
                    Decimal(str(ticker["last"]))
                    if ticker.get("last") is not None
                    else None
                ),
            }

            if is_perp:
                funding = await self._exchange.fetchFundingRate(symbol)
                quote["funding_rate"] = (
                    Decimal(str(funding["fundingRate"]))
                    if funding.get("fundingRate") is not None
                    else None
                )
                quote["funding_timestamp"] = funding.get("fundingTimestamp")

            yield quote
            await asyncio.sleep(1)

    def _map_ccxt_type(self, ccxt_type: Optional[str]) -> Optional[ContractType]:
        mapping = {
            "spot": ContractType.SPOT,
            "margin": ContractType.SPOT,
            "swap": ContractType.PERPETUAL,
            "future": ContractType.FUTURE,
            "option": ContractType.OPTION,
            "cfd": ContractType.CFD,
        }
        return mapping.get(ccxt_type)

    async def get_instruments(self) -> List[Instrument]:
        markets = await self._exchange.load_markets()
        instruments: List[Instrument] = []

        for symbol, market in markets.items():
            ctype = self._map_ccxt_type(market.get("type"))
            if ctype is None:
                continue
            instruments.append(
                Instrument(
                    symbol=symbol,
                    venue=self.name,
                    contract_type=ctype,
                    exchange=self._exchange_id,
                    underlying=market.get("base", ""),
                    currency=market.get("quote", ""),
                    meta=market,
                )
            )

        return instruments

    async def place_order(self, order: OrderRequest) -> str:
        symbol = self._ccxt_symbol(order.instrument)
        side = order.side.lower()
        order_type = order.order_type.lower()
        amount = float(order.quantity)
        price = float(order.price) if order.price is not None else None

        result = await self._exchange.create_order(
            symbol, order_type, side, amount, price
        )
        return str(result["id"])

    async def cancel_order(self, order_id: str) -> bool:
        try:
            await self._exchange.cancel_order(order_id)
            return True
        except Exception:
            return False

    async def get_positions(self) -> List[Dict[str, Any]]:
        positions = await self._exchange.fetch_positions()
        return [
            {
                "symbol": p.get("symbol"),
                "contracts": p.get("contracts"),
                "notional": p.get("notional"),
                "unrealized_pnl": p.get("unrealizedPnl"),
                "side": p.get("side"),
            }
            for p in positions
        ]

    async def get_account(self) -> Dict[str, Any]:
        balance = await self._exchange.fetch_balance()
        return {
            "exchange": self._exchange_id,
            "free": balance.get("free", {}),
            "total": balance.get("total", {}),
            "used": balance.get("used", {}),
        }

    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        return await self._exchange.fetch_order(order_id)
