from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional


class ContractType(str, Enum):
    SPOT = "spot"
    FUTURE = "future"
    PERPETUAL = "perpetual"
    OPTION = "option"
    CFD = "cfd"


@dataclass
class Instrument:
    symbol: str
    venue: str
    contract_type: ContractType
    exchange: str = ""
    underlying: str = ""
    multiplier: Decimal = Decimal("1")
    currency: str = "USD"
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Bar:
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal = Decimal("0")


@dataclass
class OrderRequest:
    instrument: Instrument
    side: str
    quantity: Decimal
    order_type: str
    price: Optional[Decimal] = None
    time_in_force: str = "day"


class DataProvider(ABC):
    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def get_historical_bars(
        self,
        instrument: Instrument,
        start: datetime,
        end: datetime,
        timeframe: str,
    ) -> List[Bar]: ...

    @abstractmethod
    async def subscribe_quotes(
        self, instrument: Instrument
    ) -> AsyncIterator[Dict[str, Any]]: ...

    @abstractmethod
    async def get_instruments(self) -> List[Instrument]: ...


class BrokerAdapter(ABC):
    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def place_order(self, order: OrderRequest) -> str: ...

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool: ...

    @abstractmethod
    async def get_positions(self) -> List[Dict[str, Any]]: ...

    @abstractmethod
    async def get_account(self) -> Dict[str, Any]: ...

    @abstractmethod
    async def get_order_status(self, order_id: str) -> Dict[str, Any]: ...
