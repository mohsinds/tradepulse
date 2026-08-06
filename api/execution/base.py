from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from ..adapters.base import BrokerAdapter, Instrument, OrderRequest


class ExecutionEngine(ABC):
    @abstractmethod
    async def submit(self, broker: BrokerAdapter, order: OrderRequest) -> Dict[str, Any]: ...

    @abstractmethod
    async def route(self, instrument: Instrument, signal: Dict[str, Any]) -> OrderRequest: ...

    @abstractmethod
    async def execute_idea(self, idea: Dict[str, Any]) -> Dict[str, Any]: ...
