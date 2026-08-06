from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class WebhookHandler(ABC):
    @abstractmethod
    async def handle(self, payload: Dict[str, Any]) -> Dict[str, Any]: ...
