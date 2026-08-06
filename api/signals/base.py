from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from ..adapters.base import Instrument


class Signal(ABC):
    name: str

    @abstractmethod
    def generate(self, instrument: Instrument, data: Any) -> Dict[str, Any]: ...
