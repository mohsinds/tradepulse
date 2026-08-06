from __future__ import annotations

from typing import Any, Dict, List, Type

from .base import Signal
from .breakout import BreakoutSignal
from .funding_rate import FundingRateSignal
from .meanrev import MeanReversionSignal
from .momentum import MomentumSignal
from .trend import TrendSignal
from .volatility import VolatilitySignal
from .volume import VolumeSignal

SIGNAL_REGISTRY: Dict[str, Type[Signal]] = {
    TrendSignal.name: TrendSignal,
    MomentumSignal.name: MomentumSignal,
    MeanReversionSignal.name: MeanReversionSignal,
    BreakoutSignal.name: BreakoutSignal,
    VolatilitySignal.name: VolatilitySignal,
    VolumeSignal.name: VolumeSignal,
    FundingRateSignal.name: FundingRateSignal,
}


def build_signals(merged_config: Dict[str, Any]) -> List[Signal]:
    """Instantiate every known signal present in a merged instrument config.

    Accepts either the merged signal block itself
    (``{"trend": {...}, "momentum": {...}}``) or a full instrument config
    containing a ``signals`` key. Unknown block names are ignored.
    """
    if not merged_config:
        return []

    blocks = merged_config
    if "signals" in merged_config and isinstance(merged_config["signals"], dict):
        blocks = merged_config["signals"]

    signals: List[Signal] = []
    for name, cls in SIGNAL_REGISTRY.items():
        if name in blocks:
            block = blocks.get(name) or {}
            if not isinstance(block, dict):
                block = {}
            signals.append(cls(block))  # type: ignore[call-arg]
    return signals


__all__ = [
    "Signal",
    "TrendSignal",
    "MomentumSignal",
    "MeanReversionSignal",
    "BreakoutSignal",
    "VolatilitySignal",
    "VolumeSignal",
    "FundingRateSignal",
    "SIGNAL_REGISTRY",
    "build_signals",
]
