from __future__ import annotations

from typing import Any, Dict

from ..adapters.base import Instrument
from ._util import (
    ConfiguredSignal,
    atr,
    clamp,
    direction_from_score,
    ema,
    extract_frame,
    safe_div,
)


class TrendSignal(ConfiguredSignal):
    """Fast/slow EMA relationship plus EMA slope, both ATR-normalized."""

    name = "trend"
    defaults: Dict[str, Any] = {
        "weight": 0.25,
        "fast": 20,
        "slow": 50,
        "atr_period": 14,
    }

    def generate(self, instrument: Instrument, data: Any) -> Dict[str, Any]:
        fast_period = self._int("fast", 20)
        slow_period = self._int("slow", 50)
        atr_period = self._int("atr_period", 14)

        frame = extract_frame(data)
        required = max(slow_period, atr_period) + 1
        if len(frame) < required:
            return self.insufficient(
                "insufficient_data",
                {"bars": len(frame), "required": required},
            )

        close = frame["close"]
        fast_ema = ema(close, fast_period)
        slow_ema = ema(close, slow_period)
        atr_series = atr(frame, atr_period)

        atr_now = float(atr_series.iloc[-1])
        fast_now = float(fast_ema.iloc[-1])
        slow_now = float(slow_ema.iloc[-1])

        # EMA separation in ATR units
        separation = safe_div(fast_now - slow_now, atr_now)

        # Slope of the fast EMA over the fast window, in ATR units per bar
        lookback = min(fast_period, len(fast_ema) - 1)
        slope_raw = (fast_now - float(fast_ema.iloc[-1 - lookback])) / max(lookback, 1)
        slope = safe_div(slope_raw, atr_now)

        separation_score = clamp(separation / 2.0)
        slope_score = clamp(slope * 10.0)
        score = clamp(0.6 * separation_score + 0.4 * slope_score)

        metrics = {
            "fast_ema": fast_now,
            "slow_ema": slow_now,
            "atr": atr_now,
            "separation_atr": separation,
            "slope_atr_per_bar": slope,
            "separation_score": separation_score,
            "slope_score": slope_score,
            "close": float(close.iloc[-1]),
            "bars": len(frame),
        }
        return self.result(score, direction_from_score(score), metrics)
