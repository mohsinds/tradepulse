from __future__ import annotations

from typing import Any, Dict

from ..adapters.base import Instrument
from ._util import (
    ConfiguredSignal,
    atr,
    clamp,
    direction_from_score,
    extract_frame,
    safe_div,
)


class BreakoutSignal(ConfiguredSignal):
    """Donchian channel breakout with an ATR buffer around the channel edges."""

    name = "breakout"
    defaults: Dict[str, Any] = {
        "weight": 0.15,
        "channel_period": 20,
        "atr_period": 14,
        "atr_buffer": 0.25,
    }

    def generate(self, instrument: Instrument, data: Any) -> Dict[str, Any]:
        channel_period = max(self._int("channel_period", 20), 2)
        atr_period = max(self._int("atr_period", 14), 1)
        atr_buffer = self._float("atr_buffer", 0.25)

        frame = extract_frame(data)
        required = max(channel_period, atr_period) + 1
        if len(frame) < required:
            return self.insufficient(
                "insufficient_data",
                {"bars": len(frame), "required": required},
            )

        # Channel is built from the prior N bars, excluding the current bar.
        prior = frame.iloc[-1 - channel_period : -1]
        upper = float(prior["high"].max())
        lower = float(prior["low"].min())
        mid = (upper + lower) / 2.0

        atr_now = float(atr(frame, atr_period).iloc[-1])
        buffer_abs = atr_buffer * atr_now
        upper_trigger = upper + buffer_abs
        lower_trigger = lower - buffer_abs

        close = float(frame["close"].iloc[-1])

        if close > upper_trigger:
            breakout = "up"
            distance = safe_div(close - upper_trigger, atr_now)
            score = clamp(0.5 + distance)
        elif close < lower_trigger:
            breakout = "down"
            distance = safe_div(lower_trigger - close, atr_now)
            score = -clamp(0.5 + distance)
        else:
            breakout = "none"
            width = max(upper - lower, 1e-12)
            # Mild tilt toward whichever edge price is leaning on.
            score = clamp(2.0 * (close - mid) / width * 0.25)
            distance = 0.0

        metrics = {
            "close": close,
            "channel_upper": upper,
            "channel_lower": lower,
            "channel_mid": mid,
            "channel_width": upper - lower,
            "atr": atr_now,
            "atr_buffer": atr_buffer,
            "upper_trigger": upper_trigger,
            "lower_trigger": lower_trigger,
            "breakout": breakout,
            "distance_atr": distance,
            "bars": len(frame),
        }
        return self.result(score, direction_from_score(score), metrics)
