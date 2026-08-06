from __future__ import annotations

from typing import Any, Dict

from ..adapters.base import Instrument
from ._util import (
    ConfiguredSignal,
    clamp,
    direction_from_score,
    extract_frame,
    safe_div,
)


class MomentumSignal(ConfiguredSignal):
    """Rate-of-change over ``lookback`` bars measured against ``threshold``."""

    name = "momentum"
    defaults: Dict[str, Any] = {
        "weight": 0.2,
        "lookback": 20,
        "threshold": 0.05,
    }

    def generate(self, instrument: Instrument, data: Any) -> Dict[str, Any]:
        lookback = max(self._int("lookback", 20), 1)
        threshold = abs(self._float("threshold", 0.05)) or 0.05

        frame = extract_frame(data)
        required = lookback + 1
        if len(frame) < required:
            return self.insufficient(
                "insufficient_data",
                {"bars": len(frame), "required": required},
            )

        close = frame["close"].astype(float)
        now = float(close.iloc[-1])
        past = float(close.iloc[-1 - lookback])
        roc = safe_div(now - past, abs(past))

        score = clamp(roc / threshold)

        metrics = {
            "close": now,
            "reference_close": past,
            "roc": roc,
            "roc_pct": roc * 100.0,
            "threshold": threshold,
            "threshold_multiple": safe_div(roc, threshold),
            "lookback": lookback,
            "bars": len(frame),
        }
        return self.result(score, direction_from_score(score), metrics)
