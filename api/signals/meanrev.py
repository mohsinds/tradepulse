from __future__ import annotations

import numpy as np

from typing import Any, Dict

from ..adapters.base import Instrument
from ._util import (
    ConfiguredSignal,
    clamp,
    direction_from_score,
    extract_frame,
    rsi,
    zscore,
)


class MeanReversionSignal(ConfiguredSignal):
    """RSI extremes combined with a close-price z-score.

    Mean reversion is contrarian: stretched-high readings are bearish, stretched
    -low readings are bullish.
    """

    name = "meanrev"
    defaults: Dict[str, Any] = {
        "weight": 0.15,
        "rsi_period": 14,
        "oversold": 30,
        "overbought": 70,
        "zscore_period": 20,
    }

    def generate(self, instrument: Instrument, data: Any) -> Dict[str, Any]:
        rsi_period = max(self._int("rsi_period", 14), 2)
        zscore_period = max(self._int("zscore_period", 20), 2)
        oversold = self._float("oversold", 30.0)
        overbought = self._float("overbought", 70.0)

        frame = extract_frame(data)
        required = max(rsi_period + 1, zscore_period)
        if len(frame) < required:
            return self.insufficient(
                "insufficient_data",
                {"bars": len(frame), "required": required},
            )

        close = frame["close"].astype(float)
        rsi_series = rsi(close, rsi_period)
        rsi_now = float(rsi_series.iloc[-1])
        if not np.isfinite(rsi_now):
            return self.insufficient(
                "insufficient_data",
                {"bars": len(frame), "required": required},
            )

        z = zscore(close, zscore_period)

        # RSI component: 0 inside the band, ramping to +/-1 at 0 / 100.
        if rsi_now < oversold:
            rsi_score = min((oversold - rsi_now) / max(oversold, 1e-9), 1.0)
        elif rsi_now > overbought:
            span = max(100.0 - overbought, 1e-9)
            rsi_score = -min((rsi_now - overbought) / span, 1.0)
        else:
            rsi_score = 0.0

        # Z-score component: fade the move, saturating around 2 sigma.
        z_score_component = clamp(-z / 2.0)

        score = clamp(0.6 * rsi_score + 0.4 * z_score_component)

        stretched = rsi_now < oversold or rsi_now > overbought or abs(z) >= 2.0

        metrics = {
            "rsi": rsi_now,
            "oversold": oversold,
            "overbought": overbought,
            "zscore": z,
            "rsi_score": rsi_score,
            "zscore_score": z_score_component,
            "stretched": bool(stretched),
            "close": float(close.iloc[-1]),
            "bars": len(frame),
        }
        return self.result(score, direction_from_score(score), metrics)
