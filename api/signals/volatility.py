from __future__ import annotations

from typing import Any, Dict

from ..adapters.base import Instrument
from ._util import (
    ConfiguredSignal,
    atr,
    clamp,
    extract_frame,
    percentile_rank,
    safe_div,
)


class VolatilitySignal(ConfiguredSignal):
    """ATR level, ATR percentile against the regime window, and squeeze detection.

    Volatility is a *regime* read, not a directional one, so ``direction`` is
    always ``"flat"``. ``score`` expresses regime favorability: compression
    (a squeeze) scores positive because it precedes expansion, while an already
    elevated / blown-out ATR scores negative.
    """

    name = "volatility"
    defaults: Dict[str, Any] = {
        "weight": 0.1,
        "atr_period": 14,
        "regime_period": 100,
        "squeeze_percentile": 0.25,
    }

    def generate(self, instrument: Instrument, data: Any) -> Dict[str, Any]:
        atr_period = max(self._int("atr_period", 14), 1)
        regime_period = max(self._int("regime_period", 100), 2)
        squeeze_percentile = self._float("squeeze_percentile", 0.25)

        frame = extract_frame(data)
        required = atr_period + 2
        if len(frame) < required:
            return self.insufficient(
                "insufficient_data",
                {"bars": len(frame), "required": required},
            )

        atr_series = atr(frame, atr_period)
        atr_now = float(atr_series.iloc[-1])
        close = float(frame["close"].iloc[-1])

        window = atr_series.tail(regime_period)
        pct = percentile_rank(window, atr_now)

        atr_pct_of_price = safe_div(atr_now, abs(close))
        squeeze = pct <= squeeze_percentile
        expansion = pct >= 1.0 - squeeze_percentile

        if squeeze:
            score = clamp((squeeze_percentile - pct) / max(squeeze_percentile, 1e-9))
        elif expansion:
            top = max(squeeze_percentile, 1e-9)
            score = -clamp((pct - (1.0 - squeeze_percentile)) / top)
        else:
            score = 0.0

        metrics = {
            "atr": atr_now,
            "atr_percentile": pct,
            "atr_pct_of_price": atr_pct_of_price,
            "regime_period": regime_period,
            "regime_samples": int(len(window)),
            "squeeze": bool(squeeze),
            "expansion": bool(expansion),
            "squeeze_percentile": squeeze_percentile,
            "regime": "squeeze" if squeeze else ("expansion" if expansion else "normal"),
            "close": close,
            "bars": len(frame),
        }
        # Non-directional by construction.
        return self.result(score, "flat", metrics)
