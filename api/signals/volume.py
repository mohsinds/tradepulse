from __future__ import annotations

import numpy as np

from typing import Any, Dict

from ..adapters.base import Instrument
from ._util import (
    ConfiguredSignal,
    clamp,
    direction_from_score,
    extract_frame,
    percentile_rank,
    safe_div,
)


class VolumeSignal(ConfiguredSignal):
    """Wyckoff-style effort (volume) vs result (spread / close location).

    * ``spread`` is the bar range, compared against its trailing average and
      percentile rank.
    * ``climax`` = abnormally high volume (>= ``climax_sigma`` std) on a wide
      spread bar (spread percentile >= ``spread_percentile``).
    * ``absorption`` = abnormally high volume producing almost no result
      (spread <= ``absorption_ratio`` x average spread); the side doing the
      pushing is failing, so the score flips against the bar's direction.
    """

    name = "volume"
    defaults: Dict[str, Any] = {
        "weight": 0.15,
        "lookback": 20,
        "climax_sigma": 2.0,
        "absorption_ratio": 0.35,
        "spread_percentile": 0.7,
    }

    def generate(self, instrument: Instrument, data: Any) -> Dict[str, Any]:
        lookback = max(self._int("lookback", 20), 2)
        climax_sigma = self._float("climax_sigma", 2.0)
        absorption_ratio = self._float("absorption_ratio", 0.35)
        spread_percentile = self._float("spread_percentile", 0.7)

        frame = extract_frame(data)
        required = lookback + 1
        if len(frame) < required:
            return self.insufficient(
                "insufficient_data",
                {"bars": len(frame), "required": required},
            )

        high = frame["high"].astype(float)
        low = frame["low"].astype(float)
        close = frame["close"].astype(float)
        open_ = frame["open"].astype(float)
        volume = frame["volume"].astype(float)

        spread = (high - low).abs()
        spread_hist = spread.iloc[-1 - lookback : -1]
        vol_hist = volume.iloc[-1 - lookback : -1]

        spread_now = float(spread.iloc[-1])
        volume_now = float(volume.iloc[-1])

        avg_spread = float(spread_hist.mean())
        avg_volume = float(vol_hist.mean())
        std_volume = float(vol_hist.std(ddof=0))

        volume_ratio = safe_div(volume_now, avg_volume, 1.0)
        result_ratio = safe_div(spread_now, avg_spread, 1.0)
        if std_volume > 0.0:
            volume_z = safe_div(volume_now - avg_volume, std_volume, 0.0)
        else:
            # Degenerate history (constant volume): fall back to the ratio so a
            # genuine volume spike is still flagged.
            volume_z = climax_sigma * (volume_ratio - 1.0)
        spread_pct = percentile_rank(spread_hist, spread_now)

        # Close location value within the bar: 1 = close on the high.
        rng = spread_now
        if rng > 0:
            clv = (float(close.iloc[-1]) - float(low.iloc[-1])) / rng
        else:
            clv = 0.5
        clv_signed = clamp(2.0 * clv - 1.0)

        bar_dir = float(np.sign(float(close.iloc[-1]) - float(open_.iloc[-1])))

        high_effort = volume_z >= climax_sigma
        wide_spread = spread_pct >= spread_percentile
        climax = bool(high_effort and wide_spread)
        absorption = bool(high_effort and result_ratio <= absorption_ratio)

        effort_factor = clamp(volume_ratio / 2.0, 0.0, 1.0)
        result_factor = clamp(result_ratio, 0.0, 1.0)

        if absorption:
            # Effort without result: fade the side that is pushing.
            score = clamp(-0.6 * (bar_dir if bar_dir != 0.0 else clv_signed))
            pattern = "absorption"
        elif climax:
            score = clamp(clv_signed * effort_factor)
            pattern = "buying_climax" if clv_signed >= 0 else "selling_climax"
        else:
            score = clamp(clv_signed * effort_factor * result_factor)
            pattern = "normal"

        metrics = {
            "volume": volume_now,
            "avg_volume": avg_volume,
            "volume_ratio": volume_ratio,
            "volume_zscore": volume_z,
            "spread": spread_now,
            "avg_spread": avg_spread,
            "spread_ratio": result_ratio,
            "spread_percentile_rank": spread_pct,
            "close_location": clv,
            "close_location_signed": clv_signed,
            "bar_direction": bar_dir,
            "climax": climax,
            "absorption": absorption,
            "wide_spread": bool(wide_spread),
            "high_effort": bool(high_effort),
            "pattern": pattern,
            "bars": len(frame),
        }
        return self.result(score, direction_from_score(score), metrics)
