from __future__ import annotations

import numpy as np

from typing import Any, Dict, Optional

from ..adapters.base import ContractType, Instrument
from ._util import (
    ConfiguredSignal,
    clamp,
    direction_from_score,
    extract_frame,
    extract_quote,
    finite,
)


def _find_funding_rate(data: Any) -> Optional[float]:
    """Locate a funding rate on the flexible ``data`` payload.

    Looks at ``data["quote"]["funding_rate"]`` first (what the CCXT adapter
    emits on perpetual quotes), then a top-level ``funding_rate``.
    """
    quote = extract_quote(data)
    for source in (quote, data if isinstance(data, dict) else {}):
        for key in ("funding_rate", "fundingRate"):
            if key in source and source[key] is not None:
                value = finite(source[key], float("nan"))
                if np.isfinite(value):
                    return value
    return None


class FundingRateSignal(ConfiguredSignal):
    """Perpetual funding rate as a directional tilt and an annualized carry cost.

    Positive funding => longs pay shorts => bearish tilt for longs (negative
    score). Negative funding => shorts pay longs => bullish tilt.

    Only active for ``ContractType.PERPETUAL`` instruments; anything else
    returns a neutral, inactive result.
    """

    name = "funding_rate"
    defaults: Dict[str, Any] = {
        "weight": 0.2,
        "neutral_band": 0.0001,
        "extreme": 0.0005,
        "periods_per_day": 3,
        "unfavorable_penalty": 0.5,
    }

    def generate(self, instrument: Instrument, data: Any) -> Dict[str, Any]:
        contract_type = getattr(instrument, "contract_type", None)
        if contract_type != ContractType.PERPETUAL:
            return self.result(
                0.0,
                "flat",
                {
                    "contract_type": (
                        contract_type.value
                        if isinstance(contract_type, ContractType)
                        else str(contract_type)
                    ),
                    "applicable": False,
                },
                active=False,
                reason="not_perpetual",
            )

        neutral_band = abs(self._float("neutral_band", 0.0001))
        extreme = abs(self._float("extreme", 0.0005))
        periods_per_day = max(self._int("periods_per_day", 3), 1)
        unfavorable_penalty = self._float("unfavorable_penalty", 0.5)

        rate = _find_funding_rate(data)
        if rate is None:
            metrics = {
                "applicable": True,
                "contract_type": ContractType.PERPETUAL.value,
                "bars": len(extract_frame(data)),
            }
            payload = self.insufficient("missing_funding_rate", metrics)
            payload["active"] = True
            return payload

        annualized = rate * periods_per_day * 365.0
        daily = rate * periods_per_day

        if rate > 0:
            payer, receiver = "long", "short"
        elif rate < 0:
            payer, receiver = "short", "long"
        else:
            payer, receiver = "none", "none"

        span = max(extreme - neutral_band, 1e-12)
        magnitude = max(abs(rate) - neutral_band, 0.0)
        score = -float(np.sign(rate)) * clamp(magnitude / span, 0.0, 1.0)
        score = clamp(score)

        metrics = {
            "applicable": True,
            "contract_type": ContractType.PERPETUAL.value,
            "funding_rate": rate,
            "periods_per_day": periods_per_day,
            "daily_rate": daily,
            "annualized_rate": annualized,
            "annualized_pct": annualized * 100.0,
            "pays": payer,
            "receives": receiver,
            "neutral_band": neutral_band,
            "extreme": extreme,
            "neutral": bool(abs(rate) <= neutral_band),
            "is_extreme": bool(abs(rate) >= extreme),
            "unfavorable_penalty": unfavorable_penalty,
        }

        return self.result(
            score,
            direction_from_score(score),
            metrics,
            active=True,
            annualized_rate=annualized,
            pays=payer,
        )
