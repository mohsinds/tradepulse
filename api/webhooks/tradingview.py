from __future__ import annotations

import hmac
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .base import WebhookHandler

logger = logging.getLogger(__name__)

DEFAULT_WEIGHT = 0.2


class TradingViewWebhookHandler(WebhookHandler):
    """Inbound TradingView Pine alert handler.

    Validates a shared secret, normalizes the Pine alert JSON into the same
    signal-result dict that internal calculators produce, and returns it so the
    caller can feed it into the ranking agent as an *additional* input.

    This handler never places orders directly.
    """

    name = "tradingview"

    def __init__(self, secret: Optional[str] = None, signal_weight: Optional[float] = None):
        self.secret = secret if secret is not None else os.getenv("TRADINGVIEW_WEBHOOK_SECRET", "")
        self.signal_weight = (
            signal_weight
            if signal_weight is not None
            else float(os.getenv("TRADINGVIEW_SIGNAL_WEIGHT", DEFAULT_WEIGHT))
        )

    def _verify(self, payload: Dict[str, Any]) -> bool:
        """Constant-time compare of the provided secret against the env secret."""
        if not self.secret:
            logger.warning("TRADINGVIEW_WEBHOOK_SECRET not configured; accepting alert unverified")
            return True

        provided = payload.get("secret") or payload.get("token") or ""
        return hmac.compare_digest(str(provided), self.secret)

    @staticmethod
    def _normalize_ticker(raw: str) -> str:
        """Best-effort map from Pine ticker to instrument symbol."""
        sym = str(raw).upper().strip()
        # TradingView perps often include ".P" or "PERP".
        if sym.endswith("PERP"):
            sym = sym[:-4] + ".P"
        if "XAUUSD" in sym and sym.endswith(".P"):
            return "XAUUSDT.P"
        return sym

    @staticmethod
    def _direction_and_score(action: str) -> tuple[str, float]:
        act = str(action).lower().strip()
        if act in ("buy", "long", "bullish"):
            return "long", 1.0
        if act in ("sell", "short", "bearish"):
            return "short", -1.0
        return "flat", 0.0

    async def handle(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self._verify(payload):
            raise PermissionError("TradingView webhook secret mismatch")

        ticker = self._normalize_ticker(payload.get("ticker", payload.get("symbol", "")))
        action = payload.get("action", payload.get("side", "flat"))
        direction, score = self._direction_and_score(action)

        try:
            price = float(payload.get("price") or payload.get("close") or 0)
        except (TypeError, ValueError):
            price = 0.0

        result: Dict[str, Any] = {
            "name": "tradingview",
            "score": score,
            "direction": direction,
            "weight": self.signal_weight,
            "metrics": {
                "source": "tradingview",
                "ticker": ticker,
                "raw_ticker": payload.get("ticker", payload.get("symbol", "")),
                "action": action,
                "price": price,
                "interval": payload.get("interval"),
                "exchange": payload.get("exchange"),
                "time": payload.get("time") or datetime.now(timezone.utc).isoformat(),
                "verified": bool(self.secret),
            },
            "ok": True,
        }

        logger.info("TradingView alert normalized: %s %s", ticker, direction)
        return result

    @classmethod
    def from_env(cls) -> "TradingViewWebhookHandler":
        return cls()
