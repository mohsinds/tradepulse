from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional

from ..adapters.base import Instrument, OrderRequest

logger = logging.getLogger(__name__)


@dataclass
class RiskResult:
    ok: bool
    current_notional_pct: Decimal
    proposed_notional_pct: Decimal
    combined_notional_pct: Decimal
    max_allowed_pct: Decimal
    underlying: str
    reason: Optional[str] = None
    flagged_symbols: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "current_notional_pct": float(self.current_notional_pct),
            "proposed_notional_pct": float(self.proposed_notional_pct),
            "combined_notional_pct": float(self.combined_notional_pct),
            "max_allowed_pct": float(self.max_allowed_pct),
            "underlying": self.underlying,
            "reason": self.reason,
            "flagged_symbols": self.flagged_symbols,
        }


class UnderlyingRiskGuard:
    """Enforce a combined notional cap across all instruments sharing an underlying.

    This is a *hard backend check* (AGENTS.md). It runs before any order is
    submitted, and also before an advisory recommendation is logged, so the cap
    is enforced regardless of whether live execution is enabled.
    """

    def __init__(
        self,
        account_value: Decimal,
        underlying_configs: Dict[str, Dict[str, Any]],
    ) -> None:
        self.account_value = account_value
        self._underlying_configs = underlying_configs

    @classmethod
    def from_config_loader(
        cls,
        account_value: Decimal,
        loaded_config: Optional[Dict[str, Any]] = None,
    ) -> "UnderlyingRiskGuard":
        """Build a guard from the output of ``api.config.load_all()``."""
        if loaded_config is None:
            from ..config import load_all

            loaded_config = load_all()
        underlyings = loaded_config.get("underlyings", {})
        return cls(account_value, underlyings)

    def _max_pct(self, underlying: str) -> Decimal:
        cfg = self._underlying_configs.get(underlying, {})
        raw = cfg.get("max_combined_position_pct")
        if raw is None:
            return Decimal("0.05")  # safe default
        return Decimal(str(raw))

    @staticmethod
    def _notional_pct(
        positions: List[Dict[str, Any]], account_value: Decimal
    ) -> Decimal:
        """Sum absolute notional values of a position list as a pct of NAV.

        Position dicts are expected to have either:
            - ``notional`` / ``notional_pct``
            - ``quantity`` + ``last`` / ``mark_price``
        """
        if account_value <= 0:
            return Decimal("0")

        total = Decimal("0")
        for pos in positions:
            notional = pos.get("notional")
            if notional is not None:
                total += abs(Decimal(str(notional)))
            else:
                qty = Decimal(str(pos.get("quantity") or 0))
                price = Decimal(str(pos.get("last") or pos.get("mark_price") or pos.get("price") or 0))
                multiplier = Decimal(str(pos.get("multiplier") or 1))
                total += abs(qty * price * multiplier)

        return (total / account_value).quantize(Decimal("0.0001"))

    @staticmethod
    def _proposed_notional_pct(order: OrderRequest, account_value: Decimal) -> Decimal:
        """Notional value of the order being considered."""
        if account_value <= 0:
            return Decimal("0")
        price = order.price or order.instrument.meta.get("last") or 0
        price = Decimal(str(price)) if price is not None else Decimal("0")
        multiplier = Decimal(str(order.instrument.multiplier or 1))
        notional = abs(order.quantity) * price * multiplier
        return (notional / account_value).quantize(Decimal("0.0001"))

    def _positions_for_underlying(
        self, underlying: str, positions_by_symbol: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Gather current positions that share the same underlying."""
        # positions_by_symbol maps symbol -> list of position dicts.
        # The caller can populate this from one or many venues/adapters.
        grouped: List[Dict[str, Any]] = []
        for _, positions in positions_by_symbol.items():
            for pos in positions:
                # each position carries its instrument's underlying or is keyed externally
                if pos.get("underlying") == underlying:
                    grouped.append(pos)
                # also accept if a symbol field matches a known instrument config
                elif pos.get("symbol") in self._symbol_to_underlying:
                    if self._symbol_to_underlying[pos["symbol"]] == underlying:
                        grouped.append(pos)
        return grouped

    @property
    def _symbol_to_underlying(self) -> Dict[str, str]:
        """Best-effort symbol -> underlying map from loaded underlyings."""
        # Underlying configs do not enumerate symbols, so callers should pass
        # positions with an explicit ``underlying`` field when possible.
        mapping: Dict[str, str] = {}
        for underlying, cfg in self._underlying_configs.items():
            for sym in cfg.get("symbols", []):
                mapping[str(sym)] = underlying
        return mapping

    async def check_order(
        self,
        order: OrderRequest,
        positions_by_symbol: Dict[str, List[Dict[str, Any]]],
    ) -> RiskResult:
        """Check a proposed order against the underlying cap.

        Returns a ``RiskResult``. If the cap is exceeded, ``ok`` is ``False``.
        The caller decides whether to reject (order submission) or flag
        (advisory logging).
        """
        if self.account_value <= 0:
            return RiskResult(
                ok=False,
                current_notional_pct=Decimal("0"),
                proposed_notional_pct=Decimal("0"),
                combined_notional_pct=Decimal("0"),
                max_allowed_pct=Decimal("0"),
                underlying=(order.instrument.underlying or order.instrument.symbol).upper(),
                reason="account value must be positive to compute position percentages",
                flagged_symbols=[order.instrument.symbol],
            )

        instrument = order.instrument
        underlying = (instrument.underlying or instrument.symbol).upper()
        max_pct = self._max_pct(underlying)

        current_positions = self._positions_for_underlying(underlying, positions_by_symbol)
        current_pct = self._notional_pct(current_positions, self.account_value)
        proposed_pct = self._proposed_notional_pct(order, self.account_value)
        combined = min(current_pct + proposed_pct, Decimal("1"))

        flagged: List[str] = []
        reason: Optional[str] = None
        ok = True

        if combined > max_pct:
            ok = False
            flagged = [p.get("symbol", "?") for p in current_positions]
            if instrument.symbol not in flagged:
                flagged.append(instrument.symbol)
            reason = (
                f"Combined {underlying} exposure {float(combined):.4%} "
                f"exceeds max {float(max_pct):.4%} (current {float(current_pct):.4%} "
                f"+ proposed {float(proposed_pct):.4%})"
            )
            logger.warning("Risk check failed: %s", reason)

        return RiskResult(
            ok=ok,
            current_notional_pct=current_pct,
            proposed_notional_pct=proposed_pct,
            combined_notional_pct=combined,
            max_allowed_pct=max_pct,
            underlying=underlying,
            reason=reason,
            flagged_symbols=flagged,
        )

    async def check_advisory(
        self,
        idea: Dict[str, Any],
        account_value: Decimal,
        positions_by_symbol: Dict[str, List[Dict[str, Any]]],
    ) -> RiskResult:
        """Apply the same underlying cap to an advisory idea before it is logged.

        ``idea`` must contain at least ``symbol`` and ``proposed_notional_pct``
        (or ``quantity`` + ``price``). Current positions are aggregated by
        underlying as in ``check_order``.
        """
        if account_value <= 0:
            return RiskResult(
                ok=False,
                current_notional_pct=Decimal("0"),
                proposed_notional_pct=Decimal("0"),
                combined_notional_pct=Decimal("0"),
                max_allowed_pct=Decimal("0"),
                underlying=idea.get("underlying") or idea.get("symbol", ""),
                reason="account value must be positive to compute position percentages",
                flagged_symbols=[idea.get("symbol", "")],
            )

        symbol = idea.get("symbol", "")
        underlying = idea.get("underlying") or symbol
        max_pct = self._max_pct(underlying)

        # Build a fake OrderRequest only for the notional helper
        class _FakeInstrument:
            def __init__(self, symbol: str, underlying: str) -> None:
                self.symbol = symbol
                self.underlying = underlying
                self.multiplier = Decimal("1")
                self.meta = {}

        fake = _FakeInstrument(symbol, underlying)
        proposed_pct = Decimal(str(idea.get("proposed_notional_pct") or 0))
        if proposed_pct == 0 and idea.get("quantity") and idea.get("price"):
            qty = Decimal(str(idea["quantity"]))
            price = Decimal(str(idea["price"]))
            if account_value > 0:
                proposed_pct = (abs(qty) * price / account_value).quantize(Decimal("0.0001"))

        current_positions = self._positions_for_underlying(underlying, positions_by_symbol)
        current_pct = self._notional_pct(current_positions, account_value)
        combined = min(current_pct + proposed_pct, Decimal("1"))

        ok = combined <= max_pct
        reason = None
        flagged: List[str] = [p.get("symbol", "?") for p in current_positions]
        if symbol not in flagged:
            flagged.append(symbol)

        if not ok:
            reason = (
                f"Advisory {underlying} exposure {float(combined):.4%} "
                f"exceeds max {float(max_pct):.4%} (current {float(current_pct):.4%} "
                f"+ proposed {float(proposed_pct):.4%})"
            )
            logger.warning("Advisory risk check failed: %s", reason)

        return RiskResult(
            ok=ok,
            current_notional_pct=current_pct,
            proposed_notional_pct=proposed_pct,
            combined_notional_pct=combined,
            max_allowed_pct=max_pct,
            underlying=underlying,
            reason=reason,
            flagged_symbols=flagged,
        )
