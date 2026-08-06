from __future__ import annotations

import logging
import uuid
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional

from ..adapters.base import BrokerAdapter, ContractType, Instrument, OrderRequest
from ..config import load_all
from .base import ExecutionEngine
from .risk import UnderlyingRiskGuard

logger = logging.getLogger(__name__)


class ExposureUnavailableError(RuntimeError):
    """Raised when current exposure cannot be established for every venue.

    The underlying-level cap spans venues, so a venue that cannot report its
    positions makes the check unsound. Failing closed is mandatory: a cap that
    silently treats an unreachable venue as flat is worse than no cap.
    """


def _to_decimal(value: Any, default: Optional[Decimal] = None) -> Optional[Decimal]:
    if value is None:
        return default
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except InvalidOperation:
        return default


class PaperExecutionEngine(ExecutionEngine):
    """Advisory-aware execution engine that routes ranked ideas to venue adapters.

    By default the engine operates in paper mode: orders are simulated unless
    ``paper=False`` or the chosen broker adapter is explicitly configured with
    ``sandbox=True``. All orders pass through :class:`UnderlyingRiskGuard`
    before submission.
    """

    def __init__(
        self,
        adapters: Dict[str, BrokerAdapter],
        account_value: Decimal,
        underlying_configs: Dict[str, Any],
        db_session_factory: Optional[Any] = None,
        paper: bool = True,
    ) -> None:
        self.adapters = adapters
        self.account_value = account_value
        self.underlying_configs = underlying_configs
        self.db_session_factory = db_session_factory
        self.paper = paper

    @staticmethod
    def _resolve_underlying_name(
        cfg: Dict[str, Any], config: Dict[str, Any]
    ) -> str:
        """Recover the underlying symbol from a merged instrument config."""
        underlying_cfg = cfg.get("underlying")
        if isinstance(underlying_cfg, str):
            return underlying_cfg
        if isinstance(underlying_cfg, dict):
            underlyings = config.get("underlyings", {})
            for u_sym, u_def in underlyings.items():
                if underlying_cfg is u_def:
                    return u_sym
            # Fallback: use the human-readable name, upper-cased.
            name = underlying_cfg.get("name", "")
            if name:
                return name.upper()
        return ""

    def _symbol_to_underlying(self, config: Dict[str, Any]) -> Dict[str, str]:
        """Build a best-effort symbol -> underlying map from instrument configs."""
        mapping: Dict[str, str] = {}
        for symbol, cfg in config.get("instruments", {}).items():
            underlying = self._resolve_underlying_name(cfg, config)
            if underlying:
                mapping[symbol] = underlying
        return mapping

    def _instrument_from_config(
        self, symbol: str, cfg: Dict[str, Any], config: Dict[str, Any]
    ) -> Instrument:
        """Build an :class:`Instrument` from the merged config."""
        raw_contract = cfg.get("contract_type", "spot")
        if isinstance(raw_contract, ContractType):
            contract_type = raw_contract
        else:
            try:
                contract_type = ContractType(str(raw_contract).lower())
            except ValueError:
                contract_type = ContractType.SPOT

        underlying = self._resolve_underlying_name(cfg, config)
        multiplier = Decimal(str(cfg.get("multiplier", 1)))

        # Copy the full instrument config into ``meta`` so adapters and the
        # routing layer can access keys like ``paper_quantity`` or
        # ``last_price`` without inventing instrument-specific fields.
        meta = dict(cfg)

        return Instrument(
            symbol=symbol,
            venue=cfg.get("venue", ""),
            contract_type=contract_type,
            exchange=cfg.get("exchange", ""),
            underlying=underlying,
            currency=cfg.get("currency", "USD"),
            multiplier=multiplier,
            meta=meta,
        )

    async def _positions_by_symbol(self) -> Dict[str, List[Dict[str, Any]]]:
        """Aggregate positions from every configured adapter, keyed by symbol.

        Connects adapters that are not connected yet. Raises
        :class:`ExposureUnavailableError` if any adapter cannot report its
        positions, so the caller rejects rather than under-counting exposure.
        """
        all_positions: Dict[str, List[Dict[str, Any]]] = {}
        config = load_all()
        symbol_to_underlying = self._symbol_to_underlying(config)

        for adapter_name, adapter in self.adapters.items():
            if not getattr(adapter, "connected", False):
                try:
                    await adapter.connect()
                except Exception as exc:  # noqa: BLE001
                    raise ExposureUnavailableError(
                        f"Cannot connect adapter '{adapter_name}' to read positions: {exc}"
                    ) from exc
            try:
                positions = await adapter.get_positions()
            except Exception as exc:  # noqa: BLE001
                raise ExposureUnavailableError(
                    f"Adapter '{adapter_name}' could not report positions: {exc}"
                ) from exc

            if positions is None:
                raise ExposureUnavailableError(
                    f"Adapter '{adapter_name}' returned no position data"
                )
            for pos in positions:
                if not isinstance(pos, dict):
                    continue
                pos = dict(pos)
                symbol = pos.get("symbol")
                if symbol and not pos.get("underlying"):
                    underlying = symbol_to_underlying.get(symbol)
                    if underlying:
                        pos["underlying"] = underlying
                key = symbol or adapter_name
                all_positions.setdefault(key, []).append(pos)

        return all_positions

    async def route(
        self, instrument: Instrument, signal: Dict[str, Any]
    ) -> OrderRequest:
        """Translate a ranked signal into an :class:`OrderRequest`."""
        direction = str(signal.get("direction", "long")).lower()
        if direction == "long":
            side = "buy"
        elif direction == "short":
            side = "sell"
        else:
            side = "buy"

        quantity = (
            _to_decimal(signal.get("quantity"), None)
            or _to_decimal(instrument.meta.get("paper_quantity"), None)
            or Decimal("1")
        )
        price = (
            _to_decimal(signal.get("price"), None)
            or _to_decimal(instrument.meta.get("last_price"), None)
            or None
        )

        return OrderRequest(
            instrument=instrument,
            side=side,
            quantity=quantity,
            order_type="market",
            price=price,
            time_in_force="day",
        )

    def _order_to_dict(self, order: OrderRequest) -> Dict[str, Any]:
        """Serialize an order for logging / responses."""
        return {
            "symbol": order.instrument.symbol,
            "side": order.side,
            "quantity": str(order.quantity),
            "price": str(order.price) if order.price is not None else None,
            "order_type": order.order_type,
            "time_in_force": order.time_in_force,
        }

    async def submit(
        self, broker: BrokerAdapter, order: OrderRequest
    ) -> Dict[str, Any]:
        """Risk-check and submit an order (paper or live)."""
        risk_guard = UnderlyingRiskGuard(
            account_value=self.account_value,
            underlying_configs=self.underlying_configs,
        )
        try:
            positions_by_symbol = await self._positions_by_symbol()
        except ExposureUnavailableError as exc:
            logger.warning("Risk check could not run: %s", exc)
            return {
                "status": "risk_rejected",
                "risk": {"ok": False, "reason": str(exc)},
                "order": self._order_to_dict(order),
            }
        risk = await risk_guard.check_order(order, positions_by_symbol)

        if not risk.ok:
            return {
                "status": "risk_rejected",
                "risk": risk.to_dict(),
                "order": self._order_to_dict(order),
            }

        broker_config = getattr(broker, "config", {}) or {}
        is_sandbox = bool(broker_config.get("sandbox"))

        if self.paper and not is_sandbox:
            order_id = f"paper-{uuid.uuid4()}"
            return {
                "status": "paper_filled",
                "order_id": order_id,
                "risk": risk.to_dict(),
                "order": self._order_to_dict(order),
            }

        try:
            order_id = await broker.place_order(order)
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "error": str(exc)}

        return {
            "status": "filled",
            "order_id": order_id,
            "risk": risk.to_dict(),
        }

    async def execute_idea(self, idea: Dict[str, Any]) -> Dict[str, Any]:
        """Load the instrument, connect the venue adapter, and execute the idea."""
        symbol = idea.get("instrument")
        if not symbol:
            return {"status": "error", "error": "idea missing instrument"}

        config = load_all()
        cfg = config.get("instruments", {}).get(symbol)
        if cfg is None:
            return {"status": "error", "error": f"Unknown instrument '{symbol}'"}

        instrument = self._instrument_from_config(symbol, cfg, config)
        adapter = self.adapters.get(instrument.venue)
        if adapter is None:
            return {
                "status": "error",
                "error": f"No adapter for venue '{instrument.venue}'",
            }

        signal: Dict[str, Any] = {
            "name": "ranked_idea",
            "direction": idea.get("direction", "long"),
        }
        if "price" in idea:
            signal["price"] = idea["price"]
        if "quantity" in idea:
            signal["quantity"] = idea["quantity"]
        signal["rationale"] = idea.get("rationale")

        order = await self.route(instrument, signal)

        # Without a price the proposed notional is 0%, which makes the
        # underlying cap inert. Tolerable for a simulated fill, never for a
        # live one.
        if order.price is None and not self.paper:
            return {
                "status": "risk_rejected",
                "risk": {
                    "ok": False,
                    "reason": (
                        f"No price available for {symbol}; cannot size exposure "
                        "for a live order"
                    ),
                },
                "order": self._order_to_dict(order),
            }

        return await self.submit(adapter, order)
