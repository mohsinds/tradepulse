"""Underlying-level combined position risk cap."""

from __future__ import annotations

import asyncio
import unittest
from decimal import Decimal

from api.adapters.base import ContractType, Instrument, OrderRequest
from api.execution.risk import UnderlyingRiskGuard

ACCOUNT_VALUE = Decimal("100000")


def _inst(symbol: str, contract_type: ContractType, underlying: str = "GOLD", multiplier: str = "1") -> Instrument:
    return Instrument(
        symbol=symbol,
        venue="ibkr" if contract_type in (ContractType.FUTURE, ContractType.SPOT) else "ccxt",
        contract_type=contract_type,
        exchange="COMEX" if contract_type == ContractType.FUTURE else "SMART",
        underlying=underlying,
        currency="USD",
        multiplier=Decimal(multiplier),
    )


class TestUnderlyingRiskGuard(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        underlyings = {
            "GOLD": {
                "name": "Gold",
                "currency": "USD",
                "max_combined_position_pct": 0.05,
            }
        }
        self.guard = UnderlyingRiskGuard(
            account_value=ACCOUNT_VALUE,
            underlying_configs=underlyings,
        )

    def _order(self, instrument: Instrument, qty: Decimal, price: Decimal) -> OrderRequest:
        return OrderRequest(
            instrument=instrument,
            side="buy",
            quantity=qty,
            order_type="market",
            price=price,
        )

    def _positions(
        self,
        mgc_qty: Decimal = Decimal("0"),
        xauusd_qty: Decimal = Decimal("0"),
        xauusdt_qty: Decimal = Decimal("0"),
    ) -> dict:
        """Current positions keyed by symbol; each carries explicit underlying."""
        return {
            "MGC": [
                {
                    "symbol": "MGC",
                    "underlying": "GOLD",
                    "quantity": mgc_qty,
                    "last": "2500",
                    "multiplier": "10",
                }
            ],
            "XAUUSD": [
                {
                    "symbol": "XAUUSD",
                    "underlying": "GOLD",
                    "quantity": xauusd_qty,
                    "last": "2500",
                    "multiplier": "1",
                }
            ],
            "XAUUSDT.P": [
                {
                    "symbol": "XAUUSDT.P",
                    "underlying": "GOLD",
                    "quantity": xauusdt_qty,
                    "last": "2500",
                    "multiplier": "1",
                }
            ],
        }

    async def test_single_order_within_cap_passes(self) -> None:
        # 0.01 MGC, multiplier=10, price=2500 => $250 notional => 0.25% < 5%
        mgc = _inst("MGC", ContractType.FUTURE, multiplier="10")
        order = self._order(mgc, Decimal("0.01"), Decimal("2500"))
        result = await self.guard.check_order(
            order=order,
            positions_by_symbol={},
        )
        self.assertTrue(result.ok)

    async def test_order_with_existing_positions_exceeds_cap(self) -> None:
        # Existing exposure: 0.01 MGC ($250), 0.01 XAUUSD ($25), 0.01 XAUUSDT.P ($25)
        positions = self._positions(
            mgc_qty=Decimal("0.01"),
            xauusd_qty=Decimal("0.01"),
            xauusdt_qty=Decimal("0.01"),
        )
        current_notional = Decimal("0.01") * Decimal("2500") * Decimal("10") + Decimal("0.01") * Decimal("2500") + Decimal("0.01") * Decimal("2500")
        current_pct = current_notional / ACCOUNT_VALUE  # 250+25+25 = 300 / 100000 = 0.003

        # Propose 0.2 MGC, multiplier=10 => $5000 notional
        # current 300 + 5000 = 5300 / 100000 = 0.053 -> FAIL
        mgc = _inst("MGC", ContractType.FUTURE, multiplier="10")
        order = self._order(mgc, Decimal("0.2"), Decimal("2500"))
        result = await self.guard.check_order(order, positions)
        self.assertFalse(result.ok)
        self.assertEqual(result.underlying, "GOLD")
        self.assertGreater(result.combined_notional_pct, result.max_allowed_pct)
        self.assertIn("MGC", result.flagged_symbols)
        self.assertIn("XAUUSD", result.flagged_symbols)
        self.assertIn("XAUUSDT.P", result.flagged_symbols)

    async def test_combined_three_instruments_under_cap_passes(self) -> None:
        positions = self._positions(
            mgc_qty=Decimal("0.01"),
            xauusd_qty=Decimal("0.01"),
            xauusdt_qty=Decimal("0.01"),
        )
        # Add another 0.01 MGC, multiplier=10 => $250 -> total 550 / 100000 = 0.55% < 5%
        order = self._order(_inst("MGC", ContractType.FUTURE, multiplier="10"), Decimal("0.01"), Decimal("2500"))
        result = await self.guard.check_order(order, positions)
        self.assertTrue(result.ok)

    async def test_advisory_same_cap_as_order(self) -> None:
        positions = self._positions(
            mgc_qty=Decimal("0.01"),
            xauusd_qty=Decimal("0.01"),
            xauusdt_qty=Decimal("0.01"),
        )
        idea = {
            "symbol": "XAUUSDT.P",
            "underlying": "GOLD",
            "proposed_notional_pct": 0.06,  # would push over 5%
        }
        result = await self.guard.check_advisory(
            idea=idea,
            account_value=ACCOUNT_VALUE,
            positions_by_symbol=positions,
        )
        self.assertFalse(result.ok)
        self.assertIn("exceeds max", result.reason or "")

    async def test_missing_cap_uses_default(self) -> None:
        guard = UnderlyingRiskGuard(
            account_value=ACCOUNT_VALUE,
            underlying_configs={},
        )
        # 0.01 MGC ($250) < default 5%
        result = await guard.check_order(
            self._order(_inst("MGC", ContractType.FUTURE), Decimal("0.01"), Decimal("2500")),
            {},
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.max_allowed_pct, Decimal("0.05"))

    async def test_zero_account_value_returns_safe_result(self) -> None:
        guard = UnderlyingRiskGuard(
            account_value=Decimal("0"),
            underlying_configs={"GOLD": {"max_combined_position_pct": 0.05}},
        )
        result = await guard.check_order(
            self._order(_inst("MGC", ContractType.FUTURE), Decimal("1"), Decimal("2500")),
            {},
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.combined_notional_pct, Decimal("0"))
        self.assertIn("account value must be positive", result.reason or "")


if __name__ == "__main__":
    unittest.main()
