"""Instrument-specific holding-cost overlays for backtests.

Each overlay estimates the daily drag of holding a position and expresses it
as a per-bar reduction in portfolio returns.  The cost is applied only when a
position is held (``position_size`` > 0).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import pandas as pd
import vectorbt as vbt

from api.adapters import ContractType, Instrument


def _bars_per_day(price_or_index: Union[pd.Series, pd.DatetimeIndex]) -> float:
    """Infer the number of bars in a trading day from the index frequency.

    Defaults to 24 bars/day when frequency cannot be inferred.
    """
    idx = price_or_index if isinstance(price_or_index, pd.DatetimeIndex) else price_or_index.index
    freq = pd.infer_freq(idx)
    if freq is None:
        # Try vectorbt wrapper freq as a fallback.
        return 24.0
    freq = freq.lower()
    if freq == "h" or freq == "1h":
        return 24.0
    if freq.endswith("m"):
        try:
            minutes = int(freq[:-1])
            return (24 * 60) / max(minutes, 1)
        except ValueError:
            return 24.0
    if freq.endswith("h"):
        try:
            hours = int(freq[:-1])
            return 24 / max(hours, 1)
        except ValueError:
            return 24.0
    if freq == "d" or freq.endswith("d"):
        return 1.0
    return 24.0


def _as_series(value: Any, index: pd.Index, name: str) -> pd.Series:
    if isinstance(value, pd.Series):
        return value.reindex(index, method="ffill").ffill().fillna(0)
    return pd.Series(float(value), index=index, name=name)


class CostOverlay(ABC):
    """Abstract base for holding-cost overlays."""

    def __init__(self, instrument: Instrument) -> None:
        self.instrument = instrument

    def apply_cost(
        self,
        pf: vbt.Portfolio,
        price: Optional[pd.Series] = None,
    ) -> pd.Series:
        """Return the portfolio return series after subtracting holding costs.

        If ``price`` is omitted, the portfolio's close price is used.
        """
        if price is None:
            price = pf.close
        position_size = pf.gross_exposure()
        cost = self.cost_per_bar(price, position_size)
        return pf.returns().fillna(0.0) - cost.reindex(pf.returns().index).fillna(0.0)

    @abstractmethod
    def cost_per_bar(
        self,
        price: pd.Series,
        position_size: Optional[pd.Series] = None,
    ) -> pd.Series:
        """Return the fractional cost per bar as a fraction of notional exposure.

        ``position_size`` is the fractional notional exposure (0..1).  When
        omitted the cost is computed for a fully-invested position.
        """
        ...


class NoCost(CostOverlay):
    """Zero holding cost (e.g. spot equity with no financing)."""

    def cost_per_bar(
        self,
        price: pd.Series,
        position_size: Optional[pd.Series] = None,
    ) -> pd.Series:
        return pd.Series(0.0, index=price.index)


class FuturesRollCost(CostOverlay):
    """Roll cost for a futures contract.

    Daily drag = ``roll_cost_pct * rolls_per_year / 252``.
    Per-bar cost = daily drag / bars-per-day, applied only when holding.
    """

    def __init__(
        self,
        instrument: Instrument,
        roll_cost_pct: float = 0.0002,
        rolls_per_year: float = 12.0,
    ) -> None:
        super().__init__(instrument)
        self.roll_cost_pct = roll_cost_pct
        self.rolls_per_year = rolls_per_year
        self.daily_drag = roll_cost_pct * rolls_per_year / 252.0

    def cost_per_bar(
        self,
        price: pd.Series,
        position_size: Optional[pd.Series] = None,
    ) -> pd.Series:
        position_size = (
            pd.Series(1.0, index=price.index)
            if position_size is None
            else position_size.reindex(price.index).fillna(0.0)
        )
        bpd = _bars_per_day(price)
        return position_size.abs() * (self.daily_drag / bpd)


class CFDSwapCost(CostOverlay):
    """Overnight swap/financing cost for a CFD.

    Daily drag = ``annual_swap_rate / 252``.
    """

    def __init__(
        self,
        instrument: Instrument,
        annual_swap_rate: float = 0.03,
    ) -> None:
        super().__init__(instrument)
        self.annual_swap_rate = annual_swap_rate
        self.daily_drag = annual_swap_rate / 252.0

    def cost_per_bar(
        self,
        price: pd.Series,
        position_size: Optional[pd.Series] = None,
    ) -> pd.Series:
        position_size = (
            pd.Series(1.0, index=price.index)
            if position_size is None
            else position_size.reindex(price.index).fillna(0.0)
        )
        bpd = _bars_per_day(price)
        return position_size.abs() * (self.daily_drag / bpd)


class PerpetualFundingCost(CostOverlay):
    """Funding-rate cost for a perpetual swap.

    If ``funding_rate`` is a constant, the per-period rate is scaled by
    ``periods_per_day`` to obtain a daily rate and then divided by the number
    of bars per day.  If a ``pandas.Series`` of per-bar funding rates is
    supplied it is used directly per bar with the same scaling.
    """

    def __init__(
        self,
        instrument: Instrument,
        periods_per_day: float = 3.0,
        funding_rate: Union[float, pd.Series] = 0.0001,
    ) -> None:
        super().__init__(instrument)
        self.periods_per_day = periods_per_day
        self.funding_rate = funding_rate

    def cost_per_bar(
        self,
        price: pd.Series,
        position_size: Optional[pd.Series] = None,
    ) -> pd.Series:
        position_size = (
            pd.Series(1.0, index=price.index)
            if position_size is None
            else position_size.reindex(price.index).fillna(0.0)
        )
        funding = _as_series(self.funding_rate, price.index, "funding_rate")
        bpd = _bars_per_day(price)
        daily_rate = funding * self.periods_per_day
        return position_size.abs() * (daily_rate / bpd)


def cost_overlay_for(instrument: Instrument, **kwargs: Any) -> CostOverlay:
    """Return the appropriate ``CostOverlay`` for ``instrument.contract_type``."""
    ctype = instrument.contract_type
    if ctype == ContractType.FUTURE:
        return FuturesRollCost(instrument, **kwargs)
    if ctype == ContractType.CFD:
        return CFDSwapCost(instrument, **kwargs)
    if ctype == ContractType.PERPETUAL:
        return PerpetualFundingCost(instrument, **kwargs)
    # spot / option / unknown: assume no holding cost
    return NoCost(instrument)
