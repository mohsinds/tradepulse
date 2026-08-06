"""Portfolio scoring helpers for the walk-forward grid search."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd
import vectorbt as vbt


def _annualization_factor(pf: vbt.Portfolio, returns: pd.Series) -> float:
    """Return ``sqrt(periods_per_year)`` for annualizing a Sharpe ratio."""
    freq = getattr(pf.wrapper, "freq", None)
    if freq is not None:
        try:
            hours = freq.total_seconds() / 3600.0
        except AttributeError:
            hours = 1.0
    else:
        inferred = pd.infer_freq(returns.index)
        if inferred is None:
            hours = 1.0
        else:
            inferred = inferred.lower()
            if inferred == "h" or inferred == "1h":
                hours = 1.0
            elif inferred.endswith("m"):
                try:
                    minutes = int(inferred[:-1])
                    hours = max(minutes, 1) / 60.0
                except ValueError:
                    hours = 1.0
            elif inferred.endswith("h"):
                try:
                    hours = int(inferred[:-1])
                except ValueError:
                    hours = 1.0
            elif inferred == "d" or inferred.endswith("d"):
                hours = 24.0
            else:
                hours = 1.0
    # 24-hour markets: 252 trading days * 24 hours; traditional equity would be
    # 252 * 6.5, but crypto/forex-style 24h data is the default here.
    periods_per_year = 252.0 * 24.0 / max(hours, 1e-9)
    return math.sqrt(periods_per_year)


def _sharpe(returns: pd.Series, ann_factor: float) -> float:
    rets = returns.dropna()
    if len(rets) < 2 or rets.std() == 0 or not np.isfinite(rets.std()):
        return 0.0
    return float(rets.mean() / rets.std() * ann_factor)


def _max_drawdown(returns: pd.Series) -> float:
    rets = returns.dropna()
    if len(rets) < 2:
        return 0.0
    cumulative = (1.0 + rets).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return float(drawdown.min())


def _win_rate(pf: vbt.Portfolio) -> float:
    try:
        wr = pf.trades.win_rate()
    except Exception:
        return 0.0
    if isinstance(wr, (pd.Series, pd.DataFrame)):
        wr = wr.iloc[0] if len(wr) else 0.0
    if wr is None or (isinstance(wr, float) and math.isnan(wr)):
        return 0.0
    return float(wr)


def score_portfolio(
    pf: vbt.Portfolio,
    adjusted_returns: Optional[pd.Series] = None,
) -> dict:
    """Score a vectorbt portfolio.

    Parameters
    ----------
    pf:
        The vectorbt portfolio to score.
    adjusted_returns:
        Return series after applying holding costs.  If omitted, ``pf.returns()``
        is used.

    Returns
    -------
    dict
        ``{"sharpe": float, "max_drawdown": float, "win_rate": float,
        "composite": float}``

    Notes
    -----
    The composite rewards positive Sharpe, penalizes drawdown, and rewards win
    rate, but saturates the Sharpe term at ``-1.0`` so disastrous strategies
    cannot produce a positive composite by accident:

        composite = max(-1.0, sharpe)
                    * (1.0 - abs(max_drawdown))
                    * max(0.0, win_rate)

    ``max_drawdown`` is expected to be negative or zero, so ``1.0 - abs(dd)``
    shrinks the score as the drawdown deepens.
    """
    if adjusted_returns is None:
        adjusted_returns = pf.returns()
    adjusted_returns = adjusted_returns.fillna(0.0)

    ann_factor = _annualization_factor(pf, adjusted_returns)
    sharpe = _sharpe(adjusted_returns, ann_factor)
    max_dd = _max_drawdown(adjusted_returns)
    win_rate = _win_rate(pf)

    composite = (
        max(-1.0, sharpe)
        * (1.0 - abs(max_dd))
        * max(0.0, win_rate)
    )

    return {
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "win_rate": float(win_rate),
        "composite": float(composite),
    }
