"""Standalone signal/entry-exit generators for research backtests.

These functions are intentionally independent of ``api.signals`` and use only
pandas/numpy.  They are consumed by ``param_sweep.py`` for the walk-forward
grid search.
"""

from __future__ import annotations

import itertools
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd


def _atr_from_ohlc(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """Average True Range from full OHLC data."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=max(period, 1), min_periods=period).mean()


def _atr_from_price(price: pd.Series, period: int) -> pd.Series:
    """Close-to-close volatility proxy used when only a price series is available."""
    tr = price.diff().abs()
    return tr.rolling(window=max(period, 1), min_periods=period).mean()


def _rsi(price: pd.Series, period: int) -> pd.Series:
    """Wilder's RSI."""
    delta = price.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    alpha = 1.0 / max(period, 1)
    avg_gain = gain.ewm(alpha=alpha, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=alpha, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _percentile_rank(series: pd.Series, window: int) -> pd.Series:
    """Percentile rank of the current observation in the previous ``window`` bars.

    The current bar is excluded from the historical sample, mirroring the
    lookback logic in the production signal calculators.
    """
    window = max(window, 2)

    def _pct(x: np.ndarray) -> float:
        if len(x) < 2:
            return 0.5
        return float(np.mean(x[:-1] <= x[-1]))

    return series.rolling(window=window, min_periods=window).apply(_pct, raw=True)


def trend_signals(
    price: pd.Series,
    fast: int = 20,
    slow: int = 50,
    atr_period: int = 14,
) -> Tuple[pd.Series, pd.Series]:
    """Fast/slow moving-average crossover with an ATR-volatility filter.

    * Entry when the fast MA crosses above the slow MA while ATR is above its
      recent average (avoid flat-market whipsaws).
    * Exit when the fast MA crosses below the slow MA.
    """
    fast = max(int(fast), 1)
    slow = max(int(slow), fast + 1)
    atr_period = max(int(atr_period), 1)

    fast_ma = price.rolling(window=fast).mean()
    slow_ma = price.rolling(window=slow).mean()
    atr = _atr_from_price(price, atr_period)
    atr_avg = atr.rolling(window=atr_period).mean()

    entries = (
        (fast_ma > slow_ma)
        & (fast_ma.shift(1) <= slow_ma.shift(1))
        & (atr > atr_avg.shift(1))
    )
    exits = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
    return entries, exits


def momentum_signals(
    price: pd.Series,
    lookback: int = 20,
    threshold: float = 0.05,
) -> Tuple[pd.Series, pd.Series]:
    """Rate-of-change momentum.

    * Entry when price rises more than ``threshold`` over ``lookback`` bars.
    * Exit when it falls more than ``threshold`` over the same window.
    """
    lookback = max(int(lookback), 1)
    threshold = abs(float(threshold))

    roc = price / price.shift(lookback) - 1.0
    entries = (roc > threshold) & (roc.shift(1) <= threshold)
    exits = (roc < -threshold) & (roc.shift(1) >= -threshold)
    return entries, exits


def meanrev_signals(
    price: pd.Series,
    rsi_period: int = 14,
    oversold: float = 30.0,
    overbought: float = 70.0,
) -> Tuple[pd.Series, pd.Series]:
    """RSI mean-reversion.

    * Long when RSI crosses below ``oversold`` (buy dips).
    * Flat when RSI crosses above ``overbought`` (sell rallies).
    """
    rsi_period = max(int(rsi_period), 2)
    oversold = float(oversold)
    overbought = float(overbought)

    rsi = _rsi(price, rsi_period)
    entries = (rsi < oversold) & (rsi.shift(1) >= oversold)
    exits = (rsi > overbought) & (rsi.shift(1) <= overbought)
    return entries, exits


def breakout_signals(
    price: pd.Series,
    high: pd.Series = None,
    low: pd.Series = None,
    channel_period: int = 20,
    atr_buffer: float = 0.25,
    atr_period: int = 14,
) -> Tuple[pd.Series, pd.Series]:
    """Donchian-channel breakout buffered by ATR.

    If ``high``/``low`` are omitted, the bar's ``close`` is used for both
    (useful for tests); in production pass the real high/low series.
    """
    channel_period = max(int(channel_period), 2)
    atr_period = max(int(atr_period), 1)
    atr_buffer = float(atr_buffer)

    high = price if high is None else high
    low = price if low is None else low

    upper = high.rolling(window=channel_period).max().shift(1)
    lower = low.rolling(window=channel_period).min().shift(1)
    atr = _atr_from_ohlc(high, low, price, atr_period)
    buffer = atr_buffer * atr

    entries = (price > upper + buffer) & (price.shift(1) <= upper.shift(1) + buffer.shift(1))
    exits = (price < lower - buffer) & (price.shift(1) >= lower.shift(1) - buffer.shift(1))
    return entries, exits


def volatility_signals(
    price: pd.Series,
    atr_period: int = 14,
    squeeze_percentile: float = 0.25,
    regime_period: int = 100,
) -> Tuple[pd.Series, pd.Series]:
    """Volatility regime: long compression, flat expansion.

    Directional interpretation used here:

    * Enter long when ATR drops into the bottom ``squeeze_percentile`` of its
      recent distribution (a volatility squeeze that often precedes expansion).
    * Exit when ATR rises into the top ``squeeze_percentile`` (high volatility;
      the move has already occurred or is exhausted).
    """
    atr_period = max(int(atr_period), 1)
    squeeze_percentile = float(squeeze_percentile)
    regime_period = max(int(regime_period), atr_period + 2)

    atr = _atr_from_price(price, atr_period)
    atr_pct = _percentile_rank(atr, regime_period)

    entries = (atr_pct < squeeze_percentile) & (atr_pct.shift(1) >= squeeze_percentile)
    exits = (atr_pct > 1.0 - squeeze_percentile) & (atr_pct.shift(1) <= 1.0 - squeeze_percentile)
    return entries, exits


def volume_signals(
    df: pd.DataFrame,
    lookback: int = 20,
    climax_sigma: float = 2.0,
    absorption_ratio: float = 0.35,
    spread_percentile: float = 0.7,
) -> Tuple[pd.Series, pd.Series]:
    """Volume-effort vs result.

    * A volume spike (>= ``climax_sigma`` std above mean) with the close in the
      upper portion of the bar is treated as a buying climax -> entry.
    * A volume spike with the close in the lower portion is treated as a selling
      climax -> exit.
    * An absorption bar (high volume, small spread relative to average) flips
      against the bar's direction and is handled the same way.
    """
    lookback = max(int(lookback), 2)
    climax_sigma = float(climax_sigma)
    absorption_ratio = float(absorption_ratio)
    spread_percentile = float(spread_percentile)

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    open_ = df["open"].astype(float)
    volume = df["volume"].astype(float)

    spread = (high - low).abs()
    avg_volume = volume.rolling(window=lookback, min_periods=lookback).mean().shift(1)
    std_volume = volume.rolling(window=lookback, min_periods=lookback).std(ddof=0).shift(1)
    avg_spread = spread.rolling(window=lookback, min_periods=lookback).mean().shift(1)
    spread_pct = _percentile_rank(spread, lookback)

    volume_z = (volume - avg_volume) / std_volume.replace(0, np.nan)
    spread_ratio = spread / avg_spread.replace(0, np.nan)

    high_effort = volume_z >= climax_sigma
    wide_spread = spread_pct >= spread_percentile
    climax = high_effort & wide_spread
    absorption = high_effort & (spread_ratio <= absorption_ratio)
    trigger = climax | absorption

    rng = spread.replace(0, np.nan)
    clv = ((close - low) / rng).fillna(0.5)
    signed = 2.0 * clv - 1.0

    bar_dir = np.sign(close - open_).fillna(0.0)
    # For absorption, fade the side that is pushing.
    signed = signed.where(~absorption, -bar_dir)

    entries = trigger & (signed > 0) & ((trigger & (signed > 0)).shift(1).fillna(False) == False)
    exits = trigger & (signed < 0) & ((trigger & (signed < 0)).shift(1).fillna(False) == False)
    return entries, exits


SIGNAL_FUNCS: Dict[str, Callable[..., Tuple[pd.Series, pd.Series]]] = {
    "trend": trend_signals,
    "momentum": momentum_signals,
    "meanrev": meanrev_signals,
    "breakout": breakout_signals,
    "volatility": volatility_signals,
    "volume": volume_signals,
}


def build_signal(signal_name: str, params: Dict[str, Any], df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Build ``(entries, exits)`` for ``signal_name`` using ``df`` OHLCV data."""
    if signal_name not in SIGNAL_FUNCS:
        raise ValueError(f"Unknown signal: {signal_name}")
    price = df["close"]
    if signal_name == "trend":
        return trend_signals(
            price,
            fast=params["fast"],
            slow=params["slow"],
            atr_period=params.get("atr_period", 14),
        )
    if signal_name == "momentum":
        return momentum_signals(
            price,
            lookback=params["lookback"],
            threshold=params["threshold"],
        )
    if signal_name == "meanrev":
        return meanrev_signals(
            price,
            rsi_period=params["rsi_period"],
            oversold=params["oversold"],
            overbought=params["overbought"],
        )
    if signal_name == "breakout":
        return breakout_signals(
            price,
            high=df["high"],
            low=df["low"],
            channel_period=params["channel_period"],
            atr_buffer=params["atr_buffer"],
            atr_period=params.get("atr_period", 14),
        )
    if signal_name == "volatility":
        return volatility_signals(
            price,
            atr_period=params["atr_period"],
            squeeze_percentile=params["squeeze_percentile"],
            regime_period=params.get("regime_period", 100),
        )
    if signal_name == "volume":
        return volume_signals(
            df,
            lookback=params["lookback"],
            climax_sigma=params["climax_sigma"],
            absorption_ratio=params["absorption_ratio"],
            spread_percentile=params.get("spread_percentile", 0.7),
        )
    raise ValueError(f"Unhandled signal: {signal_name}")


def _param_grid(ranges: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(ranges.keys())
    values = [ranges[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


# Parameter grids used by the walk-forward search.  Keys mirror technical.yaml.
SIGNAL_PARAMS: Dict[str, List[Dict[str, Any]]] = {
    "trend": _param_grid(
        {
            "fast": [10, 20],
            "slow": [40, 50],
            "atr_period": [10, 14],
        }
    ),
    "momentum": _param_grid(
        {
            "lookback": [10, 20],
            "threshold": [0.02, 0.05],
        }
    ),
    "meanrev": _param_grid(
        {
            "rsi_period": [10, 14],
            "oversold": [25, 30],
            "overbought": [70, 75],
        }
    ),
    "breakout": _param_grid(
        {
            "channel_period": [15, 20],
            "atr_period": [10, 14],
            "atr_buffer": [0.1, 0.25, 0.5],
        }
    ),
    "volatility": _param_grid(
        {
            "atr_period": [10, 14],
            "squeeze_percentile": [0.2, 0.25],
            "regime_period": [80, 100],
        }
    ),
    "volume": _param_grid(
        {
            "lookback": [15, 20],
            "climax_sigma": [1.5, 2.0],
            "absorption_ratio": [0.3, 0.35],
        }
    ),
}
