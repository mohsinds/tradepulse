"""Market data fetching helpers for research.

``fetch_bars`` routes to the correct venue adapter, fetches historical OHLCV,
and returns a uniform ``pandas.DataFrame``.  A fake adapter can be injected
via the private ``_adapter`` keyword for testing.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from api.adapters import Bar, CCXTVenue, IBKRVenue, Instrument


def _bars_to_df(bars: List[Bar]) -> pd.DataFrame:
    if not bars:
        return pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"]
        ).astype(float)
    records = [
        {
            "timestamp": b.timestamp,
            "open": float(b.open),
            "high": float(b.high),
            "low": float(b.low),
            "close": float(b.close),
            "volume": float(b.volume),
        }
        for b in bars
    ]
    df = pd.DataFrame(records).set_index("timestamp").sort_index()
    return df


# Module-level adapter registry so tests can inject fakes without touching
# application code.
_ADAPTER_CLS: Dict[str, Any] = {
    "ibkr": IBKRVenue,
    "ccxt": CCXTVenue,
}


def _make_adapter(venue: str, adapter_config: Dict[str, Any]) -> Any:
    venue = venue.lower()
    if venue not in _ADAPTER_CLS:
        raise ValueError(f"Unsupported venue: {venue}")
    config = adapter_config.get(venue, {}) if isinstance(adapter_config, dict) else {}
    return _ADAPTER_CLS[venue](config)


async def fetch_bars(
    instrument: Instrument,
    adapter_config: Dict[str, Any],
    start: datetime,
    end: datetime,
    timeframe: str,
    *,
    _adapter: Optional[Any] = None,
) -> pd.DataFrame:
    """Fetch historical bars for ``instrument`` and return a DataFrame.

    Parameters
    ----------
    instrument:
        The instrument to fetch.
    adapter_config:
        Mapping of venue name -> constructor config dict.
    start, end:
        Time range (naive datetimes are assumed in the adapter's local time).
    timeframe:
        Bar granularity, e.g. ``"1h"``.
    _adapter:
        Optional pre-built adapter instance (used by tests to avoid network).

    Returns
    -------
    pandas.DataFrame
        Columns ``open, high, low, close, volume`` indexed by timestamp.
    """
    adapter = _adapter if _adapter is not None else _make_adapter(instrument.venue, adapter_config)
    own_connection = _adapter is None
    if own_connection:
        await adapter.connect()
    try:
        bars = await adapter.get_historical_bars(instrument, start, end, timeframe)
    finally:
        if own_connection and hasattr(adapter, "disconnect"):
            await adapter.disconnect()
    return _bars_to_df(bars)


def make_synthetic_bars(
    n: int,
    *,
    trend_drift: float = 0.0,
    start_price: float = 100.0,
    funding_rate: Optional[float] = None,
    volatility: float = 1.0,
    start: datetime = datetime(2024, 1, 1),
    freq: str = "h",
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Generate a deterministic OHLCV DataFrame for dry runs and tests.

    Parameters
    ----------
    n:
        Number of bars.
    trend_drift:
        Average price change per bar.
    start_price:
        Initial close price.
    funding_rate:
        Optional constant funding rate to attach as a ``funding_rate`` column.
    volatility:
        Standard deviation of the per-bar shock (in price units).
    start:
        First bar timestamp.
    freq:
        Pandas frequency string; default ``"h"``.
    seed:
        Optional random seed for reproducibility.

    Returns
    -------
    pandas.DataFrame
        Columns ``open, high, low, close, volume`` and optionally
        ``funding_rate`` indexed by timestamp.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    index = pd.date_range(start=start, periods=n, freq=freq)
    shocks = rng.normal(trend_drift, volatility, size=n)
    close = start_price + np.cumsum(shocks)
    close = np.maximum(close, 0.01)

    # Build OHLC around the close with a random intrabar range.
    ranges = np.abs(rng.normal(0.0, volatility, size=n))
    ranges = np.clip(ranges, 0.001, None)
    opens = close - rng.uniform(-ranges, ranges)
    high = np.maximum(np.maximum(opens, close), opens + ranges * 0.5)
    low = np.minimum(np.minimum(opens, close), opens - ranges * 0.5)
    low = np.clip(low, 0.0001, None)

    volume = rng.lognormal(8.0, 0.5, size=n)

    df = pd.DataFrame(
        {
            "open": opens,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=index,
    )
    if funding_rate is not None:
        df["funding_rate"] = float(funding_rate)
    return df
