"""Internal math/extraction helpers shared by the signal calculators.

Pure numpy/pandas. No I/O, no network, no LLM.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .base import Signal

OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


def _to_float(value: Any, default: float = float("nan")) -> float:
    if value is None:
        return default
    if isinstance(value, Decimal):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def extract_quote(data: Any) -> Dict[str, Any]:
    """Pull the optional quote dict out of a flexible ``data`` payload."""
    if isinstance(data, dict):
        quote = data.get("quote")
        if isinstance(quote, dict):
            return quote
    return {}


def extract_frame(data: Any) -> pd.DataFrame:
    """Normalise ``data`` into an ohlcv DataFrame.

    Accepts:
      * a list/tuple of ``Bar`` dataclasses (or ohlcv dicts)
      * a pandas DataFrame with ohlcv columns
      * a dict containing ``bars`` (plus an optional ``quote``)

    Always returns a DataFrame with float columns
    ``open, high, low, close, volume`` (possibly empty).
    """
    if data is None:
        return _empty_frame()

    if isinstance(data, dict):
        return extract_frame(data.get("bars"))

    if isinstance(data, pd.DataFrame):
        frame = data.copy()
        frame.columns = [str(c).lower() for c in frame.columns]
        missing = [c for c in OHLCV_COLUMNS if c not in frame.columns]
        if "volume" in missing and len(missing) == 1:
            frame["volume"] = 0.0
            missing = []
        if missing:
            return _empty_frame()
        out = frame.loc[:, list(OHLCV_COLUMNS)].reset_index(drop=True)
        return out.apply(pd.to_numeric, errors="coerce").astype(float)

    if isinstance(data, (list, tuple)):
        rows = []
        for bar in data:
            if isinstance(bar, dict):
                get = bar.get
            else:
                def get(key, _bar=bar):  # type: ignore[misc]
                    return getattr(_bar, key, None)
            rows.append(
                {
                    "open": _to_float(get("open")),
                    "high": _to_float(get("high")),
                    "low": _to_float(get("low")),
                    "close": _to_float(get("close")),
                    "volume": _to_float(get("volume"), 0.0),
                }
            )
        if not rows:
            return _empty_frame()
        return pd.DataFrame(rows, columns=list(OHLCV_COLUMNS)).astype(float)

    return _empty_frame()


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in OHLCV_COLUMNS})


def ema(series: pd.Series, period: int) -> pd.Series:
    period = max(int(period), 1)
    return series.astype(float).ewm(span=period, adjust=False).mean()


def true_range(frame: pd.DataFrame) -> pd.Series:
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    prev_close = frame["close"].astype(float).shift(1)
    ranges = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    )
    tr = ranges.max(axis=1)
    tr.iloc[0] = float(high.iloc[0] - low.iloc[0]) if len(frame) else np.nan
    return tr


def atr(frame: pd.DataFrame, period: int) -> pd.Series:
    period = max(int(period), 1)
    return true_range(frame).ewm(alpha=1.0 / period, adjust=False).mean()


def rsi(series: pd.Series, period: int) -> pd.Series:
    period = max(int(period), 1)
    values = series.astype(float)
    delta = values.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    # all-gain windows -> RSI 100, all-loss windows -> RSI 0
    out = out.where(avg_loss != 0.0, 100.0)
    out = out.where(avg_gain != 0.0, 0.0)
    return out


def zscore(series: pd.Series, period: int) -> float:
    period = max(int(period), 2)
    window = series.astype(float).tail(period)
    if len(window) < 2:
        return 0.0
    std = float(window.std(ddof=0))
    if not np.isfinite(std) or std <= 0.0:
        return 0.0
    return float((float(window.iloc[-1]) - float(window.mean())) / std)


def percentile_rank(series: pd.Series, value: float, period: Optional[int] = None) -> float:
    """Fraction of the (trailing) window that is <= ``value``, in [0, 1]."""
    window = series.astype(float).dropna()
    if period:
        window = window.tail(int(period))
    if window.empty or not np.isfinite(value):
        return 0.5
    return float((window <= value).sum()) / float(len(window))


def clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(min(max(value, low), high))


def direction_from_score(score: float, deadband: float = 0.05) -> str:
    if score > deadband:
        return "long"
    if score < -deadband:
        return "short"
    return "flat"


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator is None or not np.isfinite(denominator) or denominator == 0.0:
        return default
    result = numerator / denominator
    return float(result) if np.isfinite(result) else default


def finite(value: Any, default: float = 0.0) -> float:
    number = _to_float(value, float("nan"))
    return float(number) if np.isfinite(number) else default


class ConfiguredSignal(Signal):
    """Small base that stores a config block and builds the shared result dict."""

    name: str = "signal"
    defaults: Dict[str, Any] = {}

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        merged: Dict[str, Any] = dict(self.defaults)
        merged.update(config or {})
        self.config = merged

    @property
    def weight(self) -> float:
        return finite(self.config.get("weight"), 0.0)

    def _param(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

    def _int(self, key: str, default: int) -> int:
        try:
            return int(self.config.get(key, default))
        except (TypeError, ValueError):
            return default

    def _float(self, key: str, default: float) -> float:
        return finite(self.config.get(key, default), default)

    def result(
        self,
        score: float,
        direction: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
        **extra: Any,
    ) -> Dict[str, Any]:
        clamped = clamp(score)
        payload: Dict[str, Any] = {
            "name": self.name,
            "score": clamped,
            "direction": direction or direction_from_score(clamped),
            "weight": self.weight,
            "metrics": metrics or {},
            "ok": True,
        }
        payload.update(extra)
        return payload

    def insufficient(
        self,
        reason: str,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self.result(
            0.0,
            direction="flat",
            metrics=metrics or {},
            ok=False,
            reason=reason,
        )
