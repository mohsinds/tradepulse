"""Walk-forward parameter sweep that produces per-underlying shared configs."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import vectorbt as vbt
import yaml

from api.adapters import ContractType, Instrument
from api.config import load_all

from .costs import CostOverlay, cost_overlay_for
from .data import fetch_bars, make_synthetic_bars
from .metrics import score_portfolio
from .strategy import SIGNAL_PARAMS, build_signal


def _split_train_test(df: pd.DataFrame, train_frac: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split = int(len(df) * train_frac)
    if split < 30 or len(df) - split < 10:
        # Too small for a meaningful split; use the whole series as train.
        return df, df
    return df.iloc[:split].copy(), df.iloc[split:].copy()


def _build_cost_overlay(instrument: Instrument, df: pd.DataFrame) -> CostOverlay:
    """Return a holding-cost overlay configured for this instrument's data."""
    kwargs: Dict[str, Any] = {}
    if instrument.contract_type == ContractType.PERPETUAL:
        funding = df.get("funding_rate")
        if funding is not None:
            kwargs["funding_rate"] = funding
    elif instrument.contract_type == ContractType.FUTURE:
        kwargs["roll_cost_pct"] = float(instrument.meta.get("roll_cost_pct", 0.0002))
        kwargs["rolls_per_year"] = float(instrument.meta.get("rolls_per_year", 12))
    elif instrument.contract_type == ContractType.CFD:
        kwargs["annual_swap_rate"] = float(instrument.meta.get("annual_swap_rate", 0.03))
    return cost_overlay_for(instrument, **kwargs)


def _simulate(
    signal_name: str,
    params: Dict[str, Any],
    df: pd.DataFrame,
    instrument: Instrument,
) -> Dict[str, Any]:
    """Run a single train or test simulation and return metrics."""
    entries, exits = build_signal(signal_name, params, df)
    price = df["close"].astype(float)
    pf = vbt.Portfolio.from_signals(
        price,
        entries,
        exits,
        init_cash=10000.0,
        freq="1h",
        direction="longonly",
    )
    overlay = _build_cost_overlay(instrument, df)
    adjusted_returns = overlay.apply_cost(pf, price)
    return score_portfolio(pf, adjusted_returns)


def _evaluate_params_for_signal(
    signal_name: str,
    params: Dict[str, Any],
    data_map: List[Tuple[Instrument, pd.DataFrame]],
    train_frac: float,
) -> Tuple[float, Dict[str, Any]]:
    """Average train composite and aggregated test metrics across instruments."""
    train_scores: List[float] = []
    test_metrics_by_inst: List[Dict[str, float]] = []

    for instrument, df in data_map:
        train_df, test_df = _split_train_test(df, train_frac)
        train_result = _simulate(signal_name, params, train_df, instrument)
        train_scores.append(train_result["composite"])

        test_result = _simulate(signal_name, params, test_df, instrument)
        test_metrics_by_inst.append(test_result)

    avg_train = float(np.nanmean(train_scores))
    # Aggregate test metrics by averaging across instruments.
    aggregated: Dict[str, Any] = {}
    if test_metrics_by_inst:
        for key in test_metrics_by_inst[0]:
            values = [m[key] for m in test_metrics_by_inst]
            aggregated[key] = float(np.nanmean(values))
    return avg_train, aggregated


def _best_params_for_signal(
    signal_name: str,
    data_map: List[Tuple[Instrument, pd.DataFrame]],
    train_frac: float,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Select the parameter combo with the highest average train composite."""
    grid = SIGNAL_PARAMS.get(signal_name, [])
    if not grid:
        raise ValueError(f"No parameter grid for signal {signal_name}")

    best_params: Optional[Dict[str, Any]] = None
    best_score = -np.inf
    best_test_metrics: Dict[str, Any] = {}

    for params in grid:
        avg_train, test_metrics = _evaluate_params_for_signal(
            signal_name, params, data_map, train_frac
        )
        if avg_train > best_score or best_params is None:
            best_score = avg_train
            best_params = dict(params)
            best_test_metrics = test_metrics

    return best_params, best_test_metrics


async def sweep_underlying(
    underlying: str,
    instruments: List[Instrument],
    adapters_config: Dict[str, Any],
    start: datetime,
    end: datetime,
    timeframe: str,
    train_frac: float = 0.7,
    *,
    dry_run: bool = False,
    synthetic_bars: int = 500,
) -> Dict[str, Dict[str, Any]]:
    """Run a walk-forward grid search for one underlying across its instruments.

    Parameters
    ----------
    underlying:
        Underlying symbol, e.g. ``"GOLD"``.
    instruments:
        Instruments that share this underlying.
    adapters_config:
        Mapping ``venue -> config`` passed to ``fetch_bars``.
    start, end, timeframe:
        Historical bar request details.
    train_frac:
        Fraction of data used for in-sample parameter selection.
    dry_run:
        If True, use synthetic data and skip adapter connections.
    synthetic_bars:
        Number of bars to generate in dry-run mode.

    Returns
    -------
    dict
        ``{signal_name: {params..., test_metrics: {...}}}``.
    """
    data_map: List[Tuple[Instrument, pd.DataFrame]] = []
    for instrument in instruments:
        if dry_run:
            funding = 0.0001 if instrument.contract_type == ContractType.PERPETUAL else None
            df = make_synthetic_bars(
                n=synthetic_bars,
                start=start,
                trend_drift=0.02 if instrument.contract_type == ContractType.FUTURE else 0.0,
                volatility=1.0,
                funding_rate=funding,
                seed=hash(instrument.symbol) % 2**31,
            )
        else:
            df = await fetch_bars(instrument, adapters_config, start, end, timeframe)
        data_map.append((instrument, df))

    best_params: Dict[str, Dict[str, Any]] = {}
    for signal_name in SIGNAL_PARAMS:
        params, test_metrics = _best_params_for_signal(signal_name, data_map, train_frac)
        best_params[signal_name] = {**params, "test_metrics": test_metrics}
    return best_params


def _load_technical_weights() -> Dict[str, Any]:
    """Load the global technical signal weights from api/config/signals/technical.yaml."""
    path = Path(__file__).parents[1] / "api" / "config" / "signals" / "technical.yaml"
    if not path.exists():
        return {}
    with path.open() as f:
        content = yaml.safe_load(f)
    return content if isinstance(content, dict) else {}


def write_underlying_shared_config(
    underlying: str,
    best_params: Dict[str, Dict[str, Any]],
    output_dir: str = "api/config/signals",
) -> Path:
    """Write a ``<underlying>_shared.yaml`` file for ``api.config.load_all``.

    Each signal block contains the original ``weight`` from technical.yaml and
    the optimized parameters.  ``test_metrics`` and ``funding_rate`` are stripped
    because they are not tunable signal parameters.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    weights = _load_technical_weights()

    out: Dict[str, Any] = {}
    for signal_name, params in best_params.items():
        block: Dict[str, Any] = {}
        global_block = weights.get(signal_name, {})
        if isinstance(global_block, dict):
            block.update({k: v for k, v in global_block.items() if k == "weight"})
        # Add optimized parameters, excluding non-tunable metadata.
        for k, v in params.items():
            if k == "test_metrics" or k == "funding_rate":
                continue
            block[k] = v
        out[signal_name] = block

    file_path = output_path / f"{underlying.upper()}_shared.yaml"
    with file_path.open("w") as f:
        yaml.safe_dump(out, f, default_flow_style=False, sort_keys=True)
    return file_path


def _instrument_from_config(symbol: str, cfg: Dict[str, Any]) -> Instrument:
    """Build an ``Instrument`` from the merged config loaded by ``load_all``."""
    contract_str = cfg.get("contract_type", "spot")
    return Instrument(
        symbol=symbol,
        venue=cfg.get("venue", ""),
        contract_type=ContractType(contract_str),
        exchange=cfg.get("exchange", ""),
        underlying="",
        multiplier=Decimal(str(cfg.get("multiplier", 1))),
        currency=cfg.get("currency", "USD"),
        meta=cfg.get("meta", {}),
    )


def _group_instruments_by_underlying(
    config: Dict[str, Any],
) -> Dict[str, List[Instrument]]:
    """Group instrument configs by underlying symbol."""
    underlyings = config.get("underlyings", {})
    instruments = config.get("instruments", {})
    groups: Dict[str, List[Instrument]] = {k: [] for k in underlyings}
    for symbol, cfg in instruments.items():
        underlying_cfg = cfg.get("underlying")
        for underlying_symbol, ucfg in underlyings.items():
            if underlying_cfg is ucfg:
                inst = _instrument_from_config(symbol, cfg)
                groups[underlying_symbol].append(inst)
                break
    return {k: v for k, v in groups.items() if v}


async def run_sweep(
    start: datetime,
    end: datetime,
    timeframe: str = "1h",
    *,
    underlying: Optional[str] = None,
    dry_run: bool = False,
    adapters_config: Optional[Dict[str, Any]] = None,
    output_dir: str = "api/config/signals",
    train_frac: float = 0.7,
    synthetic_bars: int = 500,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Orchestrate the full sweep across all underlyings.

    Loads config, fetches data, runs one walk-forward search per underlying,
    writes ``<underlying>_shared.yaml`` files, and returns a report.
    """
    config = load_all()
    groups = _group_instruments_by_underlying(config)
    if underlying is not None:
        underlying = underlying.upper()
        groups = {k: v for k, v in groups.items() if k == underlying}
        if not groups:
            raise ValueError(f"No instruments found for underlying {underlying}")

    adapters_config = adapters_config or {}
    report: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for underlying_symbol, instruments in groups.items():
        best_params = await sweep_underlying(
            underlying_symbol,
            instruments,
            adapters_config,
            start,
            end,
            timeframe,
            train_frac=train_frac,
            dry_run=dry_run,
            synthetic_bars=synthetic_bars,
        )
        write_underlying_shared_config(underlying_symbol, best_params, output_dir)
        report[underlying_symbol] = best_params
    return report
