from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import yaml

CONFIG_DIR = Path(__file__).parent


def _load_signals() -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Load global signal blocks plus per-underlying ``<underlying>_shared.yaml`` files.

    Global files contribute to ``signals`` keyed by signal name.
    Files ending in ``_shared.yaml`` are treated as underlying-specific shared
    signal definitions; their contents are keyed under
    ``signals_by_underlying[<underlying>][<signal>]``.
    """
    signals: Dict[str, Any] = {}
    by_underlying: Dict[str, Dict[str, Any]] = {}
    directory = CONFIG_DIR / "signals"
    if not directory.exists():
        return signals, by_underlying

    for path in sorted(directory.iterdir()):
        if path.suffix not in (".yaml", ".yml", ".json"):
            continue
        with path.open() as f:
            content = yaml.safe_load(f)
        if not isinstance(content, dict):
            continue

        stem = path.stem
        if stem.endswith("_shared"):
            underlying = stem[: -len("_shared")].upper()
            by_underlying.setdefault(underlying, {}).update(content)
        else:
            signals.update(content)

    return signals, by_underlying


def _load_dir(name: str) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    directory = CONFIG_DIR / name
    if not directory.exists():
        return data
    for path in sorted(directory.iterdir()):
        if path.suffix in (".yaml", ".yml", ".json"):
            with path.open() as f:
                content = yaml.safe_load(f)
                if isinstance(content, dict):
                    data.update(content)
    return data


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Return a new dict: ``override`` keys recursively update ``base``."""
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_all() -> Dict[str, Any]:
    underlyings = _load_dir("underlyings")
    signals, signals_by_underlying = _load_signals()
    instruments = _load_dir("instruments")

    for symbol, cfg in instruments.items():
        merged: Dict[str, Any] = {}

        # merge shared signal blocks first (global)
        for block_name in cfg.get("shared_signals", []):
            shared = signals.get(block_name, {})
            if shared:
                merged[block_name] = dict(shared)

        # then underlying-specific shared blocks (research-optimized parameters)
        underlying_key = cfg.get("underlying")
        if underlying_key:
            for block_name, block in (signals_by_underlying.get(underlying_key) or {}).items():
                merged[block_name] = _deep_merge(merged.get(block_name, {}), block)

        # finally instrument-specific overrides
        for block_name, block in (cfg.get("signals") or {}).items():
            merged[block_name] = _deep_merge(merged.get(block_name, {}), block)

        cfg["signals"] = merged

        # link underlying definition
        if underlying_key and underlying_key in underlyings:
            cfg["underlying"] = underlyings[underlying_key]

    return {
        "underlyings": underlyings,
        "signals": signals,
        "signals_by_underlying": signals_by_underlying,
        "instruments": instruments,
    }
