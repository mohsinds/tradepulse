"""CLI entry point for running the research parameter sweep."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import yaml

from research.param_sweep import run_sweep


def _parse_date(s: str) -> datetime:
    return datetime.fromisoformat(s)


def _load_adapters_config(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    with p.open() as f:
        if p.suffix in (".yaml", ".yml"):
            return yaml.safe_load(f) or {}
        return json.load(f)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a walk-forward parameter sweep per underlying."
    )
    parser.add_argument(
        "--start",
        type=_parse_date,
        default="2024-01-01",
        help="Start datetime (ISO format).",
    )
    parser.add_argument(
        "--end",
        type=_parse_date,
        default="2024-06-01",
        help="End datetime (ISO format).",
    )
    parser.add_argument(
        "--timeframe",
        default="1h",
        help="Bar timeframe (default: 1h).",
    )
    parser.add_argument(
        "--underlying",
        default=None,
        help="Limit sweep to a single underlying, e.g. GOLD.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use synthetic data and skip adapter connections.",
    )
    parser.add_argument(
        "--adapters-config",
        default=None,
        help="Path to a JSON/YAML file mapping venue -> adapter config.",
    )
    parser.add_argument(
        "--output-dir",
        default="api/config/signals",
        help="Directory to write <underlying>_shared.yaml files.",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.7,
        help="Fraction of data used for in-sample parameter selection.",
    )
    parser.add_argument(
        "--synthetic-bars",
        type=int,
        default=500,
        help="Number of bars to generate in dry-run mode.",
    )
    return parser


async def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    adapters_config: Dict[str, Any] = {}
    if args.adapters_config:
        adapters_config = _load_adapters_config(args.adapters_config)

    report = await run_sweep(
        start=args.start,
        end=args.end,
        timeframe=args.timeframe,
        underlying=args.underlying,
        dry_run=args.dry_run,
        adapters_config=adapters_config,
        output_dir=args.output_dir,
        train_frac=args.train_frac,
        synthetic_bars=args.synthetic_bars,
    )

    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
