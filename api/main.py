from __future__ import annotations

from fastapi import FastAPI

from .adapters import CCXTVenue, IBKRVenue, Instrument
from .config import load_all

app = FastAPI(title="TradePulse API")


def _build_venues(config: dict):
    return {
        "ibkr": IBKRVenue(config.get("ibkr", {})),
        "ccxt": CCXTVenue(config.get("ccxt", {})),
    }


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/config")
async def get_config() -> dict:
    return load_all()


@app.get("/instruments")
async def list_instruments() -> list:
    cfg = load_all()
    return list(cfg["instruments"].keys())
