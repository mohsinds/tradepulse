from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from decimal import Decimal
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request, Response

from .adapters import CCXTVenue, IBKRVenue
from .config import load_all

logger = logging.getLogger(__name__)


def _broker_configs() -> Dict[str, Any]:
    return {
        "ibkr": {
            "host": os.getenv("IBKR_HOST", "127.0.0.1"),
            "port": int(os.getenv("IBKR_PORT", "7497")),
            "client_id": int(os.getenv("IBKR_CLIENT_ID", "1")),
            "account": os.getenv("IBKR_ACCOUNT", ""),
        },
        "ccxt": {
            "exchange_id": os.getenv("CCXT_EXCHANGE_ID", "bybit"),
            "api_key": os.getenv("CCXT_API_KEY", ""),
            "api_secret": os.getenv("CCXT_API_SECRET", ""),
            "password": os.getenv("CCXT_PASSWORD", ""),
            "sandbox": os.getenv("CCXT_SANDBOX", "false").lower() == "true",
        },
    }


def _build_venues(config: dict):
    return {
        "ibkr": IBKRVenue(config.get("ibkr", {})),
        "ccxt": CCXTVenue(config.get("ccxt", {})),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the DB, execution engine, and Slack app on startup."""
    from . import db
    from .execution import PaperExecutionEngine
    from .integrations.slack import get_slack_app

    if os.getenv("DATABASE_URL"):
        try:
            await db.init_db()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Database initialization failed: %s", exc)

    cfg = load_all()
    adapters = _build_venues(_broker_configs())
    account_value = Decimal(os.getenv("PAPER_ACCOUNT_VALUE", "100000"))
    paper = os.getenv("PAPER_TRADING", "true").lower() != "false"

    execution_engine = PaperExecutionEngine(
        adapters=adapters,
        account_value=account_value,
        underlying_configs=cfg.get("underlyings", {}),
        db_session_factory=db.AsyncSessionLocal,
        paper=paper,
    )
    app.state.execution_engine = execution_engine
    app.state.slack_app = get_slack_app(execution_engine, db.AsyncSessionLocal)

    yield


app = FastAPI(title="TradePulse API", lifespan=lifespan)


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


@app.post("/webhooks/tradingview")
async def tradingview_webhook(request: Request) -> Dict[str, Any]:
    """Accept a TradingView Pine alert as an additional advisory input.

    Validates the shared secret, normalizes the alert into the same signal
    result model used by internal calculators, and runs it through the ranking
    agent. The response is advisory only; no order is placed.
    """
    from .agents import RankerAgent
    from .webhooks import TradingViewWebhookHandler

    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}") from exc

    handler = TradingViewWebhookHandler.from_env()
    try:
        signal = await handler.handle(payload)
    except PermissionError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Malformed alert: {exc}") from exc

    ticker = signal["metrics"]["ticker"]
    cfg = load_all()
    instrument_cfg = cfg["instruments"].get(ticker)
    if instrument_cfg is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown ticker '{ticker}' — add it to api/config/instruments/",
        )

    # Feed the TradingView signal into the ranker as an additional input.
    # The ranker still weights it against config, applies funding adjustments,
    # enforces advisory-only output, and never emits an order.
    instruments = {ticker: instrument_cfg}
    signal_results = {ticker: [signal]}

    try:
        ranker = RankerAgent()
        result = await ranker.run(
            {
                "instruments": instruments,
                "signal_results": signal_results,
                "tags": ["tradingview", "webhook"],
                "metadata": {
                    "source": "tradingview_webhook",
                    "alert_price": signal["metrics"].get("price"),
                },
            }
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Ranking failed: {exc}"
        ) from exc

    return {
        "advisory_only": True,
        "execution_enabled": False,
        "source_signal": signal,
        "ranking": result,
    }


@app.post("/slack/events")
async def slack_events(request: Request) -> Response:
    """Inbound Slack events and interactivity requests."""
    from .integrations.slack import slack_request_handler

    slack_app = getattr(request.app.state, "slack_app", None)
    return await slack_request_handler(request, app=slack_app)


@app.post("/advisory/notify")
async def advisory_notify(request: Request) -> Dict[str, Any]:
    """Run the ranker on a single instrument and notify Slack with the top idea.

    If ``SLACK_BOT_TOKEN`` is not configured, the idea is still logged to the
    decision database and a JSON response is returned describing what would
    have been posted.
    """
    from . import db
    from .agents import RankerAgent
    from .integrations.slack import generate_approval_id, post_ranked_signal

    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}") from exc

    ticker = payload.get("instrument")
    if not ticker:
        raise HTTPException(status_code=400, detail="instrument is required")

    cfg = load_all()
    instrument_cfg = cfg["instruments"].get(ticker)
    if instrument_cfg is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown ticker '{ticker}'",
        )

    signal_results = payload.get("signal_results", {})
    if not isinstance(signal_results, dict) or ticker not in signal_results:
        raise HTTPException(
            status_code=400,
            detail="signal_results must be a dict containing the instrument",
        )

    instruments = {ticker: instrument_cfg}

    try:
        ranker = RankerAgent()
        result = await ranker.run(
            {
                "instruments": instruments,
                "signal_results": signal_results,
                "tags": ["advisory", "notify"],
                "metadata": {"source": "advisory_notify"},
            }
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ranking failed: {exc}") from exc

    ideas = result.get("ideas", [])
    if not ideas:
        return {"advisory_only": True, "message": "No ranked ideas generated."}

    top_idea = ideas[0]
    run_id = generate_approval_id()
    channel = os.getenv("SLACK_CHANNEL", "#signals")
    slack_app = getattr(request.app.state, "slack_app", None)

    if os.getenv("SLACK_BOT_TOKEN") and slack_app is not None:
        ts = await post_ranked_signal(
            slack_app,
            channel,
            top_idea,
            run_id,
            session_factory=db.AsyncSessionLocal,
        )
        return {
            "advisory_only": True,
            "run_id": run_id,
            "posted": True,
            "ts": ts,
            "idea": top_idea,
        }

    # Slack not configured: log the decision and return an advisory payload.
    if db.AsyncSessionLocal is not None:
        try:
            async with db.AsyncSessionLocal() as session:
                await db.log_decision(
                    session,
                    run_id=run_id,
                    source="slack",
                    action="slack_post",
                    instrument=top_idea.get("instrument"),
                    underlying=top_idea.get("underlying") or None,
                    direction=top_idea.get("direction"),
                    payload=top_idea,
                    rationale=top_idea.get("rationale"),
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to log advisory notify decision: %s", exc)

    return {
        "advisory_only": True,
        "run_id": run_id,
        "posted": False,
        "message": "Slack not configured; idea logged only",
        "idea": top_idea,
    }
