from __future__ import annotations

import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from starlette.responses import Response

from .. import db
from ..execution import PaperExecutionEngine

logger = logging.getLogger(__name__)

# Default Slack app instance set by ``get_slack_app``.  ``slack_request_handler``
# uses this unless an app is passed explicitly.
_default_slack_app: Optional[Any] = None


def _approval_id() -> str:
    return uuid.uuid4().hex[:12]


def generate_approval_id() -> str:
    """Return a short UUID string suitable for Slack button values."""
    return _approval_id()


def _truncated_rationale(idea: Dict[str, Any], max_len: int = 300) -> str:
    rationale = idea.get("rationale", "") or ""
    if len(rationale) > max_len:
        return rationale[: max_len - 3] + "..."
    return rationale


def _build_signal_blocks(idea: Dict[str, Any], run_id: str) -> list:
    instrument = idea.get("instrument", "unknown")
    direction = idea.get("direction", "flat")
    rank = idea.get("rank", "?")
    raw_score = idea.get("raw_score", 0.0)
    adjusted_score = idea.get("adjusted_score", raw_score)
    rationale = _truncated_rationale(idea)

    header_text = f"Trade Idea: {instrument} {direction} (rank {rank})"
    fallback = f"Trade idea: {instrument} {direction} (rank {rank})"

    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": header_text},
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Instrument:* {instrument}"},
                {"type": "mrkdwn", "text": f"*Direction:* {direction}"},
                {"type": "mrkdwn", "text": f"*Raw Score:* {raw_score}"},
                {"type": "mrkdwn", "text": f"*Adjusted Score:* {adjusted_score}"},
                {"type": "mrkdwn", "text": f"*Rank:* {rank}"},
                {"type": "mrkdwn", "text": f"*Rationale:* {rationale}"},
            ],
        },
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Approve"},
                    "action_id": "approve_idea",
                    "value": run_id,
                    "style": "primary",
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Reject"},
                    "action_id": "reject_idea",
                    "value": run_id,
                    "style": "danger",
                },
            ],
        },
    ]
    return blocks, fallback


async def post_ranked_signal(
    app: Any,
    channel: str,
    idea: Dict[str, Any],
    run_id: str,
    session_factory: Optional[Any] = None,
) -> str:
    """Post a ranked trade idea to Slack and log it to the decision database.

    Returns the Slack message timestamp.
    """
    session_factory = session_factory or db.AsyncSessionLocal
    if session_factory is not None:
        try:
            async with session_factory() as session:
                await db.log_decision(
                    session,
                    run_id=run_id,
                    source="slack",
                    action="slack_post",
                    instrument=idea.get("instrument"),
                    underlying=idea.get("underlying") or None,
                    direction=idea.get("direction"),
                    payload=idea,
                    rationale=idea.get("rationale"),
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to log slack_post decision: %s", exc)
    else:
        logger.warning("No database session factory configured; skipping slack_post log")

    blocks, fallback = _build_signal_blocks(idea, run_id)
    response = await app.client.chat_postMessage(
        channel=channel,
        blocks=blocks,
        text=fallback,
    )
    return response["ts"]


def _user_label(user: Dict[str, Any]) -> str:
    return user.get("name") or user.get("id") or "unknown"


def _action_from_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    actions = payload.get("actions") or []
    return actions[0] if actions else {}


async def handle_action(
    payload: Dict[str, Any],
    app: Any,
    execution_engine: PaperExecutionEngine,
    session_factory: Any,
) -> str:
    """Handle an Approve/Reject button click from Slack.

    ``session_factory`` is typically ``db.AsyncSessionLocal``; tests may pass a
    callable returning a context manager with a mocked session.
    """
    action = _action_from_payload(payload)
    action_id = action.get("action_id")
    run_id = action.get("value")

    user = payload.get("user", {})
    user_name = _user_label(user)
    user_id = user.get("id") or user_name
    channel_id = (payload.get("channel", {}) or {}).get("id", "")

    if not run_id:
        return "Missing run_id."

    if session_factory is None:
        return "Database not configured."

    async with session_factory() as session:
        decision = await db.get_decision(session, run_id)
        if decision is None:
            return "Idea not found."

        idea = decision.payload or {}
        instrument = idea.get("instrument")
        underlying = idea.get("underlying") or None
        direction = idea.get("direction")

        await db.log_decision(
            session,
            run_id=run_id,
            source="slack",
            action="reject" if action_id == "reject_idea" else "approve",
            instrument=instrument,
            underlying=underlying,
            direction=direction,
            payload=idea,
            rationale=f"Slack action by {user_name}",
        )

        if action_id == "reject_idea":
            text = f"Rejected by <@{user_id}>"
            try:
                await app.client.chat_postEphemeral(
                    channel=channel_id,
                    user=user_id,
                    text=text,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to post ephemeral reject message: %s", exc)
            return text

        if action_id == "approve_idea":
            result = await execution_engine.execute_idea(idea)
            status = result.get("status")
            risk_result = result.get("risk")

            if status == "risk_rejected":
                log_action = "risk_reject"
                reason = (risk_result or {}).get("reason", "unknown")
                text = f"Risk check rejected: {reason}"
                order_result = None
            elif status in ("filled", "paper_filled"):
                log_action = "place_order"
                order_id = result.get("order_id")
                text = f"Paper order placed: {order_id}"
                order_result = {
                    "order_id": order_id,
                    "status": status,
                    "order": result.get("order"),
                }
            else:
                log_action = "error"
                error = result.get("error", "unknown error")
                text = f"Error placing order: {error}"
                order_result = {"error": error, "status": status}

            await db.log_decision(
                session,
                run_id=run_id,
                source="execution",
                action=log_action,
                instrument=instrument,
                underlying=underlying,
                direction=direction,
                payload=idea,
                risk_result=risk_result,
                order_result=order_result,
                rationale=text,
            )

            try:
                await app.client.chat_postMessage(channel=channel_id, text=text)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to post order result message: %s", exc)
            return text

    return "Unknown action."


def get_slack_app(
    execution_engine: Optional[PaperExecutionEngine] = None,
    session_factory: Optional[Any] = None,
) -> Optional[Any]:
    """Create and configure a Bolt AsyncApp with action handlers.

    Returns ``None`` unless both ``SLACK_BOT_TOKEN`` and
    ``SLACK_SIGNING_SECRET`` are set (Bolt refuses to construct or verify
    requests without them), so the rest of the app can run in log-only mode
    without Slack credentials.  The returned app is also stored as the module
    default for ``slack_request_handler``.
    """
    from slack_bolt.async_app import AsyncApp

    token = os.getenv("SLACK_BOT_TOKEN")
    signing_secret = os.getenv("SLACK_SIGNING_SECRET")
    if not token or not signing_secret:
        logger.warning(
            "SLACK_BOT_TOKEN/SLACK_SIGNING_SECRET not set; Slack integration disabled"
        )
        return None

    app = AsyncApp(token=token, signing_secret=signing_secret)

    @app.action("approve_idea")
    async def approve_handler(ack, body):  # type: ignore[no-untyped-def]
        await ack()
        if execution_engine is None:
            logger.warning("Approve clicked but no execution engine configured")
            return
        await handle_action(body, app, execution_engine, session_factory)

    @app.action("reject_idea")
    async def reject_handler(ack, body):  # type: ignore[no-untyped-def]
        await ack()
        if execution_engine is None:
            logger.warning("Reject clicked but no execution engine configured")
            return
        await handle_action(body, app, execution_engine, session_factory)

    global _default_slack_app
    _default_slack_app = app
    return app


async def slack_request_handler(request: Any, app: Optional[Any] = None) -> Response:
    """Translate a FastAPI/Starlette request into a Bolt request and dispatch it.

    If ``app`` is not provided, the last app created by ``get_slack_app`` is
    used.
    """
    from slack_bolt.request.async_request import AsyncBoltRequest

    app = app or _default_slack_app
    if app is None:
        return Response(
            content="Slack integration is not configured",
            status_code=503,
        )

    body = await request.body()
    body_str = body.decode("utf-8") if isinstance(body, bytes) else str(body)
    query = str(request.query_params) or ""
    headers = {k: v for k, v in request.headers.items()}

    bolt_req = AsyncBoltRequest(body=body_str, query=query, headers=headers)
    bolt_resp = await app.async_dispatch(bolt_req)

    response = Response(content=bolt_resp.body, status_code=bolt_resp.status)

    # Bolt returns a header map whose values may be lists (e.g. multiple
    # Set-Cookie).  Starlette's ``headers=`` argument only accepts a mapping of
    # single values, so append to the raw headers instead, replacing any
    # defaults Response already set for the same key.
    overridden = {key.lower().encode("latin-1") for key in bolt_resp.headers}
    response.raw_headers[:] = [
        (key, value) for key, value in response.raw_headers if key not in overridden
    ]
    for key, value in bolt_resp.headers.items():
        values = value if isinstance(value, list) else [value]
        for item in values:
            response.raw_headers.append(
                (key.lower().encode("latin-1"), str(item).encode("latin-1"))
            )

    return response
