from __future__ import annotations

import os
import unittest
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

from api.execution import PaperExecutionEngine
from api.integrations.slack import (
    generate_approval_id,
    get_slack_app,
    handle_action,
    post_ranked_signal,
    slack_request_handler,
)


def _fake_session_factory():
    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    return lambda: session


class TestSlackIntegration(unittest.IsolatedAsyncioTestCase):
    async def test_post_ranked_signal_builds_blocks_and_logs(self) -> None:
        app = MagicMock()
        app.client.chat_postMessage = AsyncMock(return_value={"ts": "12345.67"})

        idea = {
            "instrument": "XAUUSDT.P",
            "direction": "long",
            "raw_score": 0.5,
            "adjusted_score": 0.4,
            "rank": 1,
            "rationale": "Strong momentum and funding tailwind.",
        }
        run_id = "run-abc"

        with patch("api.integrations.slack.db.log_decision", new=AsyncMock()) as mock_log:
            ts = await post_ranked_signal(
                app,
                "#signals",
                idea,
                run_id,
                session_factory=_fake_session_factory(),
            )

        self.assertEqual(ts, "12345.67")
        app.client.chat_postMessage.assert_awaited_once()
        call_kwargs = app.client.chat_postMessage.call_args[1]
        self.assertEqual(call_kwargs["channel"], "#signals")
        blocks = call_kwargs["blocks"]
        self.assertEqual(blocks[0]["type"], "header")
        self.assertEqual(blocks[1]["type"], "section")
        self.assertEqual(blocks[2]["type"], "actions")
        actions = blocks[2]["elements"]
        self.assertEqual(len(actions), 2)
        self.assertEqual(actions[0]["action_id"], "approve_idea")
        self.assertEqual(actions[0]["value"], run_id)
        self.assertEqual(actions[0]["style"], "primary")
        self.assertEqual(actions[1]["action_id"], "reject_idea")
        self.assertEqual(actions[1]["value"], run_id)
        self.assertEqual(actions[1]["style"], "danger")

        mock_log.assert_awaited_once()
        log_kwargs = mock_log.call_args[1]
        self.assertEqual(log_kwargs["action"], "slack_post")
        self.assertEqual(log_kwargs["run_id"], run_id)
        self.assertEqual(log_kwargs["payload"], idea)

    async def test_handle_action_approve_executes_and_posts(self) -> None:
        idea = {"instrument": "MGC", "direction": "long"}
        execution_engine = MagicMock(spec=PaperExecutionEngine)
        execution_engine.execute_idea = AsyncMock(
            return_value={
                "status": "paper_filled",
                "order_id": "paper-123",
                "risk": {"ok": True},
            }
        )

        app = MagicMock()
        app.client.chat_postMessage = AsyncMock()

        payload = {
            "actions": [{"action_id": "approve_idea", "value": "run-1"}],
            "user": {"id": "U1", "name": "trader"},
            "channel": {"id": "C1"},
        }

        decision = MagicMock(payload=idea)

        with patch(
            "api.integrations.slack.db.get_decision", new=AsyncMock(return_value=decision)
        ) as mock_get, patch(
            "api.integrations.slack.db.log_decision", new=AsyncMock()
        ) as mock_log:
            text = await handle_action(
                payload, app, execution_engine, _fake_session_factory()
            )

        self.assertEqual(text, "Paper order placed: paper-123")
        execution_engine.execute_idea.assert_awaited_once_with(idea)
        app.client.chat_postMessage.assert_awaited_once_with(
            channel="C1", text="Paper order placed: paper-123"
        )

        mock_get.assert_awaited_once()
        log_calls = [c.kwargs for c in mock_log.call_args_list]
        self.assertEqual(log_calls[0]["action"], "approve")
        self.assertEqual(log_calls[1]["action"], "place_order")
        self.assertEqual(log_calls[1]["order_result"]["order_id"], "paper-123")

    async def test_handle_action_reject_logs_and_does_not_execute(self) -> None:
        idea = {"instrument": "MGC", "direction": "long"}
        execution_engine = MagicMock(spec=PaperExecutionEngine)
        execution_engine.execute_idea = AsyncMock()

        app = MagicMock()
        app.client.chat_postEphemeral = AsyncMock()

        payload = {
            "actions": [{"action_id": "reject_idea", "value": "run-2"}],
            "user": {"id": "U2", "name": "risk"},
            "channel": {"id": "C1"},
        }

        decision = MagicMock(payload=idea)

        with patch(
            "api.integrations.slack.db.get_decision", new=AsyncMock(return_value=decision)
        ), patch("api.integrations.slack.db.log_decision", new=AsyncMock()) as mock_log:
            text = await handle_action(
                payload, app, execution_engine, _fake_session_factory()
            )

        self.assertEqual(text, "Rejected by <@U2>")
        execution_engine.execute_idea.assert_not_awaited()
        app.client.chat_postEphemeral.assert_awaited_once()

        log_calls = [c.kwargs for c in mock_log.call_args_list]
        self.assertEqual(log_calls[0]["action"], "reject")

    async def test_handle_action_missing_run_id(self) -> None:
        app = MagicMock()
        execution_engine = MagicMock(spec=PaperExecutionEngine)
        payload = {"actions": [{"action_id": "approve_idea"}]}

        text = await handle_action(payload, app, execution_engine, _fake_session_factory())
        self.assertEqual(text, "Missing run_id.")

    def test_generate_approval_id_is_short(self) -> None:
        run_id = generate_approval_id()
        self.assertEqual(len(run_id), 12)


class TestSlackAppLifecycle(unittest.IsolatedAsyncioTestCase):
    def test_get_slack_app_returns_none_without_token(self) -> None:
        with patch.dict(
            os.environ, {"SLACK_BOT_TOKEN": "", "SLACK_SIGNING_SECRET": "shh"}
        ):
            self.assertIsNone(get_slack_app())

    def test_get_slack_app_returns_none_without_signing_secret(self) -> None:
        with patch.dict(
            os.environ, {"SLACK_BOT_TOKEN": "xoxb-x", "SLACK_SIGNING_SECRET": ""}
        ):
            self.assertIsNone(get_slack_app())

    async def test_request_handler_returns_503_without_app(self) -> None:
        request = MagicMock()
        with patch("api.integrations.slack._default_slack_app", None):
            response = await slack_request_handler(request, app=None)
        self.assertEqual(response.status_code, 503)

    async def test_request_handler_flattens_bolt_headers(self) -> None:
        app = MagicMock()
        bolt_resp = MagicMock()
        bolt_resp.body = "ok"
        bolt_resp.status = 200
        bolt_resp.headers = {"content-type": ["text/plain"], "x-slack": "1"}
        app.async_dispatch = AsyncMock(return_value=bolt_resp)

        request = MagicMock()
        request.body = AsyncMock(return_value=b"{}")
        request.query_params = ""
        request.headers = {}

        response = await slack_request_handler(request, app=app)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "text/plain")
        self.assertEqual(response.headers["x-slack"], "1")
