from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, MagicMock

from api.db import DecisionLog, get_decision, log_decision


class TestDecisionLog(unittest.IsolatedAsyncioTestCase):
    async def test_log_decision_adds_commits_and_refreshes(self) -> None:
        session = MagicMock()
        session.commit = AsyncMock()
        session.refresh = AsyncMock()

        entry = await log_decision(
            session,
            run_id="run-123",
            source="slack",
            action="approve",
            instrument="MGC",
            underlying="GOLD",
            direction="long",
            payload={"instrument": "MGC", "direction": "long"},
            risk_result={"ok": True},
            order_result={"order_id": "paper-xyz"},
            rationale="test",
        )

        self.assertIsInstance(entry, DecisionLog)
        self.assertEqual(entry.run_id, "run-123")
        self.assertEqual(entry.source, "slack")
        self.assertEqual(entry.action, "approve")
        self.assertEqual(entry.instrument, "MGC")
        self.assertEqual(entry.underlying, "GOLD")
        self.assertEqual(entry.direction, "long")
        self.assertEqual(entry.payload, {"instrument": "MGC", "direction": "long"})
        self.assertEqual(entry.risk_result, {"ok": True})
        self.assertEqual(entry.order_result, {"order_id": "paper-xyz"})
        self.assertEqual(entry.rationale, "test")

        session.add.assert_called_once_with(entry)
        session.commit.assert_awaited_once()
        session.refresh.assert_awaited_once_with(entry)

    async def test_get_decision_queries_by_run_id(self) -> None:
        expected = DecisionLog(
            run_id="run-456",
            source="slack",
            action="slack_post",
            instrument="XAUUSDT.P",
        )
        mock_result = MagicMock()
        mock_result.scalars.return_value.first.return_value = expected
        session = MagicMock()
        session.execute = AsyncMock(return_value=mock_result)

        decision = await get_decision(session, "run-456")

        self.assertIs(decision, expected)
        session.execute.assert_awaited_once()
        call_args = session.execute.call_args[0][0]
        # The compiled query should reference the run_id and decision_logs table.
        compiled = str(call_args.compile(compile_kwargs={"literal_binds": True}))
        self.assertIn("run-456", compiled)
        self.assertIn("decision_logs", compiled)
