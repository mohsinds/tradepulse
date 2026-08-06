from __future__ import annotations

import asyncio
import unittest
from unittest import mock

from api.agents import Agent
from api.agents.ranker import RankerAgent


def sig(name, score, weight=1.0, direction=None, **metrics):
    if direction is None:
        direction = "long" if score > 0 else "short" if score < 0 else "flat"
    return {
        "name": name,
        "score": score,
        "direction": direction,
        "weight": weight,
        "metrics": metrics,
    }


PERP_CFG = {
    "venue": "ccxt",
    "contract_type": "perpetual",
    "exchange": "bybit",
    "llm": {"provider": "openrouter", "model": "anthropic/claude-3.5-sonnet"},
    "signals": {
        "trend": {"weight": 0.25},
        "momentum": {"weight": 0.2},
        "funding_rate": {
            "weight": 0.2,
            "neutral_band": 0.0001,
            "extreme": 0.0005,
            "unfavorable_penalty": 0.5,
        },
    },
}

SPOT_CFG = {
    "venue": "ccxt",
    "contract_type": "spot",
    "llm": {"provider": "openai", "model": "gpt-4o-mini"},
    "signals": {"trend": {"weight": 0.25}, "momentum": {"weight": 0.2}},
}


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeModel:
    """Minimal stand-in for a LangChain chat model."""

    def __init__(self, content="LLM rationale.", fail=False):
        self.content = content
        self.fail = fail
        self.calls = []

    async def ainvoke(self, prompt, *args, **kwargs):
        self.calls.append(prompt)
        if self.fail:
            raise RuntimeError("llm exploded")
        return FakeResponse(self.content)


class RankerTestBase(unittest.TestCase):
    def run_agent(self, context, model=None, fail=False, agent=None):
        model = model or FakeModel(fail=fail)
        agent = agent or RankerAgent()
        with mock.patch(
            "api.agents.ranker.get_chat_model_for_config", return_value=model
        ) as m_cfg, mock.patch(
            "api.agents.ranker.get_chat_model", return_value=model
        ) as m_def:
            result = asyncio.run(agent.run(context))
        self.model = model
        self.m_cfg = m_cfg
        self.m_default = m_def
        return result

    @staticmethod
    def by_symbol(result):
        return {i["instrument"]: i for i in result["ideas"]}


class TestDeterministicScoring(RankerTestBase):
    def test_weighted_aggregation(self):
        ctx = {
            "instruments": {"SPOTX": SPOT_CFG},
            "signal_results": {
                "SPOTX": [sig("trend", 0.8), sig("momentum", -0.2)],
            },
        }
        result = self.run_agent(ctx)
        idea = self.by_symbol(result)["SPOTX"]
        expected = (0.25 * 0.8 + 0.2 * -0.2) / (0.25 + 0.2)
        self.assertAlmostEqual(idea["raw_score"], expected)
        self.assertAlmostEqual(idea["adjusted_score"], expected)
        self.assertEqual(idea["direction"], "long")
        self.assertAlmostEqual(idea["contributions"]["trend"], 0.25 * 0.8)
        self.assertAlmostEqual(idea["contributions"]["momentum"], 0.2 * -0.2)
        self.assertTrue(result["advisory_only"])
        self.assertFalse(result["execution_enabled"])

    def test_config_weights_override_signal_weights(self):
        ctx = {
            "instruments": {"SPOTX": SPOT_CFG},
            # deliberately bogus weights embedded in the signal results
            "signal_results": {
                "SPOTX": [
                    sig("trend", 0.8, weight=99.0),
                    sig("momentum", -0.2, weight=99.0),
                ]
            },
        }
        idea = self.by_symbol(self.run_agent(ctx))["SPOTX"]
        expected = (0.25 * 0.8 + 0.2 * -0.2) / 0.45
        self.assertAlmostEqual(idea["raw_score"], expected)

    def test_signal_weight_used_when_config_has_none(self):
        cfg = {"contract_type": "spot", "signals": {}}
        ctx = {
            "instruments": {"S": cfg},
            "signal_results": {"S": [sig("trend", 0.5, weight=2.0), sig("x", -0.5, 1.0)]},
        }
        idea = self.by_symbol(self.run_agent(ctx))["S"]
        self.assertAlmostEqual(idea["raw_score"], (2.0 * 0.5 - 0.5) / 3.0)

    def test_no_signals_scores_zero_flat(self):
        ctx = {"instruments": {"S": SPOT_CFG}, "signal_results": {"S": []}}
        idea = self.by_symbol(self.run_agent(ctx))["S"]
        self.assertEqual(idea["raw_score"], 0.0)
        self.assertEqual(idea["direction"], "flat")


class TestFundingAdjustment(RankerTestBase):
    def _perp_ctx(self, funding_rate, trend=0.8):
        return {
            "instruments": {"PERP": PERP_CFG},
            "signal_results": {
                "PERP": [
                    sig("trend", trend),
                    sig("momentum", trend),
                    sig(
                        "funding_rate",
                        0.0,
                        funding_rate=funding_rate,
                        side_paying="long" if funding_rate > 0 else "short",
                    ),
                ]
            },
        }

    def test_long_penalised_when_funding_positive(self):
        # extreme funding => full 0.5 penalty
        idea = self.by_symbol(self.run_agent(self._perp_ctx(0.0005)))["PERP"]
        self.assertEqual(idea["direction"], "long")
        self.assertLess(idea["adjusted_score"], idea["raw_score"])
        self.assertAlmostEqual(idea["adjusted_score"], idea["raw_score"] * 0.5)
        self.assertTrue(idea["funding"]["unfavorable"])
        self.assertEqual(idea["funding"]["pays"], "long")

    def test_penalty_scales_with_severity(self):
        # half of extreme => factor 1 - 0.5*0.5 = 0.75
        idea = self.by_symbol(self.run_agent(self._perp_ctx(0.00025)))["PERP"]
        self.assertAlmostEqual(idea["adjusted_score"], idea["raw_score"] * 0.75)
        self.assertAlmostEqual(idea["funding"]["severity"], 0.5)

    def test_no_penalty_when_funding_favorable(self):
        # long idea, negative funding => shorts pay, longs collect
        idea = self.by_symbol(self.run_agent(self._perp_ctx(-0.0005)))["PERP"]
        self.assertEqual(idea["direction"], "long")
        self.assertAlmostEqual(idea["adjusted_score"], idea["raw_score"])
        self.assertFalse(idea["funding"]["unfavorable"])
        self.assertEqual(idea["funding"]["factor"], 1.0)

    def test_short_penalised_when_funding_negative(self):
        idea = self.by_symbol(self.run_agent(self._perp_ctx(-0.0005, trend=-0.8)))["PERP"]
        self.assertEqual(idea["direction"], "short")
        self.assertAlmostEqual(idea["adjusted_score"], idea["raw_score"] * 0.5)
        self.assertTrue(idea["funding"]["unfavorable"])

    def test_neutral_band_suppresses_penalty(self):
        idea = self.by_symbol(self.run_agent(self._perp_ctx(0.00005)))["PERP"]
        self.assertAlmostEqual(idea["adjusted_score"], idea["raw_score"])
        self.assertFalse(idea["funding"]["unfavorable"])

    def test_spot_untouched_by_funding_node(self):
        ctx = {
            "instruments": {"SPOTX": SPOT_CFG},
            "signal_results": {
                "SPOTX": [
                    sig("trend", 0.8),
                    sig("momentum", 0.8),
                    # even if a funding signal leaks in, spot must not be adjusted
                    sig("funding_rate", 0.0, funding_rate=0.001, side_paying="long"),
                ]
            },
        }
        idea = self.by_symbol(self.run_agent(ctx))["SPOTX"]
        self.assertIsNone(idea["funding"])
        self.assertAlmostEqual(idea["adjusted_score"], idea["raw_score"])

    def test_perpetual_without_funding_signal_is_unadjusted(self):
        ctx = {
            "instruments": {"PERP": PERP_CFG},
            "signal_results": {"PERP": [sig("trend", 0.6)]},
        }
        idea = self.by_symbol(self.run_agent(ctx))["PERP"]
        self.assertIsNone(idea["funding"])
        self.assertAlmostEqual(idea["adjusted_score"], idea["raw_score"])


class TestRanking(RankerTestBase):
    def test_ranked_by_absolute_adjusted_score(self):
        ctx = {
            "instruments": {"A": SPOT_CFG, "B": SPOT_CFG, "C": SPOT_CFG},
            "signal_results": {
                "A": [sig("trend", 0.2)],
                "B": [sig("trend", -0.9)],
                "C": [sig("trend", 0.5)],
            },
        }
        result = self.run_agent(ctx)
        self.assertEqual([i["instrument"] for i in result["ideas"]], ["B", "C", "A"])
        self.assertEqual([i["rank"] for i in result["ideas"]], [1, 2, 3])
        self.assertEqual(result["ideas"][0]["direction"], "short")

    def test_funding_penalty_can_reorder_ranking(self):
        ctx = {
            "instruments": {"PERP": PERP_CFG, "SPOTX": SPOT_CFG},
            "signal_results": {
                "PERP": [
                    sig("trend", 0.9),
                    sig("momentum", 0.9),
                    sig("funding_rate", 0.0, funding_rate=0.001, side_paying="long"),
                ],
                "SPOTX": [sig("trend", 0.6), sig("momentum", 0.6)],
            },
        }
        result = self.run_agent(ctx)
        ideas = self.by_symbol(result)
        self.assertGreater(ideas["PERP"]["raw_score"], ideas["SPOTX"]["raw_score"])
        self.assertLess(ideas["PERP"]["adjusted_score"], ideas["SPOTX"]["adjusted_score"])
        self.assertEqual(result["ideas"][0]["instrument"], "SPOTX")


class TestExplainNode(RankerTestBase):
    def test_llm_rationale_used(self):
        ctx = {
            "instruments": {"SPOTX": SPOT_CFG},
            "signal_results": {"SPOTX": [sig("trend", 0.8)]},
        }
        model = FakeModel(content="Trend is strongly bullish.")
        result = self.run_agent(ctx, model=model)
        self.assertEqual(result["ideas"][0]["rationale"], "Trend is strongly bullish.")
        self.assertEqual(len(model.calls), 1)
        self.m_cfg.assert_called_once()
        self.assertEqual(self.m_cfg.call_args[0][0], SPOT_CFG)

    def test_llm_failure_falls_back_to_template(self):
        ctx = {
            "instruments": {"SPOTX": SPOT_CFG},
            "signal_results": {"SPOTX": [sig("trend", 0.8)]},
        }
        result = self.run_agent(ctx, fail=True)
        rationale = result["ideas"][0]["rationale"]
        self.assertIn("SPOTX", rationale)
        self.assertIn("Advisory only", rationale)
        self.assertIn("trend", rationale)

    def test_model_resolution_falls_back_to_default(self):
        ctx = {
            "instruments": {"SPOTX": SPOT_CFG},
            "signal_results": {"SPOTX": [sig("trend", 0.8)]},
        }
        model = FakeModel(content="Default model rationale.")
        agent = RankerAgent()
        with mock.patch(
            "api.agents.ranker.get_chat_model_for_config",
            side_effect=RuntimeError("bad provider"),
        ), mock.patch("api.agents.ranker.get_chat_model", return_value=model) as m_def:
            result = asyncio.run(agent.run(ctx))
        m_def.assert_called_once()
        self.assertEqual(result["ideas"][0]["rationale"], "Default model rationale.")

    def test_llm_cannot_change_numeric_scores(self):
        ctx = {
            "instruments": {"PERP": PERP_CFG, "SPOTX": SPOT_CFG},
            "signal_results": {
                "PERP": [
                    sig("trend", 0.9),
                    sig("funding_rate", 0.0, funding_rate=0.001, side_paying="long"),
                ],
                "SPOTX": [sig("trend", 0.4)],
            },
        }
        honest = self.run_agent(ctx, model=FakeModel(content="ok"))
        adversarial = self.run_agent(
            ctx,
            model=FakeModel(
                content=(
                    '{"adjusted_score": 99.0, "raw_score": -99.0, "rank": 1, '
                    '"direction": "short", "instrument": "HACKED"}'
                )
            ),
        )
        numeric = lambda res: [  # noqa: E731
            (i["instrument"], i["direction"], i["raw_score"], i["adjusted_score"], i["rank"])
            for i in res["ideas"]
        ]
        self.assertEqual(numeric(honest), numeric(adversarial))
        self.assertTrue(all(abs(i["adjusted_score"]) <= 1.0 for i in adversarial["ideas"]))


class TestAgentContract(RankerTestBase):
    def test_subclasses_agent_and_has_graph(self):
        agent = RankerAgent()
        self.assertIsInstance(agent, Agent)
        self.assertEqual(agent.name, "ranker")
        graph = agent.graph
        self.assertIs(graph, agent.graph)  # cached
        nodes = set(graph.get_graph().nodes)
        for node in ("score_signals", "apply_funding_adjustment", "rank", "explain"):
            self.assertIn(node, nodes)

    def test_tracing_configured_and_graph_tagged(self):
        ctx = {
            "instruments": {"SPOTX": SPOT_CFG},
            "signal_results": {"SPOTX": [sig("trend", 0.8)]},
            "tags": ["unit-test"],
        }
        agent = RankerAgent()
        real_ainvoke = agent.graph.ainvoke
        captured = {}

        async def spy(state, config=None, **kwargs):
            captured["config"] = config
            return await real_ainvoke(state, config=config, **kwargs)

        with mock.patch("api.agents.ranker.configure_tracing") as tracing, mock.patch.object(
            agent.graph, "ainvoke", side_effect=spy
        ):
            self.run_agent(ctx, agent=agent)

        tracing.assert_called_once()
        cfg = captured["config"]
        self.assertEqual(cfg["run_name"], "tradepulse.ranker")
        self.assertTrue(cfg["metadata"]["advisory_only"])
        self.assertIn("ranker", cfg["tags"])
        self.assertIn("unit-test", cfg["tags"])

    def test_accepts_signals_key_alias(self):
        ctx = {
            "instruments": {"SPOTX": SPOT_CFG},
            "signals": {"SPOTX": [sig("trend", 0.8)]},
        }
        result = self.run_agent(ctx)
        self.assertAlmostEqual(result["ideas"][0]["raw_score"], 0.8)


if __name__ == "__main__":
    unittest.main()
