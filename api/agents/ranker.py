"""LangGraph ranking agent.

Produces *advisory-only* trade ideas by combining already-computed signal
results into a deterministic weighted score, applying deterministic funding
adjustments for perpetuals, ranking, and finally asking an LLM for a short
prose rationale.

Per AGENTS.md:
  * all scoring / risk math is hard backend math, never delegated to the LLM
  * the LLM is only reached through ``api.llm_providers``
  * output is advisory-only
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, TypedDict

from ..adapters.base import ContractType, Instrument
from ..llm_providers import configure_tracing, get_chat_model, get_chat_model_for_config
from .base import Agent

logger = logging.getLogger(__name__)

FUNDING_SIGNAL = "funding_rate"
DEFAULT_UNFAVORABLE_PENALTY = 0.5
DEFAULT_FUNDING_EXTREME = 0.0005
DEFAULT_FUNDING_NEUTRAL_BAND = 0.0

#: Signals excluded from the *directional* weighted aggregate.
#:
#: ``volatility`` is a regime measure, not a directional call (it always
#: reports ``direction == "flat"``), so folding its score into a long/short
#: aggregate would flip directions for no economic reason.
#:
#: ``funding_rate`` is excluded because it is already applied as a
#: deterministic penalty in :meth:`RankerAgent.apply_funding_adjustment`;
#: scoring it here as well would double-count funding.
NON_DIRECTIONAL_SIGNALS = frozenset({"volatility", FUNDING_SIGNAL})


class RankerState(TypedDict, total=False):
    """State threaded through the ranking graph."""

    instruments: Dict[str, Dict[str, Any]]
    signal_results: Dict[str, List[Dict[str, Any]]]
    scores: Dict[str, Dict[str, Any]]
    funding: Dict[str, Optional[Dict[str, Any]]]
    ideas: List[Dict[str, Any]]
    rationales: Dict[str, str]
    errors: List[str]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _contract_type(cfg: Dict[str, Any]) -> Optional[ContractType]:
    raw = cfg.get("contract_type")
    if isinstance(raw, ContractType):
        return raw
    if isinstance(raw, str):
        try:
            return ContractType(raw.lower())
        except ValueError:
            return None
    instrument = cfg.get("instrument")
    if isinstance(instrument, Instrument):
        return instrument.contract_type
    return None


def _config_weight(cfg: Dict[str, Any], signal_name: str) -> Optional[float]:
    """Weight for ``signal_name`` from the merged instrument config, if any."""
    blocks = cfg.get("signals") or {}
    block = blocks.get(signal_name)
    if isinstance(block, dict) and block.get("weight") is not None:
        return _as_float(block.get("weight"))
    return None


def _funding_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    blocks = cfg.get("signals") or {}
    block = blocks.get(FUNDING_SIGNAL)
    return block if isinstance(block, dict) else {}


def _direction_from_score(score: float) -> str:
    if score > 0:
        return "long"
    if score < 0:
        return "short"
    return "flat"


def _extract_funding_rate(result: Dict[str, Any]) -> Optional[float]:
    metrics = result.get("metrics") or {}
    for key in ("funding_rate", "rate", "current_funding_rate", "funding"):
        if key in metrics and metrics[key] is not None:
            return _as_float(metrics[key])
    return None


def _paying_side(rate: float, metrics: Dict[str, Any]) -> str:
    side = metrics.get("side_paying") or metrics.get("pays") or metrics.get("payer")
    if isinstance(side, str) and side.lower() in ("long", "longs", "short", "shorts"):
        return "long" if side.lower().startswith("long") else "short"
    # positive funding => longs pay shorts
    return "long" if rate > 0 else "short"


# --------------------------------------------------------------------------
# agent
# --------------------------------------------------------------------------
class RankerAgent(Agent):
    """Rank pre-computed signal results into advisory trade ideas."""

    name = "ranker"

    def __init__(self, max_rationale_sentences: int = 3) -> None:
        self.max_rationale_sentences = max_rationale_sentences
        self._graph = None

    # -- graph construction -------------------------------------------------
    def build_graph(self):
        """Compile the LangGraph state graph (langgraph imported lazily)."""
        from langgraph.graph import END, START, StateGraph

        builder = StateGraph(RankerState)
        builder.add_node("score_signals", self.score_signals)
        builder.add_node("apply_funding_adjustment", self.apply_funding_adjustment)
        builder.add_node("rank", self.rank)
        builder.add_node("explain", self.explain)

        builder.add_edge(START, "score_signals")
        builder.add_edge("score_signals", "apply_funding_adjustment")
        builder.add_edge("apply_funding_adjustment", "rank")
        builder.add_edge("rank", "explain")
        builder.add_edge("explain", END)
        return builder.compile()

    @property
    def graph(self):
        if self._graph is None:
            self._graph = self.build_graph()
        return self._graph

    # -- node 1: deterministic weighted aggregation -------------------------
    async def score_signals(self, state: RankerState) -> Dict[str, Any]:
        instruments = state.get("instruments") or {}
        signal_results = state.get("signal_results") or {}
        scores: Dict[str, Dict[str, Any]] = {}

        for symbol, cfg in instruments.items():
            results = signal_results.get(symbol) or []
            contributions: Dict[str, float] = {}
            weights: Dict[str, float] = {}
            context_scores: Dict[str, float] = {}
            weighted_sum = 0.0
            weight_total = 0.0

            for result in results:
                sig_name = result.get("name")
                if not sig_name:
                    continue
                score = max(-1.0, min(1.0, _as_float(result.get("score"))))
                # config weights win over any weight embedded in the result
                weight = _config_weight(cfg, sig_name)
                if weight is None:
                    weight = _as_float(result.get("weight"))
                weights[sig_name] = weight

                # Non-directional signals inform context but must not move the
                # long/short aggregate. Funding is applied separately as a
                # penalty, volatility is a regime measure.
                if sig_name in NON_DIRECTIONAL_SIGNALS:
                    contributions[sig_name] = 0.0
                    context_scores[sig_name] = score
                    continue

                contributions[sig_name] = weight * score
                weighted_sum += weight * score
                weight_total += abs(weight)

            raw_score = weighted_sum / weight_total if weight_total else 0.0
            scores[symbol] = {
                "instrument": symbol,
                "raw_score": raw_score,
                "direction": _direction_from_score(raw_score),
                "contributions": contributions,
                "context_scores": context_scores,
                "weights": weights,
                "weight_total": weight_total,
            }

        return {"scores": scores}

    # -- node 2: deterministic funding adjustment ---------------------------
    async def apply_funding_adjustment(self, state: RankerState) -> Dict[str, Any]:
        instruments = state.get("instruments") or {}
        signal_results = state.get("signal_results") or {}
        scores = dict(state.get("scores") or {})
        funding_info: Dict[str, Optional[Dict[str, Any]]] = {}

        for symbol, entry in scores.items():
            cfg = instruments.get(symbol) or {}
            entry = dict(entry)
            entry.setdefault("adjusted_score", entry["raw_score"])
            entry["adjusted_score"] = entry["raw_score"]
            funding_info[symbol] = None

            if _contract_type(cfg) is not ContractType.PERPETUAL:
                scores[symbol] = entry
                continue

            funding_result = next(
                (
                    r
                    for r in (signal_results.get(symbol) or [])
                    if r.get("name") == FUNDING_SIGNAL
                ),
                None,
            )
            if funding_result is None:
                scores[symbol] = entry
                continue

            rate = _extract_funding_rate(funding_result)
            if rate is None:
                scores[symbol] = entry
                continue

            fcfg = _funding_config(cfg)
            penalty = _as_float(
                fcfg.get("unfavorable_penalty", DEFAULT_UNFAVORABLE_PENALTY),
                DEFAULT_UNFAVORABLE_PENALTY,
            )
            extreme = _as_float(
                fcfg.get("extreme", DEFAULT_FUNDING_EXTREME), DEFAULT_FUNDING_EXTREME
            )
            neutral_band = _as_float(
                fcfg.get("neutral_band", DEFAULT_FUNDING_NEUTRAL_BAND),
                DEFAULT_FUNDING_NEUTRAL_BAND,
            )

            metrics = funding_result.get("metrics") or {}
            pays = _paying_side(rate, metrics)
            direction = entry["direction"]

            unfavorable = direction in ("long", "short") and pays == direction
            if abs(rate) <= abs(neutral_band):
                unfavorable = False

            severity = 0.0
            factor = 1.0
            if unfavorable:
                severity = min(1.0, abs(rate) / extreme) if extreme else 1.0
                # scale penalty by how extreme funding is
                factor = 1.0 - (1.0 - penalty) * severity
                entry["adjusted_score"] = entry["raw_score"] * factor

            funding_info[symbol] = {
                "rate": rate,
                "pays": pays,
                "direction": direction,
                "unfavorable": unfavorable,
                "severity": severity,
                "penalty": penalty,
                "factor": factor,
                "extreme": extreme,
                "neutral_band": neutral_band,
            }
            scores[symbol] = entry

        return {"scores": scores, "funding": funding_info}

    # -- node 3: ranking ----------------------------------------------------
    async def rank(self, state: RankerState) -> Dict[str, Any]:
        scores = state.get("scores") or {}
        funding = state.get("funding") or {}

        ideas: List[Dict[str, Any]] = []
        for symbol, entry in scores.items():
            ideas.append(
                {
                    "instrument": symbol,
                    "direction": entry.get("direction", "flat"),
                    "raw_score": entry.get("raw_score", 0.0),
                    "adjusted_score": entry.get(
                        "adjusted_score", entry.get("raw_score", 0.0)
                    ),
                    "contributions": entry.get("contributions", {}),
                    "weights": entry.get("weights", {}),
                    "funding": funding.get(symbol),
                }
            )

        ideas.sort(key=lambda i: (-abs(i["adjusted_score"]), i["instrument"]))
        for position, idea in enumerate(ideas, start=1):
            idea["rank"] = position

        return {"ideas": ideas}

    # -- node 4: the ONLY LLM node -----------------------------------------
    def _fallback_rationale(self, idea: Dict[str, Any]) -> str:
        contributions = idea.get("contributions") or {}
        top = sorted(contributions.items(), key=lambda kv: -abs(kv[1]))[:3]
        drivers = ", ".join(f"{name} {value:+.3f}" for name, value in top) or "no signals"
        text = (
            f"Rank {idea.get('rank')}: {idea['instrument']} {idea['direction']} "
            f"with adjusted score {idea['adjusted_score']:+.3f} "
            f"(raw {idea['raw_score']:+.3f}). Top drivers: {drivers}."
        )
        funding = idea.get("funding")
        if funding and funding.get("unfavorable"):
            text += (
                f" Funding at {funding['rate']:.5f} is unfavorable for the "
                f"{idea['direction']} side, so the score was down-weighted."
            )
        return text + " Advisory only."

    def _build_prompt(self, idea: Dict[str, Any]) -> str:
        return (
            "You are writing a short explanatory note for an advisory-only "
            "trading idea. Do NOT change, question, or restate any numbers as "
            "different values, and do not give risk or sizing advice.\n"
            f"Instrument: {idea['instrument']}\n"
            f"Direction: {idea['direction']}\n"
            f"Rank: {idea['rank']}\n"
            f"Raw score: {idea['raw_score']:.4f}\n"
            f"Adjusted score: {idea['adjusted_score']:.4f}\n"
            f"Signal contributions: {idea.get('contributions')}\n"
            f"Funding: {idea.get('funding')}\n"
            f"Write 1-{self.max_rationale_sentences} sentences of plain prose."
        )

    async def _rationale_for(self, idea: Dict[str, Any], cfg: Dict[str, Any]) -> str:
        try:
            try:
                model = get_chat_model_for_config(cfg)
            except Exception:  # noqa: BLE001 - fall back to the default model
                model = get_chat_model()
            if model is None:
                raise RuntimeError("no chat model available")

            prompt = self._build_prompt(idea)
            if hasattr(model, "ainvoke"):
                response = await model.ainvoke(prompt)
            else:
                response = await asyncio.to_thread(model.invoke, prompt)

            content = getattr(response, "content", response)
            if isinstance(content, list):
                content = " ".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                )
            text = str(content).strip()
            if not text:
                raise ValueError("empty rationale from LLM")
            return text
        except Exception as exc:  # noqa: BLE001 - LLM must never break ranking
            logger.warning(
                "LLM rationale failed for %s, using deterministic fallback: %s",
                idea.get("instrument"),
                exc,
            )
            return self._fallback_rationale(idea)

    async def explain(self, state: RankerState) -> Dict[str, Any]:
        instruments = state.get("instruments") or {}
        ideas = state.get("ideas") or []

        rationales: Dict[str, str] = {}
        for idea in ideas:
            cfg = instruments.get(idea["instrument"]) or {}
            rationales[idea["instrument"]] = await self._rationale_for(idea, cfg)

        # LLM output is prose only: attach it without touching any numbers.
        annotated = [
            {**idea, "rationale": rationales.get(idea["instrument"], "")}
            for idea in ideas
        ]
        return {"ideas": annotated, "rationales": rationales}

    # -- entry point --------------------------------------------------------
    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        configure_tracing()

        instruments: Dict[str, Dict[str, Any]] = context.get("instruments") or {}
        signal_results: Dict[str, List[Dict[str, Any]]] = (
            context.get("signal_results") or context.get("signals") or {}
        )

        initial: RankerState = {
            "instruments": instruments,
            "signal_results": signal_results,
            "scores": {},
            "funding": {},
            "ideas": [],
            "rationales": {},
            "errors": [],
        }

        run_config = {
            "run_name": context.get("run_name", "tradepulse.ranker"),
            "metadata": {
                "agent": self.name,
                "instruments": sorted(instruments.keys()),
                "instrument_count": len(instruments),
                "advisory_only": True,
                **(context.get("metadata") or {}),
            },
            "tags": ["tradepulse", "ranker", "advisory-only"]
            + list(context.get("tags") or []),
        }

        final_state = await self.graph.ainvoke(initial, config=run_config)

        ideas = [
            {
                "instrument": idea["instrument"],
                "direction": idea["direction"],
                "raw_score": idea["raw_score"],
                "adjusted_score": idea["adjusted_score"],
                "rank": idea["rank"],
                "rationale": idea.get("rationale", ""),
                "contributions": idea.get("contributions", {}),
                "funding": idea.get("funding"),
            }
            for idea in (final_state.get("ideas") or [])
        ]

        return {
            "agent": self.name,
            "ideas": ideas,
            "advisory_only": True,
            "execution_enabled": False,
            "notice": (
                "Advisory only: these ranked ideas are not orders and no "
                "execution is performed."
            ),
            "instrument_count": len(instruments),
        }
