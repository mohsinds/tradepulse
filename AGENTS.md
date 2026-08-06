# AGENTS.md

## Architecture rules

- **Async-only backend.** All I/O, market data, broker, and execution calls must be `async`/`await`.
- **All instrument-specific logic lives in venue adapters.** Route every instrument through `api/adapters/ibkr_venue.py` or `api/adapters/ccxt_venue.py`. Do not add MGC, XAUUSD, XAUUSDT.P, or any other instrument-specific code outside `api/adapters/`.
- **Adapters are organized by venue, not asset.** Each venue adapter supports multiple `ContractType`s.

## LLM access

- **Always use `llm_providers.get_chat_model()`.** Never instantiate an OpenAI, Anthropic, OpenRouter, or other provider client directly in application code.

## Execution and risk

- **Execution is advisory-only until explicitly enabled in config.** Any advisory or live-trading behavior must be off by default and toggled through configuration.
- **Risk checks in `api/execution/` are hard backend checks.** Use math, limits, and pre-trade validation. Never delegate risk decisions to the LLM.

## Signals and ranking

- Signal calculators in `api/signals/` are **sync** pure math over already-fetched bars; every `generate()` returns `{name, score in [-1,1] (positive = bullish), direction, weight, metrics, ok}`.
- Non-directional signals must not move the directional aggregate. `api/agents/ranker.py` excludes `NON_DIRECTIONAL_SIGNALS` (`volatility`, `funding_rate`) from the weighted long/short score; funding is applied once, as a deterministic penalty.
- The ranker's LLM node writes prose only. Scores, funding adjustments, and ranks are frozen before the LLM is called.

## Verification

```bash
python -m unittest discover -s tests   # 120 tests, fully offline (no network, no API keys)
```

## Scope

- This is a personal tool, not for redistribution. Do not add packaging, distribution, licensing, or publishing workflows unless explicitly asked.
