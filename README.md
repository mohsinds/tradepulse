# TradePulse

Personal multi-venue trading research and signal platform. TradePulse pulls market data
from Interactive Brokers and CCXT exchanges, runs a set of deterministic signal
calculators over the bars, ranks the resulting trade ideas, and exposes everything
through a FastAPI backend with a Next.js frontend.

Trade ideas are **advisory-only**: nothing is routed to a broker unless live execution
is explicitly enabled in configuration, and all risk math is enforced in the backend
rather than by the LLM.

## Contents

- [Architecture](#architecture)
- [Repository layout](#repository-layout)
- [Getting started](#getting-started)
- [Configuration](#configuration)
- [Signals](#signals)
- [Ranking agent](#ranking-agent)
- [Risk](#risk)
- [Research and parameter sweeps](#research-and-parameter-sweeps)
- [API endpoints](#api-endpoints)
- [Testing](#testing)
- [Development rules](#development-rules)

## Architecture

```
                 ┌────────────────────┐
                 │  web (Next.js 15)  │
                 └─────────┬──────────┘
                           │ NEXT_PUBLIC_API_URL
                 ┌─────────▼──────────┐
                 │  api (FastAPI)     │
                 │                    │
   venues  ──────┤  adapters/         │  IBKR (ib-async), CCXT
   config  ──────┤  config/           │  YAML instruments/underlyings/signals
   math    ──────┤  signals/          │  sync pure-math calculators
   agent   ──────┤  agents/ranker.py  │  LangGraph scoring + LLM prose
   risk    ──────┤  execution/        │  hard pre-trade checks
                 └─────────┬──────────┘
                           │
              ┌────────────┴────────────┐
              │ TimescaleDB      Redis  │
              └─────────────────────────┘

   research/  ── offline vectorbt backtests and walk-forward sweeps that
                 write optimized parameters back into api/config/signals/
```

Key ideas:

- **Async-only backend.** Every market data, broker, and execution call is `async`.
- **Venue adapters, not asset adapters.** Instrument-specific behaviour lives entirely
  in `api/adapters/`; each adapter serves multiple `ContractType`s (spot, future,
  perpetual, option, CFD).
- **Deterministic scoring.** Signal scores, funding adjustments, and ranks are frozen
  before the LLM is called. The LLM only writes the rationale prose.

## Repository layout

| Path | Purpose |
| --- | --- |
| `api/main.py` | FastAPI app and HTTP routes |
| `api/adapters/` | `IBKRVenue`, `CCXTVenue`, and the `DataProvider` / `BrokerAdapter` interfaces plus `Instrument`, `Bar`, `OrderRequest` dataclasses |
| `api/config/` | YAML config for instruments, underlyings, and signals, plus the merging `load_all()` loader |
| `api/signals/` | Sync pure-math signal calculators over already-fetched bars |
| `api/agents/ranker.py` | LangGraph ranking agent (score → funding adjust → rank → explain) |
| `api/execution/` | Execution engine interface and `UnderlyingRiskGuard` pre-trade checks |
| `api/webhooks/` | `WebhookHandler` interface for inbound event handling |
| `api/llm_providers.py` | The only place LLM clients are constructed; LangSmith tracing setup |
| `research/` | vectorbt backtesting, cost overlays, metrics, and the walk-forward parameter sweep |
| `web/` | Next.js 15 + Tailwind frontend |
| `tests/` | Fully offline unit tests |

## Getting started

### Docker Compose (recommended)

```bash
cp .env.example .env      # fill in the keys you need
docker compose up --build
```

This starts four services:

| Service | Port | Notes |
| --- | --- | --- |
| `db` | 5432 | TimescaleDB (Postgres 15) |
| `redis` | 6379 | Cache |
| `api` | 8000 | uvicorn with `--reload`, `./api` bind-mounted |
| `web` | 3000 | `next dev`, `./web` bind-mounted |

Open http://localhost:3000 for the UI and http://localhost:8000/docs for the API docs.

### Local (without Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r api/requirements.txt
uvicorn api.main:app --reload
```

```bash
cd web && npm install && npm run dev
```

For research work install the extra dependencies:

```bash
pip install -r research/requirements.txt
```

## Configuration

All runtime secrets and connection details come from environment variables — see
`.env.example` for the full list (LLM provider keys, LangSmith tracing, Postgres,
Redis, IBKR connection, CCXT credentials, and `NEXT_PUBLIC_API_URL`).

Trading configuration lives in YAML under `api/config/` and is assembled by
`api.config.load_all()`:

```
api/config/
├── instruments/     MGC.yaml, XAUUSD.yaml, XAUUSDT_P.yaml, BTCUSD.yaml
├── underlyings/     GOLD.yaml, BTC.yaml
└── signals/
    ├── technical.yaml      global defaults for trend/momentum/meanrev/...
    ├── funding.yaml        global funding_rate defaults
    └── GOLD_shared.yaml    per-underlying overrides written by the sweep
```

An instrument declares its venue, contract type, underlying, LLM model, and the list of
shared signal blocks it uses:

```yaml
MGC:
  venue: ibkr
  contract_type: future
  underlying: GOLD
  exchange: COMEX
  currency: USD
  multiplier: 10
  roll_days: 5
  llm:
    provider: anthropic
    model: claude-3-5-sonnet-latest
  shared_signals: [trend, momentum, meanrev, breakout, volatility, volume]
  signals: {}
```

Signal parameters are merged in increasing order of precedence:

1. global blocks from `signals/*.yaml` listed in `shared_signals`
2. per-underlying overrides from `signals/<UNDERLYING>_shared.yaml`
3. instrument-specific overrides from the instrument's own `signals:` block

Underlying files carry portfolio-level metadata, including the risk cap:

```yaml
GOLD:
  name: Gold
  currency: USD
  tick_size: 0.1
  max_combined_position_pct: 0.05
```

## Signals

Signal calculators are **sync pure math** over bars that have already been fetched.
Every `generate()` returns the same shape:

```python
{
  "name": "trend",
  "score": 0.42,          # [-1, 1], positive = bullish
  "direction": "long",    # long | short | flat
  "weight": 0.25,
  "metrics": {...},
  "ok": True,
}
```

Registered signals (`api/signals/__init__.py`):

| Signal | Module | Directional |
| --- | --- | --- |
| `trend` | `trend.py` | yes |
| `momentum` | `momentum.py` | yes |
| `meanrev` | `meanrev.py` | yes |
| `breakout` | `breakout.py` | yes |
| `volume` | `volume.py` | yes |
| `volatility` | `volatility.py` | no — regime measure |
| `funding_rate` | `funding_rate.py` | no — applied as a cost penalty |

`build_signals(merged_config)` instantiates every signal present in an instrument's
merged config, ignoring unknown block names.

## Ranking agent

`api/agents/ranker.py` implements a LangGraph pipeline:

1. **`score_signals`** — weighted aggregate of the directional signal scores.
   `NON_DIRECTIONAL_SIGNALS` (`volatility`, `funding_rate`) are excluded so a regime
   measure never flips a long/short call.
2. **`apply_funding_adjustment`** — for perpetuals, funding is applied exactly once as a
   deterministic penalty when the funding cost works against the proposed side.
3. **`rank`** — orders the resulting ideas.
4. **`explain`** — asks the per-instrument LLM (via `llm_providers.get_chat_model()`)
   for a short rationale. Scores and ranks are already frozen at this point; the LLM
   writes prose only, and a deterministic fallback rationale is used if the call fails.

## Risk

`UnderlyingRiskGuard` (`api/execution/risk.py`) enforces a combined notional cap across
every instrument sharing an underlying — e.g. MGC, XAUUSD, and XAUUSDT.P all count
against `GOLD.max_combined_position_pct`. This is a hard backend check that runs before
an order is submitted *and* before an advisory recommendation is logged, so the cap
applies whether or not live execution is enabled. It returns a `RiskResult` with the
current, proposed, and combined notional percentages plus the flagged symbols.

## Research and parameter sweeps

The `research/` package is offline and independent of `api.signals`:

- `data.py` — loads bars through the venue adapters, or generates synthetic bars
- `strategy.py` — pandas/numpy entry–exit generators used by the sweep
- `costs.py` — contract-aware cost overlays (`FuturesRollCost`, `CFDSwapCost`,
  `PerpetualFundingCost`, `NoCost`) selected by `cost_overlay_for(instrument)`
- `metrics.py` — Sharpe, max drawdown, win rate, and `score_portfolio()`
- `param_sweep.py` — walk-forward grid search per underlying, writing the winning
  parameters to `api/config/signals/<UNDERLYING>_shared.yaml`
- `scripts/backtest.py` — a standalone vectorbt sample backtest

Run a sweep:

```bash
# offline smoke run on synthetic data
python -m research.scripts.run_sweep --dry-run --underlying GOLD

# real data through the venue adapters
python -m research.scripts.run_sweep \
  --start 2024-01-01 --end 2024-06-01 \
  --timeframe 1h --underlying GOLD \
  --adapters-config adapters.yaml \
  --train-frac 0.7 --output-dir api/config/signals
```

The script prints a JSON report and rewrites the per-underlying shared signal config.

## API endpoints

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Liveness probe |
| `GET` | `/config` | Full merged configuration from `load_all()` |
| `GET` | `/instruments` | List of configured instrument symbols |

Interactive docs are served at `/docs`.

## Testing

```bash
python -m unittest discover -s tests
```

The suite is fully offline — no network access and no API keys required.

## Development rules

`AGENTS.md` is the authoritative contributor guide. In short:

- Backend I/O is async-only.
- No instrument-specific logic outside `api/adapters/`; adapters are organized by venue.
- Always obtain LLM clients through `llm_providers.get_chat_model()`.
- Execution stays advisory-only until explicitly enabled in config.
- Risk checks are backend math, never delegated to the LLM.
- This is a personal tool — no packaging, distribution, or publishing workflows.
