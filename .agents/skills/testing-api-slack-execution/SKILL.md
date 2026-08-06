---
name: testing-api-slack-execution
description: How to run TradePulse's FastAPI backend end-to-end for runtime testing — Timescale/Redis infra, offline venue adapters, a fake Slack Web API, correctly-signed Slack requests, and how to assert against decision_logs. Use when verifying webhooks, Slack approvals, the execution engine, the risk guard, or DecisionLog persistence.
---

# Runtime-testing the TradePulse API (webhooks, Slack, execution, risk)

Unit tests (`python -m unittest discover -s tests`) are fully offline and prove none of the
wiring. For end-to-end verification, boot the real `api.main:app` and drive it over HTTP,
mocking **only** the two external network boundaries: the venue adapters and the Slack Web API.

## 1. Infra

```bash
# docker-compose.yml reads a .env that is gitignored and may not exist -> create it first
cat > .env <<'EOF'
POSTGRES_USER=tradepulse
POSTGRES_PASSWORD=tradepulse
POSTGRES_DB=tradepulse
EOF
docker compose up -d db redis
pip install asyncpg psycopg2-binary   # asyncpg is NOT in api/requirements.txt but DATABASE_URL needs it
export DATABASE_URL=postgresql+asyncpg://tradepulse:tradepulse@localhost:5432/tradepulse
```

Prefer real TimescaleDB over sqlite: `init_db()` calls `create_hypertable('decision_logs','timestamp')`,
and failures there are swallowed with a warning ("expected if not TimescaleDB"), so sqlite/plain
Postgres will hide hypertable bugs. Always verify explicitly:

```sql
SELECT hypertable_name FROM timescaledb_information.hypertables;
SELECT column_name FROM timescaledb_information.dimensions WHERE hypertable_name='decision_logs';
```

If you change the `DecisionLog` model, `DROP TABLE decision_logs` before re-running `init_db()` —
`create_all` will not migrate an existing table and you will test a stale schema.

## 2. Offline venue adapters

Monkeypatch `api.main._build_venues` from a harness module that imports `api.main` and re-exports
`app`; run uvicorn against the harness. Have each fake adapter append every
`connect`/`get_positions`/`place_order` call to a log file — that log is the primary evidence for
"the risk guard ran before the venue was touched" and for "the webhook cannot reach an order path".

Two traps:
- `app.on_event("startup")` is **ignored** when the app defines a `lifespan`. To make positions
  visible from the start, construct the fakes with `connected=True` instead.
- Test risk rejections with `PAPER_TRADING=false`. In paper mode `place_order` is never called
  anyway, so its absence proves nothing. Always pair an over-cap case (expect no `place_order`)
  with an under-cap control (expect exactly one) — otherwise a broken adapter path looks like a
  working risk guard.

## 3. Slack without a real workspace

`get_slack_app()` returns `None` unless **both** `SLACK_BOT_TOKEN` and `SLACK_SIGNING_SECRET` are
set; with Slack unconfigured `POST /slack/events` returns 503 by design.

Sign requests exactly as Slack does — HMAC-SHA256 over `v0:{timestamp}:{body}` with your own
`SLACK_SIGNING_SECRET`, sent as `X-Slack-Signature: v0=<hex>` plus `X-Slack-Request-Timestamp`.
`block_actions` must be form-encoded as `payload=<json>`. A tampered signature should give
401 `{"error":"invalid request"}`.

A **dummy bot token is not enough** to reach the listeners: Bolt's authorization middleware calls
`auth.test` and answers 200 with "your installation with this app is no longer available", so the
listener silently never runs. Work around it by serving a local fake Slack Web API
(`auth.test`, `chat.postMessage`, `chat.postEphemeral` → `{"ok": true, ...}`) and repointing the SDK
in the harness before the app starts:

```python
import slack_sdk.web.async_base_client as abc
_orig = abc.AsyncBaseClient.__init__
def _init(self, *a, **kw):
    kw["base_url"] = "http://127.0.0.1:8099/api/"
    return _orig(self, *a, **kw)
abc.AsyncBaseClient.__init__ = _init
```

Logging every call the fake receives also gives you the outbound Block Kit payload, which is how you
assert the Approve/Reject buttons carry the right `action_id` and `run_id`.

## 4. Driving and asserting the flow

Approve/reject handlers look the idea up by `run_id` from the latest `slack_post` row, so you can
seed a `decision_logs` row with an explicit `payload` (including `quantity` and `price`) and then
click Approve on that `run_id` — no ranker run needed. Assert on the row sequence per `run_id`:
`slack_post → approve → place_order` or `slack_post → approve → risk_reject`, and read
`risk_result` / `order_result` back with a separate DB connection so you know it was committed.

Note that ideas produced by the real ranker carry **no price or quantity**, so
`proposed_notional_pct` computes to 0 and exposure caps cannot bind — seed explicit values when you
want to exercise a breach.

## 5. Things worth checking specifically (all have been broken before)

- The API boots with Slack env vars **unset** (`/health`, `/config`, `/instruments`,
  `/webhooks/tradingview`, `/advisory/notify` all working, `/slack/events` → 503).
- `POST /slack/events` returns 200/401, never 500 — passing a list to `Response(headers=...)`
  used to break every inbound Slack request.
- The hypertable genuinely exists (see §1).
- Exposure caps aggregate across **all** configured venues. `_positions_by_symbol()` skips adapters
  whose `.connected` is false and `execute_idea()` only connects the idea's own venue, so a
  same-underlying position on another venue may be ignored and the cap can fail open. A/B the
  identical order with the other adapter connected vs not; if the outcomes differ, the cap is
  venue-local rather than combined.
- Send a TradingView alert carrying order-ish fields (`quantity`, `execute`, `place_order`) and
  confirm `execution_enabled:false` with zero venue calls.

## Devin Secrets Needed

None. No real Slack, IBKR or Bybit credentials are required for any of the above; a real
`SLACK_BOT_TOKEN` would only add proof that messages render and buttons click in the Slack UI.
