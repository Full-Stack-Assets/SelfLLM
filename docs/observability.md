# Observability & Operations

The serving API is instrumented for production operation: Prometheus metrics,
durable usage persistence, and a one-command continuous-deployment pipeline.

## Metrics (`GET /metrics`)

Every request is timed by a FastAPI middleware and recorded into a
dependency-free Prometheus collector (`selfllm/serving/metrics.py`). Scrape
`GET /metrics` with Prometheus, Grafana Agent, or an OpenTelemetry collector —
no extra libraries, so it stays inside the Fly.io image budget.

Exposed series:

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `selfllm_requests_total` | counter | `endpoint`, `method`, `status` | Total HTTP requests |
| `selfllm_request_latency_seconds` | histogram | `endpoint` | Request duration (`_bucket`/`_sum`/`_count`) |
| `selfllm_requests_in_flight` | gauge | — | Requests being processed right now |

The `endpoint` label uses the matched **route template** (e.g.
`/v1/chat/completions`), not the raw path, so metric cardinality stays bounded.

```
# Example Prometheus scrape config
scrape_configs:
  - job_name: selfllm
    metrics_path: /metrics
    static_configs:
      - targets: ["selfllm.fly.dev"]
```

The Vercel gateway also exposes `GET /metrics`, proxying the upstream so a
single scrape target covers the whole system. The endpoint is unauthenticated
by convention — restrict it at the ingress/network layer if it is public.

## Durable usage metering

By default the billing layer meters usage **in memory**, which resets on every
restart — a real problem for daily quotas across deploys. Set
`SELFLLM_USAGE_DB` to a file path to persist usage in **SQLite** instead
(`selfllm/serving/usage_store.py`):

```bash
# Fly.io: persist on a mounted volume so usage survives restarts.
flyctl secrets set SELFLLM_USAGE_DB=/data/usage.db
```

The store is keyed by `(api_key, day)` in UTC, so the daily quota rollover is
automatic. Backends are pluggable via the `UsageStore` protocol
(`InMemoryUsageStore`, `SQLiteUsageStore`) — drop in Redis/Postgres by
implementing `get`/`add`.

## Continuous deployment (Fly.io)

`.github/workflows/deploy-fly.yml` deploys the inference API to Fly.io on every
push to `main` that touches app code (and on manual dispatch). It is a no-op
until a `FLY_API_TOKEN` secret is configured, so forks are never blocked:

1. `flyctl auth token` to mint a deploy token.
2. Add it as the `FLY_API_TOKEN` repository secret
   (**Settings → Secrets and variables → Actions**).
3. Push to `main` — the workflow runs `flyctl deploy --remote-only`.

The image (`Dockerfile`) installs the CPU torch wheel and serves
`selfllm.serving.server:app` via uvicorn on port 8080 (matching `fly.toml`).
The docs site deploys separately to GitHub Pages
(`.github/workflows/deploy-pages.yml`), and the Vercel gateway deploys on push
via the Vercel GitHub integration.
