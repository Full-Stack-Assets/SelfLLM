# Hosted API & Monetization

SelfLLM ships a complete monetization layer for its OpenAI-compatible
serving API: tiered API keys, per-key usage metering, daily quotas with an
HTTP 402 upgrade path, and public pricing/usage endpoints. This page
documents how to operate it — both the reference deployment topology and
the configuration for running your own.

## Deployment topology

```
                    ┌─────────────────────────────┐
  Users ──────────▶ │  Vercel gateway (no torch)  │  landing page, /pricing,
                    │  vercel_gateway:app         │  reverse-proxies /v1/*
                    └──────────────┬──────────────┘
                                   │  Authorization header passes through
                    ┌──────────────▼──────────────┐
                    │  Fly.io inference API       │  auth, quotas, metering,
                    │  selfllm.serving.server:app │  continuous batching
                    └─────────────────────────────┘
```

Why two layers: the CPU torch wheel alone pushes a Python bundle to
~780 MB — over Vercel's 500 MB serverless function limit — so inference
cannot run there. Vercel hosts the slim public front door (marketing,
pricing, proxy); Fly.io (`fly.toml`) runs the real model. Key validation,
quota enforcement, and metering all happen at the upstream, so the gateway
holds no secrets and keeps no state.

## Configuring keys and tiers

The serving API reads its billing configuration from the environment:

| Variable | Meaning |
|---|---|
| `SELFLLM_API_KEYS` | Comma-separated `key:tier` pairs, e.g. `sk-abc:free,sk-def:pro`. Tier defaults to `free` when omitted. |
| `SELFLLM_API_KEY` | Legacy single key (still supported); mapped to the unlimited `enterprise` tier. |
| `SELFLLM_UPGRADE_URL` | Payment/upgrade link surfaced in 402 responses and `/v1/pricing` — point this at your Stripe payment link (or similar). |
| `SELFLLM_UPSTREAM_URL` | (Gateway only) the inference API the Vercel gateway proxies to. |

With **no** keys configured, the server runs open and unmetered — identical
to the historical behavior, suitable for local development.

Built-in tiers (see `selfllm/serving/billing.py` to customize):

| Tier | Requests/day | Tokens/day | Price |
|---|---|---|---|
| `free` | 200 | 50,000 | $0 |
| `pro` | 10,000 | 5,000,000 | $29/mo |
| `enterprise` | Unlimited | Unlimited | Custom |

## Pricing and usage endpoints

- **`GET /v1/pricing`** *(public)* — the tier table and upgrade URL as JSON.
- **`GET /v1/usage`** *(authenticated)* — the calling key's tier, today's
  request/token consumption, limits, and remaining quota:

```json
{
  "object": "usage",
  "tier": "pro",
  "day": "2026-06-28",
  "requests": 41,
  "prompt_tokens": 5210,
  "completion_tokens": 11834,
  "total_tokens": 17044,
  "limits":    {"requests_per_day": 10000, "tokens_per_day": 5000000},
  "remaining": {"requests": 9959, "tokens": 4982956},
  "upgrade_url": "https://…/pricing/"
}
```

## Quota exhaustion (402)

When a key exceeds its daily quota, inference endpoints reject with
**HTTP 402 Payment Required**:

```json
{"detail": "Daily request quota exhausted for the 'free' tier (200 requests/day). Upgrade at https://…"}
```

with the upgrade link duplicated in the `X-Upgrade-Url` header. Quotas
reset at UTC midnight. Successful responses carry
`X-RateLimit-Limit-*` / `X-RateLimit-Remaining-*` headers so clients can
back off before hitting 402.

## Activating payments

The 402 upgrade path and pricing pages are wired to whatever
`SELFLLM_UPGRADE_URL` points at. To take real payments:

1. Create a payment link (e.g. a Stripe Payment Link per paid tier).
2. Set `SELFLLM_UPGRADE_URL` on both the inference API and the gateway.
3. On purchase, provision a key for the customer and append it to
   `SELFLLM_API_KEYS` (e.g. `flyctl secrets set` on the Fly.io app), then
   restart/redeploy.

Sponsorship-based funding is also scaffolded via `.github/FUNDING.yml` —
fill in your GitHub Sponsors / funding handles to activate the *Sponsor*
button on the repository.
