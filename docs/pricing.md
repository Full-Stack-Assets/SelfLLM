# Pricing

SelfLLM's hosted, OpenAI-compatible inference API is available in three
tiers. Quotas reset daily (UTC); usage is metered per API key across both
requests and tokens.

| | **Free** | **Pro** | **Enterprise** |
|---|---|---|---|
| **Price** | $0 | $29 / month | Contact us |
| **Requests / day** | 200 | 10,000 | Unlimited |
| **Tokens / day** | 50,000 | 5,000,000 | Unlimited |
| **Support** | Community | Priority | Dedicated + custom SLAs |
| **Best for** | Evaluation | Production workloads | Scale deployments |

!!! tip "Live pricing endpoint"
    The API itself always serves the authoritative table at
    [`GET /v1/pricing`](hosted-api.md#pricing-and-usage-endpoints) — no
    authentication required.

## How quotas behave

- Every request is metered against the calling API key: request count plus
  prompt and completion tokens (streaming responses are metered too).
- When a tier's daily quota is exhausted, the API returns **HTTP 402** with
  an upgrade link in the body and an `X-Upgrade-Url` response header.
- Responses carry OpenAI-style rate-limit headers
  (`X-RateLimit-Limit-Requests`, `X-RateLimit-Remaining-Requests`,
  `X-RateLimit-Limit-Tokens`, `X-RateLimit-Remaining-Tokens`) so clients can
  self-throttle before hitting the limit.
- Check your live consumption at any time via
  [`GET /v1/usage`](hosted-api.md#pricing-and-usage-endpoints).

## Getting a key / upgrading

Keys are provisioned by the operator of the deployment (see
[Hosted API & Monetization](hosted-api.md) to run your own). For the
reference deployment, use the upgrade link surfaced by the API's 402
responses and `/v1/pricing` payload.

## Self-hosting is always free

SelfLLM is MIT-licensed. The entire stack — model, training loop, serving
engine, and this billing layer — is open source. The paid tiers price the
*hosted convenience*, not the software.
