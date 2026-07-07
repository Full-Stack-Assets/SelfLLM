"""Slim, torch-free front door for the SelfLLM Vercel deployment.

Why this exists: the full serving stack (``selfllm.serving.server:app``)
imports torch, and the CPU torch wheel alone pushes the Vercel Python
function bundle to ~780 MB -- far over Vercel's 500 MB limit, which is why
every deployment of the full app failed. Inference therefore lives on
Fly.io (see ``fly.toml``); this module is the lightweight public surface
Vercel *can* host:

- ``GET /``          -- landing page (pitch, pricing, links)
- ``GET /health``    -- gateway health
- ``GET /pricing``   -- pricing tiers as JSON
- ``/v1/*``          -- reverse proxy to the real inference API

The upstream inference API is configured with the ``SELFLLM_UPSTREAM_URL``
env var (defaults to the Fly.io deployment). Authorization headers pass
through untouched, so API keys are validated -- and usage metered -- by the
upstream, never here.

This module deliberately imports neither torch nor the ``selfllm`` package
(whose import graph reaches torch). The pricing table mirrors
``selfllm/serving/billing.py``; the upstream's ``/v1/pricing`` endpoint is
the runtime source of truth.
"""

from __future__ import annotations

import json
import os

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse

UPSTREAM_URL = os.environ.get(
    "SELFLLM_UPSTREAM_URL", "https://selfllm.fly.dev"
).rstrip("/")
UPGRADE_URL = os.environ.get(
    "SELFLLM_UPGRADE_URL", "https://full-stack-assets.github.io/SelfLLM/pricing/"
)
DOCS_URL = "https://full-stack-assets.github.io/SelfLLM/"
REPO_URL = "https://github.com/Full-Stack-Assets/SelfLLM"

# Mirrors selfllm/serving/billing.py DEFAULT_TIERS (display copy only; the
# upstream /v1/pricing endpoint is authoritative at runtime).
PRICING_TIERS = [
    {
        "name": "free",
        "price_per_month_usd": 0.0,
        "requests_per_day": 200,
        "tokens_per_day": 50_000,
        "description": "Evaluate the API. Community support.",
    },
    {
        "name": "pro",
        "price_per_month_usd": 29.0,
        "requests_per_day": 10_000,
        "tokens_per_day": 5_000_000,
        "description": "Production workloads. Priority scheduling.",
    },
    {
        "name": "enterprise",
        "price_per_month_usd": None,
        "requests_per_day": None,
        "tokens_per_day": None,
        "description": "Unlimited usage, custom SLAs, dedicated support.",
    },
]

app = FastAPI(title="SelfLLM Gateway", version="1.0.0")


def _fmt_quota(value) -> str:
    return "Unlimited" if value is None else f"{value:,}"


def _fmt_price(value) -> str:
    if value is None:
        return "Contact us"
    if value == 0:
        return "$0"
    return f"${value:g}/mo"


_LANDING_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>SelfLLM — a recursively self-improving language model</title>
<style>
  :root {{ color-scheme: light dark; }}
  * {{ box-sizing: border-box; margin: 0; }}
  body {{
    font: 16px/1.6 system-ui, -apple-system, "Segoe UI", sans-serif;
    background: #0b1020; color: #e6e9f2; padding: 0 1.25rem 4rem;
  }}
  main {{ max-width: 60rem; margin: 0 auto; }}
  header {{ text-align: center; padding: 4rem 0 2.5rem; }}
  h1 {{ font-size: 2.4rem; letter-spacing: -0.02em; }}
  h1 span {{ color: #7aa2ff; }}
  .tag {{ color: #9aa3b8; margin-top: .6rem; }}
  .cta {{ margin-top: 1.6rem; display: flex; gap: .8rem; justify-content: center;
         flex-wrap: wrap; }}
  .cta a {{
    display: inline-block; padding: .65rem 1.2rem; border-radius: .55rem;
    text-decoration: none; font-weight: 600;
  }}
  .cta a.primary {{ background: #4f7cff; color: #fff; }}
  .cta a.ghost {{ border: 1px solid #39415a; color: #cdd4e4; }}
  section {{ margin-top: 3rem; }}
  h2 {{ font-size: 1.35rem; margin-bottom: 1rem; }}
  .grid {{ display: grid; gap: 1rem;
          grid-template-columns: repeat(auto-fit, minmax(15rem, 1fr)); }}
  .card {{
    background: #121a33; border: 1px solid #232c4a; border-radius: .8rem;
    padding: 1.2rem 1.3rem;
  }}
  .card h3 {{ font-size: 1.05rem; text-transform: capitalize; }}
  .price {{ font-size: 1.7rem; font-weight: 700; margin: .4rem 0 .6rem; }}
  .card ul {{ padding-left: 1.1rem; color: #aab3c8; font-size: .93rem; }}
  .card.pro {{ border-color: #4f7cff; }}
  .card .pick {{
    display: inline-block; margin-top: .9rem; padding: .45rem .9rem;
    border-radius: .5rem; background: #4f7cff; color: #fff;
    text-decoration: none; font-size: .9rem; font-weight: 600;
  }}
  code, pre {{
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    background: #121a33; border: 1px solid #232c4a; border-radius: .5rem;
  }}
  pre {{ padding: 1rem; overflow-x: auto; font-size: .88rem; }}
  code {{ padding: .1rem .35rem; }}
  footer {{ margin-top: 3.5rem; text-align: center; color: #77809a;
           font-size: .88rem; }}
  a {{ color: #8fb0ff; }}
</style>
</head>
<body>
<main>
  <header>
    <h1>Self<span>LLM</span></h1>
    <p class="tag">A recursively self-improving foundation language model,
    built from scratch in PyTorch — served through an OpenAI-compatible API.</p>
    <div class="cta">
      <a class="primary" href="{upgrade_url}">Get an API key</a>
      <a class="ghost" href="{docs_url}">Documentation</a>
      <a class="ghost" href="{repo_url}">GitHub</a>
    </div>
  </header>

  <section>
    <h2>Drop-in OpenAI-compatible</h2>
    <pre>curl {self_url}/v1/chat/completions \\
  -H "Authorization: Bearer $SELFLLM_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{{"model": "selfllm", "messages": [{{"role": "user", "content": "Hello"}}]}}'</pre>
    <p style="color:#9aa3b8">Point any OpenAI SDK at this base URL and it just
    works — chat and text completions, streaming, usage accounting, and
    opt-in test-time-compute reasoning.</p>
  </section>

  <section>
    <h2>Pricing</h2>
    <div class="grid">{pricing_cards}</div>
  </section>

  <section>
    <h2>What's inside</h2>
    <div class="grid">
      <div class="card"><h3>Recursive self-improvement</h3>
        <ul><li>Generate → filter → train → evaluate loop</li>
        <li>Reward-model-guided sample selection</li>
        <li>Automatic rollback on regression</li></ul></div>
      <div class="card"><h3>Serving engine</h3>
        <ul><li>Continuous batching + paged KV cache</li>
        <li>Streaming attention sinks for unbounded chats</li>
        <li>Speculative decoding</li></ul></div>
      <div class="card"><h3>Reasoning &amp; evals</h3>
        <ul><li>Self-consistency, best-of-N, beam search</li>
        <li>MMLU / GSM8K / HumanEval suite built in</li>
        <li>Regression detection across iterations</li></ul></div>
    </div>
  </section>

  <footer>MIT-licensed research project ·
    <a href="{docs_url}">docs</a> · <a href="{repo_url}">source</a> ·
    <a href="/pricing">pricing API</a> · <a href="/health">status</a>
  </footer>
</main>
</body>
</html>"""


def _pricing_cards() -> str:
    cards = []
    for tier in PRICING_TIERS:
        highlight = " pro" if tier["name"] == "pro" else ""
        cards.append(
            f'<div class="card{highlight}"><h3>{tier["name"]}</h3>'
            f'<div class="price">{_fmt_price(tier["price_per_month_usd"])}</div>'
            f'<ul><li>{_fmt_quota(tier["requests_per_day"])} requests/day</li>'
            f'<li>{_fmt_quota(tier["tokens_per_day"])} tokens/day</li>'
            f'<li>{tier["description"]}</li></ul>'
            f'<a class="pick" href="{UPGRADE_URL}">Choose {tier["name"]}</a>'
            "</div>"
        )
    return "".join(cards)


@app.get("/", response_class=HTMLResponse)
async def landing(request: Request) -> HTMLResponse:
    """Marketing landing page: pitch, quickstart, pricing, links."""
    self_url = str(request.base_url).rstrip("/")
    return HTMLResponse(
        _LANDING_HTML.format(
            upgrade_url=UPGRADE_URL,
            docs_url=DOCS_URL,
            repo_url=REPO_URL,
            self_url=self_url,
            pricing_cards=_pricing_cards(),
        )
    )


@app.get("/health")
async def health() -> dict:
    """Gateway health (does not call the upstream)."""
    return {"status": "healthy", "role": "gateway", "upstream": UPSTREAM_URL}


@app.get("/pricing")
async def pricing() -> dict:
    """Pricing tiers as JSON (display mirror; upstream /v1/pricing is live)."""
    return {
        "object": "pricing",
        "currency": "USD",
        "upgrade_url": UPGRADE_URL,
        "tiers": PRICING_TIERS,
    }


# Hop-by-hop headers that must not be forwarded by a proxy (RFC 9110 §7.6.1).
_HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length",
}


@app.api_route(
    "/v1/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
)
async def proxy_v1(path: str, request: Request) -> Response:
    """Reverse-proxy ``/v1/*`` to the real inference API.

    Bodies, query strings, and the ``Authorization`` header pass through, so
    key validation, quota enforcement, and usage metering all happen at the
    upstream -- the gateway holds no secrets and keeps no state.
    """
    url = f"{UPSTREAM_URL}/v1/{path}"
    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in _HOP_BY_HOP
    }
    body = await request.body()
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            upstream = await client.request(
                request.method,
                url,
                params=dict(request.query_params),
                headers=headers,
                content=body,
            )
    except httpx.HTTPError as exc:
        detail = {
            "error": {
                "message": (
                    "Upstream inference API unreachable: "
                    f"{type(exc).__name__}"
                ),
                "type": "upstream_unavailable",
            }
        }
        return Response(
            content=json.dumps(detail),
            status_code=502,
            media_type="application/json",
        )
    response_headers = {
        k: v for k, v in upstream.headers.items()
        if k.lower() not in _HOP_BY_HOP
    }
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers=response_headers,
        media_type=upstream.headers.get("content-type"),
    )
