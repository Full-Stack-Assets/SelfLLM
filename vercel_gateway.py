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
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

# Largest request body the proxy will forward (protects the upstream from
# oversized payloads at the edge).
MAX_PROXY_BODY_BYTES = 10 * 1024 * 1024  # 10 MiB

UPSTREAM_URL = os.environ.get(
    "SELFLLM_UPSTREAM_URL", "https://selfllm.fly.dev"
).rstrip("/")
# Where the "get a key / upgrade" CTAs point. Defaults to the pricing section
# on this same page (never 404s); operators set this to their real payment
# link (e.g. a Stripe Payment Link) in production.
UPGRADE_URL = os.environ.get("SELFLLM_UPGRADE_URL", "/#pricing")
DOCS_URL = "https://fullstackassets.com/SelfLLM/"
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

# Security headers applied to every response (front-end hardening). The landing
# page uses inline <style>, so style-src allows 'unsafe-inline'; everything else
# is locked to same-origin.
_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "SAMEORIGIN",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Strict-Transport-Security": "max-age=63072000; includeSubDomains",
    "Content-Security-Policy": (
        "default-src 'self'; style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; base-uri 'none'; frame-ancestors 'self'"
    ),
}


@app.middleware("http")
async def _security_headers(request: Request, call_next):
    response = await call_next(request)
    for header, value in _SECURITY_HEADERS.items():
        response.headers.setdefault(header, value)
    return response


# Inline SVG favicon so browsers don't log a 404 on every page load.
_FAVICON_SVG = (
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'>"
    "<rect width='64' height='64' rx='12' fill='#4f7cff'/>"
    "<text x='50%' y='55%' text-anchor='middle' dominant-baseline='middle' "
    "font-family='system-ui,sans-serif' font-size='34' font-weight='700' "
    "fill='white'>S</text></svg>"
)


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Response:
    """Serve a favicon (prevents a 404 on the browser's automatic request)."""
    return Response(content=_FAVICON_SVG, media_type="image/svg+xml")


@app.get("/robots.txt", include_in_schema=False)
async def robots() -> Response:
    """Allow crawling of the marketing pages; keep the API surface out."""
    body = "User-agent: *\nAllow: /\nDisallow: /v1/\nDisallow: /metrics\n"
    return Response(content=body, media_type="text/plain")


def _render_error_page(status_code: int, message: str) -> str:
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{status_code} — SelfLLM</title>
<style>
  html {{ color-scheme: light dark; }}
  body {{ font: 16px/1.6 system-ui, sans-serif; background: #0b1020; color: #e6e9f2;
         display: grid; place-items: center; min-height: 100vh; margin: 0;
         text-align: center; padding: 1.5rem; }}
  h1 {{ font-size: 3rem; margin: 0 0 .3rem; color: #7aa2ff; }}
  a {{ color: #8fb0ff; }}
</style></head>
<body><main>
  <h1>{status_code}</h1>
  <p>{message}</p>
  <p><a href="/">← Back to SelfLLM</a></p>
</main></body></html>"""


@app.exception_handler(StarletteHTTPException)
async def _http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Branded error responses: HTML for browsers, JSON for API clients.

    Turns the bare ``{"detail": "Not Found"}`` (and other HTTP errors) into a
    styled page for humans and a structured error object for programmatic
    callers, keyed off the ``Accept`` header.
    """
    wants_html = "text/html" in request.headers.get("accept", "")
    if wants_html:
        message = exc.detail if exc.status_code != 404 else (
            "That page doesn't exist. The API lives under /v1/."
        )
        return HTMLResponse(
            _render_error_page(exc.status_code, message), status_code=exc.status_code
        )
    return JSONResponse(
        {"error": {"message": exc.detail, "type": "http_error",
                   "status": exc.status_code}},
        status_code=exc.status_code,
    )


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

  <section id="pricing">
    <h2>Pricing</h2>
    <div class="grid">{pricing_cards}</div>
    <p style="color:#9aa3b8;margin-top:1rem">Quotas reset daily. Need a key?
    Reach out via the <a href="{repo_url}">GitHub repo</a> — or self-host the
    whole stack, it's MIT-licensed.</p>
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


@app.get("/metrics")
async def metrics(request: Request) -> Response:
    """Proxy the upstream's Prometheus metrics so a single scrape target
    (this gateway) covers the whole system."""
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            upstream = await client.get(f"{UPSTREAM_URL}/metrics")
    except httpx.HTTPError:
        return Response(
            content="# upstream metrics unavailable\n",
            media_type="text/plain; version=0.0.4; charset=utf-8",
            status_code=502,
        )
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get(
            "content-type", "text/plain; version=0.0.4; charset=utf-8"
        ),
    )


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
    if len(body) > MAX_PROXY_BODY_BYTES:
        return JSONResponse(
            {"error": {"message": "Request body too large.",
                       "type": "payload_too_large"}},
            status_code=413,
        )
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
