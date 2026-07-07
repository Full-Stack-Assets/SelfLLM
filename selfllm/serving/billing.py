"""Usage metering, API-key tiers, and quota enforcement for the serving API.

This is the monetization layer for the OpenAI-compatible server. It is
deliberately dependency-free (stdlib only) and thread-safe so it can sit in
the request path of both the plain and continuous-batching serving modes.

Concepts
~~~~~~~~
- **Tier**: a named plan (``free`` / ``pro`` / ``enterprise``) with daily
  request and token quotas and display pricing. ``None`` quota = unlimited.
- **BillingManager**: maps API keys to tiers, meters per-key usage with a
  daily rollover, enforces quotas (HTTP 402 with an upgrade URL when
  exceeded), and reports usage/rate-limit headers.

Configuration (environment)
~~~~~~~~~~~~~~~~~~~~~~~~~~~
- ``SELFLLM_API_KEYS``: comma-separated ``key:tier`` pairs, e.g.
  ``"sk-abc123:free,sk-def456:pro"``. Tier defaults to ``free`` when omitted.
- ``SELFLLM_API_KEY``: legacy single key (kept for backward compatibility);
  mapped to the ``enterprise`` (unlimited) tier.
- ``SELFLLM_UPGRADE_URL``: payment/upgrade link surfaced in 402 responses and
  ``/v1/pricing`` (e.g. a Stripe payment link). Defaults to the pricing docs.

When no keys are configured at all, the server stays open (no auth, no
quotas) -- identical to the historical behavior.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional

__all__ = ["Tier", "BillingManager", "DEFAULT_TIERS", "DEFAULT_UPGRADE_URL"]

DEFAULT_UPGRADE_URL = "https://full-stack-assets.github.io/SelfLLM/pricing/"


@dataclass(frozen=True)
class Tier:
    """A billing plan with daily quotas. ``None`` means unlimited."""

    name: str
    requests_per_day: Optional[int]
    tokens_per_day: Optional[int]
    price_per_month_usd: Optional[float]  # None = custom / contact
    description: str = ""

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "requests_per_day": self.requests_per_day,
            "tokens_per_day": self.tokens_per_day,
            "price_per_month_usd": self.price_per_month_usd,
            "description": self.description,
        }


DEFAULT_TIERS: Dict[str, Tier] = {
    "free": Tier(
        name="free",
        requests_per_day=200,
        tokens_per_day=50_000,
        price_per_month_usd=0.0,
        description="Evaluate the API. Community support.",
    ),
    "pro": Tier(
        name="pro",
        requests_per_day=10_000,
        tokens_per_day=5_000_000,
        price_per_month_usd=29.0,
        description="Production workloads. Priority scheduling.",
    ),
    "enterprise": Tier(
        name="enterprise",
        requests_per_day=None,
        tokens_per_day=None,
        price_per_month_usd=None,
        description="Unlimited usage, custom SLAs, dedicated support.",
    ),
}


@dataclass
class _Usage:
    """Mutable per-key usage counters for a single day."""

    day: str
    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def _today() -> str:
    """UTC day key used for the daily quota rollover."""
    return time.strftime("%Y-%m-%d", time.gmtime())


class BillingManager:
    """Maps API keys to tiers and meters/enforces per-key daily usage.

    Args:
        keys: ``api_key -> tier_name`` mapping. Empty = open server.
        tiers: tier definitions; defaults to :data:`DEFAULT_TIERS`.
        upgrade_url: payment/upgrade link surfaced on quota exhaustion.
    """

    def __init__(
        self,
        keys: Optional[Dict[str, str]] = None,
        tiers: Optional[Dict[str, Tier]] = None,
        upgrade_url: str = DEFAULT_UPGRADE_URL,
    ) -> None:
        self.tiers = dict(tiers) if tiers else dict(DEFAULT_TIERS)
        self.upgrade_url = upgrade_url
        self._keys: Dict[str, str] = {}
        for key, tier_name in (keys or {}).items():
            if tier_name not in self.tiers:
                raise ValueError(
                    f"Unknown tier {tier_name!r} for key; "
                    f"expected one of {sorted(self.tiers)}"
                )
            self._keys[key] = tier_name
        self._usage: Dict[str, _Usage] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Construction from environment
    # ------------------------------------------------------------------ #

    @classmethod
    def from_env(cls, environ: Optional[Dict[str, str]] = None) -> "BillingManager":
        """Build a manager from environment variables (see module docstring).

        The legacy single ``SELFLLM_API_KEY`` maps to the unlimited
        ``enterprise`` tier so existing deployments keep working unchanged.
        """
        env = environ if environ is not None else os.environ
        keys: Dict[str, str] = {}

        legacy = env.get("SELFLLM_API_KEY")
        if legacy:
            keys[legacy] = "enterprise"

        multi = env.get("SELFLLM_API_KEYS", "")
        for entry in multi.split(","):
            entry = entry.strip()
            if not entry:
                continue
            if ":" in entry:
                key, _, tier_name = entry.rpartition(":")
                key, tier_name = key.strip(), tier_name.strip() or "free"
            else:
                key, tier_name = entry, "free"
            keys[key] = tier_name

        upgrade_url = env.get("SELFLLM_UPGRADE_URL", DEFAULT_UPGRADE_URL)
        return cls(keys=keys, upgrade_url=upgrade_url)

    # ------------------------------------------------------------------ #
    # Auth
    # ------------------------------------------------------------------ #

    @property
    def enabled(self) -> bool:
        """Whether any API keys are configured (False = open server)."""
        return bool(self._keys)

    def authenticate(self, authorization: Optional[str]) -> Optional[str]:
        """Validate a ``Authorization: Bearer <key>`` header value.

        Returns the API key on success, or ``None`` for an open server.
        Raises ``PermissionError`` on a bad/missing key (the HTTP layer maps
        this to 401 so this module stays framework-free).
        """
        if not self.enabled:
            return None
        if not authorization or not authorization.startswith("Bearer "):
            raise PermissionError("Invalid or missing API key.")
        key = authorization[len("Bearer "):]
        if key not in self._keys:
            raise PermissionError("Invalid or missing API key.")
        return key

    def tier_of(self, key: Optional[str]) -> Optional[Tier]:
        """Return the tier for a key (None for the open-server case).

        A key that authenticated outside this manager (the legacy single
        ``SELFLLM_API_KEY`` path) is treated as unlimited ``enterprise`` so it
        still gets metered without gaining a quota it never had.
        """
        if key is None:
            return None
        return self.tiers[self._keys.get(key, "enterprise")]

    # ------------------------------------------------------------------ #
    # Metering + quota
    # ------------------------------------------------------------------ #

    def _usage_for(self, key: str) -> _Usage:
        """Return today's usage record for a key (rolls over at UTC midnight).

        Caller must hold ``self._lock``.
        """
        day = _today()
        usage = self._usage.get(key)
        if usage is None or usage.day != day:
            usage = _Usage(day=day)
            self._usage[key] = usage
        return usage

    def check_quota(self, key: Optional[str]) -> None:
        """Raise ``QuotaExceeded`` if the key is out of daily quota."""
        if key is None:
            return
        tier = self.tier_of(key)
        with self._lock:
            usage = self._usage_for(key)
            if (
                tier.requests_per_day is not None
                and usage.requests >= tier.requests_per_day
            ):
                raise QuotaExceeded(
                    f"Daily request quota exhausted for the '{tier.name}' tier "
                    f"({tier.requests_per_day} requests/day). "
                    f"Upgrade at {self.upgrade_url}",
                    upgrade_url=self.upgrade_url,
                )
            if (
                tier.tokens_per_day is not None
                and usage.total_tokens >= tier.tokens_per_day
            ):
                raise QuotaExceeded(
                    f"Daily token quota exhausted for the '{tier.name}' tier "
                    f"({tier.tokens_per_day} tokens/day). "
                    f"Upgrade at {self.upgrade_url}",
                    upgrade_url=self.upgrade_url,
                )

    def record(
        self, key: Optional[str], prompt_tokens: int = 0, completion_tokens: int = 0
    ) -> None:
        """Meter one request's usage against a key (no-op for open server)."""
        if key is None:
            return
        with self._lock:
            usage = self._usage_for(key)
            usage.requests += 1
            usage.prompt_tokens += int(prompt_tokens)
            usage.completion_tokens += int(completion_tokens)

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #

    def usage_report(self, key: str) -> Dict:
        """Structured usage/limits/remaining report for ``/v1/usage``."""
        tier = self.tier_of(key)
        with self._lock:
            usage = self._usage_for(key)
            remaining_requests = (
                None
                if tier.requests_per_day is None
                else max(0, tier.requests_per_day - usage.requests)
            )
            remaining_tokens = (
                None
                if tier.tokens_per_day is None
                else max(0, tier.tokens_per_day - usage.total_tokens)
            )
            return {
                "object": "usage",
                "tier": tier.name,
                "day": usage.day,
                "requests": usage.requests,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
                "limits": {
                    "requests_per_day": tier.requests_per_day,
                    "tokens_per_day": tier.tokens_per_day,
                },
                "remaining": {
                    "requests": remaining_requests,
                    "tokens": remaining_tokens,
                },
                "upgrade_url": self.upgrade_url,
            }

    def rate_limit_headers(self, key: Optional[str]) -> Dict[str, str]:
        """OpenAI-style X-RateLimit-* headers for responses (empty if open)."""
        if key is None:
            return {}
        tier = self.tier_of(key)
        with self._lock:
            usage = self._usage_for(key)
            headers: Dict[str, str] = {}
            if tier.requests_per_day is not None:
                headers["X-RateLimit-Limit-Requests"] = str(tier.requests_per_day)
                headers["X-RateLimit-Remaining-Requests"] = str(
                    max(0, tier.requests_per_day - usage.requests)
                )
            if tier.tokens_per_day is not None:
                headers["X-RateLimit-Limit-Tokens"] = str(tier.tokens_per_day)
                headers["X-RateLimit-Remaining-Tokens"] = str(
                    max(0, tier.tokens_per_day - usage.total_tokens)
                )
            return headers

    def pricing(self) -> Dict:
        """Public pricing table for ``/v1/pricing`` and the docs site."""
        return {
            "object": "pricing",
            "currency": "USD",
            "upgrade_url": self.upgrade_url,
            "tiers": [tier.to_dict() for tier in self.tiers.values()],
        }


class QuotaExceeded(Exception):
    """Raised when a key is over its daily quota (HTTP layer maps to 402)."""

    def __init__(self, message: str, upgrade_url: str) -> None:
        super().__init__(message)
        self.upgrade_url = upgrade_url
