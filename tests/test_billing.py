"""Tests for the monetization layer: billing module, server integration,
and the torch-free Vercel gateway."""

from unittest.mock import MagicMock

import pytest
import torch

from selfllm.serving.billing import (
    DEFAULT_TIERS,
    BillingManager,
    QuotaExceeded,
    Tier,
)


# ---------------------------------------------------------------------------
# BillingManager unit tests
# ---------------------------------------------------------------------------


class TestBillingManager:
    def test_from_env_parses_multi_keys(self):
        mgr = BillingManager.from_env(
            {"SELFLLM_API_KEYS": "sk-a:free, sk-b:pro ,sk-c"}
        )
        assert mgr.enabled
        assert mgr.tier_of("sk-a").name == "free"
        assert mgr.tier_of("sk-b").name == "pro"
        assert mgr.tier_of("sk-c").name == "free"  # tier defaults to free

    def test_from_env_legacy_key_is_enterprise(self):
        mgr = BillingManager.from_env({"SELFLLM_API_KEY": "sk-legacy"})
        assert mgr.tier_of("sk-legacy").name == "enterprise"

    def test_from_env_empty_is_open(self):
        mgr = BillingManager.from_env({})
        assert not mgr.enabled
        assert mgr.authenticate(None) is None  # open server: no auth

    def test_unknown_tier_rejected(self):
        with pytest.raises(ValueError):
            BillingManager(keys={"sk-a": "platinum"})

    def test_authenticate_success_and_failure(self):
        mgr = BillingManager(keys={"sk-a": "free"})
        assert mgr.authenticate("Bearer sk-a") == "sk-a"
        for bad in [None, "sk-a", "Bearer wrong", "Basic sk-a"]:
            with pytest.raises(PermissionError):
                mgr.authenticate(bad)

    def test_request_quota_enforced(self):
        tiers = dict(DEFAULT_TIERS)
        tiers["tiny"] = Tier("tiny", requests_per_day=2, tokens_per_day=None,
                             price_per_month_usd=0.0)
        mgr = BillingManager(keys={"sk-a": "tiny"}, tiers=tiers)
        mgr.check_quota("sk-a")
        mgr.record("sk-a", 10, 10)
        mgr.record("sk-a", 10, 10)
        with pytest.raises(QuotaExceeded) as exc:
            mgr.check_quota("sk-a")
        assert "request quota" in str(exc.value)
        assert exc.value.upgrade_url == mgr.upgrade_url

    def test_token_quota_enforced(self):
        tiers = dict(DEFAULT_TIERS)
        tiers["tiny"] = Tier("tiny", requests_per_day=None, tokens_per_day=15,
                             price_per_month_usd=0.0)
        mgr = BillingManager(keys={"sk-a": "tiny"}, tiers=tiers)
        mgr.record("sk-a", prompt_tokens=10, completion_tokens=5)
        with pytest.raises(QuotaExceeded) as exc:
            mgr.check_quota("sk-a")
        assert "token quota" in str(exc.value)

    def test_enterprise_is_unlimited(self):
        mgr = BillingManager(keys={"sk-a": "enterprise"})
        for _ in range(1000):
            mgr.record("sk-a", 1000, 1000)
        mgr.check_quota("sk-a")  # never raises

    def test_open_server_noops(self):
        mgr = BillingManager()
        mgr.check_quota(None)
        mgr.record(None, 5, 5)
        assert mgr.rate_limit_headers(None) == {}

    def test_daily_rollover_resets_usage(self, monkeypatch):
        mgr = BillingManager(keys={"sk-a": "free"})
        mgr.record("sk-a", 10, 10)
        assert mgr.usage_report("sk-a")["requests"] == 1
        # Advance the clock to the next day: counters reset.
        monkeypatch.setattr(
            "selfllm.serving.billing._today", lambda: "2099-01-01"
        )
        report = mgr.usage_report("sk-a")
        assert report["requests"] == 0
        assert report["total_tokens"] == 0

    def test_usage_report_shape(self):
        mgr = BillingManager(keys={"sk-a": "free"})
        mgr.record("sk-a", prompt_tokens=7, completion_tokens=3)
        report = mgr.usage_report("sk-a")
        assert report["tier"] == "free"
        assert report["prompt_tokens"] == 7
        assert report["completion_tokens"] == 3
        assert report["total_tokens"] == 10
        assert report["remaining"]["requests"] == 199
        assert report["remaining"]["tokens"] == 49_990

    def test_rate_limit_headers(self):
        mgr = BillingManager(keys={"sk-a": "free"})
        mgr.record("sk-a", 5, 5)
        headers = mgr.rate_limit_headers("sk-a")
        assert headers["X-RateLimit-Limit-Requests"] == "200"
        assert headers["X-RateLimit-Remaining-Requests"] == "199"
        assert headers["X-RateLimit-Remaining-Tokens"] == "49990"

    def test_legacy_unregistered_key_meters_as_enterprise(self):
        mgr = BillingManager(keys={"sk-a": "free"})
        # A key authenticated outside the manager (legacy path) still meters.
        mgr.record("sk-legacy", 5, 5)
        assert mgr.usage_report("sk-legacy")["tier"] == "enterprise"

    def test_pricing_table(self):
        mgr = BillingManager()
        pricing = mgr.pricing()
        names = {t["name"] for t in pricing["tiers"]}
        assert {"free", "pro", "enterprise"} <= names
        assert pricing["upgrade_url"]


# ---------------------------------------------------------------------------
# Server integration
# ---------------------------------------------------------------------------


class TestServerBilling:
    """Tiered auth, quotas, and usage endpoints on the FastAPI server."""

    @pytest.fixture
    def mock_model(self):
        model = MagicMock()
        model.generate.return_value = {"sequences": torch.tensor([[42, 99]])}
        param = MagicMock()
        param.device = torch.device("cpu")
        model.parameters.side_effect = lambda: iter([param])
        return model

    @pytest.fixture
    def mock_tokenizer(self):
        tok = MagicMock()
        tok.eos_token_id = 1
        tok.encode.return_value = [10]
        tok.decode.return_value = "hello world"
        return tok

    @pytest.fixture
    def client(self, mock_model, mock_tokenizer, monkeypatch):
        from fastapi.testclient import TestClient
        import selfllm.serving.server as server_module

        monkeypatch.setattr(server_module, "_model", mock_model)
        monkeypatch.setattr(server_module, "_tokenizer", mock_tokenizer)
        monkeypatch.setattr(server_module, "_scheduler", None)
        monkeypatch.setattr(server_module, "_api_key", None)
        tiers = dict(DEFAULT_TIERS)
        tiers["tiny"] = Tier("tiny", requests_per_day=2, tokens_per_day=None,
                             price_per_month_usd=1.0)
        monkeypatch.setattr(
            server_module,
            "_billing",
            BillingManager(
                keys={"sk-free": "tiny", "sk-pro": "pro"}, tiers=tiers
            ),
        )
        return TestClient(server_module.app)

    BODY = {"model": "selfllm",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4}

    def test_tiered_key_authenticates(self, client):
        r = client.post("/v1/chat/completions", json=self.BODY,
                        headers={"Authorization": "Bearer sk-pro"})
        assert r.status_code == 200
        assert "X-RateLimit-Remaining-Requests" in r.headers

    def test_bad_key_is_401(self, client):
        r = client.post("/v1/chat/completions", json=self.BODY,
                        headers={"Authorization": "Bearer nope"})
        assert r.status_code == 401

    def test_quota_exhaustion_is_402_with_upgrade_url(self, client):
        headers = {"Authorization": "Bearer sk-free"}
        for _ in range(2):  # tiny tier: 2 requests/day
            assert client.post("/v1/chat/completions", json=self.BODY,
                               headers=headers).status_code == 200
        r = client.post("/v1/chat/completions", json=self.BODY,
                        headers=headers)
        assert r.status_code == 402
        assert "Upgrade" in r.json()["detail"]
        assert r.headers["X-Upgrade-Url"]

    def test_usage_endpoint_reports_consumption(self, client):
        headers = {"Authorization": "Bearer sk-pro"}
        client.post("/v1/chat/completions", json=self.BODY, headers=headers)
        r = client.get("/v1/usage", headers=headers)
        assert r.status_code == 200
        data = r.json()
        assert data["tier"] == "pro"
        assert data["requests"] == 1
        assert data["total_tokens"] > 0

    def test_pricing_endpoint_is_public(self, client):
        r = client.get("/v1/pricing")
        assert r.status_code == 200
        assert {t["name"] for t in r.json()["tiers"]} >= {"free", "pro"}

    def test_completions_endpoint_metered(self, client):
        headers = {"Authorization": "Bearer sk-pro"}
        r = client.post(
            "/v1/completions",
            json={"model": "selfllm", "prompt": "Hello", "max_tokens": 4},
            headers=headers,
        )
        assert r.status_code == 200
        usage = client.get("/v1/usage", headers=headers).json()
        assert usage["requests"] == 1

    def test_legacy_single_key_still_works(self, client, monkeypatch):
        import selfllm.serving.server as server_module

        monkeypatch.setattr(server_module, "_api_key", "sk-legacy")
        r = client.post("/v1/chat/completions", json=self.BODY,
                        headers={"Authorization": "Bearer sk-legacy"})
        assert r.status_code == 200
        # Legacy key meters on the unlimited enterprise tier.
        usage = client.get(
            "/v1/usage", headers={"Authorization": "Bearer sk-legacy"}
        ).json()
        assert usage["tier"] == "enterprise"
        assert usage["requests"] == 1

    def test_open_server_usage_reports_disabled(
        self, mock_model, mock_tokenizer, monkeypatch
    ):
        from fastapi.testclient import TestClient
        import selfllm.serving.server as server_module

        monkeypatch.setattr(server_module, "_model", mock_model)
        monkeypatch.setattr(server_module, "_tokenizer", mock_tokenizer)
        monkeypatch.setattr(server_module, "_api_key", None)
        monkeypatch.setattr(server_module, "_billing", BillingManager())
        client = TestClient(server_module.app)
        assert client.get("/v1/usage").json()["billing"] == "disabled"


# ---------------------------------------------------------------------------
# Vercel gateway
# ---------------------------------------------------------------------------


class TestVercelGateway:
    """The slim torch-free front door: landing, pricing, health, proxy."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        import vercel_gateway

        return TestClient(vercel_gateway.app)

    def test_landing_page(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "SelfLLM" in r.text
        assert "Pricing" in r.text
        assert "/v1/chat/completions" in r.text  # quickstart snippet

    def test_health(self, client):
        data = client.get("/health").json()
        assert data["status"] == "healthy"
        assert data["role"] == "gateway"

    def test_pricing_tiers(self, client):
        tiers = {t["name"] for t in client.get("/pricing").json()["tiers"]}
        assert tiers == {"free", "pro", "enterprise"}

    def test_gateway_does_not_import_torch(self):
        import importlib
        import sys

        import vercel_gateway

        importlib.reload(vercel_gateway)
        # The whole point of the gateway: deployable without the ~780 MB
        # torch bundle. Its import graph must never reach torch.
        gateway_modules = {
            name for name, mod in sys.modules.items()
            if mod is not None and name == "vercel_gateway"
        }
        assert gateway_modules  # imported fine
        import inspect

        src = inspect.getsource(vercel_gateway)
        assert "import torch" not in src
        assert "import selfllm" not in src

    def test_proxy_forwards_to_upstream(self, client, monkeypatch):
        import vercel_gateway

        captured = {}

        class _StubResponse:
            status_code = 200
            content = b'{"object": "list", "data": []}'
            headers = {"content-type": "application/json"}

        class _StubClient:
            def __init__(self, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def request(self, method, url, **kwargs):
                captured["method"] = method
                captured["url"] = url
                captured["headers"] = kwargs.get("headers", {})
                return _StubResponse()

        monkeypatch.setattr(
            vercel_gateway.httpx, "AsyncClient", _StubClient
        )
        r = client.get("/v1/models",
                       headers={"Authorization": "Bearer sk-abc"})
        assert r.status_code == 200
        assert captured["url"].endswith("/v1/models")
        # Authorization passes through so the upstream does auth + metering.
        assert captured["headers"].get("authorization") == "Bearer sk-abc"

    def test_proxy_maps_upstream_failure_to_502(self, client, monkeypatch):
        import httpx as httpx_module

        import vercel_gateway

        class _FailingClient:
            def __init__(self, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def request(self, method, url, **kwargs):
                raise httpx_module.ConnectError("no route to upstream")

        monkeypatch.setattr(
            vercel_gateway.httpx, "AsyncClient", _FailingClient
        )
        r = client.post("/v1/chat/completions", json={"messages": []})
        assert r.status_code == 502
        assert r.json()["error"]["type"] == "upstream_unavailable"
