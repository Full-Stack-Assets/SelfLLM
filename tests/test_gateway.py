"""Tests for the Vercel gateway front door and its hardening.

Covers the fixes for the reported 404s (favicon, robots, branded 404, no
dangling GitHub-Pages pricing links), security headers, the proxy, and the
guarantee that the gateway never imports torch (so it fits Vercel's function
size limit)."""

import sys

import pytest
from fastapi.testclient import TestClient

import vercel_gateway


@pytest.fixture
def client():
    return TestClient(vercel_gateway.app)


class TestLandingAndPages:
    def test_landing_ok_and_self_contained_pricing(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "Self" in r.text and "Pricing" in r.text
        assert 'id="pricing"' in r.text
        # Regression: the old landing linked "Get an API key" etc. at the
        # GitHub-Pages /pricing/ page, which 404s on main. It must not anymore.
        assert "github.io/SelfLLM/pricing" not in r.text
        assert 'href="/#pricing"' in r.text

    def test_quickstart_snippet_present(self, client):
        assert "/v1/chat/completions" in client.get("/").text

    def test_health(self, client):
        assert client.get("/health").json()["role"] == "gateway"

    def test_pricing_json_backward_compatible(self, client):
        tiers = {t["name"] for t in client.get("/pricing").json()["tiers"]}
        assert tiers == {"free", "pro", "enterprise"}


class TestNoMore404s:
    def test_favicon_served(self, client):
        r = client.get("/favicon.ico")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("image/svg")

    def test_robots_served(self, client):
        r = client.get("/robots.txt")
        assert r.status_code == 200
        assert "Disallow: /v1/" in r.text

    def test_unknown_path_html_404_is_branded(self, client):
        r = client.get("/does-not-exist", headers={"accept": "text/html"})
        assert r.status_code == 404
        assert "Back to SelfLLM" in r.text
        assert "<title>404" in r.text

    def test_unknown_path_json_404_is_structured(self, client):
        r = client.get("/does-not-exist", headers={"accept": "application/json"})
        assert r.status_code == 404
        body = r.json()
        assert body["error"]["type"] == "http_error"
        assert body["error"]["status"] == 404


class TestSecurityHeaders:
    @pytest.mark.parametrize("path", ["/", "/health", "/pricing", "/robots.txt"])
    def test_headers_present_on_all_surfaces(self, client, path):
        h = client.get(path).headers
        assert h["X-Content-Type-Options"] == "nosniff"
        assert h["X-Frame-Options"] == "SAMEORIGIN"
        assert "Content-Security-Policy" in h
        assert "Strict-Transport-Security" in h


class TestProxyHardening:
    def test_body_too_large_is_413(self, client, monkeypatch):
        monkeypatch.setattr(vercel_gateway, "MAX_PROXY_BODY_BYTES", 8)
        r = client.post("/v1/chat/completions", content=b"x" * 100)
        assert r.status_code == 413
        assert r.json()["error"]["type"] == "payload_too_large"

    def test_proxy_forwards_auth_header(self, client, monkeypatch):
        captured = {}

        class _Resp:
            status_code = 200
            content = b'{"ok": true}'
            headers = {"content-type": "application/json"}

        class _Client:
            def __init__(self, **kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def request(self, method, url, **kw):
                captured["url"] = url
                captured["headers"] = kw.get("headers", {})
                return _Resp()

        monkeypatch.setattr(vercel_gateway.httpx, "AsyncClient", _Client)
        r = client.get("/v1/models", headers={"Authorization": "Bearer sk-x"})
        assert r.status_code == 200
        assert captured["url"].endswith("/v1/models")
        assert captured["headers"].get("authorization") == "Bearer sk-x"

    def test_proxy_upstream_failure_is_502(self, client, monkeypatch):
        import httpx as httpx_module

        class _Failing:
            def __init__(self, **kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def request(self, *a, **kw):
                raise httpx_module.ConnectError("down")

        monkeypatch.setattr(vercel_gateway.httpx, "AsyncClient", _Failing)
        r = client.post("/v1/chat/completions", json={"messages": []})
        assert r.status_code == 502
        assert r.json()["error"]["type"] == "upstream_unavailable"


def test_gateway_source_is_torch_free():
    import inspect

    src = inspect.getsource(vercel_gateway)
    assert "import torch" not in src
    assert "import selfllm" not in src


def test_gateway_import_does_not_pull_torch():
    # The gateway must stay torch-free to fit Vercel's 500 MB function limit.
    # Check in a clean subprocess, since other tests in this process load torch.
    import subprocess

    code = (
        "import vercel_gateway, sys; "
        "assert 'torch' not in sys.modules; "
        "print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
