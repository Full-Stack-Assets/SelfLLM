"""Tests for the Prometheus metrics collector and /metrics endpoints."""


import pytest

from selfllm.serving.metrics import DEFAULT_BUCKETS, MetricsCollector


class TestMetricsCollector:
    def test_request_counter(self):
        m = MetricsCollector()
        m.observe_request("/v1/chat/completions", "POST", 200, 0.02)
        m.observe_request("/v1/chat/completions", "POST", 200, 0.03)
        m.observe_request("/v1/chat/completions", "POST", 402, 0.01)
        text = m.render()
        assert 'status="200"} 2' in text
        assert 'status="402"} 1' in text
        assert "# TYPE selfllm_requests_total counter" in text

    def test_latency_histogram_is_cumulative(self):
        m = MetricsCollector()
        m.observe_request("/e", "GET", 200, 0.03)  # falls in le=0.05
        m.observe_request("/e", "GET", 200, 2.0)   # falls in le=2.5
        text = m.render()
        # le=0.05 bucket holds only the fast request...
        assert 'le="0.05"} 1' in text
        # ...le=2.5 holds both (cumulative)...
        assert 'le="2.5"} 2' in text
        assert 'le="+Inf"} 2' in text
        assert '_count{endpoint="/e"} 2' in text
        assert '_sum{endpoint="/e"} 2.03' in text

    def test_in_flight_gauge(self):
        m = MetricsCollector()
        m.inc_in_flight()
        m.inc_in_flight()
        assert "selfllm_requests_in_flight 2" in m.render()
        m.dec_in_flight()
        assert "selfllm_requests_in_flight 1" in m.render()

    def test_in_flight_never_negative(self):
        m = MetricsCollector()
        m.dec_in_flight()
        assert "selfllm_requests_in_flight 0" in m.render()

    def test_label_escaping(self):
        m = MetricsCollector()
        m.observe_request('/weird"path', "GET", 200, 0.01)
        text = m.render()
        assert '\\"path' in text  # quote escaped, render stays valid

    def test_reset(self):
        m = MetricsCollector()
        m.observe_request("/e", "GET", 200, 0.01)
        m.reset()
        snap = m.snapshot()
        assert snap["requests"] == {}
        assert snap["in_flight"] == 0

    def test_default_buckets_sorted(self):
        assert list(DEFAULT_BUCKETS) == sorted(DEFAULT_BUCKETS)


class TestMetricsEndpoint:
    @pytest.fixture
    def client(self, monkeypatch):
        from fastapi.testclient import TestClient
        import selfllm.serving.server as server_module

        server_module.METRICS.reset()
        monkeypatch.setattr(server_module, "_model", None)
        monkeypatch.setattr(server_module, "_tokenizer", None)
        monkeypatch.setattr(server_module, "_scheduler", None)
        monkeypatch.setattr(server_module, "_api_key", None)
        return TestClient(server_module.app)

    def test_metrics_endpoint_served_and_records_traffic(self, client):
        # Generate a couple of requests through the middleware first.
        client.get("/health")
        client.get("/health")
        r = client.get("/metrics")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/plain")
        body = r.text
        assert "selfllm_requests_total" in body
        # The middleware labelled by route template, so /health appears.
        assert 'endpoint="/health"' in body

    def test_middleware_records_error_status(self, client, monkeypatch):
        import selfllm.serving.server as server_module

        # No model -> chat completions returns 503; the middleware should
        # still record it with the real status code.
        server_module.METRICS.reset()
        client.post(
            "/v1/chat/completions",
            json={"model": "selfllm",
                  "messages": [{"role": "user", "content": "hi"}]},
        )
        body = client.get("/metrics").text
        assert 'endpoint="/v1/chat/completions"' in body
        assert 'status="503"' in body


class TestGatewayMetrics:
    def test_gateway_proxies_upstream_metrics(self, monkeypatch):
        from fastapi.testclient import TestClient
        import vercel_gateway

        class _Resp:
            status_code = 200
            content = b"selfllm_requests_total 5\n"
            headers = {"content-type": "text/plain; version=0.0.4"}

        class _Client:
            def __init__(self, **kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def get(self, url):
                assert url.endswith("/metrics")
                return _Resp()

        monkeypatch.setattr(vercel_gateway.httpx, "AsyncClient", _Client)
        r = TestClient(vercel_gateway.app).get("/metrics")
        assert r.status_code == 200
        assert "selfllm_requests_total 5" in r.text
