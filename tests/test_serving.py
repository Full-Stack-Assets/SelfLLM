"""Tests for the serving system (scheduler + server)."""

import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from selfllm.serving.paged_cache import BlockManager
from selfllm.serving.scheduler import ContinuousBatchingScheduler, Request


class _StubModel:
    """Minimal model exposing the ``forward(...)`` contract the scheduler needs.

    Returns deterministic logits (peaked at ``force_token`` so greedy decoding
    is predictable) and correctly-shaped KV caches, without a real transformer.
    """

    def __init__(self, vocab=64, n_layers=2, n_heads=2, head_dim=8, force_token=7):
        self.vocab = vocab
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.force_token = force_token
        self.config = SimpleNamespace(
            max_seq_len=128, vocab_size=vocab, n_layers=n_layers,
            n_heads=n_heads, d_model=n_heads * head_dim,
        )

    def parameters(self):
        yield torch.nn.Parameter(torch.zeros(1))

    def forward(self, token_ids, past_key_values=None, use_cache=False,
                positions=None, key_padding_mask=None, targets=None):
        B, T = token_ids.shape
        logits = torch.zeros(B, T, self.vocab)
        if self.force_token is not None:
            logits[:, :, self.force_token] = 10.0
        result = {"logits": logits, "hidden_states": torch.zeros(B, T, self.config.d_model)}
        if use_cache:
            cl = past_key_values[0][0].shape[2] if past_key_values is not None else 0
            nl = cl + T
            result["past_key_values"] = [
                (torch.zeros(B, self.n_heads, nl, self.head_dim),
                 torch.zeros(B, self.n_heads, nl, self.head_dim))
                for _ in range(self.n_layers)
            ]
        return result

    __call__ = forward


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_model():
    """Create a mock model with a generate() method (for server fallback path)."""
    model = MagicMock()
    model.generate.return_value = {
        "sequences": torch.tensor([[42, 99]]),  # 42 is prompt, 99 is generated
    }
    # Mock parameters() for device detection — must return an iterator
    param = MagicMock()
    param.device = torch.device("cpu")
    model.parameters.return_value = iter([param])
    return model


@pytest.fixture
def stub_model():
    """Deterministic stub model implementing forward() for scheduler tests."""
    return _StubModel(force_token=7)


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 1
    tokenizer.encode.return_value = [10, 20, 30]
    tokenizer.decode.return_value = "hello world"
    return tokenizer


@pytest.fixture
def block_manager():
    """Create a BlockManager for testing."""
    return BlockManager(
        num_blocks=20,
        block_size=4,
        num_layers=2,
        num_heads=4,
        head_dim=32,
        dtype=torch.float32,
        device="cpu",
    )


@pytest.fixture
def scheduler(stub_model, mock_tokenizer, block_manager):
    """Create a ContinuousBatchingScheduler for testing."""
    return ContinuousBatchingScheduler(
        model=stub_model,
        tokenizer=mock_tokenizer,
        block_manager=block_manager,
        max_batch_size=4,
        device="cpu",
    )


# ---------------------------------------------------------------------------
# Request tests
# ---------------------------------------------------------------------------


class TestRequest:
    """Tests for the Request dataclass."""

    def test_request_creation(self):
        """Request can be created with default values."""
        req = Request(id="test-1", prompt_tokens=[1, 2, 3])
        assert req.id == "test-1"
        assert req.prompt_tokens == [1, 2, 3]
        assert req.max_new_tokens == 256
        assert req.temperature == 1.0
        assert req.top_p == 1.0
        assert req.status == "waiting"
        assert req.generated_tokens == []
        assert not req.is_finished

    def test_request_total_tokens(self):
        """total_tokens counts prompt + generated."""
        req = Request(id="test-1", prompt_tokens=[1, 2, 3], generated_tokens=[4, 5])
        assert req.total_tokens == 5

    def test_request_is_finished(self):
        """is_finished is True when status is 'finished'."""
        req = Request(id="test-1", prompt_tokens=[1, 2, 3], status="finished")
        assert req.is_finished

    def test_request_latency(self):
        """latency is computed correctly when finished."""
        now = time.time()
        req = Request(
            id="test-1",
            prompt_tokens=[1, 2, 3],
            status="finished",
            created_at=now,
            finished_at=now + 1.5,
        )
        assert req.latency is not None
        assert abs(req.latency - 1.5) < 0.01

    def test_request_latency_not_finished(self):
        """latency is None when not finished."""
        req = Request(id="test-1", prompt_tokens=[1, 2, 3], status="running")
        assert req.latency is None


# ---------------------------------------------------------------------------
# Scheduler tests
# ---------------------------------------------------------------------------


class TestSchedulerInit:
    """Tests for ContinuousBatchingScheduler initialization."""

    def test_scheduler_init(self, scheduler, stub_model, mock_tokenizer, block_manager):
        """Scheduler initializes correctly."""
        assert scheduler.model is stub_model
        assert scheduler.tokenizer is mock_tokenizer
        assert scheduler.block_manager is block_manager
        assert scheduler.max_batch_size == 4
        assert scheduler.num_active_requests == 0

    def test_scheduler_empty_stats(self, scheduler):
        """Stats reflect empty state."""
        stats = scheduler.stats
        assert stats["active"] == 0
        assert stats["waiting"] == 0
        assert stats["completed"] == 0
        assert stats["free_blocks"] == 20
        assert stats["total_served"] == 0


class TestSchedulerAddRequest:
    """Tests for adding requests to the scheduler."""

    def test_add_request(self, scheduler):
        """add_request() adds to waiting queue."""
        req = Request(id="test-1", prompt_tokens=[1, 2, 3])
        scheduler.add_request(req)
        assert scheduler.stats["waiting"] == 1
        assert scheduler.stats["active"] == 0

    def test_create_request(self, scheduler):
        """create_request() creates and enqueues a request."""
        req = scheduler.create_request(prompt_tokens=[1, 2, 3], max_new_tokens=10)
        assert isinstance(req, Request)
        assert req.prompt_tokens == [1, 2, 3]
        assert req.max_new_tokens == 10
        assert scheduler.stats["waiting"] == 1


class TestSchedulerStep:
    """Tests for scheduler.step()."""

    def test_step_moves_waiting_to_active(self, scheduler):
        """step() moves waiting requests to active."""
        req = scheduler.create_request(prompt_tokens=[1, 2, 3])
        assert scheduler.stats["waiting"] == 1

        scheduler.step()
        assert scheduler.stats["active"] == 1
        assert scheduler.stats["waiting"] == 0
        assert req.status == "running"

    def test_step_generates_token(self, scheduler):
        """step() generates a token for active requests."""
        req = scheduler.create_request(
            prompt_tokens=[1, 2, 3], max_new_tokens=5, temperature=0.0
        )
        scheduler.step()  # Move to active and prefill (first token)

        assert len(req.generated_tokens) == 1
        # Greedy decoding against the stub picks force_token deterministically.
        assert req.generated_tokens[0] == scheduler.model.force_token

    def test_step_completes_on_eos(self, scheduler, mock_tokenizer):
        """step() marks request finished on EOS token."""
        # Make the stub emit the EOS token deterministically (greedy).
        mock_tokenizer.eos_token_id = scheduler.model.force_token
        req = scheduler.create_request(
            prompt_tokens=[1, 2, 3], max_new_tokens=5, temperature=0.0
        )

        scheduler.step()  # Prefill emits the EOS token
        assert req.is_finished
        assert req.finish_reason == "stop"
        assert scheduler.stats["active"] == 0
        assert scheduler.stats["completed"] == 1

    def test_step_completes_on_max_length(self, scheduler):
        """step() marks request finished when max_new_tokens reached."""
        req = scheduler.create_request(prompt_tokens=[1, 2, 3], max_new_tokens=1)
        scheduler.step()

        # After generating 1 token, should be finished due to max length
        assert req.is_finished
        assert req.finish_reason == "length"

    def test_step_no_requests(self, scheduler):
        """step() with no requests returns empty list."""
        completed = scheduler.step()
        assert completed == []

    def test_step_multiple_requests(self, scheduler):
        """step() handles multiple active requests."""
        req1 = scheduler.create_request(prompt_tokens=[1, 2], max_new_tokens=1)
        req2 = scheduler.create_request(prompt_tokens=[3, 4], max_new_tokens=1)

        scheduler.step()

        assert scheduler.stats["completed"] == 2
        assert len(req1.generated_tokens) == 1
        assert len(req2.generated_tokens) == 1

    def test_step_respects_max_batch_size(self, scheduler):
        """step() respects max_batch_size limit."""
        scheduler.max_batch_size = 2

        # Create 4 requests
        for i in range(4):
            scheduler.create_request(prompt_tokens=[i], max_new_tokens=1)

        assert scheduler.stats["waiting"] == 4
        scheduler.step()

        # Only 2 should be active/completed, 2 still waiting
        assert scheduler.stats["completed"] == 2
        assert scheduler.stats["waiting"] == 2


class TestSchedulerRun:
    """Tests for scheduler.run() (continuous loop)."""

    def test_run_and_shutdown(self, scheduler):
        """run() processes requests and shutdown() stops it."""
        req = scheduler.create_request(prompt_tokens=[1, 2, 3], max_new_tokens=1)

        # Run in background thread
        thread = threading.Thread(target=scheduler.run)
        thread.start()

        # Wait for request to complete
        timeout = 5.0
        start = time.time()
        while not req.is_finished and (time.time() - start) < timeout:
            time.sleep(0.05)

        scheduler.shutdown()
        thread.join(timeout=2.0)

        assert req.is_finished

    def test_run_idle_sleep(self, scheduler):
        """run() sleeps when idle."""
        thread = threading.Thread(target=scheduler.run)
        thread.start()

        time.sleep(0.05)  # Let it enter idle loop

        scheduler.shutdown()
        thread.join(timeout=2.0)

        assert not thread.is_alive()


class TestSchedulerGetRequestResult:
    """Tests for scheduler.get_request_result()."""

    def test_get_active_request(self, scheduler):
        """get_request_result() finds active requests."""
        req = scheduler.create_request(prompt_tokens=[1, 2, 3], max_new_tokens=5)
        scheduler.step()  # Moves to active

        found = scheduler.get_request_result(req.id)
        assert found is not None
        assert found.id == req.id

    def test_get_waiting_request(self, scheduler):
        """get_request_result() finds waiting requests."""
        req = scheduler.create_request(prompt_tokens=[1, 2, 3])

        found = scheduler.get_request_result(req.id)
        assert found is not None
        assert found.id == req.id

    def test_get_completed_request(self, scheduler):
        """get_request_result() finds completed requests."""
        req = scheduler.create_request(prompt_tokens=[1, 2, 3], max_new_tokens=1)
        scheduler.step()  # Completes immediately

        found = scheduler.get_request_result(req.id)
        assert found is not None
        assert found.id == req.id
        assert found.is_finished

    def test_get_missing_request(self, scheduler):
        """get_request_result() returns None for unknown IDs."""
        found = scheduler.get_request_result("nonexistent")
        assert found is None


class TestSchedulerStats:
    """Tests for scheduler.stats."""

    def test_stats_with_active(self, scheduler):
        """stats reflect active requests."""
        scheduler.create_request(prompt_tokens=[1, 2, 3], max_new_tokens=5)
        scheduler.step()

        stats = scheduler.stats
        assert stats["active"] == 1
        assert stats["waiting"] == 0
        assert stats["free_blocks"] < 20  # Some blocks allocated

    def test_stats_with_waiting(self, scheduler):
        """stats reflect waiting requests."""
        scheduler.create_request(prompt_tokens=[1, 2, 3])

        stats = scheduler.stats
        assert stats["active"] == 0
        assert stats["waiting"] == 1

    def test_stats_total_served(self, scheduler):
        """total_served increments as requests complete."""
        scheduler.create_request(prompt_tokens=[1, 2, 3], max_new_tokens=1)
        scheduler.step()

        assert scheduler.stats["total_served"] == 1

        scheduler.create_request(prompt_tokens=[4, 5], max_new_tokens=1)
        scheduler.step()

        assert scheduler.stats["total_served"] == 2

    def test_stats_after_free(self, scheduler):
        """Free blocks return to pool after request completes."""
        initial_free = scheduler.block_manager.num_free_blocks
        scheduler.create_request(prompt_tokens=[1, 2, 3], max_new_tokens=1)
        scheduler.step()

        # After completion, blocks should be freed
        assert scheduler.block_manager.num_free_blocks == initial_free


# ---------------------------------------------------------------------------
# Server endpoint tests (using FastAPI TestClient)
# ---------------------------------------------------------------------------


class TestServerEndpoints:
    """Tests for FastAPI server endpoints.

    These tests use httpx's TestClient to test the FastAPI app without
    actually starting a server.
    """

    @pytest.fixture
    def client(self, mock_model, mock_tokenizer, monkeypatch):
        """Create a TestClient with mocked global state."""
        from fastapi.testclient import TestClient
        from selfllm.serving.server import app

        # We need to reload the module to pick up the app
        import selfllm.serving.server as server_module

        # Set the globals
        monkeypatch.setattr(server_module, "_model", mock_model)
        monkeypatch.setattr(server_module, "_tokenizer", mock_tokenizer)
        monkeypatch.setattr(server_module, "_scheduler", None)

        return TestClient(app)

    def test_health_endpoint_no_model(self):
        """/health returns not loaded when model is None."""
        from fastapi.testclient import TestClient
        from selfllm.serving.server import app

        with TestClient(app) as client:
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["model_loaded"] is False

    def test_health_endpoint_with_model(self, mock_model, monkeypatch):
        """/health returns loaded when model is set."""
        from fastapi.testclient import TestClient
        from selfllm.serving.server import app
        import selfllm.serving.server as server_module

        monkeypatch.setattr(server_module, "_model", mock_model)

        with TestClient(app) as client:
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["model_loaded"] is True

    def test_models_endpoint(self):
        """/v1/models lists available models."""
        from fastapi.testclient import TestClient
        from selfllm.serving.server import app

        with TestClient(app) as client:
            response = client.get("/v1/models")
            assert response.status_code == 200
            data = response.json()
            assert data["object"] == "list"
            assert len(data["data"]) == 1
            assert data["data"][0]["id"] == "selfllm"

    def test_stats_endpoint_no_scheduler(self):
        """/v1/stats returns error when scheduler not initialized."""
        from fastapi.testclient import TestClient
        from selfllm.serving.server import app

        with TestClient(app) as client:
            response = client.get("/v1/stats")
            assert response.status_code == 200
            data = response.json()
            assert "error" in data

    def test_stats_endpoint_with_scheduler(self, scheduler, monkeypatch):
        """/v1/stats returns scheduler stats when available."""
        from fastapi.testclient import TestClient
        from selfllm.serving.server import app
        import selfllm.serving.server as server_module

        monkeypatch.setattr(server_module, "_scheduler", scheduler)

        with TestClient(app) as client:
            response = client.get("/v1/stats")
            assert response.status_code == 200
            data = response.json()
            assert "active" in data
            assert "waiting" in data
            assert "free_blocks" in data

    def test_chat_completions_no_model(self):
        """/v1/chat/completions returns 503 when model not loaded."""
        from fastapi.testclient import TestClient
        from selfllm.serving.server import app

        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "selfllm",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )
            assert response.status_code == 503
            assert "Model not loaded" in response.json()["detail"]

    def test_chat_completions_with_model(self, mock_model, monkeypatch):
        """/v1/chat/completions returns valid response."""
        from fastapi.testclient import TestClient
        from selfllm.serving.server import app
        import selfllm.serving.server as server_module

        # Build a decode result where slicing off the prompt leaves the response
        prompt_text = "User: Hello\nAssistant:"
        full_text = prompt_text + " hello world"

        mock_tok = MagicMock()
        mock_tok.eos_token_id = 1
        mock_tok.encode = MagicMock(return_value=[10, 20, 30])
        mock_tok.decode = MagicMock(return_value=full_text)

        monkeypatch.setattr(server_module, "_model", mock_model)
        monkeypatch.setattr(server_module, "_tokenizer", mock_tok)

        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "selfllm",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 10,
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert data["object"] == "chat.completion"
            assert data["model"] == "selfllm"
            assert len(data["choices"]) == 1
            assert data["choices"][0]["message"]["role"] == "assistant"
            assert "hello" in data["choices"][0]["message"]["content"]
            assert "usage" in data
            assert "prompt_tokens" in data["usage"]
            assert "completion_tokens" in data["usage"]

    def test_completions_no_model(self):
        """/v1/completions returns 503 when model not loaded."""
        from fastapi.testclient import TestClient
        from selfllm.serving.server import app

        with TestClient(app) as client:
            response = client.post(
                "/v1/completions",
                json={"model": "selfllm", "prompt": "Hello"},
            )
            assert response.status_code == 503

    def test_completions_with_model(self, mock_model, monkeypatch):
        """/v1/completions returns valid response."""
        from fastapi.testclient import TestClient
        from selfllm.serving.server import app
        import selfllm.serving.server as server_module

        mock_tok = MagicMock()
        mock_tok.eos_token_id = 1
        mock_tok.encode = MagicMock(return_value=[10, 20, 30])
        mock_tok.decode = MagicMock(return_value="Hello world result")

        monkeypatch.setattr(server_module, "_model", mock_model)
        monkeypatch.setattr(server_module, "_tokenizer", mock_tok)

        with TestClient(app) as client:
            response = client.post(
                "/v1/completions",
                json={"model": "selfllm", "prompt": "Hello", "max_tokens": 10},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["object"] == "text_completion"
            assert "choices" in data
            assert len(data["choices"]) == 1
            assert "usage" in data

    def test_chat_completions_validation(self, mock_model, monkeypatch):
        """/v1/chat/completions validates request parameters."""
        from fastapi.testclient import TestClient
        from selfllm.serving.server import app
        import selfllm.serving.server as server_module

        monkeypatch.setattr(server_module, "_model", mock_model)
        monkeypatch.setattr(server_module, "_tokenizer", MagicMock(
            eos_token_id=1,
            encode=MagicMock(return_value=[10, 20]),
            decode=MagicMock(return_value="test"),
        ))

        with TestClient(app) as client:
            # max_tokens must be >= 1
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "selfllm",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 0,
                },
            )
            assert response.status_code == 422  # Validation error

            # temperature must be >= 0
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "selfllm",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "temperature": -1.0,
                },
            )
            assert response.status_code == 422  # Validation error

    def test_api_key_auth(self, monkeypatch):
        """When an API key is configured, inference endpoints require Bearer auth."""
        from fastapi.testclient import TestClient
        import selfllm.serving.server as server_module

        monkeypatch.setattr(server_module, "_api_key", "secret-key")
        monkeypatch.setattr(server_module, "_model", None)
        monkeypatch.setattr(server_module, "_tokenizer", None)
        monkeypatch.setattr(server_module, "_scheduler", None)

        body = {"model": "selfllm", "messages": [{"role": "user", "content": "hi"}]}
        with TestClient(server_module.app) as client:
            # No key -> 401
            assert client.post("/v1/chat/completions", json=body).status_code == 401
            # Wrong key -> 401
            assert client.post(
                "/v1/chat/completions", json=body,
                headers={"Authorization": "Bearer nope"},
            ).status_code == 401
            # Correct key -> passes auth (503 because no model is loaded)
            assert client.post(
                "/v1/chat/completions", json=body,
                headers={"Authorization": "Bearer secret-key"},
            ).status_code == 503
            # /v1/models is also protected
            assert client.get("/v1/models").status_code == 401
            assert client.get(
                "/v1/models", headers={"Authorization": "Bearer secret-key"},
            ).status_code == 200
            # /health stays open
            assert client.get("/health").status_code == 200

    def test_no_api_key_is_open(self, monkeypatch):
        """With no API key configured, endpoints are open (no auth required)."""
        from fastapi.testclient import TestClient
        import selfllm.serving.server as server_module

        monkeypatch.setattr(server_module, "_api_key", None)
        monkeypatch.setattr(server_module, "_model", None)
        monkeypatch.setattr(server_module, "_tokenizer", None)
        with TestClient(server_module.app) as client:
            # No auth header, but unprotected -> reaches handler -> 503 (no model)
            assert client.get("/v1/models").status_code == 200
            assert client.post(
                "/v1/chat/completions",
                json={"model": "selfllm", "messages": [{"role": "user", "content": "hi"}]},
            ).status_code == 503

    def test_chat_completions_reasoning_knob(self, monkeypatch):
        """/v1/chat/completions with a `reasoning` option runs a strategy."""
        from fastapi.testclient import TestClient
        import selfllm.serving.server as server_module
        from selfllm.model.config import ModelConfig
        from selfllm.model.model import SelfImprovingLLM
        from selfllm.model.tokenizer import BPETokenizer

        cfg = ModelConfig(vocab_size=256, d_model=64, n_layers=2, n_heads=4,
                          d_ff=128, max_seq_len=128, dropout=0.0)
        model = SelfImprovingLLM(cfg).eval()
        tok = BPETokenizer(vocab_size=256)
        tok.train(["The answer is 4.", "Step by step reasoning to a result."])

        monkeypatch.setattr(server_module, "_model", model)
        monkeypatch.setattr(server_module, "_tokenizer", tok)
        monkeypatch.setattr(server_module, "_scheduler", None)

        with TestClient(server_module.app) as client:
            resp = client.post("/v1/chat/completions", json={
                "model": "selfllm",
                "messages": [{"role": "user", "content": "What is 2 + 2?"}],
                "reasoning": {"strategy": "self_consistency", "num_samples": 2,
                              "answer_type": "free", "max_new_tokens": 6},
            })
            assert resp.status_code == 200
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            assert isinstance(content, str)

    def test_chat_completions_via_scheduler(
        self, stub_model, mock_tokenizer, block_manager, monkeypatch
    ):
        """/v1/chat/completions routes through the continuous-batching scheduler."""
        from fastapi.testclient import TestClient
        import selfllm.serving.server as server_module

        sched = ContinuousBatchingScheduler(
            model=stub_model, tokenizer=mock_tokenizer,
            block_manager=block_manager, max_batch_size=4, device="cpu",
        )
        thread = threading.Thread(target=sched.run, daemon=True)
        thread.start()

        monkeypatch.setattr(server_module, "_model", stub_model)
        monkeypatch.setattr(server_module, "_tokenizer", mock_tokenizer)
        monkeypatch.setattr(server_module, "_scheduler", sched)

        try:
            with TestClient(server_module.app) as client:
                response = client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "selfllm",
                        "messages": [{"role": "user", "content": "Hi"}],
                        "max_tokens": 3,
                    },
                )
                assert response.status_code == 200
                data = response.json()
                assert data["choices"][0]["message"]["role"] == "assistant"
                assert data["usage"]["completion_tokens"] >= 1
                # The request actually flowed through the scheduler.
                assert sched.stats["total_served"] >= 1
        finally:
            sched.shutdown()
            thread.join(timeout=2.0)

    def test_streaming_chat_completions(self, mock_model, monkeypatch):
        """/v1/chat/completions with stream=True returns SSE."""
        from fastapi.testclient import TestClient
        from selfllm.serving.server import app
        import selfllm.serving.server as server_module

        mock_tok = MagicMock()
        mock_tok.eos_token_id = 1
        mock_tok.encode = MagicMock(return_value=[10, 20])
        mock_tok.decode = MagicMock(return_value="t")

        monkeypatch.setattr(server_module, "_model", mock_model)
        monkeypatch.setattr(server_module, "_tokenizer", mock_tok)

        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "selfllm",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 1,
                    "stream": True,
                },
            )
            assert response.status_code == 200
            assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

            content = response.content.decode()
            assert "data:" in content
            assert "[DONE]" in content


# ---------------------------------------------------------------------------
# End-to-end integration tests
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """End-to-end integration tests for the full serving pipeline."""

    def test_full_request_lifecycle(self, scheduler):
        """Full lifecycle: waiting -> running -> finished."""
        req = scheduler.create_request(
            prompt_tokens=[1, 2, 3], max_new_tokens=2
        )
        assert req.status == "waiting"

        # First step: waiting -> running
        scheduler.step()
        assert req.status == "running"
        assert len(req.block_ids) > 0  # Blocks allocated

        # Second step: still running
        scheduler.step()
        assert len(req.generated_tokens) == 2

        # Third step: finished (max length reached)
        scheduler.step()
        assert req.status == "finished"
        assert req.is_finished
        assert req.finished_at is not None
        assert req.finish_reason == "length"

    def test_multiple_requests_interleaved(self, scheduler):
        """Multiple requests are processed concurrently."""
        req1 = scheduler.create_request(prompt_tokens=[1, 2], max_new_tokens=2)
        req2 = scheduler.create_request(prompt_tokens=[3, 4, 5], max_new_tokens=2)

        # Both should be moved to active
        scheduler.step()
        assert scheduler.stats["active"] == 2

        # Continue until completion
        for _ in range(5):
            scheduler.step()
            if scheduler.stats["completed"] == 2:
                break

        assert scheduler.stats["completed"] == 2
        assert len(req1.generated_tokens) == 2
        assert len(req2.generated_tokens) == 2

    def test_blocks_freed_on_completion(self, scheduler):
        """Blocks are freed when a request completes."""
        initial_free = scheduler.block_manager.num_free_blocks

        scheduler.create_request(prompt_tokens=[1, 2, 3], max_new_tokens=1)
        scheduler.step()  # Completes, frees blocks

        # All blocks should be returned
        assert scheduler.block_manager.num_free_blocks == initial_free

    def test_request_with_zero_temperature(self, scheduler):
        """Request with temperature=0 triggers the greedy decoding path."""
        req = scheduler.create_request(
            prompt_tokens=[1, 2, 3], max_new_tokens=1, temperature=0.0
        )
        scheduler.step()
        assert req.is_finished
        assert req.generated_tokens[0] == scheduler.model.force_token

    def test_request_ordering(self, scheduler):
        """Requests are processed in FIFO order."""
        req1 = scheduler.create_request(prompt_tokens=[1], max_new_tokens=1)
        req2 = scheduler.create_request(prompt_tokens=[2], max_new_tokens=1)
        req3 = scheduler.create_request(prompt_tokens=[3], max_new_tokens=1)

        # All should be completed
        for _ in range(5):
            scheduler.step()
            if scheduler.stats["completed"] == 3:
                break

        # Check completion order
        assert scheduler.completed_requests[0].id == req1.id
        assert scheduler.completed_requests[1].id == req2.id
        assert scheduler.completed_requests[2].id == req3.id


# ---------------------------------------------------------------------------
# Streaming-window (StreamingLLM) KV eviction in the scheduler
# ---------------------------------------------------------------------------


class TestStreamingWindowEviction:
    """A sliding-window model keeps each request's decode cache bounded so the
    server can hold unbounded / long-running conversations."""

    def _build(self, sliding_window, sinks, max_seq_len=16, max_batch_size=4):
        """Build a real tiny windowed model + scheduler on CPU."""
        from selfllm.model.config import ModelConfig
        from selfllm.model.model import SelfImprovingLLM

        cfg = ModelConfig(
            vocab_size=64,
            d_model=32,
            n_layers=2,
            n_heads=2,
            d_ff=64,
            max_seq_len=max_seq_len,
            dropout=0.0,
            sliding_window=sliding_window,
            attention_sinks=sinks,
        )
        model = SelfImprovingLLM(cfg).eval()
        tok = MagicMock()
        tok.eos_token_id = -1  # never emitted -> request runs to its length cap
        bm = BlockManager(
            num_blocks=64, block_size=4, num_layers=cfg.n_layers,
            num_heads=cfg.n_heads, head_dim=cfg.d_model // cfg.n_heads,
            dtype=torch.float32, device="cpu",
        )
        sched = ContinuousBatchingScheduler(
            model, tok, bm, max_batch_size=max_batch_size, device="cpu"
        )
        return sched, cfg

    def test_cache_stays_bounded_while_position_grows(self):
        """cache_len never exceeds sinks+window even as abs_pos grows past it
        and past the model's max_seq_len (RoPE saturation)."""
        sched, cfg = self._build(sliding_window=8, sinks=2, max_seq_len=16)
        budget = cfg.sliding_window + cfg.attention_sinks  # 10

        req = sched.create_request(
            prompt_tokens=[3, 4, 5], max_new_tokens=40, temperature=0.0
        )
        max_abs_pos = 0
        steps = 0
        while not req.is_finished and steps < 200:
            sched.step()
            steps += 1
            state = sched._req_cache.get(req.id)
            if state is not None:
                # Physical cache memory is bounded by the window budget.
                assert state["cache_len"] <= budget
                max_abs_pos = max(max_abs_pos, state["abs_pos"])

        assert req.is_finished
        assert len(req.generated_tokens) == 40
        # Absolute position climbed well past both the window budget and the
        # model's max_seq_len -- proving position is decoupled from cache size.
        assert max_abs_pos > budget
        assert max_abs_pos > cfg.max_seq_len

    def test_no_eviction_when_window_unset(self):
        """Without a sliding window the cache grows with the sequence and the
        scheduler behaves exactly as before (abs_pos == cache_len)."""
        sched, _ = self._build(sliding_window=None, sinks=0, max_seq_len=64)
        assert sched._use_evict is False

        req = sched.create_request(
            prompt_tokens=[3, 4, 5], max_new_tokens=10, temperature=0.0
        )
        seen = []
        while not req.is_finished:
            sched.step()
            state = sched._req_cache.get(req.id)
            if state is not None:
                # No eviction: physical length equals the absolute position.
                assert state["cache_len"] == state["abs_pos"]
                seen.append(state["cache_len"])
        assert req.is_finished
        assert seen == sorted(seen)  # monotonically non-decreasing
        assert seen[-1] > 3  # grew past the prompt length

    def test_concurrent_windowed_requests(self):
        """Several windowed requests of differing lengths decode together and
        all stay bounded -- exercises the batched, right-padded decode path."""
        sched, cfg = self._build(sliding_window=6, sinks=1, max_seq_len=16)
        budget = cfg.sliding_window + cfg.attention_sinks

        reqs = [
            sched.create_request(
                prompt_tokens=[2, 3], max_new_tokens=25, temperature=0.0
            ),
            sched.create_request(
                prompt_tokens=[4, 5, 6, 7], max_new_tokens=30, temperature=0.0
            ),
            sched.create_request(
                prompt_tokens=[8], max_new_tokens=20, temperature=0.0
            ),
        ]
        steps = 0
        while not all(r.is_finished for r in reqs) and steps < 300:
            sched.step()
            steps += 1
            for r in reqs:
                state = sched._req_cache.get(r.id)
                if state is not None:
                    assert state["cache_len"] <= budget

        assert all(r.is_finished for r in reqs)
        assert [len(r.generated_tokens) for r in reqs] == [25, 30, 20]
