"""
Root pytest conftest — shared fixtures and CLI options for the SelfLLM test suite.

Fixtures provided
-----------------
tiny_model_config   ModelConfig  d_model=32, 2 layers — fast enough for unit tests
small_model_config  ModelConfig  d_model=64, 2 layers — for integration tests
tiny_model          SelfImprovingLLM  built from tiny_model_config, eval mode
tiny_tokenizer      BPETokenizer  minimal 256-vocab tokenizer trained on a short corpus
fixed_prompt_ids    torch.Tensor  [1, 8] deterministic token sequence

CLI options
-----------
--run-slow   include @pytest.mark.slow tests (excluded by default)
--device     "cpu" (default) or "cuda" — passed through to fixtures

All models are created on CPU. GPU tests are auto-skipped when CUDA is
unavailable, regardless of --device.
"""

from __future__ import annotations

import pytest
import torch

from selfllm.model.config import ModelConfig
from selfllm.model.model import SelfImprovingLLM
from selfllm.model.tokenizer import BPETokenizer


# ---------------------------------------------------------------------------
# CLI option hooks
# ---------------------------------------------------------------------------


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run tests marked @pytest.mark.slow (excluded by default).",
    )
    parser.addoption(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for model fixtures (default: cpu).",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip slow tests unless --run-slow is passed; skip gpu tests without CUDA."""
    skip_slow = pytest.mark.skip(reason="Pass --run-slow to run slow tests.")
    skip_gpu  = pytest.mark.skip(reason="No CUDA device available.")

    for item in items:
        if "slow" in item.keywords and not config.getoption("--run-slow"):
            item.add_marker(skip_slow)
        if "gpu" in item.keywords and not torch.cuda.is_available():
            item.add_marker(skip_gpu)


# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def tiny_model_config() -> ModelConfig:
    """Minimal config — fastest possible model for unit tests."""
    return ModelConfig(
        vocab_size=256,
        d_model=32,
        n_layers=2,
        n_heads=2,
        d_ff=64,
        max_seq_len=64,
        dropout=0.0,
        use_rope=False,
        use_swiglu=False,
    )


@pytest.fixture(scope="session")
def small_model_config() -> ModelConfig:
    """Small but realistic config for integration tests."""
    return ModelConfig(
        vocab_size=512,
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=128,
        max_seq_len=128,
        dropout=0.0,
        use_rope=True,
        use_swiglu=True,
    )


# ---------------------------------------------------------------------------
# Model fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def tiny_model(tiny_model_config: ModelConfig) -> SelfImprovingLLM:
    """Seeded tiny model in eval mode — shared across the whole test session."""
    torch.manual_seed(0)
    model = SelfImprovingLLM(tiny_model_config)
    model.eval()
    return model


@pytest.fixture()
def fresh_tiny_model(tiny_model_config: ModelConfig) -> SelfImprovingLLM:
    """Fresh tiny model instance per test — use when tests mutate model state."""
    torch.manual_seed(0)
    model = SelfImprovingLLM(tiny_model_config)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Tokenizer fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def tiny_tokenizer(tiny_model_config: ModelConfig) -> BPETokenizer:
    """
    BPETokenizer with vocab_size=256, trained on a short deterministic corpus.

    Training is done once per session so the fixture is fast on repeated runs.
    The vocabulary is large enough that most ASCII characters get their own token,
    making encode/decode tests predictable.
    """
    corpus = [
        "the quick brown fox jumps over the lazy dog",
        "hello world this is a test of the tokenizer",
        "once upon a time in a land far far away",
        "the science of artificial intelligence enables computers",
        "in the beginning there was light and it was good",
        "machine learning models improve with more data and training",
        "recursive self improvement requires careful evaluation of quality",
        "tokens are the basic unit of language model processing",
    ] * 8   # repeat so BPE has enough frequency signal
    tokenizer = BPETokenizer(vocab_size=tiny_model_config.vocab_size)
    tokenizer.train(corpus)
    return tokenizer


# ---------------------------------------------------------------------------
# Input fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def fixed_prompt_ids() -> torch.Tensor:
    """A deterministic [1, 8] prompt tensor — used across multiple test modules."""
    return torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)


@pytest.fixture(scope="session")
def device(request: pytest.FixtureRequest) -> str:
    """The device string from --device CLI option."""
    return request.config.getoption("--device")
