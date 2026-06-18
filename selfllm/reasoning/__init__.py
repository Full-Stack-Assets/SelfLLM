"""Test-time-compute reasoning strategies (V6).

Spend extra inference compute to reason better. V6 Wave 1 ships the reasoning
core: a uniform :class:`ReasoningStrategy` interface, a shared answer-extraction
layer, and the chain-of-thought strategy. Later waves add self-consistency,
best-of-N with a verifier, and search.

    >>> from selfllm.reasoning import CoTStrategy
    >>> strategy = CoTStrategy(model, tokenizer, answer_type="numeric", device="cpu")
    >>> result = strategy.solve("What is 2 + 3?")
    >>> result.answer, result.confidence
"""

from .base import ReasoningResult, ReasoningStrategy
from .cot import CoTStrategy
from .extract import extract_answer, extract_tagged_answer, normalize

__all__ = [
    "ReasoningResult",
    "ReasoningStrategy",
    "CoTStrategy",
    "extract_answer",
    "extract_tagged_answer",
    "normalize",
]
