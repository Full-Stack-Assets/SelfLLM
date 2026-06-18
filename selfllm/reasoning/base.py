"""Reasoning strategy interface and result type (V6 test-time compute).

A :class:`ReasoningStrategy` spends extra inference compute to produce a better
answer to a question -- via chain-of-thought, self-consistency, best-of-N, or
search. Every strategy returns a uniform :class:`ReasoningResult` so they are
interchangeable in the eval harness and the serving stack.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class ReasoningResult:
    """The outcome of a reasoning strategy on a single question.

    Attributes:
        answer: The extracted final answer (normalized string), or ``None`` if
            nothing could be extracted.
        confidence: A ``[0, 1]`` confidence in the answer. For single-sample
            strategies this is ``1.0`` when an answer was extracted; for
            voting strategies it is the agreement fraction.
        traces: The full reasoning text of every sampled path (length matches
            ``num_samples``).
        num_samples: Number of reasoning paths generated.
        details: Free-form strategy-specific extras (per-path answers, vote
            counts, verifier scores, ...).
    """

    answer: Optional[str]
    confidence: float
    traces: List[str] = field(default_factory=list)
    num_samples: int = 1
    details: dict = field(default_factory=dict)

    def __str__(self) -> str:
        ans = "None" if self.answer is None else repr(self.answer)
        return f"ReasoningResult(answer={ans}, confidence={self.confidence:.3f}, n={self.num_samples})"


class ReasoningStrategy(ABC):
    """Abstract base class for a test-time reasoning strategy.

    Args:
        model: A model exposing ``generate``.
        tokenizer: A tokenizer with ``encode`` / ``decode`` / ``eos_token_id``.
        answer_type: How to extract the final answer -- ``"numeric"`` (GSM8K),
            ``"choice"`` (MMLU multiple-choice), or ``"free"`` (default).
        num_choices: Number of options when ``answer_type == "choice"``.
        device: Torch device string.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        answer_type: str = "free",
        num_choices: int = 4,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.answer_type = answer_type
        self.num_choices = num_choices
        self.device = device

    @abstractmethod
    def solve(self, question: str) -> ReasoningResult:
        """Solve ``question`` and return a :class:`ReasoningResult`."""
        raise NotImplementedError
