"""Chain-of-Thought reasoning strategy.

Wraps the existing :class:`selfllm.cot.cot_generator.ChainOfThoughtGenerator`
(so the prompt templates and ``<think>/<answer>`` generation live in one place)
behind the uniform :class:`ReasoningStrategy` interface, and applies the shared
answer extractor.
"""

from __future__ import annotations

from typing import Any

from selfllm.cot.cot_generator import ChainOfThoughtGenerator

from .base import ReasoningResult, ReasoningStrategy
from .extract import extract_answer


class CoTStrategy(ReasoningStrategy):
    """Single chain-of-thought pass: think step by step, then extract the answer.

    Args:
        model: A model exposing ``generate``.
        tokenizer: A tokenizer.
        answer_type: ``"numeric"`` / ``"choice"`` / ``"free"`` (see
            :class:`ReasoningStrategy`).
        num_choices: Options for ``"choice"`` extraction.
        device: Torch device string.
        template_idx: Which CoT prompt template to use.
        max_think_tokens: Token budget for the reasoning section.
        max_answer_tokens: Token budget for the answer section.
        temperature: Sampling temperature (``0.0`` for greedy).
        top_p: Nucleus sampling threshold.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        answer_type: str = "free",
        num_choices: int = 4,
        device: str = "cpu",
        template_idx: int = 0,
        max_think_tokens: int = 256,
        max_answer_tokens: int = 64,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> None:
        super().__init__(model, tokenizer, answer_type, num_choices, device)
        self.template_idx = template_idx
        self.max_think_tokens = max_think_tokens
        self.max_answer_tokens = max_answer_tokens
        self.temperature = temperature
        self.top_p = top_p
        self._generator = ChainOfThoughtGenerator(model, tokenizer, device=device)

    def _generate_trace(self, question: str, template_idx: int) -> dict:
        """Produce one CoT trace dict (``thinking`` / ``answer`` / ``full_text``)."""
        return self._generator.generate_cot_response(
            question,
            template_idx=template_idx,
            max_think_tokens=self.max_think_tokens,
            max_answer_tokens=self.max_answer_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )

    def solve(self, question: str) -> ReasoningResult:
        """Run one CoT pass and extract the final answer."""
        trace = self._generate_trace(question, self.template_idx)
        # Prefer the model's explicit <answer> section, then fall back to the
        # full text so the extractor can still find a numeric/choice answer.
        answer = extract_answer(
            trace["answer"], self.answer_type, self.num_choices
        )
        if answer is None:
            answer = extract_answer(
                trace["full_text"], self.answer_type, self.num_choices
            )
        return ReasoningResult(
            answer=answer,
            confidence=1.0 if answer is not None else 0.0,
            traces=[trace["full_text"]],
            num_samples=1,
            details={"thinking": trace["thinking"], "raw_answer": trace["answer"]},
        )
