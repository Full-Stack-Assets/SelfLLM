"""Self-consistency reasoning strategy (sample N chains, majority-vote)."""

from __future__ import annotations

from typing import Any, List, Optional

from selfllm.cot.cot_generator import ChainOfThoughtGenerator
from selfllm.debate.debate import majority_vote

from .base import ReasoningResult, ReasoningStrategy
from .cot import sample_cot_traces
from .extract import extract_answer


class SelfConsistencyStrategy(ReasoningStrategy):
    """Sample ``num_samples`` CoT chains and take the majority-vote answer.

    Confidence is the fraction of *all* samples (not just the ones that yielded
    an answer) that agree with the winning answer, so failed extractions lower
    confidence.

    Args:
        num_samples: Number of reasoning chains to sample.
        temperature / top_p: Sampling parameters (use temperature > 0 for
            diversity).
        vary_templates: Use a different prompt template per sample for extra
            diversity.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        answer_type: str = "free",
        num_choices: int = 4,
        device: str = "cpu",
        num_samples: int = 5,
        temperature: float = 0.8,
        top_p: float = 0.9,
        max_think_tokens: int = 256,
        max_answer_tokens: int = 64,
        vary_templates: bool = True,
    ) -> None:
        super().__init__(model, tokenizer, answer_type, num_choices, device)
        self.num_samples = num_samples
        self.temperature = temperature
        self.top_p = top_p
        self.max_think_tokens = max_think_tokens
        self.max_answer_tokens = max_answer_tokens
        self.vary_templates = vary_templates
        self._generator = ChainOfThoughtGenerator(model, tokenizer, device=device)

    def _extract(self, trace: dict) -> Optional[str]:
        return extract_answer(trace["answer"], self.answer_type, self.num_choices) or \
            extract_answer(trace["full_text"], self.answer_type, self.num_choices)

    def solve(self, question: str) -> ReasoningResult:
        traces = sample_cot_traces(
            self._generator, question, self.num_samples,
            temperature=self.temperature, top_p=self.top_p,
            max_think_tokens=self.max_think_tokens,
            max_answer_tokens=self.max_answer_tokens,
            vary_templates=self.vary_templates,
        )
        answers: List[Optional[str]] = [self._extract(t) for t in traces]
        full_texts = [t["full_text"] for t in traces]
        valid = [a for a in answers if a]

        if not valid:
            return ReasoningResult(
                answer=None, confidence=0.0, traces=full_texts,
                num_samples=self.num_samples, details={"answers": answers},
            )

        winner, _, counts = majority_vote(valid)
        winner_count = max(counts.values()) if counts else 0
        confidence = winner_count / self.num_samples
        return ReasoningResult(
            answer=winner,
            confidence=confidence,
            traces=full_texts,
            num_samples=self.num_samples,
            details={"answers": answers, "counts": counts},
        )
