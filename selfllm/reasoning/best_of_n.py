"""Best-of-N reasoning strategy (sample N chains, pick the verifier's best)."""

from __future__ import annotations

from typing import Any, Optional

from selfllm.cot.cot_generator import ChainOfThoughtGenerator

from .base import ReasoningResult, ReasoningStrategy
from .cot import sample_cot_traces
from .extract import extract_answer, normalize
from .verifier import Candidate, Verifier


class BestOfNStrategy(ReasoningStrategy):
    """Sample ``num_samples`` CoT chains and return the verifier's top pick.

    Args:
        verifier: A :class:`Verifier` that scores the candidate set.
        num_samples: Number of reasoning chains to sample.
        temperature / top_p / vary_templates: Sampling diversity controls.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        verifier: Verifier,
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
        self.verifier = verifier
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
        candidates = [
            Candidate(text=t["full_text"], answer=self._extract(t)) for t in traces
        ]
        scores = self.verifier.score(question, candidates)
        best_idx = max(range(len(candidates)), key=lambda i: scores[i])
        best = candidates[best_idx]

        # Confidence: agreement of the chosen answer across all samples.
        norms = [normalize(c.answer) for c in candidates]
        best_norm = normalize(best.answer)
        confidence = (
            sum(1 for x in norms if x and x == best_norm) / self.num_samples
            if best_norm else 0.0
        )
        return ReasoningResult(
            answer=best.answer,
            confidence=confidence,
            traces=[c.text for c in candidates],
            num_samples=self.num_samples,
            details={
                "scores": scores,
                "answers": [c.answer for c in candidates],
                "best_idx": best_idx,
            },
        )
