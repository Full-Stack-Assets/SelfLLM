"""Verifier-guided beam search over reasoning steps (V6 Wave 2, Task 4).

A tractable, from-scratch stand-in for MCTS: expand each partial reasoning
chain by sampling a few short continuations ("steps"), score the resulting
partial chains with a :class:`Verifier`, and keep the top-``beam_width``. After
a fixed number of steps, return the highest-scoring chain's answer.
"""

from __future__ import annotations

from typing import Any, List

import torch

from selfllm.cot.cot_generator import ChainOfThoughtPrompts

from .base import ReasoningResult, ReasoningStrategy
from .extract import extract_answer
from .verifier import Candidate, Verifier


class BeamSearchReasoner(ReasoningStrategy):
    """Beam search over reasoning steps, guided by a verifier value estimate.

    Args:
        verifier: Scores partial chains (value function). Both
            :class:`QualityVerifier` (per-chain) and
            :class:`SelfConsistencyVerifier` (agreement across the beam) work.
        beam_width: Number of partial chains kept after each step.
        num_expansions: Continuations sampled per beam item per step.
        max_steps: Number of expansion rounds.
        tokens_per_step: New tokens generated per continuation.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        verifier: Verifier,
        answer_type: str = "free",
        num_choices: int = 4,
        device: str = "cpu",
        beam_width: int = 3,
        num_expansions: int = 2,
        max_steps: int = 3,
        tokens_per_step: int = 16,
        temperature: float = 0.8,
        top_p: float = 0.9,
        template_idx: int = 0,
    ) -> None:
        super().__init__(model, tokenizer, answer_type, num_choices, device)
        self.verifier = verifier
        self.beam_width = beam_width
        self.num_expansions = num_expansions
        self.max_steps = max_steps
        self.tokens_per_step = tokens_per_step
        self.temperature = temperature
        self.top_p = top_p
        self.template_idx = template_idx

    def _initial_prefix(self, question: str) -> str:
        template = ChainOfThoughtPrompts.TEMPLATES[
            self.template_idx % len(ChainOfThoughtPrompts.TEMPLATES)
        ]
        return template.format(problem=question) + "\n<think>"

    def _expand(self, prefix_text: str) -> List[str]:
        """Sample ``num_expansions`` short continuations of ``prefix_text``.

        Returns the full (prefix + continuation) text of each continuation.
        """
        max_seq = getattr(getattr(self.model, "config", None), "max_seq_len", 512)
        ids = self.tokenizer.encode(prefix_text)[-max_seq:]
        prompt = torch.tensor([ids], device=self.device)
        out: List[str] = []
        for _ in range(self.num_expansions):
            gen = self.model.generate(
                prompt,
                max_new_tokens=self.tokens_per_step,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            out.append(self.tokenizer.decode(gen["sequences"][0].tolist()))
        return out

    def _score(self, question: str, chains: List[str]) -> List[float]:
        cands = [
            Candidate(text=t, answer=extract_answer(t, self.answer_type, self.num_choices))
            for t in chains
        ]
        return self.verifier.score(question, cands)

    def solve(self, question: str) -> ReasoningResult:
        beam: List[str] = [self._initial_prefix(question)]

        for _ in range(self.max_steps):
            expanded: List[str] = []
            for prefix in beam:
                expanded.extend(self._expand(prefix))
            if not expanded:
                break
            scores = self._score(question, expanded)
            order = sorted(range(len(expanded)), key=lambda i: scores[i], reverse=True)
            beam = [expanded[i] for i in order[: self.beam_width]]

        final_scores = self._score(question, beam)
        best_idx = max(range(len(beam)), key=lambda i: final_scores[i])
        best_text = beam[best_idx]
        best_answer = extract_answer(best_text, self.answer_type, self.num_choices)
        # Both shipped verifiers produce scores in [0, 1]; clamp for safety.
        confidence = min(max(final_scores[best_idx], 0.0), 1.0)

        return ReasoningResult(
            answer=best_answer,
            confidence=confidence,
            traces=beam,
            num_samples=len(beam),
            details={
                "scores": final_scores,
                "best_idx": best_idx,
                "max_steps": self.max_steps,
                "beam_width": self.beam_width,
            },
        )
