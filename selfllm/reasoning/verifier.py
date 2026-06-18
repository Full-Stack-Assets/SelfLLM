"""Verifiers for best-of-N reasoning.

A :class:`Verifier` scores a set of candidate reasoning traces so the best one
can be selected. V6 ships two lightweight verifiers (no reward-model training):

- :class:`SelfConsistencyVerifier` -- scores a candidate by how many other
  candidates agree with its final answer.
- :class:`QualityVerifier` -- scores a candidate's text with the existing
  :class:`selfllm.model.quality_model.QualityModel`.

The interface scores the whole candidate set at once so cross-candidate
verifiers (like self-consistency) are expressible.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional

import torch

from .extract import normalize


@dataclass
class Candidate:
    """A single reasoning candidate: its full text and extracted answer."""

    text: str
    answer: Optional[str]


class Verifier(ABC):
    """Abstract base: score a list of candidates for one question."""

    @abstractmethod
    def score(self, question: str, candidates: List[Candidate]) -> List[float]:
        """Return one score per candidate (higher is better)."""
        raise NotImplementedError


class SelfConsistencyVerifier(Verifier):
    """Score each candidate by the fraction of candidates sharing its answer.

    A candidate with no extractable answer scores ``0.0``. The score is in
    ``[0, 1]`` and equals the agreement fraction of that candidate's answer.
    """

    def score(self, question: str, candidates: List[Candidate]) -> List[float]:
        n = len(candidates)
        if n == 0:
            return []
        norms = [normalize(c.answer) for c in candidates]
        scores: List[float] = []
        for ni in norms:
            if not ni:
                scores.append(0.0)
                continue
            agree = sum(1 for x in norms if x == ni)
            scores.append(agree / n)
        return scores


class QualityVerifier(Verifier):
    """Score each candidate's text with a :class:`QualityModel`.

    Args:
        quality_model: A model exposing ``score_sequences([B, T]) -> [B]``.
        tokenizer: A tokenizer with ``encode``.
        device: Torch device string.
        max_len: Truncate each candidate to this many tokens before scoring.
    """

    def __init__(
        self,
        quality_model: Any,
        tokenizer: Any,
        device: str = "cpu",
        max_len: int = 256,
    ) -> None:
        self.quality_model = quality_model
        self.tokenizer = tokenizer
        self.device = device
        self.max_len = max_len

    @torch.no_grad()
    def score(self, question: str, candidates: List[Candidate]) -> List[float]:
        if hasattr(self.quality_model, "eval"):
            self.quality_model.eval()
        scores: List[float] = []
        for c in candidates:
            ids = self.tokenizer.encode(c.text)[: self.max_len] or [0]
            seq = torch.tensor([ids], device=self.device)
            scores.append(float(self.quality_model.score_sequences(seq)[0].item()))
        return scores
