"""Multi-agent debate orchestration with deterministic consensus voting."""

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .agent import DebateAgent


# ---------------------------------------------------------------------- #
# Result container
# ---------------------------------------------------------------------- #


@dataclass
class DebateResult:
    """Structured outcome of a multi-agent debate.

    Attributes:
        question: The question that was debated.
        consensus: The selected consensus answer text.
        confidence: Agreement score in ``[0.0, 1.0]`` for the consensus.
        final_answers: Mapping of agent name to its final answer.
        transcript: Per-round log. Each entry is a dict with ``"round"`` and a
            ``"events"`` list of ``{"agent", "action", "text"}`` records.
        answer_counts: Normalised-answer -> count tally used for voting.
    """

    question: str
    consensus: str
    confidence: float
    final_answers: Dict[str, str] = field(default_factory=dict)
    transcript: List[Dict] = field(default_factory=list)
    answer_counts: Dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------- #
# Deterministic voting / consensus utilities (model-independent)
# ---------------------------------------------------------------------- #


def normalize_answer(answer: str) -> str:
    """Normalise an answer for equality comparison.

    Lowercases, strips surrounding punctuation, and collapses all whitespace so
    that superficially different strings with the same content compare equal.

    Args:
        answer: Raw answer text.

    Returns:
        The normalised answer string.
    """
    text = answer.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\n.,!?;:\"'")


def _token_set(answer: str) -> frozenset:
    """Return the set of normalised word tokens in ``answer``."""
    return frozenset(t for t in re.split(r"\W+", normalize_answer(answer)) if t)


def jaccard_similarity(a: str, b: str) -> float:
    """Compute Jaccard token-overlap similarity between two answers.

    Args:
        a: First answer.
        b: Second answer.

    Returns:
        Overlap in ``[0.0, 1.0]``. Two empty answers are treated as identical
        (``1.0``); one empty and one non-empty yields ``0.0``.
    """
    sa, sb = _token_set(a), _token_set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union


def majority_vote(answers: List[str]) -> Tuple[str, float, Dict[str, int]]:
    """Pick the most common answer by exact normalised match.

    Args:
        answers: List of answer strings.

    Returns:
        Tuple ``(best_answer, confidence, counts)`` where ``best_answer`` is the
        original (non-normalised) text of a representative for the winning
        group, ``confidence`` is the fraction of agents in that group, and
        ``counts`` maps normalised answers to their occurrence counts. Ties are
        broken by first appearance, making the result deterministic.
    """
    if not answers:
        return "", 0.0, {}

    normalized = [normalize_answer(a) for a in answers]
    counts = Counter(normalized)
    # Deterministic tie-break: highest count, then earliest first appearance.
    first_index = {n: normalized.index(n) for n in counts}
    best_norm = min(counts, key=lambda n: (-counts[n], first_index[n]))

    confidence = counts[best_norm] / len(answers)
    best_original = answers[first_index[best_norm]]
    return best_original, confidence, dict(counts)


def agreement_score(answers: List[str]) -> float:
    """Mean pairwise Jaccard similarity across all answers.

    A similarity-based agreement measure that, unlike exact majority vote,
    rewards partial overlap. Returns ``1.0`` for a single answer (or none) and
    ``0.0`` when no pair of answers shares any tokens.

    Args:
        answers: List of answer strings.

    Returns:
        Mean pairwise similarity in ``[0.0, 1.0]``.
    """
    n = len(answers)
    if n <= 1:
        return 1.0
    total = 0.0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += jaccard_similarity(answers[i], answers[j])
            pairs += 1
    return total / pairs if pairs else 1.0


def select_consensus(
    answers: List[str], use_similarity: bool = False
) -> Tuple[str, float, Dict[str, int]]:
    """Select a consensus answer and confidence from final agent answers.

    Args:
        answers: Final answers, one per agent.
        use_similarity: If ``True``, blend the majority-vote confidence with the
            mean pairwise similarity (averaged) to reward partial agreement.
            If ``False``, confidence is the pure majority-vote fraction.

    Returns:
        Tuple ``(consensus, confidence, counts)``.
    """
    best, confidence, counts = majority_vote(answers)
    if use_similarity:
        confidence = (confidence + agreement_score(answers)) / 2.0
    return best, confidence, counts


# ---------------------------------------------------------------------- #
# Orchestrator
# ---------------------------------------------------------------------- #


class DebateOrchestrator:
    """Run a multi-agent debate over ``num_rounds`` and aggregate a consensus.

    Round 1: every agent proposes an initial answer. Each subsequent round,
    every agent critiques its peers' current answers and then revises its own
    answer in light of those critiques. After all rounds, a deterministic
    voting step selects the consensus answer and a confidence score.

    Args:
        agents: The participating debate agents. They may share one model.
        num_rounds: Total number of rounds (>= 1). Round 1 is propose-only.
        use_similarity: Whether the consensus confidence blends a similarity
            measure with the majority-vote fraction (see ``select_consensus``).
    """

    def __init__(
        self,
        agents: List[DebateAgent],
        num_rounds: int = 2,
        use_similarity: bool = False,
    ) -> None:
        if not agents:
            raise ValueError("DebateOrchestrator requires at least one agent.")
        if num_rounds < 1:
            raise ValueError("num_rounds must be >= 1.")
        self.agents = agents
        self.num_rounds = num_rounds
        self.use_similarity = use_similarity

    def run(self, question: str) -> DebateResult:
        """Execute the debate for ``question`` and return a structured result.

        Args:
            question: The question to debate.

        Returns:
            A populated :class:`DebateResult`.
        """
        transcript: List[Dict] = []

        # ---- Round 1: every agent proposes. ----
        current: Dict[str, str] = {}
        events: List[Dict[str, str]] = []
        for agent in self.agents:
            answer = agent.propose(question)
            current[agent.name] = answer
            events.append(
                {"agent": agent.name, "action": "propose", "text": answer}
            )
        transcript.append({"round": 1, "events": events})

        # ---- Rounds 2..R: critique peers, then revise. ----
        for round_idx in range(2, self.num_rounds + 1):
            events = []

            # Snapshot answers so all critiques in this round see the same state.
            snapshot = dict(current)

            # Each agent collects critiques targeted at *its* answer from peers.
            critiques_for: Dict[str, List[str]] = {a.name: [] for a in self.agents}
            for agent in self.agents:
                for peer in self.agents:
                    if peer.name == agent.name:
                        continue
                    critique = agent.critique(question, snapshot[peer.name])
                    critiques_for[peer.name].append(critique)
                    events.append(
                        {
                            "agent": agent.name,
                            "action": "critique",
                            "target": peer.name,
                            "text": critique,
                        }
                    )

            # Each agent revises its own answer using critiques it received.
            revised: Dict[str, str] = {}
            for agent in self.agents:
                new_answer = agent.revise(
                    question, snapshot[agent.name], critiques_for[agent.name]
                )
                revised[agent.name] = new_answer
                events.append(
                    {"agent": agent.name, "action": "revise", "text": new_answer}
                )
            current = revised
            transcript.append({"round": round_idx, "events": events})

        # ---- Aggregation / voting. ----
        ordered_answers = [current[a.name] for a in self.agents]
        consensus, confidence, counts = select_consensus(
            ordered_answers, use_similarity=self.use_similarity
        )

        return DebateResult(
            question=question,
            consensus=consensus,
            confidence=confidence,
            final_answers=dict(current),
            transcript=transcript,
            answer_counts=counts,
        )
