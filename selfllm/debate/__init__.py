"""Multi-agent debate package: agents debate, critique, and reach consensus."""

from .agent import DebateAgent
from .debate import (
    DebateOrchestrator,
    DebateResult,
    agreement_score,
    jaccard_similarity,
    majority_vote,
    normalize_answer,
    select_consensus,
)

__all__ = [
    "DebateAgent",
    "DebateOrchestrator",
    "DebateResult",
    "normalize_answer",
    "jaccard_similarity",
    "majority_vote",
    "agreement_score",
    "select_consensus",
]
