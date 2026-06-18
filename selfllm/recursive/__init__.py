"""Recursive self-improvement engine for SelfLLM.

This module implements the core recursive self-improvement loop where the
model generates its own training data, filters by quality, and iteratively
re-trains to improve itself without external human-labeled data.

Components:
    RecursiveConfig: Dataclass with all hyperparameters for the loop.
    RecursiveSelfTrainer: Orchestrates the full recursive improvement loop.
    SelfImprovementEvaluator: Evaluates model quality across iterations.
"""

from .recursive_config import RecursiveConfig
from .recursive_trainer import RecursiveSelfTrainer
from .evaluator import SelfImprovementEvaluator

__all__ = ["RecursiveConfig", "RecursiveSelfTrainer", "SelfImprovementEvaluator"]
