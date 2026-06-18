"""MMLU multiple-choice benchmark.

Loads multiple-choice questions (from an in-memory list or a local JSONL file),
prompts the model, extracts the predicted answer letter from the generated
continuation, and reports overall (and optionally per-subject) accuracy.

No network access is required: datasets are supplied directly or read from a
local path.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

from .harness import (
    Benchmark,
    EvalResult,
    SupportsGenerate,
    SupportsTokenize,
)

# Canonical choice letters used by MMLU.
CHOICE_LETTERS = ("A", "B", "C", "D")


@dataclass
class MMLUExample:
    """A single MMLU multiple-choice question.

    Attributes:
        question: The question text.
        choices: The answer options, ordered to match ``CHOICE_LETTERS``.
        answer: The correct answer as a letter (``"A".."D"``) or 0-based index.
        subject: Optional subject/category label.
    """

    question: str
    choices: Sequence[str]
    answer: Union[str, int]
    subject: Optional[str] = None

    @property
    def answer_letter(self) -> str:
        """Return the correct answer as an uppercase letter."""
        if isinstance(self.answer, int):
            return CHOICE_LETTERS[self.answer]
        ans = str(self.answer).strip().upper()
        if ans in CHOICE_LETTERS:
            return ans
        # Allow a numeric string index.
        if ans.isdigit():
            return CHOICE_LETTERS[int(ans)]
        raise ValueError(f"Unrecognized MMLU answer: {self.answer!r}")


def load_mmlu(
    source: Union[str, Sequence[Dict[str, Any]]],
) -> List[MMLUExample]:
    """Load MMLU examples from an in-memory list or a local JSONL file.

    Each record must contain ``question``, ``choices`` (a list of 2-4 options)
    and ``answer`` (a letter ``"A".."D"`` or a 0-based index). An optional
    ``subject`` field is preserved for per-subject scoring.

    Args:
        source: Either a path to a ``.jsonl`` file (one JSON object per line) or
            an already-parsed sequence of dict records.

    Returns:
        A list of :class:`MMLUExample`.

    Raises:
        FileNotFoundError: If ``source`` is a path that does not exist.
        ValueError: If a record is missing required fields.
    """
    records: Sequence[Dict[str, Any]]
    if isinstance(source, str):
        if not os.path.exists(source):
            raise FileNotFoundError(source)
        parsed: List[Dict[str, Any]] = []
        with open(source, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    parsed.append(json.loads(line))
        records = parsed
    else:
        records = source

    examples: List[MMLUExample] = []
    for rec in records:
        if "question" not in rec or "choices" not in rec or "answer" not in rec:
            raise ValueError(f"MMLU record missing required fields: {rec!r}")
        examples.append(
            MMLUExample(
                question=str(rec["question"]),
                choices=list(rec["choices"]),
                answer=rec["answer"],
                subject=rec.get("subject"),
            )
        )
    return examples


def build_mmlu_prompt(example: MMLUExample) -> str:
    """Render a multiple-choice example into a prompt string.

    Args:
        example: The example to render.

    Returns:
        A prompt ending with ``"Answer:"`` to elicit a single answer letter.
    """
    lines = [example.question.strip()]
    for letter, choice in zip(CHOICE_LETTERS, example.choices):
        lines.append(f"{letter}. {choice}")
    lines.append("Answer:")
    return "\n".join(lines)


def extract_choice(text: str, num_choices: int = 4) -> Optional[str]:
    """Extract the predicted answer letter from generated text.

    A *standalone* choice letter is preferred: a single letter not embedded in
    a larger word (e.g. ``"B"``, ``"(A)"``, ``"D) ..."``, ``"answer: b"``),
    which avoids latching onto the first letter of an English word such as the
    ``"C"`` in ``"Capital"``. If no standalone letter is found, the first valid
    letter anywhere in the text is used as a fallback. Returns ``None`` when no
    valid letter is present.

    Args:
        text: The model's generated continuation.
        num_choices: Number of valid choices (caps the letter range).

    Returns:
        The predicted uppercase letter, or ``None``.
    """
    valid = set(CHOICE_LETTERS[:num_choices])

    # Prefer a single letter delimited by non-letter characters (start/end of
    # string, whitespace, or punctuation) so we ignore letters inside words.
    for match in re.finditer(r"(?<![A-Za-z])([A-Za-z])(?![A-Za-z])", text):
        letter = match.group(1).upper()
        if letter in valid:
            return letter

    # Fallback: first valid letter anywhere.
    for ch in text:
        if ch.upper() in valid and ch.isalpha():
            return ch.upper()
    return None


class MMLUBenchmark(Benchmark):
    """MMLU multiple-choice accuracy benchmark.

    Args:
        examples: The examples to evaluate.
    """

    def __init__(self, examples: Sequence[MMLUExample]) -> None:
        super().__init__("mmlu")
        self.examples: List[MMLUExample] = list(examples)

    @classmethod
    def from_source(
        cls, source: Union[str, Sequence[Dict[str, Any]]]
    ) -> "MMLUBenchmark":
        """Construct a benchmark from a JSONL path or in-memory records."""
        return cls(load_mmlu(source))

    def evaluate(
        self,
        model: SupportsGenerate,
        tokenizer: SupportsTokenize,
        max_new_tokens: int = 8,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        limit: Optional[int] = None,
        **kwargs: Any,
    ) -> EvalResult:
        """Run MMLU and return accuracy (overall and per-subject).

        Args:
            model: A model exposing ``generate``.
            tokenizer: A tokenizer.
            max_new_tokens: Tokens to generate per question.
            temperature: Sampling temperature (``0`` = greedy).
            top_p: Nucleus sampling threshold.
            top_k: Top-k cutoff (``0`` = disabled).
            limit: Optional cap on the number of examples evaluated.

        Returns:
            An :class:`EvalResult` with overall accuracy as ``score`` and a
            ``per_subject`` accuracy breakdown plus per-example predictions in
            ``details``.
        """
        import torch

        examples = self.examples if limit is None else self.examples[:limit]
        n = len(examples)
        correct = 0
        per_subject: Dict[str, List[int]] = {}
        predictions: List[Dict[str, Any]] = []

        for ex in examples:
            prompt = build_mmlu_prompt(ex)
            input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
            out = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stop_token_id=tokenizer.eos_token_id,
            )
            seq = out["sequences"][0].tolist()
            gen_ids = seq[input_ids.shape[1]:]
            gen_text = tokenizer.decode(gen_ids)

            pred = extract_choice(gen_text, num_choices=len(ex.choices))
            gold = ex.answer_letter
            is_correct = pred == gold
            correct += int(is_correct)

            subject = ex.subject or "unknown"
            per_subject.setdefault(subject, []).append(int(is_correct))
            predictions.append(
                {
                    "subject": subject,
                    "predicted": pred,
                    "gold": gold,
                    "correct": is_correct,
                    "generated": gen_text,
                }
            )

        score = correct / n if n else 0.0
        subject_acc = {
            subj: (sum(hits) / len(hits) if hits else 0.0)
            for subj, hits in per_subject.items()
        }
        return EvalResult(
            name=self.name,
            score=score,
            n=n,
            details={
                "correct": correct,
                "per_subject": subject_acc,
                "predictions": predictions,
            },
        )
