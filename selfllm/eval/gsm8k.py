"""GSM8K grade-school math benchmark.

Loads question/answer pairs (the gold answer is the number following ``####``,
or the last number in the answer string), prompts the model, extracts the final
numeric answer from the generated text, and reports exact-match accuracy.

No network access is required: datasets are supplied directly or read from a
local JSONL file.
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

# Matches an integer or float, optionally signed, with comma thousands
# separators (e.g. ``-1,234.5``).
_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


@dataclass
class GSM8KExample:
    """A single GSM8K example.

    Attributes:
        question: The word problem.
        answer: The reference answer string (may contain a ``####`` marker).
    """

    question: str
    answer: str

    @property
    def gold_answer(self) -> Optional[str]:
        """Return the normalized gold final numeric answer."""
        return extract_final_number(self.answer)


def extract_final_number(text: str) -> Optional[str]:
    """Extract a normalized final numeric answer from text.

    Resolution order:

    1. If a ``####`` marker is present, parse the number after it.
    2. Otherwise, take the **last** number appearing in the text.

    Commas are stripped and a trailing ``.0`` is normalized away so that
    ``"42"``, ``"42.0"`` and ``"42,0"``-style values compare equal under
    exact match.

    Args:
        text: The text to parse.

    Returns:
        The normalized number as a string, or ``None`` if none is found.
    """
    if text is None:
        return None

    if "####" in text:
        tail = text.split("####")[-1]
        match = _NUMBER_RE.search(tail)
        if match:
            return _normalize_number(match.group(0))
        # Fall through to scanning the whole string if the tail had no number.

    matches = _NUMBER_RE.findall(text)
    if not matches:
        return None
    return _normalize_number(matches[-1])


def _normalize_number(raw: str) -> str:
    """Normalize a raw numeric string for exact-match comparison.

    Removes thousands-separator commas and collapses integral floats
    (``"42.0" -> "42"``). Non-integral floats keep their fractional part with
    trailing zeros trimmed.

    Args:
        raw: The raw matched number.

    Returns:
        The normalized numeric string.
    """
    cleaned = raw.replace(",", "")
    try:
        value = float(cleaned)
    except ValueError:
        return cleaned
    if value.is_integer():
        return str(int(value))
    # Trim trailing zeros on the fractional part, e.g. "3.140" -> "3.14".
    return ("%f" % value).rstrip("0").rstrip(".")


def load_gsm8k(
    source: Union[str, Sequence[Dict[str, Any]]],
) -> List[GSM8KExample]:
    """Load GSM8K examples from an in-memory list or a local JSONL file.

    Each record must contain ``question`` and ``answer`` fields.

    Args:
        source: A path to a ``.jsonl`` file or an in-memory sequence of dicts.

    Returns:
        A list of :class:`GSM8KExample`.

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

    examples: List[GSM8KExample] = []
    for rec in records:
        if "question" not in rec or "answer" not in rec:
            raise ValueError(f"GSM8K record missing required fields: {rec!r}")
        examples.append(
            GSM8KExample(question=str(rec["question"]), answer=str(rec["answer"]))
        )
    return examples


def build_gsm8k_prompt(example: GSM8KExample) -> str:
    """Render a GSM8K example into a prompt string.

    Args:
        example: The example to render.

    Returns:
        A prompt ending with ``"Answer:"`` to elicit a numeric answer.
    """
    return f"Question: {example.question.strip()}\nAnswer:"


class GSM8KBenchmark(Benchmark):
    """GSM8K exact-match accuracy benchmark.

    Args:
        examples: The examples to evaluate.
    """

    def __init__(self, examples: Sequence[GSM8KExample]) -> None:
        super().__init__("gsm8k")
        self.examples: List[GSM8KExample] = list(examples)

    @classmethod
    def from_source(
        cls, source: Union[str, Sequence[Dict[str, Any]]]
    ) -> "GSM8KBenchmark":
        """Construct a benchmark from a JSONL path or in-memory records."""
        return cls(load_gsm8k(source))

    def evaluate(
        self,
        model: SupportsGenerate,
        tokenizer: SupportsTokenize,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        limit: Optional[int] = None,
        **kwargs: Any,
    ) -> EvalResult:
        """Run GSM8K and return exact-match accuracy.

        Args:
            model: A model exposing ``generate``.
            tokenizer: A tokenizer.
            max_new_tokens: Tokens to generate per question.
            temperature: Sampling temperature (``0`` = greedy).
            top_p: Nucleus sampling threshold.
            top_k: Top-k cutoff (``0`` = disabled).
            limit: Optional cap on the number of examples evaluated.

        Returns:
            An :class:`EvalResult` with exact-match accuracy as ``score`` and
            per-example predictions in ``details``.
        """
        import torch

        examples = self.examples if limit is None else self.examples[:limit]
        n = len(examples)
        correct = 0
        predictions: List[Dict[str, Any]] = []

        for ex in examples:
            prompt = build_gsm8k_prompt(ex)
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

            pred = extract_final_number(gen_text)
            gold = ex.gold_answer
            is_correct = pred is not None and gold is not None and pred == gold
            correct += int(is_correct)
            predictions.append(
                {
                    "predicted": pred,
                    "gold": gold,
                    "correct": is_correct,
                    "generated": gen_text,
                }
            )

        score = correct / n if n else 0.0
        return EvalResult(
            name=self.name,
            score=score,
            n=n,
            details={"correct": correct, "predictions": predictions},
        )
