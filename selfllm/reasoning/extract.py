"""Unified final-answer extraction for reasoning strategies.

One extractor layer shared by every strategy so an answer is parsed the same
way regardless of how the reasoning was produced. Reuses the eval-suite
extractors (so GSM8K numeric answers and MMLU choice letters are handled
identically to the benchmarks) and the debate answer normalizer.
"""

from __future__ import annotations

import re
from typing import Optional

from selfllm.eval.gsm8k import extract_final_number
from selfllm.eval.mmlu import extract_choice
from selfllm.debate.debate import normalize_answer

# Matches an explicit ``<answer>...</answer>`` block emitted by the CoT format.
_ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
# Matches a trailing "The answer is X" / "Answer: X" style statement.
_ANSWER_PHRASE = re.compile(
    r"(?:final answer|the answer is|answer)\s*[:\-]?\s*(.+?)(?:[.\n]|$)",
    re.IGNORECASE,
)


def extract_tagged_answer(text: str) -> Optional[str]:
    """Return the content of the last ``<answer>...</answer>`` block, if any."""
    matches = _ANSWER_TAG.findall(text)
    if matches:
        return matches[-1].strip()
    return None


def _free_answer(text: str) -> Optional[str]:
    """Best-effort free-form answer: tag, then phrase, then last non-empty line."""
    tagged = extract_tagged_answer(text)
    if tagged:
        return tagged
    phrase = _ANSWER_PHRASE.search(text)
    if phrase and phrase.group(1).strip():
        return phrase.group(1).strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else None


def extract_answer(
    text: str,
    answer_type: str = "free",
    num_choices: int = 4,
) -> Optional[str]:
    """Extract a final answer from a reasoning trace.

    Args:
        text: The full reasoning / generated text.
        answer_type: ``"numeric"`` (GSM8K-style number), ``"choice"`` (MMLU
            letter A-D), or ``"free"`` (tagged / phrased / last line).
        num_choices: Number of options for ``"choice"``.

    Returns:
        The extracted answer string, or ``None`` if nothing was found.
    """
    if answer_type == "numeric":
        return extract_final_number(text)
    if answer_type == "choice":
        return extract_choice(text, num_choices=num_choices)
    if answer_type == "free":
        return _free_answer(text)
    raise ValueError(f"Unknown answer_type: {answer_type!r}")


def normalize(answer: Optional[str]) -> str:
    """Normalize an answer for equality comparison/voting (empty if ``None``)."""
    if answer is None:
        return ""
    return normalize_answer(str(answer))
