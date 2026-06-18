"""HumanEval code-generation benchmark with sandboxed execution.

Loads programming problems (``task_id``, ``prompt``, ``canonical_solution``,
``test``, ``entry_point``), generates ``k`` completions per problem, executes
each candidate (``prompt + completion + test`` harness) in an isolated
subprocess with a hard timeout, and computes the standard unbiased
``pass@k`` estimator.

Code execution is intentionally defensive: every candidate runs in a fresh
``python3`` subprocess that is killed if it exceeds the timeout, and *any*
failure (exception, crash, timeout, non-zero exit) is treated as a failed
completion rather than propagating.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Union

from .harness import (
    Benchmark,
    EvalResult,
    SupportsGenerate,
    SupportsTokenize,
)


@dataclass
class HumanEvalProblem:
    """A single HumanEval problem.

    Attributes:
        task_id: Unique problem identifier.
        prompt: The function signature + docstring the model completes.
        canonical_solution: A reference completion (not used for scoring).
        test: A ``check(candidate)`` test harness as source code.
        entry_point: Name of the function the test harness checks.
    """

    task_id: str
    prompt: str
    canonical_solution: str
    test: str
    entry_point: str


def load_humaneval(
    source: Union[str, Sequence[Dict[str, Any]]],
) -> List[HumanEvalProblem]:
    """Load HumanEval problems from an in-memory list or local JSONL file.

    Each record must contain ``task_id``, ``prompt``, ``test`` and
    ``entry_point``. ``canonical_solution`` is optional (defaults to ``""``).

    Args:
        source: A path to a ``.jsonl`` file or an in-memory sequence of dicts.

    Returns:
        A list of :class:`HumanEvalProblem`.

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

    problems: List[HumanEvalProblem] = []
    for rec in records:
        for required in ("task_id", "prompt", "test", "entry_point"):
            if required not in rec:
                raise ValueError(
                    f"HumanEval record missing required field {required!r}: {rec!r}"
                )
        problems.append(
            HumanEvalProblem(
                task_id=str(rec["task_id"]),
                prompt=str(rec["prompt"]),
                canonical_solution=str(rec.get("canonical_solution", "")),
                test=str(rec["test"]),
                entry_point=str(rec["entry_point"]),
            )
        )
    return problems


# --------------------------------------------------------------------------- #
# pass@k estimator
# --------------------------------------------------------------------------- #


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator of pass@k.

    Computes ``1 - C(n - c, k) / C(n, k)``: the probability that at least one
    of ``k`` completions sampled (without replacement) from ``n`` total is
    correct, given ``c`` of the ``n`` are correct.

    Args:
        n: Total number of completions generated for the problem.
        c: Number of correct completions among the ``n``.
        k: The ``k`` in pass@k.

    Returns:
        The pass@k probability in ``[0, 1]``.

    Raises:
        ValueError: If ``k <= 0`` or ``k > n`` or ``c < 0`` or ``c > n``.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if n < k:
        raise ValueError("k must not exceed n")
    if c < 0 or c > n:
        raise ValueError("c must satisfy 0 <= c <= n")
    if c == 0:
        return 0.0
    if n - c < k:
        # Fewer than k incorrect completions => at least one correct is always
        # included in any draw of k.
        return 1.0
    # 1 - C(n-c, k)/C(n, k), computed stably as a product of ratios.
    prob_all_wrong = 1.0
    for i in range(k):
        prob_all_wrong *= (n - c - i) / (n - i)
    return 1.0 - prob_all_wrong


# --------------------------------------------------------------------------- #
# Sandboxed execution
# --------------------------------------------------------------------------- #


_HARNESS_TEMPLATE = """\
{program}

{test}

{entry_point_check}
"""


def _build_program(problem: HumanEvalProblem, completion: str) -> str:
    """Assemble the full runnable program for a candidate completion.

    The standard HumanEval harness is ``prompt + completion`` (the completion
    is the function body that follows the signature in ``prompt``), then the
    ``test`` source which defines ``check``, then a call to
    ``check(<entry_point>)``.

    Args:
        problem: The problem being solved.
        completion: The model-generated completion (function body).

    Returns:
        Complete Python source code as a string.
    """
    program = problem.prompt + completion
    check_call = f"check({problem.entry_point})"
    return _HARNESS_TEMPLATE.format(
        program=program,
        test=problem.test,
        entry_point_check=check_call,
    )


def execute_candidate(
    problem: HumanEvalProblem,
    completion: str,
    timeout: float = 10.0,
) -> bool:
    """Execute one candidate completion in a sandboxed subprocess.

    The assembled program runs in a fresh ``python3`` process. The process is
    killed if it exceeds ``timeout`` seconds. Any non-zero exit, exception,
    crash or timeout is reported as a failure (``False``); only a clean exit
    (the test harness asserting successfully) counts as a pass.

    Args:
        problem: The problem being solved.
        completion: The candidate completion to test.
        timeout: Hard wall-clock timeout in seconds.

    Returns:
        ``True`` if the candidate passed the tests, ``False`` otherwise.
    """
    source = _build_program(problem, completion)
    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(source)
            tmp_path = tmp.name

        proc = subprocess.run(
            [sys.executable, tmp_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            timeout=timeout,
            # Fresh, minimal environment; do not inherit anything surprising.
            env={"PATH": os.environ.get("PATH", "")},
        )
        return proc.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        # Defensive: never let a candidate's failure escape.
        return False
    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


class HumanEvalBenchmark(Benchmark):
    """HumanEval pass@k benchmark.

    Args:
        problems: The problems to evaluate.
    """

    def __init__(self, problems: Sequence[HumanEvalProblem]) -> None:
        super().__init__("humaneval")
        self.problems: List[HumanEvalProblem] = list(problems)

    @classmethod
    def from_source(
        cls, source: Union[str, Sequence[Dict[str, Any]]]
    ) -> "HumanEvalBenchmark":
        """Construct a benchmark from a JSONL path or in-memory records."""
        return cls(load_humaneval(source))

    def evaluate(
        self,
        model: SupportsGenerate,
        tokenizer: SupportsTokenize,
        n_samples: int = 1,
        k: Union[int, Sequence[int]] = 1,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 0.95,
        top_k: int = 0,
        timeout: float = 10.0,
        limit: Optional[int] = None,
        **kwargs: Any,
    ) -> EvalResult:
        """Generate completions, execute them, and compute pass@k.

        Args:
            model: A model exposing ``generate``.
            tokenizer: A tokenizer.
            n_samples: Completions generated per problem (must be ``>= max(k)``).
            k: A single ``k`` or a list of ``k`` values for pass@k.
            max_new_tokens: Tokens to generate per completion.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            top_k: Top-k cutoff (``0`` = disabled).
            timeout: Per-candidate execution timeout in seconds.
            limit: Optional cap on the number of problems evaluated.

        Returns:
            An :class:`EvalResult` whose ``score`` is pass@(min k) and whose
            ``details`` hold every requested ``pass@k`` plus per-problem
            correct counts.
        """
        import torch

        ks = [k] if isinstance(k, int) else list(k)
        problems = self.problems if limit is None else self.problems[:limit]
        n = len(problems)

        per_problem: List[Dict[str, Any]] = []
        # Track (n_completions, n_correct) per problem for the estimator.
        counts: List[Dict[str, int]] = []

        for problem in problems:
            input_ids = torch.tensor(
                [tokenizer.encode(problem.prompt)], dtype=torch.long
            )
            num_correct = 0
            for _ in range(n_samples):
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
                completion = tokenizer.decode(gen_ids)
                if execute_candidate(problem, completion, timeout=timeout):
                    num_correct += 1
            counts.append({"n": n_samples, "c": num_correct})
            per_problem.append(
                {
                    "task_id": problem.task_id,
                    "n": n_samples,
                    "correct": num_correct,
                }
            )

        pass_at: Dict[str, float] = {}
        for kk in ks:
            valid = [ct for ct in counts if ct["n"] >= kk]
            if valid:
                pass_at[f"pass@{kk}"] = sum(
                    pass_at_k(ct["n"], ct["c"], kk) for ct in valid
                ) / len(valid)
            else:
                pass_at[f"pass@{kk}"] = 0.0

        primary_k = min(ks)
        score = pass_at[f"pass@{primary_k}"]
        return EvalResult(
            name=self.name,
            score=score,
            n=n,
            details={"pass_at_k": pass_at, "per_problem": per_problem},
        )
