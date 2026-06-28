"""Tests for the evaluation suite (selfllm.eval).

These tests verify the *machinery* of the benchmarks rather than model quality:
- Loaders parse in-memory records and local JSONL files.
- Answer/number/letter extraction works on crafted strings.
- The pass@k estimator matches hand-computed values.
- HumanEval sandboxed execution correctly distinguishes pass/fail and never
  hangs (timeouts and crashes are caught).
- The harness/runner returns well-formed results.
- A tiny model end-to-end run completes on CPU without error.

All datasets are embedded inline; no network access is required.
"""

import json
import math
import os
from fractions import Fraction

import pytest

from selfllm.model.config import ModelConfig
from selfllm.model.tokenizer import BPETokenizer
from selfllm.model.model import SelfImprovingLLM

from selfllm.eval import (
    Benchmark,
    EvalResult,
    Evaluator,
    comparison_report,
    run_benchmark,
    run_all,
    MMLUBenchmark,
    MMLUExample,
    build_mmlu_prompt,
    extract_choice,
    load_mmlu,
    GSM8KBenchmark,
    GSM8KExample,
    build_gsm8k_prompt,
    extract_final_number,
    load_gsm8k,
    HumanEvalBenchmark,
    HumanEvalProblem,
    execute_candidate,
    load_humaneval,
    pass_at_k,
)


# --------------------------------------------------------------------------- #
# Embedded sample datasets
# --------------------------------------------------------------------------- #

MMLU_RECORDS = [
    {
        "question": "What is 2 + 2?",
        "choices": ["3", "4", "5", "6"],
        "answer": "B",
        "subject": "math",
    },
    {
        "question": "The capital of France is?",
        "choices": ["Berlin", "Madrid", "Paris", "Rome"],
        "answer": 2,  # index form -> "C"
        "subject": "geography",
    },
    {
        "question": "Which is a primary color?",
        "choices": ["Green", "Red", "Purple", "Brown"],
        "answer": "B",
        "subject": "math",
    },
]

GSM8K_RECORDS = [
    {
        "question": "Tom has 3 apples and buys 2 more. How many apples?",
        "answer": "He buys 2 more so 3 + 2 = 5.\n#### 5",
    },
    {
        "question": "A pie costs $1,250. How much for one?",
        "answer": "The answer is 1,250 dollars.\n#### 1,250",
    },
]

HUMANEVAL_RECORDS = [
    {
        "task_id": "test/add",
        "prompt": "def add(a, b):\n    \"\"\"Return a + b.\"\"\"\n",
        "canonical_solution": "    return a + b\n",
        "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n    assert candidate(0, 0) == 0\n",
        "entry_point": "add",
    },
]


# --------------------------------------------------------------------------- #
# Fixtures: tiny model + trained tokenizer
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def tokenizer() -> BPETokenizer:
    """A tiny BPE tokenizer trained on a small corpus."""
    tok = BPETokenizer(vocab_size=200)
    tok.train(
        [
            "What is 2 + 2? Answer: B",
            "The capital of France is Paris. Answer: C",
            "Question: how many apples? Answer: 5 #### 5",
            "def add(a, b): return a + b",
            "ABCD abcd 0123456789 the quick brown fox",
        ]
    )
    return tok


@pytest.fixture(scope="module")
def model(tokenizer: BPETokenizer) -> SelfImprovingLLM:
    """A tiny CPU model sized to the tokenizer vocabulary."""
    config = ModelConfig(
        vocab_size=max(tokenizer.vocab_size, 64),
        d_model=32,
        n_layers=2,
        n_heads=4,
        d_ff=64,
        max_seq_len=128,
        dropout=0.0,
    )
    m = SelfImprovingLLM(config)
    m.eval()
    return m


# --------------------------------------------------------------------------- #
# MMLU
# --------------------------------------------------------------------------- #


class TestMMLULoader:
    def test_load_in_memory(self) -> None:
        examples = load_mmlu(MMLU_RECORDS)
        assert len(examples) == 3
        assert isinstance(examples[0], MMLUExample)
        assert examples[0].answer_letter == "B"
        assert examples[1].answer_letter == "C"  # index 2
        assert examples[0].subject == "math"

    def test_load_jsonl(self, tmp_path) -> None:
        path = os.path.join(tmp_path, "mmlu.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for rec in MMLU_RECORDS:
                f.write(json.dumps(rec) + "\n")
        examples = load_mmlu(path)
        assert len(examples) == 3
        assert examples[2].answer_letter == "B"

    def test_missing_field_raises(self) -> None:
        with pytest.raises(ValueError):
            load_mmlu([{"question": "q", "choices": ["a"]}])

    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_mmlu("/no/such/mmlu.jsonl")

    def test_prompt_contains_choices(self) -> None:
        ex = load_mmlu(MMLU_RECORDS)[0]
        prompt = build_mmlu_prompt(ex)
        assert "A. 3" in prompt
        assert "B. 4" in prompt
        assert prompt.rstrip().endswith("Answer:")


class TestChoiceExtraction:
    @pytest.mark.parametrize(
        "text,expected",
        [
            (" B", "B"),
            ("The answer is C.", "C"),
            ("(A)", "A"),
            ("D) because", "D"),
            ("answer: b", "B"),
            ("nothing here 123", None),
            ("", None),
        ],
    )
    def test_extract_choice(self, text, expected) -> None:
        assert extract_choice(text, num_choices=4) == expected

    def test_extract_respects_num_choices(self) -> None:
        # With only 2 choices, "C"/"D" are invalid; "z" then "B" -> B.
        assert extract_choice("C then B", num_choices=2) == "B"


# --------------------------------------------------------------------------- #
# GSM8K
# --------------------------------------------------------------------------- #


class TestGSM8KLoader:
    def test_load_in_memory(self) -> None:
        examples = load_gsm8k(GSM8K_RECORDS)
        assert len(examples) == 2
        assert isinstance(examples[0], GSM8KExample)
        assert examples[0].gold_answer == "5"
        assert examples[1].gold_answer == "1250"  # commas stripped

    def test_load_jsonl(self, tmp_path) -> None:
        path = os.path.join(tmp_path, "gsm8k.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for rec in GSM8K_RECORDS:
                f.write(json.dumps(rec) + "\n")
        examples = load_gsm8k(path)
        assert examples[0].gold_answer == "5"

    def test_missing_field_raises(self) -> None:
        with pytest.raises(ValueError):
            load_gsm8k([{"question": "q"}])

    def test_prompt_format(self) -> None:
        ex = load_gsm8k(GSM8K_RECORDS)[0]
        prompt = build_gsm8k_prompt(ex)
        assert prompt.startswith("Question:")
        assert prompt.rstrip().endswith("Answer:")


class TestNumberExtraction:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("The answer is 42.", "42"),
            ("foo #### 7", "7"),
            ("step 1, step 2, result 99", "99"),  # last number
            ("price is 1,234", "1234"),  # commas
            ("value: 3.5", "3.5"),  # float
            ("answer 42.0", "42"),  # integral float normalized
            ("negative -8 here", "-8"),
            ("no numbers at all", None),
            ("#### 1,000,000", "1000000"),
        ],
    )
    def test_extract_final_number(self, text, expected) -> None:
        assert extract_final_number(text) == expected

    def test_hash_marker_priority(self) -> None:
        # The #### number wins over later text without a number after it.
        assert extract_final_number("blah 5 blah #### 12") == "12"


# --------------------------------------------------------------------------- #
# pass@k math
# --------------------------------------------------------------------------- #


class TestPassAtK:
    def _reference(self, n: int, c: int, k: int) -> float:
        """Exact reference using fractions and math.comb."""
        if c == 0:
            return 0.0
        if n - c < k:
            return 1.0
        return float(
            1 - Fraction(math.comb(n - c, k), math.comb(n, k))
        )

    @pytest.mark.parametrize(
        "n,c,k,expected",
        [
            (5, 2, 1, 0.4),   # 1 - C(3,1)/C(5,1) = 1 - 3/5
            (5, 2, 3, 0.9),   # 1 - C(3,3)/C(5,3) = 1 - 1/10
            (4, 1, 2, 0.5),   # 1 - C(3,2)/C(4,2) = 1 - 3/6
            (10, 0, 1, 0.0),  # no correct -> 0
            (3, 3, 1, 1.0),   # all correct -> 1
            (5, 4, 2, 1.0),   # n-c=1 < k=2 -> always at least one correct
            (1, 1, 1, 1.0),
        ],
    )
    def test_known_values(self, n, c, k, expected) -> None:
        assert pass_at_k(n, c, k) == pytest.approx(expected)

    @pytest.mark.parametrize("n", [1, 2, 5, 8, 12])
    def test_matches_reference(self, n) -> None:
        for c in range(0, n + 1):
            for k in range(1, n + 1):
                assert pass_at_k(n, c, k) == pytest.approx(
                    self._reference(n, c, k)
                )

    def test_invalid_args(self) -> None:
        with pytest.raises(ValueError):
            pass_at_k(5, 2, 0)
        with pytest.raises(ValueError):
            pass_at_k(2, 1, 5)  # k > n
        with pytest.raises(ValueError):
            pass_at_k(5, 6, 1)  # c > n


# --------------------------------------------------------------------------- #
# HumanEval loader + sandboxed execution
# --------------------------------------------------------------------------- #


class TestHumanEvalLoader:
    def test_load_in_memory(self) -> None:
        problems = load_humaneval(HUMANEVAL_RECORDS)
        assert len(problems) == 1
        assert isinstance(problems[0], HumanEvalProblem)
        assert problems[0].entry_point == "add"

    def test_load_jsonl(self, tmp_path) -> None:
        path = os.path.join(tmp_path, "he.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for rec in HUMANEVAL_RECORDS:
                f.write(json.dumps(rec) + "\n")
        problems = load_humaneval(path)
        assert problems[0].task_id == "test/add"

    def test_missing_field_raises(self) -> None:
        with pytest.raises(ValueError):
            load_humaneval([{"task_id": "x", "prompt": "p"}])


class TestSandboxedExecution:
    @pytest.fixture
    def problem(self) -> HumanEvalProblem:
        return load_humaneval(HUMANEVAL_RECORDS)[0]

    def test_correct_completion_passes(self, problem) -> None:
        assert execute_candidate(problem, "    return a + b\n", timeout=10.0)

    def test_wrong_completion_fails(self, problem) -> None:
        assert not execute_candidate(problem, "    return a - b\n", timeout=10.0)

    def test_syntax_error_fails(self, problem) -> None:
        assert not execute_candidate(problem, "    return a +\n", timeout=10.0)

    def test_runtime_exception_fails(self, problem) -> None:
        assert not execute_candidate(
            problem, "    raise RuntimeError('boom')\n", timeout=10.0
        )

    def test_infinite_loop_times_out(self, problem) -> None:
        # Must not hang the suite: a hard timeout kills it and returns False.
        assert not execute_candidate(
            problem, "    while True:\n        pass\n", timeout=2.0
        )


# --------------------------------------------------------------------------- #
# Harness / runner machinery
# --------------------------------------------------------------------------- #


class TestHarness:
    def test_eval_result_serialization(self) -> None:
        r = EvalResult(name="x", score=0.5, n=2, details={"a": 1})
        d = r.to_dict()
        assert d == {"name": "x", "score": 0.5, "n": 2, "details": {"a": 1}}
        assert "x" in str(r)

    def test_comparison_report_dict(self) -> None:
        results = [
            EvalResult("mmlu", 0.5, 3),
            EvalResult("gsm8k", 0.0, 2),
        ]
        report = comparison_report(results)
        assert report["mmlu"]["score"] == 0.5
        assert report["gsm8k"]["n"] == 2

    def test_comparison_report_string(self) -> None:
        results = [EvalResult("mmlu", 0.5, 3)]
        text = comparison_report(results, as_string=True)
        assert "mmlu" in text
        assert "Benchmark" in text

    def test_comparison_report_empty(self) -> None:
        assert isinstance(comparison_report([], as_string=True), str)

    def test_custom_benchmark(self, model, tokenizer) -> None:
        class Dummy(Benchmark):
            def __init__(self):
                super().__init__("dummy")

            def evaluate(self, model, tokenizer, **kwargs):
                return EvalResult("dummy", 1.0, 1)

        result = run_benchmark(Dummy(), model, tokenizer)
        assert result.score == 1.0


# --------------------------------------------------------------------------- #
# End-to-end tiny-model runs (machinery only; quality not asserted)
# --------------------------------------------------------------------------- #


class TestEndToEnd:
    def test_mmlu_runs(self, model, tokenizer) -> None:
        bench = MMLUBenchmark.from_source(MMLU_RECORDS)
        result = bench.evaluate(model, tokenizer, max_new_tokens=4)
        assert isinstance(result, EvalResult)
        assert result.n == 3
        assert 0.0 <= result.score <= 1.0
        assert "per_subject" in result.details
        assert set(result.details["per_subject"]) == {"math", "geography"}

    def test_gsm8k_runs(self, model, tokenizer) -> None:
        bench = GSM8KBenchmark.from_source(GSM8K_RECORDS)
        result = bench.evaluate(model, tokenizer, max_new_tokens=4)
        assert result.n == 2
        assert 0.0 <= result.score <= 1.0
        assert len(result.details["predictions"]) == 2

    def test_humaneval_runs(self, model, tokenizer) -> None:
        bench = HumanEvalBenchmark.from_source(HUMANEVAL_RECORDS)
        result = bench.evaluate(
            model,
            tokenizer,
            n_samples=2,
            k=[1, 2],
            max_new_tokens=4,
            timeout=5.0,
        )
        assert result.n == 1
        assert "pass@1" in result.details["pass_at_k"]
        assert "pass@2" in result.details["pass_at_k"]
        assert 0.0 <= result.score <= 1.0

    def test_evaluator_run_all(self, model, tokenizer) -> None:
        results, report = run_all(
            model,
            tokenizer,
            mmlu_source=MMLU_RECORDS,
            gsm8k_source=GSM8K_RECORDS,
            humaneval_source=HUMANEVAL_RECORDS,
            per_benchmark_kwargs={
                "humaneval": {"n_samples": 1, "k": 1, "timeout": 5.0},
            },
            max_new_tokens=4,
        )
        assert len(results) == 3
        assert {r.name for r in results} == {"mmlu", "gsm8k", "humaneval"}
        assert set(report) == {"mmlu", "gsm8k", "humaneval"}

    def test_evaluator_object(self, model, tokenizer) -> None:
        evaluator = Evaluator()
        evaluator.add(MMLUBenchmark.from_source(MMLU_RECORDS))
        evaluator.add(GSM8KBenchmark.from_source(GSM8K_RECORDS))
        results = evaluator.run(model, tokenizer, max_new_tokens=4)
        assert len(results) == 2
