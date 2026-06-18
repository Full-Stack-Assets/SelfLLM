# Plan V6: Reasoning & Test-Time Compute

**Theme:** spend more compute *at inference* to reason better — long
chain-of-thought, self-consistency, best-of-N with a verifier, and search —
then feed the best reasoning back into the recursive self-improvement loop.

Builds directly on existing pieces: `selfllm/cot/cot_generator.py`, the eval
answer extractors (`selfllm/eval/{mmlu,gsm8k}.py`), the debate voting utilities
(`selfllm/debate/debate.py`), `selfllm/model/quality_model.py`, and the
recursive loop + benchmark hook.

## Task 1 — Reasoning core (`selfllm/reasoning/`)

- New package with a `ReasoningStrategy` interface:
  `solve(question) -> ReasoningResult(answer, confidence, traces, num_samples)`.
- `CoTStrategy`: structured "think step by step" prompting + a scratchpad,
  with robust final-answer extraction (reuse/extend the eval extractors so
  numeric (GSM8K) and multiple-choice (MMLU) answers are parsed consistently).
- Greedy and sampled decoding paths; all strategies share one extractor layer.
- **Unify existing code:** `cot/cot_generator.py` already has
  `generate_cot_response`, `self_consistency_vote`, and
  `generate_cot_training_data`. V6 refactors these behind the strategy interface
  (no duplicate CoT logic) and adds the missing pieces below.

## Task 2 — Self-consistency

- `SelfConsistencyStrategy`: sample **N** CoT chains at temperature, extract the
  final answer from each, majority-vote → consensus answer + a confidence =
  winning fraction. Reuses the existing `ChainOfThoughtGenerator.self_consistency_vote`
  and the `selfllm/debate/debate.py` voting helpers (`normalize_answer`,
  `majority_vote`) so the answer-normalization logic lives in one place.
- Deterministic, unit-testable vote aggregation independent of the model.

## Task 3 — Best-of-N with a verifier

- `BestOfNStrategy`: generate **N** candidates, score each with a pluggable
  `Verifier`, return the top-scoring one.
- `Verifier` implementations (lightweight — no reward-model training in V6):
  a self-consistency verifier (agreement with the sampled majority) and a
  quality/heuristic verifier reusing `quality_model.py`. Interface left open so
  a learned verifier can drop in later.

## Task 4 — Search-based reasoning

- `BeamSearchReasoner`: step-level expansion over reasoning steps with a
  per-node value estimate (verifier-guided), keeping the top-k partial chains;
  a tractable from-scratch stand-in for MCTS. (MCTS variant optional/stretch.)

## Task 5 — Integration & measurement

- **Eval:** let `selfllm/eval` benchmarks run under a chosen reasoning strategy
  (e.g. evaluate GSM8K/MMLU with self-consistency or best-of-N) to **measure the
  lift vs greedy** — this is the headline success metric.
- **Recursive loop:** optional "reasoning self-distillation" — feed
  high-confidence self-consistent traces back as training samples, so the model
  learns from its own best reasoning (self-improvement via reasoning).
- **Serving:** an opt-in inference knob to apply a reasoning strategy per request.

## Success criteria

- Self-consistency / best-of-N **measurably improve** GSM8K (and MMLU) accuracy
  over greedy on the trained model, reported via the eval harness.
- Vote/verifier/search logic is deterministic and unit-tested independently of
  the model; everything runs CPU/tiny-model friendly in tests.

## Execution strategy

- **Wave 1:** reasoning core + extractors (Task 1) — the shared foundation.
- **Wave 2 (parallel):** self-consistency (Task 2), best-of-N + verifiers
  (Task 3), beam search (Task 4) — independent strategy modules.
- **Wave 3:** eval integration + lift measurement (Task 5), then optional
  recursive self-distillation and the serving knob.
