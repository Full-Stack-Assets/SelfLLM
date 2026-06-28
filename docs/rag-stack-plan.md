# RAG Enterprise Stack — Refined Implementation Plan

A repo-level plan that maps the design into runnable code, service boundaries, and a
training pipeline. This revision applies a build-sequencing inversion (prove the loop
before decomposing into services) and hardens the two highest-risk areas of any
enterprise RAG fine-tune: **train/serve prompt skew** and **cross-tenant data isolation**.

> Status: design doc. No code is committed for this stack yet — this is the plan to build
> against. Where it overlaps with the existing SelfLLM serving/eval code, reuse is called
> out explicitly in §9.

---

## 0. Guiding principles (what changed from the first draft)

1. **Modular monolith first, services later.** Stand up the loop end-to-end in one
   orchestrator process with guard/validator as in-process modules. Extract services only
   where scaling forces it, behind boundaries drawn up front so extraction is mechanical.
2. **One source of truth for prompts.** A shared rendering library is imported by *both*
   training-data assembly and the serving orchestrator. This is a hard contract with a
   golden test, not a convention. It is the single biggest defense against train/serve skew.
3. **Isolation is structural, not query-dependent.** Tenant data is partitioned by
   namespace/collection in the vector store, not by a metadata filter that a single buggy
   query could forget. Tenant-private content never enters shared-model training data.
4. **Every model call on the hot path costs latency.** The query path is budgeted and
   parallelized; post-hoc checks stream-then-verify rather than blocking first token.

---

## 1. Monorepo Layout

```
rag-enterprise-stack/
├── services/
│   ├── api-gateway/            # Auth, tenant routing (FastAPI)
│   ├── orchestrator/           # Core workflow engine (monolith first)
│   ├── retrieval/              # Hybrid search + reranker
│   └── llm-serving/            # vLLM wrapper + adapter manager
│       # policy-guard and response-validator start as in-process libs in the
│       # orchestrator (see libs/guard, libs/validator) and are promoted to
│       # services only when scale or independent deploy cadence demands it.
├── libs/
│   ├── contracts/              # Pydantic source-of-truth schemas (+ generated proto)
│   ├── prompting/              # SHARED prompt + citation rendering (train == serve)
│   ├── guard/                  # Prompt-injection / policy checks (in-proc lib)
│   ├── validator/              # Citation/schema/abstention checks (in-proc lib)
│   ├── model-registry/         # Checkpoint + adapter management, canary gating
│   ├── eval-sdk/               # Eval harness, metrics calculators
│   └── tracing/                # OTel instrumentation, logging
├── training/
│   ├── configs/                # Shared YAML (single source for hyperparams)
│   ├── sft/
│   ├── dpo/
│   ├── long_context/
│   └── tool_use/
├── data/
│   ├── schemas/                # Canonical JSON schema (generated from libs/contracts)
│   ├── pipelines/              # Ingestion, chunking, ACL labeling, re-index
│   ├── governance/             # Redaction, consent, retention, tenant-scoping rules
│   ├── datasets/               # Raw/processed training/eval sets (versioned)
│   └── feedback/               # Production feedback ingestion (async)
├── infra/
├── scripts/
├── tests/                      # Integration, contract, adversarial (incl. ACL red-team)
├── pyproject.toml
└── README.md
```

Two additions over the original layout carry most of the risk reduction: **`libs/prompting`**
(skew defense) and **`data/governance`** (isolation/legal defense). `policy-guard` and
`response-validator` move from `services/` to `libs/` until they need to scale independently.

---

## 2. Service Boundaries & Interaction

Each service is a Docker container. **Standardize on one transport** for internal calls —
HTTP/JSON for the first pass (simpler tracing, large payloads tolerated; revisit gRPC only
if profiling shows serialization is a bottleneck). Do not run "gRPC or HTTP" undecided.

- **api-gateway:** OAuth/OIDC, extracts tenant id + roles, rate limits, routes to
  orchestrator. Exposes `/v1/chat/completions` (OpenAI-compatible) and `/v1/rag/query`.
- **orchestrator (FastAPI):** synchronous request/response state machine —
  rewrite → retrieve → guard → route → synthesize → validate → log. **No task queue on the
  interactive path.** A Redis-backed queue (Arq) is used only for genuinely async work:
  batch evals, feedback ingestion, re-indexing. Streaming responses are served directly.
  Session/conversation state lives in a dedicated store (Redis with TTL, keyed by
  `tenant_id + session_id`); it is *owned* here, since `conversation_context` is part of the
  contract but no other service holds it.
- **retrieval (FastAPI + lexical index + vector DB):** `/search` (hybrid) and `/rerank`
  (cross-encoder). **Tenancy is structural:** one namespace/collection per tenant, selected
  from the authenticated tenant id — never a metadata filter alone. Role-based ACL filters
  apply *within* the tenant namespace. The hybrid **fusion method is part of the contract**
  (Reciprocal Rank Fusion by default; weights configurable) so scores from different scales
  are combined deterministically and identically in training-data assembly and serving.
- **llm-serving:** vLLM wrapper exposing OpenAI-compatible endpoints. Two engines (not
  "multiplexing"): a utility engine (8–14B, 1 GPU) and a primary engine (70B, tensor
  parallel). Adapter registry with **canary/shadow gating** before an adapter serves prod
  traffic. See §5 for the realistic LoRA hot-swap constraints.

All services emit OpenTelemetry spans/metrics to a tracing backend and structured logs to a
central store. **Online guardrail monitoring** (groundedness/abstention rates, ACL-violation
alarms) runs on these streams — not just offline eval.

### Latency budget (new)

The query path can chain four model round-trips: rewrite(8B) → rerank(cross-encoder) →
guard(8B) → synth(70B), plus post-hoc validation(NLI). Define an explicit p50/p95 SLO and:

- Fan out retrieval over query variants **in parallel**.
- Replace the 8B-as-classifier guard with a **fine-tuned lightweight classifier head** to
  avoid a full generation round-trip on every request.
- Make the response validator **post-hoc**: stream the answer, verify citations as a fast
  follow, and escalate/redact asynchronously rather than blocking first token. The hard
  pre-stream gate is limited to the guard (and an ACL assertion on candidate set).

---

## 3. Data Contracts (Core Schema)

`libs/contracts` is the **single source of truth** in Pydantic; the proto and JSON-schema
artifacts are *generated*, never hand-maintained in parallel. Corrections applied:

- `timestamp` is `datetime` (tz-aware), not `str`.
- `Citation` is a typed model with character offsets so the validator can mechanically check
  span alignment against chunk text.
- `Labels` carry graded scores (0–1), not just booleans — DPO pairing and abstention
  calibration both need the gradient.
- `schema_version` is embedded in every `TrainingSample` (not only in the training config).
- The exact rendered prompt and raw completion are captured (`raw_prompt`, `raw_completion`)
  so training and replay use the tokens the model actually saw.
- Retrieval fusion method + weights are recorded on `RetrievalContext`.

```python
from datetime import datetime
from pydantic import BaseModel, UUID4, Field
from typing import List, Optional, Dict, Any, Literal


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    text: str


class Candidate(BaseModel):
    doc_id: str
    chunk_id: str
    title: str
    source_type: str
    timestamp: datetime
    acl_tags: List[str]
    text: str
    score_bm25: Optional[float] = None
    score_dense: Optional[float] = None
    score_rerank: Optional[float] = None


class RetrievalContext(BaseModel):
    query_variants: List[str]
    filters: Dict[str, Any]
    fusion_method: Literal["rrf", "weighted"] = "rrf"
    fusion_weights: Optional[Dict[str, float]] = None
    candidates: List[Candidate]


class Citation(BaseModel):
    doc_id: str
    chunk_id: str
    span: str
    char_start: int
    char_end: int


class Answer(BaseModel):
    final_text: str
    citations: List[Citation]
    abstain: bool
    confidence: float


class ToolTrace(BaseModel):
    tool: str
    args: Dict[str, Any]
    result_summary: Optional[str] = None


class Labels(BaseModel):
    # Graded 0..1 scores; booleans are derived via thresholds where needed.
    grounded: float
    complete: float
    citation_faithful: float
    needs_more_retrieval: bool
    policy_safe: bool


class TrainingSample(BaseModel):
    sample_id: UUID4
    schema_version: str
    tenant_id: str
    tenant_scope: Literal["private", "shared_ok"]  # gates shared-model training
    task_type: str
    user_query: str
    conversation_context: List[Message]
    retrieval: RetrievalContext
    answer: Answer
    tool_trace: List[ToolTrace] = Field(default_factory=list)
    labels: Labels
    raw_prompt: str        # exact rendered prompt (from libs/prompting)
    raw_completion: str    # exact raw model output
```

`tenant_scope` is the contract-level enforcement point for §8: only `shared_ok` samples are
eligible for shared base-model SFT/DPO.

---

## 4. Train/Serve Prompt Rendering (new, highest-leverage)

`libs/prompting` owns the *only* function that turns a `TrainingSample` (or a live request)
into model-ready text + the citation format. Both paths import it:

- Training-data assembly (`training/*/build_dataset.py`) renders `raw_prompt`/`raw_completion`.
- The orchestrator's synthesis step renders the live prompt.

Contract test (`tests/contract/test_prompt_skew.py`): for a fixed corpus of samples, assert
the orchestrator render and the training render are **byte-identical**. Loss masking is part
of this module: only assistant tokens contribute to loss; the (long) retrieved-context and
user tokens are masked. Tokenizer + chat template are pinned and shared so there is no drift
between phases.

---

## 5. First-Pass PyTorch/vLLM Stack

- **Utility engine (8–14B):** single vLLM instance, 1 GPU, `--max-model-len 4096`.
- **Primary engine (70B):** tensor parallelism across 2–4 GPUs. Two separate engines/processes
  for the two model sizes — vLLM serves one model per engine; this is not in-process
  multiplexing.
- **LoRA adapters (realistic constraints):** vLLM supports per-request adapter selection via
  the `model` field, and runtime add/remove exists — but it is bounded by `max_loras`,
  `max_lora_rank`, and GPU memory, with an eviction policy when the active-adapter budget is
  exceeded. The adapter manager respects these caps and pre-warms hot adapters; it does not
  assume unlimited seamless hot-swap.
- **Prefix caching** is expected to help the **system-prompt prefix**, not the retrieved
  chunks (which are rarely byte-identical across queries). Don't size capacity assuming
  context reuse.
- **Adapter rollout:** `libs/model-registry` versions adapters by commit hash + eval scores
  and gates promotion through a **shadow/canary** stage before an adapter takes prod traffic.

### Inference pipeline (orchestrator, corrected)

Pre-retrieval RAG is the default path; the model may request *more* retrieval agentically via
a `needs_more_retrieval` tool call only when evidence is insufficient (this resolves the
"pre-fetch vs. agentic tools" ambiguity and matches `Labels.needs_more_retrieval`).

```python
# Simplified orchestrator pseudo-code
query, tenant_id, role = request.user_query, request.tenant_id, request.user.role

# 1. Query rewriting (utility engine) + 2. retrieval fanned out in parallel over variants
rewritten = await call_llm(model="utility", system_prompt=REWRITE_PROMPT, messages=[...])
candidates = await retrieval.search(             # tenant namespace selected structurally
    queries=rewritten, tenant=tenant_id, acl_roles=role, fusion="rrf",
)
ranked = await retrieval.rerank(query, candidates, top_k=20)

# 3. Hard pre-stream gate: guard (classifier head) + ACL assertion on candidate set
assert all(acl_visible(c, role) for c in ranked)          # structural isolation check
if await guard.is_unsafe(query, ranked):
    return abstain_response()

# 4. Synthesis (primary engine, optional per-tenant adapter), prompt from libs/prompting
answer, tool_trace = await call_llm(
    model="primary", adapter=adapter_for(tenant_id),
    messages=render_prompt(query, ranked), tools=[search_tool_spec], temperature=0.1,
)

# 5. Stream answer now; validate citations post-hoc (async escalate/redact on failure)
schedule_async(validator.validate, answer, ranked)

# 6. Log only what governance permits (see §8)
log_sample(build_training_sample(..., tenant_scope=scope_for(tenant_id)))
return answer
```

---

## 6. Training Jobs

Shared config in `training/configs/base.yaml` is the single hyperparameter source. Reconcile
the sequence-length mismatch from the first draft (`max_seq_length` must match the curriculum
target it is used in).

```yaml
model:
  base_model: "<strong-70B-instruct>"
  utility_model: "<8B-instruct>"
data:
  train_file: "data/datasets/sft_train.jsonl"
  eval_file: "data/datasets/sft_eval.jsonl"
  schema_version: "v1"
  only_tenant_scope: "shared_ok"     # hard filter for shared-model training
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  gradient_checkpointing: true
  learning_rate: 2.0e-4
  lora_r: 16
  lora_alpha: 32
  num_epochs: 3
  max_seq_length: 8192               # SFT/DPO; long-context stage overrides per-stage
  packing: true
  logging_steps: 10
  eval_steps: 100
  save_steps: 500
```

- **Phase 1 — SFT (`training/sft`):** 4-bit base + LoRA. Render text via `libs/prompting`
  (loss on assistant tokens only). Include negative examples where context contradicts or is
  insufficient, with target "Insufficient evidence." **Balance abstention** against a
  counter-set of answerable-with-weak-but-sufficient evidence to avoid teaching
  over-abstention; track abstention precision/recall as a first-class metric.
- **Phase 2 — DPO (`training/dpo`):** start from the **SFT checkpoint**; the **reference
  model is the frozen SFT policy** (adapter-disabled base via PEFT). Pairs: grounded/concise/
  citation-faithful (chosen) vs. hallucinated/verbose/drifted (rejected), plus
  correct-abstention vs. hallucination for weak-evidence cases. Graded `Labels` drive pair
  selection.
- **Phase 3 — Long-context curriculum (`training/long_context`):** stages at 2k → 8k → 16k.
  Each stage sets its own `max_seq_length`; 16k on a 70B (4-bit + LoRA) requires gradient
  checkpointing + sequence packing + flash-attention — the memory plan is explicit, not
  implied. Many distractor chunks, few relevant.
- **Phase 4 — Tool-use alignment (`training/tool_use`):** train the *insufficient-evidence →
  `needs_more_retrieval` tool call* behavior used at serving, in the OpenAI tool-call JSON
  format, rendered through the same `libs/prompting` templates.

Artifacts push to `libs/model-registry`, versioned by commit hash + eval scores, and must
clear the canary gate (§5) before serving.

### Training infrastructure

QLoRA on a single 8×A100 node for early iterations; FSDP + DeepSpeed ZeRO-3 only if full
fine-tuning is later required. Preprocessing with the `datasets` library to Arrow. Eval runs
**async on a fixed small golden set**, not a blocking full-generation pass against the 70B
every epoch.

---

## 7. Eval Harness Integration

`libs/eval-sdk` consumes `TrainingSample` batches and computes the metrics, plugged into CI
and the (async) post-epoch eval:

```python
from eval_sdk.metrics import (
    retrieval_recall_at_k, groundedness_score, citation_faithfulness,
    abstention_precision_recall, cross_tenant_leakage_assertion,
)
```

- `cross_tenant_leakage_assertion` is **both** an offline metric **and** a runtime ACL
  assertion (citations ⊆ requester's permitted candidates) **and** a CI red-team suite — not a
  single offline score.
- Golden / adversarial sets live in `data/datasets/`, versioned with code.
- Online dashboards track groundedness/abstention drift and ACL alarms in prod.

---

## 8. Data Governance & Tenant Isolation (new)

The loop logs full user queries + retrieved enterprise documents as training data. Training a
shared base on tenant A's private content and serving tenant B is a contractual/legal breach,
not a quality bug. Rules:

1. **`tenant_scope` gate:** only `shared_ok` samples enter shared base SFT/DPO. `private`
   samples train **only** that tenant's adapter.
2. **PII redaction at ingest** and at log time; consent + retention policy in
   `data/governance`.
3. **Structural vector-store isolation:** per-tenant namespace/collection, not metadata-filter
   multi-tenancy.
4. **Runtime ACL assertion** on the candidate set and on citations, enforced before streaming.
5. **Red-team CI**: adversarial cross-tenant and prompt-injection suites must pass to ship.

---

## 9. Reuse from SelfLLM

Three existing pieces seed this stack rather than starting from zero:

- **Continuous-batching scheduler** (with sliding-window KV eviction for long contexts) →
  seeds `llm-serving` if a from-scratch serving path is wanted before adopting vLLM.
- **OpenAI-compatible FastAPI server with Bearer auth** → seeds `api-gateway` /
  `llm-serving` endpoint surface.
- **Eval suite** (pass@k, sandboxed code execution) → seeds `libs/eval-sdk`.

---

## 10. Suggested Build Sequence (re-ordered)

1. **`libs/contracts` + `libs/prompting`** with the byte-identical skew contract test.
2. **Modular-monolith orchestrator** — guard/validator in-process, real retrieval (one tenant
   namespace), real vLLM. Prove the full loop end-to-end and streaming.
3. **Collect data through the loop** under the `tenant_scope` governance gate.
4. **SFT → DPO** (correct reference-model handling), then long-context and tool-use phases.
5. **Extract services** (`retrieval`, then `policy-guard`/`response-validator`) only where
   scaling or deploy-cadence demands it, behind the boundaries already drawn.

This front-loads the cheap part (process boundaries) only after the expensive risks
(prompt skew, tenant isolation, a working inference loop) are retired.
