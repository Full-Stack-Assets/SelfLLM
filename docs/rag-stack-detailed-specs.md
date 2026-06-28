# RAG Enterprise Stack — Detailed Specs (Retrieval API, SFT Templates, Terraform)

Companion to [`rag-stack-plan.md`](./rag-stack-plan.md). This document carries the
deep-dive specs with **corrections applied inline** — each correction is marked
`> ⚠️ Correction:` so the rationale is visible. The biggest fixes:

- **ACL is derived server-side from the authenticated identity, never trusted from the
  request body** (the original `acl_tags` request filter was a privilege-escalation vector).
- **Hybrid fusion uses RRF (or normalized scores), not a raw weighted sum** of BM25 and dense
  scores, which live on incompatible scales (the original worked example was internally
  inconsistent).
- **SFT loss is masked to assistant tokens**, and the rendered citation carries `chunk_id`
  (and span offsets) so the validator can mechanically align — the original `[doc_1]` form
  dropped the chunk/span the typed `Citation` contract requires.
- **Model weights do not live on EFS**; they go on instance-local NVMe / FSx synced from S3.
  EFS is the wrong store for repeated multi-hundred-GB reads at boot.
- **vLLM runtime LoRA endpoints** are `/v1/load_lora_adapter` / `/v1/unload_lora_adapter`,
  gated by `VLLM_ALLOW_RUNTIME_LORA_UPDATING`, which vLLM documents as **not for untrusted
  production**.

---

## 1. Retrieval Service API

**Service:** `retrieval` · **Base URL:** `http://retrieval-service.internal:8080`

**Auth & tenancy.** Pick **one** internal transport-auth mechanism — mTLS for service identity
(preferred); an `X-API-Key` pre-shared key is the fallback only if mTLS isn't available. Every
request carries `X-Tenant-Id` and `X-User-Roles` (comma-separated).

> ⚠️ Correction (security): the tenant's **index/collection is selected server-side** from the
> authenticated `X-Tenant-Id` — it is structural isolation (one namespace per tenant), not a
> request filter. ACL tags are **derived from `X-User-Roles` server-side**; the request body
> MUST NOT be able to widen access. If a client supplies `acl_tags`, the server **intersects**
> them with the caller's permitted tags and never unions. This closes the privilege-escalation
> hole in the original spec where `filters.acl_tags` was trusted.

### 1.1 `POST /search`

Hybrid retrieval (BM25 + dense) with metadata filtering.

Request:

```json
{
  "queries": ["password rotation policy change", "SOC2 password update March"],
  "top_k": 50,
  "filters": {
    "source": ["wiki", "tickets"],
    "timestamp_gte": "2026-01-01T00:00:00Z",
    "timestamp_lte": "2026-06-01T00:00:00Z"
  },
  "fusion": "rrf",
  "rerank": false
}
```

- `queries`: rewritten variants from the utility model; each query must be ≤ 512 tokens — the
  service returns **HTTP 422** if a query exceeds that limit.
- `filters`: metadata constraints, AND-ed. **No `acl_tags` here** — ACL comes from the
  authenticated roles (see correction above). `source`/timestamp filters are intersected with
  the tenant namespace.
- `fusion`: `"rrf"` (default, rank-based, scale-free) or `"weighted"` (only valid on
  **normalized** scores; see correction).

> ⚠️ Correction (fusion): the original `hybrid_weights` did a raw weighted sum of `score_bm25`
> (e.g. 12.3) and `score_dense` (e.g. 0.83) — different scales, so the math is meaningless and
> the worked example (`0.88`) didn't follow from its inputs. Use **Reciprocal Rank Fusion**:
> `score_final = Σ_r 1 / (k + rank_r(doc))` over the BM25 and dense rankings (`k≈60`). RRF is
> the same in training-data assembly and serving, so `RetrievalContext.fusion_method` in the
> contract reproduces ranking deterministically. `"weighted"` is offered only with min-max or
> z-score normalization applied first.

Response:

```json
{
  "request_id": "uuid",
  "candidates": [
    {
      "doc_id": "doc_1",
      "chunk_id": "18",
      "title": "Security Policy Update",
      "source_type": "wiki",
      "timestamp": "2026-03-11T00:00:00Z",
      "acl_tags": ["team-security"],
      "text": "Effective March 15, 2026, SOC2 password rotation changes from 90 to 45 days...",
      "score_bm25": 12.3,
      "score_dense": 0.83,
      "rank_bm25": 1,
      "rank_dense": 2,
      "score_final": 0.0317,
      "score_rerank": null
    }
  ],
  "total_hits": 234
}
```

`score_final` is the RRF score (so `rank_*` are surfaced for transparency). If `rerank=true`,
`score_rerank` is populated and becomes the final ordering.

> ⚠️ Correction (embedding hygiene): the dense model (`intfloat/e5-large-v2`) **requires
> `query:` / `passage:` prefixes**. Index time uses `passage:`; query time uses `query:`.
> Forgetting the prefix silently degrades recall. Pin the same embedding model + prefix
> convention across index build and query, and assert it in `/health`.

Errors: `400` missing/invalid field · `403` tenant/role not authorized · `422` query too long
(per-query, max 512 tokens).

### 1.2 `POST /rerank`

Cross-encoder rerank over caller-supplied candidates.

```json
{
  "query": "What changed in our SOC2 password rotation policy after March?",
  "candidates": [{ "doc_id": "doc_1", "chunk_id": "18", "text": "Effective March 15, 2026..." }],
  "top_k": 20,
  "return_scores": true
}
```

> ⚠️ Note (ordering invariant): `/rerank` trusts the supplied `text`, so **ACL filtering must
> already have been applied** by `/search` (or by the orchestrator) before candidates reach
> rerank. Rerank is not an authorization boundary.

Response: candidates sorted by `score_rerank` desc, truncated to `top_k`.

### 1.3 `GET /health`

```json
{
  "status": "ok",
  "bm25_index": "wiki_v2",
  "dense_model": "intfloat/e5-large-v2",
  "dense_prefix_convention": "e5",
  "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
  "indexed_documents": 124500
}
```

---

## 2. SFT Prompt Templates

Chat-style format compatible with modern instruct models. **The renderer is the single shared
function in `libs/prompting`** used by both training-data assembly and the serving orchestrator
(the skew contract from the plan, §4). Templates vary by `task_type`.

### 2.1 Shared system prompt

```
You are a precise enterprise assistant that answers only using the provided retrieved
documents. You MUST cite every factual statement with a chunk-level reference of the form
[doc_id#chunk_id]. If the documents do not contain sufficient evidence to answer, you MUST
respond with exactly "Insufficient evidence." Never use knowledge outside the retrieved
context. You may call the search_docs tool to refine retrieval, but the final answer must be
grounded in cited chunks.
```

> ⚠️ Correction (citation format): the original system prompt asked for `(doc_id, chunk_id,
> span)` but every example cited `[doc_1]` — dropping the `chunk_id` and span that the typed
> `Citation` model and `response-validator` need to align spans. Standardize on
> **`[doc_id#chunk_id]`** inline, with the validator resolving `char_start/char_end` against
> the cited chunk's text. This makes citation faithfulness mechanically checkable. Note
> `chunk_id` is the *within-document* chunk identifier only (e.g. `"18"`); the `doc_id#chunk_id`
> form is composed in exactly one place (the renderer) so it is never doubled into
> `doc_1#doc_1#18`.

### 2.2 Task templates

**`qa_single_hop` / `qa_multi_hop`** — user message:

```
Question: {user_query}

Retrieved Documents:
---
{retrieved_docs_formatted}
---
Answer the question. If fully supported, give a concise answer with inline citations like
[doc_1#18]. If evidence is incomplete or contradictory, respond exactly "Insufficient evidence."
```

Per-document format (identical in train and serve):

```
[doc_1#18] Title: Security Policy Update (wiki, 2026-03-11)
Text: Effective March 15, 2026, SOC2 password rotation changes from 90 to 45 days...
```

Grounded assistant target:

```
The password rotation policy now requires changes every 45 days instead of 90, effective
March 15, 2026 [doc_1#18].
```

Abstention target: `Insufficient evidence.`

**`summary`** — summarize only what's present, every sentence cited `[doc_id#chunk_id]`.

**`policy`** — yes/no + supporting rule, abstain if ambiguous.

> ⚠️ Correction (ACL info-leak): the original had the model answer "You are not authorised to
> view this information" when ACL filtering removed all docs. That lets the model **announce
> authorization status**, which leaks whether restricted content exists and trains a behavior
> indistinguishable from "no such document." When the permitted candidate set is empty, the
> target is the **same** `Insufficient evidence.` as genuine no-evidence — the model must not
> infer or reveal ACL state. Authorization messaging, if any, is the orchestrator's job
> outside the model.

**`agent_tool_use`** — the model emits a tool call when evidence is insufficient, then a final
answer. System prompt advertises `search_docs(query, filters)` and `rerank(query, candidates)`.

> ⚠️ Correction (tool format): the original mixed a custom `{"tool":..., "args":...}` blob with
> a claim of "OpenAI tool-calling format." Pick the model's **native** tool-call format and let
> `tokenizer.apply_chat_template(..., tools=...)` render it, so training matches what the
> serving template produces. The hand-written JSON blob and the native `tool_calls` structure
> are not interchangeable; mixing them reintroduces skew.

### 2.3 Negative / abstention data

Include misleading-or-insufficient contexts targeting `Insufficient evidence.` **Balance** them
against an answerable-with-weak-but-sufficient counter-set so the model doesn't learn to
over-abstain; track abstention precision/recall (plan §6). SFT holds only the correct answer;
the wrong/hallucinated partner is created for DPO pairs.

### 2.4 Assembly in code (corrected)

```python
def format_sft_sample(sample: TrainingSample, tokenizer) -> dict:
    """Render via the SHARED libs/prompting templates and return input_ids plus an
    assistant-only loss mask. Importing this same function in the orchestrator is what
    keeps train == serve."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += [{"role": m.role, "content": m.text} for m in sample.conversation_context]

    docs_str = "\n".join(
        f"[{c.doc_id}#{c.chunk_id}] Title: {c.title} ({c.source_type}, {c.timestamp:%Y-%m-%d})\n"
        f"Text: {c.text}\n"
        for c in sample.retrieval.candidates
    )
    messages.append({"role": "user", "content": render_user_turn(sample.task_type, sample.user_query, docs_str)})
    messages.append({"role": "assistant", "content": sample.answer.final_text})

    # return_assistant_tokens_mask gives the loss mask: ONLY assistant tokens contribute to
    # loss; the (long) system+context+user tokens are masked out.
    enc = tokenizer.apply_chat_template(
        messages, tokenize=True, return_assistant_tokens_mask=True, return_dict=True,
    )
    return {"input_ids": enc["input_ids"], "labels_mask": enc["assistant_masks"]}
```

> ⚠️ Correction (loss masking): the original returned a bare string from
> `apply_chat_template(tokenize=False)` with no mask — training would compute loss over the
> entire prompt, including the long retrieved context. That both wastes signal and teaches the
> model to *generate* context. Return the assistant-token mask and apply loss only there. For
> tool-use, split the assistant turn into the tool call and the final answer with a `tool`-role
> message between them, each masked appropriately.

---

## 3. Terraform — vLLM Serving Node Pools (AWS)

Two Auto Scaling Groups of GPU instances: a utility pool (8B) and a primary pool (70B), behind
internal Network Load Balancers exposing the OpenAI-compatible endpoint. Corrections below
focus on **model storage**, **instance sizing**, **autoscaling triggers**, and **deploy
safety** — the original had real operational gaps.

### 3.1 Variables (corrected sizing)

```hcl
variable "vllm_image" { default = "vllm/vllm-openai:v0.6.3" }  # pinned, not :latest
variable "primary_model"         { default = "<70B-instruct>" }
variable "primary_instance_type" { default = "p4de.24xlarge" } # 8×A100 80GB (see note)
variable "utility_model"         { default = "<8B-instruct>" }
variable "utility_instance_type" { default = "g5.2xlarge" }    # 1×A10G 24GB
```

> ⚠️ Correction (sizing): `p4d.24xlarge` is 8×A100 **40 GB** (320 GB total). A 70B in fp16 is
> ~140 GB of weights, leaving ~180 GB across 8 cards for KV cache at `--max-model-len 16384` —
> workable but tight under concurrency. **`p4de.24xlarge` (8×A100 80 GB)** or H100 gives real
> headroom. **`g5.48xlarge` (8×A10G, 192 GB total) cannot host a 70B in fp16** (140 GB weights
> leave too little for KV) — it would require AWQ/GPTQ quantization; don't list it as an
> fp16 option.

> ⚠️ Correction (`:latest` image): pin a vLLM version tag. `:latest` is non-reproducible and a
> supply-chain risk in `user_data`.

### 3.2 Model storage — local NVMe / FSx, **not EFS**

> ⚠️ Correction (storage): the original mounted 140 GB+ of weights from **EFS** and read them
> at every boot. EFS has per-file latency and burst-credit throttling that make large
> sequential model loads slow and unpredictable. Instead:
> - **p4d/p4de have ~8 TB local NVMe** — sync weights from **S3** to local NVMe on boot (fast,
>   no shared-fs contention), or
> - use **FSx for Lustre** linked to the S3 bucket for a shared, high-throughput cache, or
> - **bake weights into a custom AMI** for the fastest, most reproducible cold start.
>
> EFS is fine for small, frequently-updated **LoRA adapters**, but not base weights.

### 3.3 Launch template (primary, corrected user_data)

```hcl
resource "aws_launch_template" "primary_lt" {
  name_prefix            = "vllm-primary-"
  image_id               = data.aws_ami.dlami_gpu.id
  instance_type          = var.primary_instance_type
  vpc_security_group_ids = [aws_security_group.vllm_sg.id]
  iam_instance_profile { arn = aws_iam_instance_profile.vllm.arn }  # S3 read for weights/adapters

  user_data = base64encode(<<-EOF
    #!/bin/bash
    set -euo pipefail
    # Sync base weights from S3 to local NVMe (fast cold start; no EFS throttle).
    mkdir -p /mnt/nvme/models
    aws s3 sync s3://$MODELS_BUCKET/primary/ /mnt/nvme/models/primary/
    # Adapters live on EFS (small, hot-updated) and are mounted read-only.
    mkdir -p /mnt/adapters && mount -t efs -o ro ${var.efs_id}:/adapters /mnt/adapters

    docker run -d --gpus all --network host \
      -e VLLM_ALLOW_RUNTIME_LORA_UPDATING=True \
      -v /mnt/nvme/models:/models -v /mnt/adapters:/adapters \
      ${var.vllm_image} \
      --model /models/primary \
      --tensor-parallel-size 8 \
      --max-model-len 16384 \
      --gpu-memory-utilization 0.92 \
      --enable-lora --max-loras 8 --max-lora-rank 32 \
      --lora-modules finance=/adapters/finance hr=/adapters/hr \
      --served-model-name primary --port 8000
  EOF
  )

  block_device_mappings {
    device_name = "/dev/sda1"
    ebs { volume_size = 200; volume_type = "gp3" }
  }
}
```

> ⚠️ Correction (`--lora-modules`): the original `finance=/models/adapters/finance latest` had a
> dangling `latest` token; the flag takes `name=path` pairs (space-separated for multiple).
> `--max-loras` / `--max-lora-rank` are required so the active-adapter budget and eviction are
> explicit (the plan's "seamless hot-swap" caveat).

### 3.4 Autoscaling — define the trigger

```hcl
resource "aws_autoscaling_group" "primary_asg" {
  name                = "vllm-primary-asg"
  desired_capacity    = 1
  min_size            = 1          # NOT 0: cold start re-syncs ~140 GB of weights (minutes)
  max_size            = 2
  vpc_zone_identifier = var.subnet_ids
  launch_template { id = aws_launch_template.primary_lt.id; version = "$Latest" }
  target_group_arns = [aws_lb_target_group.primary_tg.arn]
  health_check_type = "ELB"

  instance_refresh {
    strategy = "Rolling"
    preferences { min_healthy_percentage = 100 }  # never drop the only primary during deploy
  }
}

# A scaling trigger is required — desired/max alone do not autoscale.
resource "aws_autoscaling_policy" "primary_scale" {
  name                   = "vllm-primary-target-tracking"
  autoscaling_group_name = aws_autoscaling_group.primary_asg.name
  policy_type            = "TargetTrackingScaling"
  target_tracking_configuration {
    customized_metric_specification {       # e.g. vLLM num_requests_waiting via CW agent
      metric_name = "vllm_num_requests_waiting"
      namespace   = "RAGStack/vLLM"
      statistic   = "Average"
    }
    target_value = 5.0
  }
}
```

> ⚠️ Correction (autoscaling): the original set `desired/max` but no scaling **policy**, so the
> ASG would never scale. Add target tracking on a real saturation signal — queue depth
> (`num_requests_waiting`) or GPU utilization emitted to CloudWatch.

> ⚠️ Correction (deploy safety): `min_healthy_percentage = 0` on a single-instance primary means
> an instance refresh can take down the **only** replica → full outage. Use `100` (and capacity
> headroom) so a new instance is healthy before the old one is replaced. Likewise `min_size = 0`
> turns every scale-up into a multi-minute cold start; keep `min_size = 1` for the primary.

### 3.5 NLB, health checks, outputs

NLB (internal) → target group port 8000, **HTTP health check on `/health`** (vLLM's health
endpoint). Utility pool mirrors the same pattern with `--tensor-parallel-size 1` and
`--max-model-len 4096`. Outputs expose `primary_endpoint` / `utility_endpoint` private DNS for
the orchestrator.

### 3.6 Adapter updates at runtime (corrected endpoints)

> ⚠️ Correction (endpoint + safety): vLLM's runtime LoRA endpoints are
> **`POST /v1/load_lora_adapter`** and **`POST /v1/unload_lora_adapter`** (payload
> `{"lora_name", "lora_path"}`), gated by **`VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`** — not the
> original `/v1/models/load`. vLLM documents this flag as carrying security risk and **"should
> not be used in production unless it is an isolated, fully trusted environment."** So:
> - Front it behind the internal-only SG + mTLS; never expose runtime-LoRA to tenant traffic.
> - Prefer a **LoRA Resolver plugin** that resolves adapters from S3 on first request to a new
>   `model` name, or bake the adapter set at launch via `--lora-modules` and roll the ASG to
>   update — both avoid leaving the runtime-update endpoint open.
> - Either way, an adapter only serves prod traffic after clearing the canary gate (plan §5).

**Storage layout** (base on S3→NVMe, adapters on EFS):

```
s3://$MODELS_BUCKET/primary/        # 70B base weights (synced to local NVMe at boot)
s3://$MODELS_BUCKET/utility/        # 8B base weights
efs:/adapters/{finance,hr,...}/     # small LoRA adapters, hot-updated
```

---

## Sources

- vLLM LoRA Adapters (runtime load/unload endpoints, `VLLM_ALLOW_RUNTIME_LORA_UPDATING`,
  Resolver plugins): https://docs.vllm.ai/en/stable/features/lora/
