# Vertex AI Gemini tuning starter

Use this path when you want SelfLLM to prepare data and evaluation workflows,
but use a managed Gemini foundation model for the actual capability.

## 1. Try retrieval before tuning

Pack local documents into a grounded prompt:

```bash
python -m selfllm rag-pack \
  --docs docs README.md \
  --query "How should we run Gemini tuning?" \
  --prompt-output-path vertex/rag_prompt.txt \
  --manifest-output-path vertex/rag_manifest.json \
  --top-k 5
```

Send `vertex/rag_prompt.txt` to Gemini first. If retrieval solves the task,
prefer this cheaper path. If you repeatedly edit the answer into the same target
style, save those prompt/ideal-response pairs for supervised tuning.

## 2. Prepare supervised examples

Create a JSON or JSONL file with prompt/response examples:

```json
{"prompt": "Summarize this support ticket: ...", "response": "Customer cannot reset password; next step is ..."}
```

Export Gemini supervised tuning JSONL:

```bash
python -m selfllm vertex-export \
  --input-path data/sft_samples.jsonl \
  --train-output-path vertex/train.jsonl \
  --validation-output-path vertex/validation.jsonl \
  --validation-ratio 0.2 \
  --system-instruction "Answer in the product team's support style."
```

Response-only pretraining chunks are skipped by default because Vertex
supervised tuning needs an input prompt and target model response.

## 3. Upload JSONL to Cloud Storage

```bash
gsutil cp vertex/train.jsonl gs://YOUR_BUCKET/selfllm/train.jsonl
gsutil cp vertex/validation.jsonl gs://YOUR_BUCKET/selfllm/validation.jsonl
```

## 4. Create a tuning request plan

```bash
python -m selfllm vertex-tune-plan \
  --project-id YOUR_PROJECT \
  --location us-central1 \
  --base-model gemini-2.5-flash \
  --training-dataset-uri gs://YOUR_BUCKET/selfllm/train.jsonl \
  --validation-dataset-uri gs://YOUR_BUCKET/selfllm/validation.jsonl \
  --adapter-size 4 \
  --tuned-model-display-name selfllm-gemini-adapter \
  --plan-output-path vertex/tuning_request.json
```

The plan file contains the REST endpoint and request body for
`tuningJobs.create`. It does not call Google Cloud, so it is safe to generate in
local and CI environments without credentials.
