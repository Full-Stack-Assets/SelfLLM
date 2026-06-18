# Using SelfLLM as a custom (self-hosted) model provider

SelfLLM is **not** a Vercel AI Gateway model — there is no `anthropic/…` style
slug. It is a from-scratch, self-hosted model served behind an
**OpenAI-compatible** HTTP API, so it integrates as a **custom provider** with
its own base URL + API key.

## 1. Deploy the endpoint

The server (`selfllm/serving/server.py`) implements the OpenAI protocol:

| Method | Path | Auth |
|---|---|---|
| POST | `/v1/chat/completions` | Bearer (when key set) |
| POST | `/v1/completions` | Bearer |
| GET | `/v1/models` | Bearer |
| GET | `/health` | open |

The trained model (the handed-off `real_model/`) is **baked into the image**, so
put it at `./real_model` before building.

**Local Docker:**

```bash
docker build -t selfllm-server .
docker run -p 8000:8000 -e SELFLLM_API_KEY=sk-pick-a-secret selfllm-server
```

**Fly.io** (recommended — simplest; one `fly.toml`, one command):

```bash
curl -L https://fly.io/install.sh | sh    # install flyctl, then: fly auth login
# edit `app = "..."` in fly.toml to a unique name; model present at ./real_model
SELFLLM_API_KEY=sk-pick-a-secret ./deploy/fly/deploy.sh
```

`fly deploy` builds the Dockerfile from your **local directory**, so the baked
`real_model/` is included even though it's gitignored (no commit, no registry,
no project/IAM setup). `fly.toml` sets 2GB/2cpu, scale-to-zero, and a `/health`
check; the Bearer key is set as a Fly secret. The app URL
`https://<app>.fly.dev` is your provider **base URL** (append `/v1`).

**Google Cloud Run** (alternative — managed, scale-to-zero, but more setup):

```bash
gcloud auth login && gcloud config set project YOUR_PROJECT
gcloud services enable run.googleapis.com cloudbuild.googleapis.com
SELFLLM_API_KEY=sk-pick-a-secret REGION=us-central1 ./deploy/cloudrun/deploy.sh
```

Builds via Cloud Build (`--source .`; `.gcloudignore` keeps `real_model/`),
`--allow-unauthenticated` + 2Gi/2cpu/300s, public but gated by your Bearer key.

> Both build from local context, so they work with the gitignored baked model.
> Git-repo builders (Render, Railway-via-GitHub) would not have `real_model/`
> unless you commit it or fetch it at startup.

When `SELFLLM_API_KEY` is set, the `/v1/*` inference endpoints require
`Authorization: Bearer <key>`.

## 2. Provider details to register

| Field | Value |
|---|---|
| Provider type | **custom** / OpenAI-compatible |
| Base URL | `https://<your-deployment>/v1`  *(after you deploy — I can't host it)* |
| Auth | `Authorization: Bearer $SELFLLM_API_KEY` |
| Model id | `selfllm-v6` *(the server is permissive on the `model` field; `/v1/models` reports `selfllm`)* |
| Modality | **text only** (not multimodal) |
| Context window | 512 tokens (the shipped "small" model; raise if you train a bigger one) |
| Pricing | **self-hosted → no vendor token cost.** Set whatever metering rate your policy wants (0 for internal/free, or a nominal cost-of-compute). I won't invent a $ rate. |
| Recommendation | add as a **selectable** option, not the default — it's a tiny experimental model |

> Note: the model is the ~6.8M-param from-scratch model. It produces rudimentary
> text; it's useful as a real custom-provider wiring, not as a frontier model.

## 3. Catalog entries (templates — adapt to the real file shapes)

I can't see the Conductor / vibe-coding-agent repos, so these match the shapes
you described; reconcile field names with the actual files.

### Conductor — `packages/coo-engine/src/catalog.js` (`MODEL_CATALOG`)

```js
{
  id: "selfllm-v6",
  label: "SelfLLM v6 (self-hosted)",
  provider: "selfllm",            // custom provider; not a gateway slug
  type: "custom",
  capability: "chat",
  pricing: { input: 0, output: 0 }, // self-hosted; set your metering policy
  multimodal: false,
  blurb: "From-scratch self-hosted model with test-time-compute reasoning (self-consistency / best-of-N).",
}
```

If the gateway resolves custom providers by base URL + key, point it at
`https://<your-deployment>/v1` with `Authorization: Bearer $SELFLLM_API_KEY`.

### vibe-coding-agent — `ai/constants.ts`

```ts
// SUPPORTED_MODELS
"selfllm-v6",
// MODEL_NAMES
"selfllm-v6": "SelfLLM v6 (self-hosted)",
// DEFAULT_MODEL: leave as-is (keep SelfLLM selectable, not default)
```

Custom OpenAI-compatible provider (e.g. via the AI SDK's `createOpenAI`):

```ts
const selfllm = createOpenAI({
  baseURL: process.env.SELFLLM_BASE_URL,   // https://<your-deployment>/v1
  apiKey: process.env.SELFLLM_API_KEY,
});
// model("selfllm-v6") -> selfllm.chat("selfllm-v6")
```

### vibe-coding-agent — `lib/billing.ts`

```ts
// MODEL_RATES (input/output $ per token) — self-hosted, set your policy:
"selfllm-v6": { input: 0, output: 0 },
// MODEL_CREDIT_COST — credits per request/token, your call:
"selfllm-v6": 0,
```

## What you still need to provide

- The **deployed base URL** (after hosting the container).
- The **API key** value (you choose it; set as `SELFLLM_API_KEY`).
- Your **metering policy** (the pricing/credit numbers above are placeholders).
