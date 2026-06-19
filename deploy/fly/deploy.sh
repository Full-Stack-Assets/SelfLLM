#!/usr/bin/env bash
# Deploy the SelfLLM OpenAI-compatible server to Fly.io.
#
# One-time:
#   curl -L https://fly.io/install.sh | sh   # install flyctl
#   fly auth login
#   # edit `app = "..."` in fly.toml to a unique name
#   # place your trained model at ./real_model (real_model/final + tokenizer.json)
#
# Usage (from the repo root):
#   SELFLLM_API_KEY=sk-pick-a-secret ./deploy/fly/deploy.sh

set -euo pipefail

: "${SELFLLM_API_KEY:?Set SELFLLM_API_KEY to the Bearer key clients must send}"

if [[ ! -d real_model/final ]]; then
  echo "ERROR: ./real_model/final not found. Put your trained model there first." >&2
  exit 1
fi

APP="$(grep -E '^app\s*=' fly.toml | head -1 | sed -E 's/.*"(.*)".*/\1/')"
if [[ "$APP" == *CHANGEME* || -z "$APP" ]]; then
  echo "ERROR: edit the 'app' name in fly.toml to a unique value first." >&2
  exit 1
fi

# Create the app if it doesn't exist yet (idempotent).
fly apps create "$APP" 2>/dev/null || true

# Bearer key as a Fly secret (not stored in fly.toml). --stage defers the
# restart so the subsequent deploy applies it in one go.
fly secrets set "SELFLLM_API_KEY=${SELFLLM_API_KEY}" --app "$APP" --stage

# Builds the Dockerfile from the local context (includes the baked real_model/).
fly deploy --app "$APP"

echo
echo "Deployed. App URL: https://${APP}.fly.dev"
echo "Register as a custom provider with base URL https://${APP}.fly.dev/v1 and"
echo "Authorization: Bearer ${SELFLLM_API_KEY}"
