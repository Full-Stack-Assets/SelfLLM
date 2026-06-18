#!/usr/bin/env bash
# Deploy the SelfLLM OpenAI-compatible server to Google Cloud Run.
#
# Prerequisites (one-time):
#   gcloud auth login
#   gcloud config set project YOUR_PROJECT
#   gcloud services enable run.googleapis.com cloudbuild.googleapis.com
#   # Place your trained model at ./real_model (real_model/final + tokenizer.json)
#
# Usage:
#   SELFLLM_API_KEY=sk-pick-a-secret REGION=us-central1 ./deploy/cloudrun/deploy.sh
#
# The model is baked into the image (see Dockerfile + .gcloudignore); Cloud Build
# builds from the repo root, so run this from the repository root.

set -euo pipefail

SERVICE="${SERVICE:-selfllm}"
REGION="${REGION:-us-central1}"
API_KEY="${SELFLLM_API_KEY:?Set SELFLLM_API_KEY to the Bearer key clients must send}"

if [[ ! -d "real_model/final" ]]; then
  echo "ERROR: ./real_model/final not found. Put your trained model there first." >&2
  exit 1
fi

# --allow-unauthenticated makes the service publicly reachable so your app can
# call it; access is gated by our own Bearer key (SELFLLM_API_KEY), not Cloud
# Run IAM. CPU-only; the small model fits comfortably in 2Gi.
gcloud run deploy "$SERVICE" \
  --source . \
  --region "$REGION" \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --concurrency 16 \
  --min-instances 0 \
  --max-instances 4 \
  --port 8080 \
  --set-env-vars "SELFLLM_API_KEY=${API_KEY}"

echo
echo "Deployed. Service URL:"
gcloud run services describe "$SERVICE" --region "$REGION" --format='value(status.url)'
echo "Register as a custom provider with base URL <that-url>/v1 and"
echo "Authorization: Bearer ${API_KEY}"
