# SelfLLM API server image: OpenAI-compatible endpoint for use as a custom provider.
#
# The trained model is BAKED into the image (Cloud Run and other --source/buildpack
# deploys can't mount volumes). Place your trained model at ./real_model before
# building (the handed-off artifact: real_model/final, real_model/tokenizer.json).
#
# Build:      docker build -t selfllm-server .
# Run local:  docker run -p 8000:8000 -e SELFLLM_API_KEY=sk-your-key selfllm-server
# Cloud Run:  see deploy/cloudrun/deploy.sh
#
# Endpoints (Bearer-auth'd when SELFLLM_API_KEY is set): POST /v1/chat/completions,
# POST /v1/completions, GET /v1/models; GET /health is open. The server listens on
# $PORT (Cloud Run injects it; defaults to 8000 locally).

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt setup.py ./
COPY selfllm ./selfllm
RUN pip install --no-cache-dir -e . && pip install --no-cache-dir uvicorn fastapi

# Bake the trained model into the image.
COPY real_model ./real_model

ENV MODEL_PATH=/app/real_model/final \
    TOKENIZER_PATH=/app/real_model/tokenizer.json \
    PORT=8000
EXPOSE 8000

# SELFLLM_API_KEY (set in the environment) enforces Bearer auth on /v1/*.
CMD ["sh", "-c", "python -m selfllm serve --model-path \"$MODEL_PATH\" --tokenizer-path \"$TOKENIZER_PATH\""]
