# SelfLLM API server image: OpenAI-compatible endpoint for use as a custom provider.
#
# Build:
#   docker build -t selfllm-server .
# Run (mount a trained model dir, set an API key):
#   docker run -p 8000:8000 \
#     -e SELFLLM_API_KEY=sk-your-key \
#     -e MODEL_PATH=/models/final \
#     -e TOKENIZER_PATH=/models/tokenizer.json \
#     -v /path/to/real_model:/models \
#     selfllm-server
#
# The server then exposes (Bearer-auth'd when SELFLLM_API_KEY is set):
#   POST /v1/chat/completions   GET /v1/models   GET /health (open)

FROM python:3.11-slim

WORKDIR /app

# System deps kept minimal; torch wheels are self-contained.
COPY requirements.txt setup.py ./
COPY selfllm ./selfllm
RUN pip install --no-cache-dir -e . && pip install --no-cache-dir uvicorn fastapi

ENV MODEL_PATH=/models/final \
    TOKENIZER_PATH=/models/tokenizer.json \
    PORT=8000
EXPOSE 8000

# SELFLLM_API_KEY (if set in the environment) is read by the server and enforces
# Bearer auth on the /v1/* inference endpoints.
CMD ["sh", "-c", "python -m selfllm serve --model-path \"$MODEL_PATH\" --tokenizer-path \"$TOKENIZER_PATH\""]
