# Container image for the SelfLLM inference API on Fly.io (see fly.toml).
# Unlike the Vercel deployment (which serves the slim torch-free gateway),
# this image runs the full server and therefore installs torch.
FROM python:3.12.13 AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

# Install the CPU torch wheel explicitly (the default build bundles multi-GB
# CUDA libraries that bloat the image and aren't used on Fly's shared CPUs),
# then the rest of the runtime dependencies.
RUN python -m venv .venv
COPY requirements.txt ./
RUN .venv/bin/pip install --no-cache-dir \
        torch==2.12.1+cpu --index-url https://download.pytorch.org/whl/cpu \
 && .venv/bin/pip install --no-cache-dir -r requirements.txt

FROM python:3.12.13-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY --from=builder /app/.venv .venv/
COPY . .

# Serve the FastAPI app via uvicorn (present in requirements.txt; more robust
# than `fastapi run`, which needs the optional fastapi-cli extra). Bind to the
# port fly.toml forwards to (8080) on all interfaces.
EXPOSE 8080
CMD ["/app/.venv/bin/uvicorn", "selfllm.serving.server:app", \
     "--host", "0.0.0.0", "--port", "8080"]
