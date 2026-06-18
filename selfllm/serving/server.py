"""OpenAI-compatible API server for SelfLLM.

Provides FastAPI endpoints implementing the OpenAI API protocol:
    POST /v1/chat/completions   -- Chat completions (with streaming support)
    POST /v1/completions        -- Text completions
    GET  /v1/models             -- List available models
    GET  /health                -- Health check
    GET  /v1/stats              -- Serving statistics

The server integrates with PagedAttention KV cache and continuous batching
for efficient concurrent request handling.
"""

import asyncio
import json
import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# --- Pydantic Models ---


class ChatMessage(BaseModel):
    """A single message in a chat conversation."""

    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """Request body for /v1/chat/completions endpoint."""

    model: str = "selfllm"
    messages: List[ChatMessage]
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    stream: bool = False
    stop: Optional[List[str]] = None


class ChatCompletionResponse(BaseModel):
    """Response body for /v1/chat/completions endpoint."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]


class CompletionRequest(BaseModel):
    """Request body for /v1/completions endpoint."""

    model: str = "selfllm"
    prompt: str
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    stream: bool = False


class ModelInfo(BaseModel):
    """Information about a single model."""

    id: str
    object: str = "model"
    created: int
    owned_by: str = "selfllm"


# --- Global State ---

_model: Optional[torch.nn.Module] = None
_tokenizer: Optional[Any] = None
_scheduler: Optional[Any] = None

# Safety cap: an endpoint never waits longer than this for a scheduled request
# to finish, so a wedged/un-schedulable request cannot hang the connection.
_REQUEST_TIMEOUT_S = 300.0

# --- FastAPI App ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events for the FastAPI application."""
    logger.info("SelfLLM API server starting...")
    yield
    if _scheduler is not None:
        _scheduler.shutdown()
    logger.info("SelfLLM API server stopped.")


app = FastAPI(title="SelfLLM API", version="2.0.0", lifespan=lifespan)


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint.

    Supports both streaming (SSE) and non-streaming responses.

    Args:
        request: ChatCompletionRequest with messages and generation params.

    Returns:
        ChatCompletionResponse for non-streaming, or StreamingResponse
        for streaming (SSE).

    Raises:
        HTTPException: 503 if model is not loaded.
    """
    if _model is None or _tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Convert messages to prompt string
    prompt = _messages_to_prompt(request.messages)
    prompt_tokens = _tokenizer.encode(prompt)

    if request.stream:
        return StreamingResponse(
            _stream_generate(prompt_tokens, request),
            media_type="text/event-stream",
        )

    # Non-streaming: generate all at once
    req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    if _scheduler is not None:
        # Route through the continuous-batching scheduler so concurrent
        # requests share batched forward passes.
        gen_ids = await _generate_via_scheduler(prompt_tokens, request)
        response_text = _tokenizer.decode(gen_ids)
        completion_tokens = len(gen_ids)
    else:
        # Fallback: decode directly when no scheduler is configured.
        device = next(_model.parameters()).device
        input_ids = torch.tensor([prompt_tokens], device=device)
        with torch.no_grad():
            output = _model.generate(
                input_ids,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop_token_id=_tokenizer.eos_token_id,
            )
        all_token_ids = output["sequences"][0].tolist()
        generated_text = _tokenizer.decode(all_token_ids)
        response_text = generated_text[len(prompt) :]
        completion_tokens = max(0, len(output["sequences"][0]) - len(prompt_tokens))

    return ChatCompletionResponse(
        id=req_id,
        created=int(time.time()),
        model=request.model,
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop",
            }
        ],
        usage={
            "prompt_tokens": len(prompt_tokens),
            "completion_tokens": completion_tokens,
            "total_tokens": len(prompt_tokens) + completion_tokens,
        },
    )


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    """Text completions endpoint.

    Args:
        request: CompletionRequest with prompt and generation params.

    Returns:
        JSON response with generated text and usage statistics.

    Raises:
        HTTPException: 503 if model is not loaded.
    """
    if _model is None or _tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    prompt_tokens = _tokenizer.encode(request.prompt)

    if _scheduler is not None:
        gen_ids = await _generate_via_scheduler(
            prompt_tokens,
            ChatCompletionRequest(
                model=request.model,
                messages=[],
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
            ),
        )
        response_text = _tokenizer.decode(gen_ids)
        completion_tokens = len(gen_ids)
    else:
        device = next(_model.parameters()).device
        input_ids = torch.tensor([prompt_tokens], device=device)
        with torch.no_grad():
            output = _model.generate(
                input_ids,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop_token_id=_tokenizer.eos_token_id,
            )
        all_token_ids = output["sequences"][0].tolist()
        generated_text = _tokenizer.decode(all_token_ids)
        response_text = generated_text[len(request.prompt) :]
        completion_tokens = max(0, len(output["sequences"][0]) - len(prompt_tokens))

    return {
        "id": f"cmpl-{uuid.uuid4().hex[:12]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {"text": response_text, "index": 0, "finish_reason": "stop"}
        ],
        "usage": {
            "prompt_tokens": len(prompt_tokens),
            "completion_tokens": completion_tokens,
            "total_tokens": len(prompt_tokens) + completion_tokens,
        },
    }


@app.get("/v1/models")
async def list_models():
    """List available models.

    Returns:
        Object list with model metadata.
    """
    return {
        "object": "list",
        "data": [
            ModelInfo(
                id="selfllm", created=int(time.time())
            ).model_dump()
        ],
    }


@app.get("/health")
async def health():
    """Health check endpoint.

    Returns:
        Status information including model and scheduler state.
    """
    return {
        "status": "healthy",
        "model_loaded": _model is not None,
        "scheduler_active": _scheduler is not None,
    }


@app.get("/v1/stats")
async def stats():
    """Serving statistics endpoint.

    Returns:
        Scheduler statistics if available.
    """
    if _scheduler is not None:
        return _scheduler.stats
    return {"error": "Scheduler not initialized"}


# --- Helpers ---


def _messages_to_prompt(messages: List[ChatMessage]) -> str:
    """Convert chat messages to a single prompt string.

    Formats messages as:
        System: {content}
        User: {content}
        Assistant: {content}
        Assistant:

    Args:
        messages: List of chat messages.

    Returns:
        Formatted prompt string ending with "Assistant:".
    """
    parts = []
    for msg in messages:
        if msg.role == "system":
            parts.append(f"System: {msg.content}\n")
        elif msg.role == "user":
            parts.append(f"User: {msg.content}\n")
        elif msg.role == "assistant":
            parts.append(f"Assistant: {msg.content}\n")
    parts.append("Assistant:")
    return "".join(parts)


async def _stream_generate(
    prompt_tokens: List[int],
    request: ChatCompletionRequest,
) -> AsyncGenerator[str, None]:
    """Stream generation results via Server-Sent Events.

    Generates one token at a time and yields SSE data events.

    Args:
        prompt_tokens: Encoded prompt token IDs.
        request: Generation parameters.

    Yields:
        SSE-formatted strings with generated token content.
    """
    # Route through the continuous-batching scheduler when one is configured.
    if _scheduler is not None:
        async for chunk in _stream_via_scheduler(prompt_tokens, request):
            yield chunk
        return

    device = next(_model.parameters()).device

    # Track the full running context. Each step regenerates from the complete
    # prompt + tokens emitted so far; feeding only the last token would strip
    # all prior context (generate() keeps no KV cache across calls).
    generated_ids: List[int] = []
    emitted_text = ""
    stop_sequences = request.stop or []

    for _ in range(request.max_tokens):
        input_ids = torch.tensor([prompt_tokens + generated_ids], device=device)
        with torch.no_grad():
            output = _model.generate(
                input_ids,
                max_new_tokens=1,
                temperature=request.temperature,
                top_p=request.top_p,
            )

        token_id = output["sequences"][0, -1].item()

        if token_id == _tokenizer.eos_token_id:
            break

        generated_ids.append(token_id)
        # Decode cumulatively so stop sequences spanning multiple tokens are
        # detected and the incremental delta is exact.
        full_text = _tokenizer.decode(generated_ids)

        stop_idx = _first_stop_index(full_text, stop_sequences)
        if stop_idx is not None:
            remaining = full_text[len(emitted_text):stop_idx]
            if remaining:
                yield f"data: {json.dumps({'choices': [{'delta': {'content': remaining}}]})}\n\n"
            break

        new_text = full_text[len(emitted_text):]
        if new_text:
            yield f"data: {json.dumps({'choices': [{'delta': {'content': new_text}}]})}\n\n"
            emitted_text = full_text

    yield "data: [DONE]\n\n"


def _first_stop_index(text: str, stop_sequences: List[str]) -> Optional[int]:
    """Return the earliest index at which any stop sequence occurs, else None."""
    best: Optional[int] = None
    for stop in stop_sequences:
        if not stop:
            continue
        idx = text.find(stop)
        if idx != -1 and (best is None or idx < best):
            best = idx
    return best


async def _generate_via_scheduler(
    prompt_tokens: List[int], request: ChatCompletionRequest
) -> List[int]:
    """Submit a request to the scheduler and await completion.

    Returns the generated token ids (excluding a trailing EOS token).
    """
    req = _scheduler.create_request(
        prompt_tokens=prompt_tokens,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
    )
    deadline = time.monotonic() + _REQUEST_TIMEOUT_S
    while not req.is_finished:
        if time.monotonic() > deadline:
            break
        await asyncio.sleep(0.005)
    gen = list(req.generated_tokens)
    if gen and gen[-1] == _tokenizer.eos_token_id:
        gen = gen[:-1]
    return gen


async def _stream_via_scheduler(
    prompt_tokens: List[int], request: ChatCompletionRequest
) -> AsyncGenerator[str, None]:
    """Stream a scheduler-driven request token-by-token as SSE events."""
    req = _scheduler.create_request(
        prompt_tokens=prompt_tokens,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
    )
    stop_sequences = request.stop or []
    emitted_text = ""
    emitted_count = 0
    deadline = time.monotonic() + _REQUEST_TIMEOUT_S

    while True:
        gen = list(req.generated_tokens)
        # Don't display a trailing EOS token.
        display = gen[:-1] if (gen and gen[-1] == _tokenizer.eos_token_id) else gen

        if len(display) > emitted_count:
            full_text = _tokenizer.decode(display)
            stop_idx = _first_stop_index(full_text, stop_sequences)
            if stop_idx is not None:
                remaining = full_text[len(emitted_text):stop_idx]
                if remaining:
                    yield f"data: {json.dumps({'choices': [{'delta': {'content': remaining}}]})}\n\n"
                break
            new_text = full_text[len(emitted_text):]
            if new_text:
                yield f"data: {json.dumps({'choices': [{'delta': {'content': new_text}}]})}\n\n"
                emitted_text = full_text
            emitted_count = len(display)

        if req.is_finished and emitted_count >= len(display):
            break
        if time.monotonic() > deadline:
            break
        await asyncio.sleep(0.005)

    yield "data: [DONE]\n\n"


def serve(
    model_path: str,
    tokenizer_path: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    max_batch_size: int = 32,
    block_size: int = 16,
    num_blocks: Optional[int] = None,
    kv_cache_mem_gb: float = 4.0,
) -> None:
    """Launch the API server with PagedAttention backend.

    Loads the model and tokenizer, sets up the PagedAttention block manager
    and continuous batching scheduler, then starts the uvicorn server.

    Args:
        model_path: Path to the model checkpoint directory.
        tokenizer_path: Path to the tokenizer file.
        host: Host address to bind to.
        port: Port to listen on.
        max_batch_size: Maximum concurrent requests in the scheduler.
        block_size: Tokens per KV cache block.
        num_blocks: Number of KV cache blocks. If ``None``, derived from
            ``kv_cache_mem_gb`` and the model's KV footprint per block.
        kv_cache_mem_gb: Target KV cache budget (GB) used to size ``num_blocks``
            when it is not given explicitly.
    """
    import uvicorn

    global _model, _tokenizer, _scheduler

    from selfllm.model.model import SelfImprovingLLM
    from selfllm.model.tokenizer import BPETokenizer
    from selfllm.serving.paged_cache import BlockManager
    from selfllm.serving.scheduler import ContinuousBatchingScheduler

    # Load tokenizer
    _tokenizer = BPETokenizer(vocab_size=32000)
    _tokenizer.load(tokenizer_path)

    # Load model
    _model = SelfImprovingLLM.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _model = _model.to(device).eval()

    # Setup PagedAttention block manager. Size the pool from a memory budget
    # instead of a hard-coded constant so it scales with model dimensions.
    config = _model.config
    head_dim = config.d_model // config.n_heads
    if num_blocks is None:
        dtype_bytes = 2  # fp16/bf16 KV cache
        bytes_per_block = (
            2  # K + V
            * config.n_layers
            * block_size
            * config.n_heads
            * head_dim
            * dtype_bytes
        )
        num_blocks = max(1, int(kv_cache_mem_gb * (1024 ** 3) // bytes_per_block))
    block_manager = BlockManager(
        num_blocks=num_blocks,
        block_size=block_size,
        num_layers=config.n_layers,
        num_heads=config.n_heads,
        head_dim=head_dim,
        device=device,
    )

    # Setup continuous batching scheduler
    _scheduler = ContinuousBatchingScheduler(
        model=_model,
        tokenizer=_tokenizer,
        block_manager=block_manager,
        max_batch_size=max_batch_size,
        device=device,
    )

    # Drive the scheduler in a background thread so the HTTP endpoints can
    # enqueue requests and await their completion while batched decoding runs.
    scheduler_thread = threading.Thread(target=_scheduler.run, daemon=True)
    scheduler_thread.start()

    param_count = sum(p.numel() for p in _model.parameters()) / 1e6
    logger.info(f"Server ready on {host}:{port}")
    logger.info(f"Model: {param_count:.1f}M params")
    logger.info(f"Device: {device}")
    logger.info(f"Max batch size: {max_batch_size}")

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SelfLLM API Server")
    parser.add_argument("--model-path", required=True, help="Path to model checkpoint")
    parser.add_argument("--tokenizer-path", required=True, help="Path to tokenizer")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument(
        "--max-batch-size", type=int, default=32, help="Maximum batch size"
    )
    args = parser.parse_args()

    serve(args.model_path, args.tokenizer_path, args.host, args.port, args.max_batch_size)
