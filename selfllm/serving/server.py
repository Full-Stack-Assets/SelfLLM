"""OpenAI-compatible API server for SelfLLM.

Provides FastAPI endpoints implementing the OpenAI API protocol:
    POST /v1/chat/completions   -- Chat completions (with streaming support)
    POST /v1/completions        -- Text completions
    GET  /v1/models             -- List available models
    GET  /health                -- Health check
    GET  /v1/stats              -- Serving statistics
    GET  /chat                  -- Browser chat UI

The server integrates with PagedAttention KV cache and continuous batching
for efficient concurrent request handling.
"""

import asyncio
import json
import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, Optional

import torch
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
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
    # Opt-in test-time-compute reasoning. e.g.
    # {"strategy": "self_consistency", "num_samples": 8, "answer_type": "free"}.
    reasoning: Optional[Dict[str, Any]] = None


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

# Optional API key for the inference endpoints. When set (via the
# SELFLLM_API_KEY env var or serve(api_key=...)), requests must send
# `Authorization: Bearer <key>` -- this is what lets SelfLLM be registered as a
# custom OpenAI-compatible provider. When unset, auth is disabled (open).
_api_key: Optional[str] = os.environ.get("SELFLLM_API_KEY")

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

_CHAT_UI_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>SelfLLM Web Chat</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0f172a;
      --panel: #111827;
      --panel-2: #1f2937;
      --border: #334155;
      --text: #e5e7eb;
      --muted: #9ca3af;
      --accent: #22c55e;
      --accent-hover: #16a34a;
      --error: #f43f5e;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
      background: linear-gradient(180deg, #0b1224, var(--bg));
      color: var(--text);
      min-height: 100vh;
      display: flex;
      justify-content: center;
      padding: 24px;
    }
    .app {
      width: min(980px, 100%);
      display: grid;
      gap: 16px;
      grid-template-columns: 280px 1fr;
    }
    .panel {
      background: rgba(17, 24, 39, 0.92);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 14px;
      backdrop-filter: blur(4px);
    }
    .settings { display: grid; gap: 10px; align-content: start; }
    label {
      font-size: 12px;
      color: var(--muted);
      display: grid;
      gap: 6px;
    }
    input, textarea, button {
      font: inherit;
      border-radius: 10px;
      border: 1px solid var(--border);
      background: var(--panel-2);
      color: var(--text);
      padding: 10px 12px;
    }
    input:focus, textarea:focus {
      outline: 2px solid #2563eb;
      outline-offset: 0;
    }
    button {
      cursor: pointer;
      background: var(--accent);
      border-color: var(--accent);
      color: #06250f;
      font-weight: 600;
    }
    button:hover { background: var(--accent-hover); border-color: var(--accent-hover); }
    button:disabled { opacity: 0.65; cursor: not-allowed; }
    .secondary {
      background: transparent;
      color: var(--text);
      border-color: var(--border);
    }
    .chat {
      display: grid;
      grid-template-rows: auto 1fr auto;
      gap: 10px;
      min-height: 72vh;
    }
    h1 { margin: 0; font-size: 18px; }
    .subtitle { margin: 0; color: var(--muted); font-size: 13px; }
    .messages {
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 12px;
      overflow-y: auto;
      background: rgba(15, 23, 42, 0.55);
      display: grid;
      gap: 10px;
      align-content: start;
    }
    .message {
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 10px 12px;
      line-height: 1.45;
      white-space: pre-wrap;
    }
    .user { background: rgba(37, 99, 235, 0.13); }
    .assistant { background: rgba(20, 83, 45, 0.22); }
    .message b {
      display: block;
      margin-bottom: 6px;
      color: #cbd5e1;
      font-size: 12px;
      letter-spacing: 0.02em;
    }
    .controls {
      display: grid;
      gap: 8px;
    }
    .controls-row {
      display: flex;
      gap: 8px;
    }
    .controls-row button { min-width: 120px; }
    .status {
      min-height: 20px;
      font-size: 12px;
      color: var(--muted);
    }
    .status.error { color: var(--error); }
    @media (max-width: 900px) {
      .app { grid-template-columns: 1fr; }
      .chat { min-height: 70vh; }
    }
  </style>
</head>
<body>
  <main class="app">
    <section class="panel settings">
      <h1>SelfLLM Web Chat</h1>
      <p class="subtitle">Talk to the currently loaded local model via <code>/v1/chat/completions</code>.</p>

      <label>Model
        <input id="model" value="selfllm" />
      </label>
      <label>Max tokens
        <input id="maxTokens" type="number" min="1" max="4096" value="256" />
      </label>
      <label>Temperature
        <input id="temperature" type="number" min="0" max="2" step="0.1" value="0.8" />
      </label>
      <label>Top-p
        <input id="topP" type="number" min="0" max="1" step="0.05" value="0.95" />
      </label>
      <label>API key (optional)
        <input id="apiKey" type="password" placeholder="Used as Bearer token for protected servers" />
      </label>
      <button id="clearBtn" class="secondary" type="button">Clear conversation</button>
    </section>

    <section class="panel chat">
      <div>
        <h1>Chat</h1>
        <p class="subtitle">Press Enter to send, Shift+Enter for newline.</p>
      </div>
      <div id="messages" class="messages"></div>
      <div class="controls">
        <textarea id="prompt" rows="5" placeholder="Type your message..."></textarea>
        <div class="controls-row">
          <button id="sendBtn" type="button">Send</button>
        </div>
        <div id="status" class="status"></div>
      </div>
    </section>
  </main>

  <script>
    const messagesEl = document.getElementById("messages");
    const promptEl = document.getElementById("prompt");
    const sendBtn = document.getElementById("sendBtn");
    const clearBtn = document.getElementById("clearBtn");
    const statusEl = document.getElementById("status");
    const modelEl = document.getElementById("model");
    const maxTokensEl = document.getElementById("maxTokens");
    const temperatureEl = document.getElementById("temperature");
    const topPEl = document.getElementById("topP");
    const apiKeyEl = document.getElementById("apiKey");

    let history = [];
    let busy = false;

    function setStatus(message, isError = false) {
      statusEl.textContent = message || "";
      statusEl.className = isError ? "status error" : "status";
    }

    function appendMessage(role, content) {
      const node = document.createElement("article");
      node.className = `message ${role}`;
      const title = document.createElement("b");
      title.textContent = role === "user" ? "User" : "Assistant";
      const text = document.createElement("div");
      text.textContent = content;
      node.appendChild(title);
      node.appendChild(text);
      messagesEl.appendChild(node);
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    function renderWelcome() {
      messagesEl.innerHTML = "";
      appendMessage("assistant", "Hello. I am your local SelfLLM model. Ask me anything.");
    }

    async function sendMessage() {
      if (busy) return;
      const text = promptEl.value.trim();
      if (!text) return;

      const payload = {
        model: modelEl.value.trim() || "selfllm",
        messages: [...history, { role: "user", content: text }],
        max_tokens: Number(maxTokensEl.value) || 256,
        temperature: Number(temperatureEl.value),
        top_p: Number(topPEl.value),
      };

      const headers = { "Content-Type": "application/json" };
      const apiKey = apiKeyEl.value.trim();
      if (apiKey) headers["Authorization"] = `Bearer ${apiKey}`;

      busy = true;
      sendBtn.disabled = true;
      promptEl.disabled = true;
      appendMessage("user", text);
      setStatus("Generating response...");

      try {
        const response = await fetch("/v1/chat/completions", {
          method: "POST",
          headers,
          body: JSON.stringify(payload),
        });

        if (!response.ok) {
          let detail = `${response.status} ${response.statusText}`;
          try {
            const err = await response.json();
            if (err && err.detail) detail = String(err.detail);
          } catch (_e) {
            // No JSON body; keep default detail.
          }
          throw new Error(detail);
        }

        const data = await response.json();
        const assistantText = data.choices?.[0]?.message?.content ?? "";
        appendMessage("assistant", assistantText || "(empty response)");

        history.push({ role: "user", content: text });
        history.push({ role: "assistant", content: assistantText });
        promptEl.value = "";
        setStatus("Done.");
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        setStatus(`Request failed: ${message}`, true);
      } finally {
        busy = false;
        sendBtn.disabled = false;
        promptEl.disabled = false;
        promptEl.focus();
      }
    }

    clearBtn.addEventListener("click", () => {
      history = [];
      promptEl.value = "";
      setStatus("");
      renderWelcome();
      promptEl.focus();
    });

    sendBtn.addEventListener("click", sendMessage);
    promptEl.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
      }
    });

    renderWelcome();
    promptEl.focus();
  </script>
</body>
</html>
"""


@app.get("/", include_in_schema=False)
async def root():
    """Redirect the root URL to the browser chat UI."""
    return RedirectResponse(url="/chat")


@app.get("/chat", response_class=HTMLResponse)
async def chat_ui():
    """Serve a minimal in-browser chat UI for the local model."""
    return HTMLResponse(_CHAT_UI_HTML)


async def _check_api_key(authorization: Optional[str] = Header(None)) -> None:
    """Bearer-token auth for the inference endpoints.

    No-op when no API key is configured (open server). When a key is set,
    requires ``Authorization: Bearer <key>`` and rejects anything else with 401
    -- the standard scheme an OpenAI-compatible custom provider uses.
    """
    if _api_key is None:
        return
    expected = f"Bearer {_api_key}"
    if authorization != expected:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )


@app.post("/v1/chat/completions", dependencies=[Depends(_check_api_key)])
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

    # Opt-in reasoning (test-time compute): bypass plain decoding and run a
    # reasoning strategy, returning its answer. Non-streaming.
    if request.reasoning:
        content = _run_reasoning(prompt, request.reasoning)
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=request.model,
            choices=[{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
            usage={
                "prompt_tokens": len(prompt_tokens),
                "completion_tokens": len(_tokenizer.encode(content)),
                "total_tokens": len(prompt_tokens) + len(_tokenizer.encode(content)),
            },
        )

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


@app.post("/v1/completions", dependencies=[Depends(_check_api_key)])
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


@app.get("/v1/models", dependencies=[Depends(_check_api_key)])
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


@app.get("/v1/stats", dependencies=[Depends(_check_api_key)])
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


def _run_reasoning(prompt: str, opts: Dict[str, Any]) -> str:
    """Apply a test-time-compute reasoning strategy to ``prompt``.

    ``opts`` selects the strategy and its parameters, e.g.
    ``{"strategy": "self_consistency", "num_samples": 8, "answer_type": "free"}``.
    Returns the strategy's answer (falling back to its best trace, then the raw
    decode) so the endpoint always yields content.
    """
    from selfllm.reasoning import (
        BestOfNStrategy,
        CoTStrategy,
        SelfConsistencyStrategy,
        SelfConsistencyVerifier,
    )

    device = str(next(_model.parameters()).device)
    name = (opts.get("strategy") or "self_consistency").lower()
    answer_type = opts.get("answer_type", "free")
    num_samples = int(opts.get("num_samples", 5))
    common = dict(answer_type=answer_type, device=device,
                  max_think_tokens=int(opts.get("max_new_tokens", 64)),
                  max_answer_tokens=32)

    if name in ("cot", "chain_of_thought"):
        strategy = CoTStrategy(_model, _tokenizer, **common)
    elif name in ("best_of_n", "bon"):
        strategy = BestOfNStrategy(
            _model, _tokenizer, SelfConsistencyVerifier(),
            num_samples=num_samples, **common)
    else:  # default: self-consistency
        strategy = SelfConsistencyStrategy(
            _model, _tokenizer, num_samples=num_samples, **common)

    result = strategy.solve(prompt)
    if result.answer:
        return result.answer
    if result.traces:
        return result.traces[0]
    return ""


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
    compile_model: bool = False,
    api_key: Optional[str] = None,
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
        compile_model: If ``True``, wrap the model with ``torch.compile`` for
            faster inference (falls back to eager on any compilation failure).
    """
    import uvicorn

    global _model, _tokenizer, _scheduler, _api_key

    # Require a Bearer key on the inference endpoints when configured (arg wins
    # over the SELFLLM_API_KEY env var). Needed to register SelfLLM as a
    # custom provider.
    if api_key is not None:
        _api_key = api_key
    if _api_key:
        logger.info("API key auth ENABLED for /v1/* inference endpoints.")
    else:
        logger.warning("API key auth DISABLED (open server). Set SELFLLM_API_KEY "
                       "or serve(api_key=...) before exposing publicly.")

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

    if compile_model:
        from selfllm.serving.optimized import compile_model as _compile
        _model = _compile(_model)

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
