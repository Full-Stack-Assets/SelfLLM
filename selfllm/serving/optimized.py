"""torch.compile-based inference optimization for SelfLLM.

This module wraps a model with ``torch.compile`` to reduce Python and kernel
launch overhead during autoregressive decoding, and provides a small benchmark
harness to measure decode throughput. Compilation degrades gracefully: if
``torch.compile`` is unavailable or fails (older PyTorch, unsupported backend,
CPU inductor issues), the original eager model is returned so callers always
receive a working module.

Public API:
    - ``compile_model(model, mode=...)`` — compile with graceful fallback.
    - ``benchmark_generation(model, ...)`` — measure decode tokens/sec + latency.
    - ``compare_baseline_vs_compiled(model, ...)`` — eager vs compiled speedup.
    - ``check_logits_match(eager, compiled, ...)`` — correctness guard.
"""

import time
import warnings
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

__all__ = [
    "compile_model",
    "benchmark_generation",
    "compare_baseline_vs_compiled",
    "check_logits_match",
    "is_compiled",
]


def is_compiled(model: nn.Module) -> bool:
    """Return ``True`` if ``model`` is a ``torch.compile`` wrapper.

    Args:
        model: A module that may or may not have been compiled.

    Returns:
        ``True`` when the module is an ``OptimizedModule`` produced by
        ``torch.compile``, ``False`` otherwise.
    """
    # torch.compile returns a torch._dynamo.OptimizedModule whose original
    # module is stored under ``_orig_mod``. This is the most reliable marker
    # across PyTorch versions.
    return hasattr(model, "_orig_mod")


def compile_model(
    model: nn.Module,
    mode: str = "reduce-overhead",
    fullgraph: bool = False,
    **compile_kwargs: Any,
) -> nn.Module:
    """Wrap ``model`` with ``torch.compile``, degrading gracefully on failure.

    The ``"reduce-overhead"`` mode uses CUDA graphs / reduced launch overhead,
    which is well suited to the small, repeated forward passes of incremental
    autoregressive decoding.

    If ``torch.compile`` is missing (older PyTorch) or raises for any reason
    (unsupported backend, CPU inductor issues, missing compiler toolchain), a
    warning is emitted and the *original* eager model is returned unchanged, so
    callers always get a usable model.

    Args:
        model: The ``nn.Module`` to compile (e.g. ``SelfImprovingLLM``).
        mode: ``torch.compile`` mode. Defaults to ``"reduce-overhead"``.
        fullgraph: Whether to require a single full graph (no graph breaks).
        **compile_kwargs: Extra keyword arguments forwarded to ``torch.compile``.

    Returns:
        The compiled module on success, otherwise the original ``model``.
    """
    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        warnings.warn(
            "torch.compile is unavailable in this PyTorch build; "
            "returning the eager model unchanged.",
            RuntimeWarning,
            stacklevel=2,
        )
        return model

    try:
        compiled = compile_fn(
            model, mode=mode, fullgraph=fullgraph, **compile_kwargs
        )
        return compiled
    except Exception as exc:  # noqa: BLE001 — intentional broad fallback
        warnings.warn(
            f"torch.compile failed ({type(exc).__name__}: {exc}); "
            "falling back to the eager model.",
            RuntimeWarning,
            stacklevel=2,
        )
        return model


def _resolve_input_ids(
    model: nn.Module,
    tokenizer_or_input: Union[torch.Tensor, Any, None],
    device: torch.device,
) -> torch.Tensor:
    """Coerce the ``tokenizer_or_input`` argument into a prompt tensor.

    Accepts, in order of preference:
        - a ``torch.Tensor`` of shape ``[batch, prompt_len]`` (used directly),
        - an object with an ``.encode(str) -> list[int]`` method (a tokenizer),
        - ``None`` (a tiny deterministic prompt is synthesised from the config).

    Args:
        model: The model whose ``config`` bounds the vocabulary / sequence.
        tokenizer_or_input: A tensor, tokenizer, or ``None``.
        device: Device on which to place the resulting tensor.

    Returns:
        A ``long`` tensor of shape ``[batch, prompt_len]`` on ``device``.
    """
    if isinstance(tokenizer_or_input, torch.Tensor):
        return tokenizer_or_input.to(device=device, dtype=torch.long)

    config = getattr(model, "config", None) or getattr(
        getattr(model, "_orig_mod", None), "config", None
    )
    vocab_size = getattr(config, "vocab_size", 32) if config is not None else 32

    if tokenizer_or_input is not None and hasattr(tokenizer_or_input, "encode"):
        ids = tokenizer_or_input.encode("hello world")
        if not ids:
            ids = [0]
        # Clamp to the model's vocabulary so a real tokenizer can't index OOB.
        ids = [int(i) % vocab_size for i in ids]
        return torch.tensor([ids], dtype=torch.long, device=device)

    # Synthesise a tiny deterministic prompt within the vocabulary.
    prompt = [i % vocab_size for i in range(4)]
    return torch.tensor([prompt], dtype=torch.long, device=device)


def benchmark_generation(
    model: nn.Module,
    tokenizer_or_input: Union[torch.Tensor, Any, None] = None,
    max_new_tokens: int = 4,
    num_runs: int = 3,
    warmup: int = 1,
    temperature: float = 0.0,
) -> Dict[str, float]:
    """Benchmark decode throughput (tokens/sec) and latency of ``model``.

    Warmup runs are executed and discarded *before* timing begins. This is
    essential for compiled models: the first ``generate`` call triggers the
    actual ``torch.compile`` graph capture / kernel compilation, which can take
    seconds and would otherwise dominate (and corrupt) the measurement. Only the
    steady-state, post-compilation runs are timed.

    Args:
        model: The (possibly compiled) model exposing ``generate(...)``.
        tokenizer_or_input: Prompt tensor, tokenizer, or ``None`` (see
            ``_resolve_input_ids``).
        max_new_tokens: New tokens to decode per run (keep small on CPU).
        num_runs: Number of timed runs to average over (must be >= 1).
        warmup: Number of untimed warmup runs (>= 0); the first compiled run
            here absorbs the compilation cost.
        temperature: Sampling temperature. ``0.0`` gives deterministic greedy
            decoding, which keeps the benchmark reproducible.

    Returns:
        A metrics dict with keys:
            - ``tokens_per_sec``: decode throughput (always > 0).
            - ``avg_latency_s``: mean wall-clock time of one timed run.
            - ``total_new_tokens``: new tokens generated per run.
            - ``num_runs`` / ``warmup``: the run configuration echoed back.
    """
    if num_runs < 1:
        raise ValueError("num_runs must be >= 1")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")

    device = next(model.parameters()).device
    input_ids = _resolve_input_ids(model, tokenizer_or_input, device)
    batch_size = input_ids.shape[0]

    if hasattr(model, "eval"):
        model.eval()

    def _run() -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            return model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

    # Warmup — discarded. The first compiled call compiles the graph here.
    for _ in range(warmup):
        _run()

    # Determine how many tokens were actually generated (a model may stop early).
    sample = _run()
    seq = sample["sequences"]
    generated_per_run = max(int(seq.shape[1] - input_ids.shape[1]), 0)
    # Guard against a model that produced nothing (e.g. immediate stop): count
    # the requested tokens so throughput stays well-defined and positive.
    new_tokens = generated_per_run if generated_per_run > 0 else max_new_tokens

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_runs):
        _run()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    # Avoid division by zero on absurdly fast / coarse-clock runs.
    elapsed = max(elapsed, 1e-9)
    avg_latency = elapsed / num_runs
    total_new_tokens = new_tokens * batch_size
    tokens_per_sec = (total_new_tokens * num_runs) / elapsed

    return {
        "tokens_per_sec": tokens_per_sec,
        "avg_latency_s": avg_latency,
        "total_new_tokens": float(total_new_tokens),
        "num_runs": float(num_runs),
        "warmup": float(warmup),
    }


def compare_baseline_vs_compiled(
    model: nn.Module,
    tokenizer_or_input: Union[torch.Tensor, Any, None] = None,
    max_new_tokens: int = 4,
    num_runs: int = 3,
    warmup: int = 1,
    mode: str = "reduce-overhead",
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """Benchmark the eager model against its compiled counterpart.

    Compiles a copy/wrapper of ``model`` via ``compile_model`` (which falls back
    to eager on failure), benchmarks both, and reports the speedup ratio
    (compiled tokens/sec divided by baseline tokens/sec).

    Args:
        model: The eager ``nn.Module`` to use as the baseline.
        tokenizer_or_input: Prompt tensor, tokenizer, or ``None``.
        max_new_tokens: New tokens to decode per run.
        num_runs: Number of timed runs per model.
        warmup: Number of untimed warmup runs per model.
        mode: ``torch.compile`` mode for the compiled variant.
        temperature: Sampling temperature (``0.0`` = deterministic greedy).

    Returns:
        A dict with keys:
            - ``baseline``: the eager benchmark metrics dict.
            - ``compiled``: the compiled benchmark metrics dict.
            - ``baseline_tokens_per_sec`` / ``compiled_tokens_per_sec``: floats.
            - ``speedup``: compiled / baseline throughput ratio (> 0).
            - ``compiled_active``: whether compilation actually took effect
              (``False`` if it fell back to eager).
    """
    baseline_metrics = benchmark_generation(
        model,
        tokenizer_or_input,
        max_new_tokens=max_new_tokens,
        num_runs=num_runs,
        warmup=warmup,
        temperature=temperature,
    )

    compiled_model = compile_model(model, mode=mode)
    compiled_active = is_compiled(compiled_model)

    compiled_metrics = benchmark_generation(
        compiled_model,
        tokenizer_or_input,
        max_new_tokens=max_new_tokens,
        num_runs=num_runs,
        warmup=warmup,
        temperature=temperature,
    )

    baseline_tps = baseline_metrics["tokens_per_sec"]
    compiled_tps = compiled_metrics["tokens_per_sec"]
    speedup = compiled_tps / baseline_tps if baseline_tps > 0 else 1.0

    return {
        "baseline": baseline_metrics,
        "compiled": compiled_metrics,
        "baseline_tokens_per_sec": baseline_tps,
        "compiled_tokens_per_sec": compiled_tps,
        "speedup": speedup,
        "compiled_active": compiled_active,
    }


def check_logits_match(
    eager_model: nn.Module,
    compiled_model: nn.Module,
    input_ids: Optional[torch.Tensor] = None,
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> Dict[str, Any]:
    """Verify the compiled model produces the same logits as the eager model.

    Runs both models on a fixed input and compares the ``"logits"`` tensors.
    Compilation should be a pure optimization; this guards against a backend
    that silently changes outputs.

    Args:
        eager_model: The reference (uncompiled) model.
        compiled_model: The compiled (or fallen-back) model.
        input_ids: Fixed input ``[batch, seq_len]``. If ``None``, a small
            deterministic input is synthesised from the model config.
        rtol: Relative tolerance for ``torch.allclose``.
        atol: Absolute tolerance for ``torch.allclose``.

    Returns:
        A dict with keys:
            - ``match``: ``True`` if logits agree within tolerance.
            - ``max_abs_diff``: maximum absolute elementwise difference.
    """
    device = next(eager_model.parameters()).device
    if input_ids is None:
        input_ids = _resolve_input_ids(eager_model, None, device)
    else:
        input_ids = input_ids.to(device=device, dtype=torch.long)

    eager_model.eval()
    if hasattr(compiled_model, "eval"):
        compiled_model.eval()

    with torch.no_grad():
        eager_logits = eager_model(input_ids)["logits"]
        compiled_logits = compiled_model(input_ids)["logits"]

    max_abs_diff = (eager_logits - compiled_logits).abs().max().item()
    match = torch.allclose(eager_logits, compiled_logits, rtol=rtol, atol=atol)

    return {"match": bool(match), "max_abs_diff": float(max_abs_diff)}
