"""Continuous batching scheduler for serving.

Implements a vLLM-style continuous batching scheduler that maintains a queue
of requests and processes them in batches. Each iteration:
1. Adds new requests from the waiting queue to the active batch (if space)
2. Runs one forward pass for all active requests
3. Removes finished requests and frees their blocks
4. Repeats
"""

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from selfllm.serving.paged_cache import BlockManager


@dataclass
class Request:
    """A single generation request.

    Tracks the full lifecycle from waiting -> running -> finished.
    """

    id: str
    prompt_tokens: List[int]
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    status: str = "waiting"  # "waiting", "running", "finished"
    generated_tokens: List[int] = field(default_factory=list)
    block_ids: List[int] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    finish_reason: Optional[str] = None  # "stop", "length", "error"

    @property
    def is_finished(self) -> bool:
        """Whether the request has completed."""
        return self.status == "finished"

    @property
    def total_tokens(self) -> int:
        """Total number of tokens (prompt + generated)."""
        return len(self.prompt_tokens) + len(self.generated_tokens)

    @property
    def latency(self) -> Optional[float]:
        """Request latency in seconds (None if not finished)."""
        if self.finished_at is not None:
            return self.finished_at - self.created_at
        return None


class ContinuousBatchingScheduler:
    """vLLM-style continuous batching scheduler.

    Maintains a queue of requests. Each step:
    1. Add new requests from queue to active batch (if space available)
    2. Run one forward pass for all active requests
    3. Remove finished requests, free their blocks
    4. Repeat

    This scheduler operates on a model's ``generate()`` method, managing
    the batching of multiple concurrent generation requests efficiently.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        block_manager: BlockManager,
        max_batch_size: int = 32,
        device: str = "cuda",
    ):
        """
        Args:
            model: The language model (e.g., SelfImprovingLLM).
            tokenizer: Tokenizer with encode/decode/eos_token_id.
            block_manager: BlockManager for KV cache allocation.
            max_batch_size: Maximum number of concurrent active requests.
            device: Device to run inference on.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.block_manager = block_manager
        self.max_batch_size = max_batch_size
        self.device = device

        self.waiting_queue: List[Request] = []
        self.active_requests: List[Request] = []
        self.completed_requests: List[Request] = []

        self._lock = threading.Lock()
        self._shutdown = False
        self._total_requests_served = 0

        # Per-request KV cache state: req.id -> {"past": [(k, v), ...], "cache_len": int}
        self._req_cache: Dict[str, Dict[str, Any]] = {}

    def add_request(self, request: Request) -> None:
        """Thread-safe add request to waiting queue.

        Args:
            request: The generation request to enqueue.
        """
        with self._lock:
            self.waiting_queue.append(request)

    def create_request(
        self,
        prompt_tokens: List[int],
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
    ) -> Request:
        """Create and enqueue a new request.

        Args:
            prompt_tokens: Encoded prompt token IDs.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            top_k: Top-k sampling limit.

        Returns:
            The created Request object.
        """
        req = Request(
            id=f"req-{uuid.uuid4().hex[:12]}",
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        self.add_request(req)
        return req

    def step(self) -> List[Request]:
        """Run one scheduler iteration.

        1. Admit waiting requests into the active batch (respecting the batch
           size and KV-cache block budget) and prefill each one.
        2. Run a single *batched* decode forward over all active requests --
           their per-request KV caches are right-padded to a common length and
           masked, so one model forward advances every sequence by one token.
        3. Retire finished requests (EOS / max length) and free their blocks.

        Returns:
            List of newly completed requests this step.
        """
        completed: List[Request] = []

        with self._lock:
            newly_admitted: List[Request] = []
            while (
                len(self.active_requests) < self.max_batch_size
                and self.waiting_queue
            ):
                req = self.waiting_queue[0]
                try:
                    req.block_ids = self.block_manager.allocate(
                        max(1, len(req.prompt_tokens))
                    )
                except RuntimeError:
                    break  # out of blocks; retry next step
                self.waiting_queue.pop(0)
                req.status = "running"
                self.active_requests.append(req)
                newly_admitted.append(req)

            active = list(self.active_requests)

        newly_ids = {r.id for r in newly_admitted}

        # Prefill newly admitted requests (build their KV cache + first token).
        for req in newly_admitted:
            try:
                self._prefill(req)
            except Exception:
                self._finish(req, "error")
                completed.append(req)
                continue
            if self._check_complete(req):
                completed.append(req)

        # Batched decode for requests that already have a cache and need more
        # tokens (exclude the ones just prefilled this step).
        decode_reqs = [
            r
            for r in active
            if r.id not in newly_ids
            and not r.is_finished
            and r.id in self._req_cache
            and len(r.generated_tokens) < r.max_new_tokens
        ]
        if decode_reqs:
            try:
                self._batched_decode(decode_reqs)
            except Exception:
                for req in decode_reqs:
                    self._finish(req, "error")
                    completed.append(req)
                decode_reqs = []
            for req in decode_reqs:
                if self._check_complete(req):
                    completed.append(req)

        # Retire finished requests and free their blocks.
        with self._lock:
            for req in completed:
                if req in self.active_requests:
                    self.active_requests.remove(req)
                self.completed_requests.append(req)
                self.block_manager.free(req.block_ids)
                self._req_cache.pop(req.id, None)
                self._total_requests_served += 1

        return completed

    # ------------------------------------------------------------------ #
    # Decoding internals
    # ------------------------------------------------------------------ #

    def _prefill(self, req: Request) -> None:
        """Prefill a request's prompt, storing its KV cache and first token."""
        max_len = getattr(getattr(self.model, "config", None), "max_seq_len", 2048)
        prompt = req.prompt_tokens[-max_len:] if req.prompt_tokens else [0]
        input_ids = torch.tensor([prompt], device=self.device)
        with torch.no_grad():
            out = self.model.forward(input_ids, use_cache=True)
        past = out["past_key_values"]
        logits = out["logits"][0, -1]
        token = self._sample(logits, req)
        self._req_cache[req.id] = {"past": past, "cache_len": past[0][0].shape[2]}
        req.generated_tokens.append(token)
        req.block_ids = self.block_manager.append_token(req.block_ids)

    def _batched_decode(self, reqs: List[Request]) -> None:
        """Advance every request in ``reqs`` by one token in a single forward."""
        n_layers = len(self._req_cache[reqs[0].id]["past"])
        sample_k = self._req_cache[reqs[0].id]["past"][0][0]
        n_heads, head_dim = sample_k.shape[1], sample_k.shape[3]
        lens = [self._req_cache[r.id]["cache_len"] for r in reqs]
        L = max(lens)

        # Right-pad each request's cache to the common length L per layer.
        padded: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for layer in range(n_layers):
            ks, vs = [], []
            for r, ln in zip(reqs, lens):
                k, v = self._req_cache[r.id]["past"][layer]
                if ln < L:
                    pad = k.new_zeros(1, n_heads, L - ln, head_dim)
                    k = torch.cat([k, pad], dim=2)
                    v = torch.cat([v, pad.clone()], dim=2)
                ks.append(k)
                vs.append(v)
            padded.append((torch.cat(ks, dim=0), torch.cat(vs, dim=0)))

        tokens = torch.tensor(
            [[r.generated_tokens[-1]] for r in reqs], device=self.device
        )
        positions = torch.tensor([[ln] for ln in lens], device=self.device)
        ar = torch.arange(L, device=self.device)
        kpm = torch.stack([ar < ln for ln in lens])  # [B, L] bool

        with torch.no_grad():
            out = self.model.forward(
                tokens,
                past_key_values=padded,
                use_cache=True,
                positions=positions,
                key_padding_mask=kpm,
            )
        logits = out["logits"][:, -1]  # [B, V]
        new_caches = out["past_key_values"]  # list of (k, v) [B, H, L+1, hd]

        for i, r in enumerate(reqs):
            ln = lens[i]
            token = self._sample(logits[i], r)
            r.generated_tokens.append(token)
            r.block_ids = self.block_manager.append_token(r.block_ids)
            # Extract this request's own contiguous cache: real keys [0, ln)
            # plus the newly appended key at index L.
            new_past: List[Tuple[torch.Tensor, torch.Tensor]] = []
            for layer in range(n_layers):
                k, v = new_caches[layer]
                kk = torch.cat([k[i : i + 1, :, :ln, :], k[i : i + 1, :, L : L + 1, :]], dim=2)
                vv = torch.cat([v[i : i + 1, :, :ln, :], v[i : i + 1, :, L : L + 1, :]], dim=2)
                new_past.append((kk, vv))
            self._req_cache[r.id] = {"past": new_past, "cache_len": ln + 1}

    def _sample(self, logits: torch.Tensor, req: Request) -> int:
        """Sample one token id from a ``[vocab]`` logits row using req params."""
        logits = logits.clone().float()
        if req.temperature == 0.0:
            return int(torch.argmax(logits).item())
        logits = logits / req.temperature
        if req.top_k and req.top_k > 0:
            k = min(req.top_k, logits.size(-1))
            vals, _ = torch.topk(logits, k)
            logits[logits < vals[-1]] = float("-inf")
        if req.top_p and req.top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cum = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cum > req.top_p
            remove[1:] = remove[:-1].clone()
            remove[0] = False
            logits[sorted_idx[remove]] = float("-inf")
        probs = torch.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())

    def _check_complete(self, req: Request) -> bool:
        """Mark req finished if it hit EOS or its length cap. Returns True if so."""
        if req.is_finished:
            return False
        last = req.generated_tokens[-1] if req.generated_tokens else None
        if last is not None and last == self.tokenizer.eos_token_id:
            self._finish(req, "stop")
            return True
        if len(req.generated_tokens) >= req.max_new_tokens:
            self._finish(req, "length")
            return True
        return False

    def _finish(self, req: Request, reason: str) -> None:
        """Set a request's terminal status."""
        req.status = "finished"
        req.finished_at = time.time()
        req.finish_reason = reason

    def run(self) -> None:
        """Run the continuous batching loop.

        This is a blocking call that runs until ``shutdown()`` is called.
        Should be run in a separate thread.
        """
        while not self._shutdown:
            self.step()
            if not self.active_requests and not self.waiting_queue:
                time.sleep(0.01)  # Small sleep when idle

    def shutdown(self) -> None:
        """Signal the scheduler to stop.

        The loop in ``run()`` will exit after the current iteration.
        """
        self._shutdown = True

    def get_request_result(self, request_id: str) -> Optional[Request]:
        """Get a completed request by ID.

        Args:
            request_id: The ID of the request to look up.

        Returns:
            The Request if found in completed requests, else None.
        """
        with self._lock:
            for req in self.completed_requests:
                if req.id == request_id:
                    return req
            for req in self.active_requests:
                if req.id == request_id:
                    return req
            for req in self.waiting_queue:
                if req.id == request_id:
                    return req
        return None

    @property
    def num_active_requests(self) -> int:
        """Number of currently active (running) requests."""
        with self._lock:
            return len(self.active_requests)

    @property
    def stats(self) -> Dict[str, Any]:
        """Return current scheduler statistics.

        Returns:
            Dictionary with active, waiting, completed request counts,
            free blocks, and total requests served.
        """
        with self._lock:
            return {
                "active": len(self.active_requests),
                "waiting": len(self.waiting_queue),
                "completed": len(self.completed_requests),
                "free_blocks": self.block_manager.num_free_blocks,
                "total_served": self._total_requests_served,
            }
