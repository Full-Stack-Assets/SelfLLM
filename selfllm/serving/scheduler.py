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
from typing import Any, Dict, List, Optional

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
        """Run one generation step.

        1. Move waiting requests to active batch
        2. For each active request, generate one token
        3. Check for completion (EOS, max length)

        Returns:
            List of newly completed requests this step.
        """
        completed: List[Request] = []

        with self._lock:
            # Move waiting -> active
            while (
                len(self.active_requests) < self.max_batch_size
                and self.waiting_queue
            ):
                req = self.waiting_queue.pop(0)
                try:
                    num_tokens = len(req.prompt_tokens)
                    req.block_ids = self.block_manager.allocate(num_tokens)
                    req.status = "running"
                    self.active_requests.append(req)
                except RuntimeError:
                    # Out of blocks, put back and stop adding more
                    self.waiting_queue.insert(0, req)
                    break

            if not self.active_requests:
                return completed

            active = list(self.active_requests)

        # For each active request, generate one token
        for req in active:
            try:
                # Feed the full running context (prompt + everything generated
                # so far). model.generate() keeps no KV cache across separate
                # calls, so passing only the last token would condition each
                # step on a single token and discard all prior context.
                context = req.prompt_tokens + req.generated_tokens
                input_ids = torch.tensor([context], device=self.device)

                # Generate one token using the model
                with torch.no_grad():
                    output = self.model.generate(
                        input_ids,
                        max_new_tokens=1,
                        temperature=req.temperature,
                        top_p=req.top_p,
                        top_k=req.top_k,
                        stop_token_id=self.tokenizer.eos_token_id,
                    )

                new_token = output["sequences"][0, -1].item()
                req.generated_tokens.append(new_token)

                # Extend block allocation if needed
                req.block_ids = self.block_manager.append_token(req.block_ids)

                # Check completion conditions
                if new_token == self.tokenizer.eos_token_id:
                    req.status = "finished"
                    req.finished_at = time.time()
                    req.finish_reason = "stop"
                    completed.append(req)

                    with self._lock:
                        self.active_requests.remove(req)
                        self.completed_requests.append(req)
                        self.block_manager.free(req.block_ids)
                        self._total_requests_served += 1

                elif len(req.generated_tokens) >= req.max_new_tokens:
                    req.status = "finished"
                    req.finished_at = time.time()
                    req.finish_reason = "length"
                    completed.append(req)

                    with self._lock:
                        self.active_requests.remove(req)
                        self.completed_requests.append(req)
                        self.block_manager.free(req.block_ids)
                        self._total_requests_served += 1

            except Exception:
                req.status = "finished"
                req.finished_at = time.time()
                req.finish_reason = "error"

                with self._lock:
                    if req in self.active_requests:
                        self.active_requests.remove(req)
                        self.completed_requests.append(req)
                        self.block_manager.free(req.block_ids)
                        self._total_requests_served += 1

        return completed

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
