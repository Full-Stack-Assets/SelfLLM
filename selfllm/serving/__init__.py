"""Serving infrastructure for SelfLLM.

Provides:
- PagedAttention KV cache management (BlockManager, PagedAttention)
- Continuous batching scheduler (ContinuousBatchingScheduler, Request)
- OpenAI-compatible API server (app, serve)
"""

from .paged_cache import BlockManager, PagedAttention
from .scheduler import ContinuousBatchingScheduler, Request
from .server import serve, app

__all__ = [
    "BlockManager",
    "PagedAttention",
    "Request",
    "ContinuousBatchingScheduler",
    "serve",
    "app",
]
