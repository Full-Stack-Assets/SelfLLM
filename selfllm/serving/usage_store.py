"""Pluggable persistence for per-key API usage counters.

The :class:`~selfllm.serving.billing.BillingManager` meters usage in memory by
default, which means quota state is lost on every process restart -- a real
robustness gap for a paid API (a restart hands every key a fresh daily quota).
This module provides a small, dependency-free storage abstraction so usage can
instead be persisted durably.

Backends
~~~~~~~~
- :class:`InMemoryUsageStore` -- the historical behavior (fast, ephemeral).
- :class:`SQLiteUsageStore` -- durable, thread-safe, stdlib-only. Survives
  restarts and is safe to share across a process's worker threads.

Both are keyed by ``(api_key, day)`` where ``day`` is a UTC ``YYYY-MM-DD``
string, so the daily quota rollover is a natural consequence of the key
changing at UTC midnight -- old rows simply stop being read.

Contract
~~~~~~~~
A store persists three monotonically-increasing counters per ``(key, day)``:
``requests``, ``prompt_tokens``, ``completion_tokens``. ``get`` returns the
current triple (zeros if unseen); ``add`` increments atomically and returns the
new triple.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from typing import Dict, Optional, Protocol, Tuple

__all__ = [
    "UsageStore",
    "InMemoryUsageStore",
    "SQLiteUsageStore",
    "usage_store_from_env",
]

# (requests, prompt_tokens, completion_tokens)
Counts = Tuple[int, int, int]


class UsageStore(Protocol):
    """Storage backend for per-``(key, day)`` usage counters."""

    def get(self, key: str, day: str) -> Counts:
        """Return ``(requests, prompt_tokens, completion_tokens)`` (zeros if new)."""
        ...

    def add(
        self, key: str, day: str, requests: int, prompt_tokens: int, completion_tokens: int
    ) -> Counts:
        """Atomically add to the counters and return the new totals."""
        ...


class InMemoryUsageStore:
    """Process-local, thread-safe usage store (ephemeral).

    This reproduces the original in-memory metering behavior and is the
    default when no persistent backend is configured.
    """

    def __init__(self) -> None:
        self._data: Dict[Tuple[str, str], list] = {}
        self._lock = threading.Lock()

    def get(self, key: str, day: str) -> Counts:
        with self._lock:
            row = self._data.get((key, day))
            return tuple(row) if row else (0, 0, 0)  # type: ignore[return-value]

    def add(
        self, key: str, day: str, requests: int, prompt_tokens: int, completion_tokens: int
    ) -> Counts:
        with self._lock:
            row = self._data.setdefault((key, day), [0, 0, 0])
            row[0] += requests
            row[1] += prompt_tokens
            row[2] += completion_tokens
            return (row[0], row[1], row[2])


class SQLiteUsageStore:
    """Durable, thread-safe usage store backed by SQLite (stdlib).

    Usage survives process restarts, so daily quotas are enforced correctly
    across deploys and crashes. A single shared connection is used with a lock
    (SQLite serializes writers anyway); this is ample for the metering write
    rate and keeps the implementation simple and portable.

    Args:
        path: SQLite database file path. The parent directory is created if
            needed. Use ``":memory:"`` for an ephemeral in-process database.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        if path != ":memory:":
            parent = os.path.dirname(os.path.abspath(path))
            os.makedirs(parent, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            path, check_same_thread=False, isolation_level=None
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS usage (
                key TEXT NOT NULL,
                day TEXT NOT NULL,
                requests INTEGER NOT NULL DEFAULT 0,
                prompt_tokens INTEGER NOT NULL DEFAULT 0,
                completion_tokens INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (key, day)
            )
            """
        )

    def get(self, key: str, day: str) -> Counts:
        with self._lock:
            cur = self._conn.execute(
                "SELECT requests, prompt_tokens, completion_tokens "
                "FROM usage WHERE key = ? AND day = ?",
                (key, day),
            )
            row = cur.fetchone()
            return (row[0], row[1], row[2]) if row else (0, 0, 0)

    def add(
        self, key: str, day: str, requests: int, prompt_tokens: int, completion_tokens: int
    ) -> Counts:
        with self._lock:
            # UPSERT then read back within the lock for a consistent total.
            self._conn.execute(
                """
                INSERT INTO usage (key, day, requests, prompt_tokens, completion_tokens)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(key, day) DO UPDATE SET
                    requests = requests + excluded.requests,
                    prompt_tokens = prompt_tokens + excluded.prompt_tokens,
                    completion_tokens = completion_tokens + excluded.completion_tokens
                """,
                (key, day, requests, prompt_tokens, completion_tokens),
            )
            cur = self._conn.execute(
                "SELECT requests, prompt_tokens, completion_tokens "
                "FROM usage WHERE key = ? AND day = ?",
                (key, day),
            )
            row = cur.fetchone()
            return (row[0], row[1], row[2])

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def usage_store_from_env(environ: Optional[Dict[str, str]] = None) -> UsageStore:
    """Select a usage store from the environment.

    ``SELFLLM_USAGE_DB`` -> :class:`SQLiteUsageStore` at that path;
    unset -> :class:`InMemoryUsageStore`.
    """
    env = environ if environ is not None else os.environ
    db_path = env.get("SELFLLM_USAGE_DB")
    if db_path:
        return SQLiteUsageStore(db_path)
    return InMemoryUsageStore()
