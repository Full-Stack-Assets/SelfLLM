"""Lightweight, dependency-free Prometheus metrics for the serving API.

Rather than pull in ``prometheus_client`` (and inflate the Vercel/Fly image),
this module implements just enough of the Prometheus data model -- counters, a
gauge, and a latency histogram -- to expose a standard ``/metrics`` text
exposition endpoint that any Prometheus/Grafana/OpenTelemetry-collector setup
can scrape.

The collector is process-global and thread-safe, so it can be driven from a
FastAPI middleware that wraps every request.

Exposed series
~~~~~~~~~~~~~~
- ``selfllm_requests_total{endpoint,method,status}`` -- request counter.
- ``selfllm_request_latency_seconds{endpoint}`` -- histogram
  (``_bucket``/``_sum``/``_count``) of request durations.
- ``selfllm_requests_in_flight`` -- gauge of concurrently-processing requests.
"""

from __future__ import annotations

import threading
from typing import Dict, List, Tuple

__all__ = ["MetricsCollector", "METRICS", "DEFAULT_BUCKETS"]

# Latency histogram bucket upper bounds, in seconds (Prometheus convention:
# each bucket is cumulative "<= le").
DEFAULT_BUCKETS: Tuple[float, ...] = (
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0
)


def _fmt_labels(labels: Dict[str, str]) -> str:
    if not labels:
        return ""
    inner = ",".join(
        f'{k}="{_escape(v)}"' for k, v in sorted(labels.items())
    )
    return "{" + inner + "}"


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


class MetricsCollector:
    """Thread-safe collector of counters, a gauge, and a latency histogram."""

    def __init__(self, buckets: Tuple[float, ...] = DEFAULT_BUCKETS) -> None:
        self.buckets = tuple(sorted(buckets))
        self._lock = threading.Lock()
        # (endpoint, method, status) -> count
        self._requests: Dict[Tuple[str, str, str], int] = {}
        # endpoint -> [count per bucket] (+inf implicit via _count)
        self._latency_buckets: Dict[str, List[int]] = {}
        self._latency_sum: Dict[str, float] = {}
        self._latency_count: Dict[str, int] = {}
        self._in_flight = 0

    # ------------------------------------------------------------------ #
    # Recording
    # ------------------------------------------------------------------ #

    def inc_in_flight(self) -> None:
        with self._lock:
            self._in_flight += 1

    def dec_in_flight(self) -> None:
        with self._lock:
            self._in_flight = max(0, self._in_flight - 1)

    def observe_request(
        self, endpoint: str, method: str, status: int, latency_s: float
    ) -> None:
        """Record one completed request: counter + latency histogram."""
        with self._lock:
            key = (endpoint, method, str(status))
            self._requests[key] = self._requests.get(key, 0) + 1

            buckets = self._latency_buckets.setdefault(
                endpoint, [0] * len(self.buckets)
            )
            for i, upper in enumerate(self.buckets):
                if latency_s <= upper:
                    buckets[i] += 1
            self._latency_sum[endpoint] = (
                self._latency_sum.get(endpoint, 0.0) + latency_s
            )
            self._latency_count[endpoint] = (
                self._latency_count.get(endpoint, 0) + 1
            )

    # ------------------------------------------------------------------ #
    # Exposition
    # ------------------------------------------------------------------ #

    def render(self) -> str:
        """Render all series in Prometheus text exposition format."""
        lines: List[str] = []
        with self._lock:
            lines.append("# HELP selfllm_requests_total Total HTTP requests.")
            lines.append("# TYPE selfllm_requests_total counter")
            for (endpoint, method, status), count in sorted(self._requests.items()):
                labels = _fmt_labels(
                    {"endpoint": endpoint, "method": method, "status": status}
                )
                lines.append(f"selfllm_requests_total{labels} {count}")

            lines.append(
                "# HELP selfllm_request_latency_seconds Request latency histogram."
            )
            lines.append("# TYPE selfllm_request_latency_seconds histogram")
            for endpoint in sorted(self._latency_count):
                buckets = self._latency_buckets[endpoint]
                cumulative = 0
                for i, upper in enumerate(self.buckets):
                    cumulative = buckets[i]
                    le = repr(upper)
                    labels = _fmt_labels({"endpoint": endpoint, "le": le})
                    lines.append(
                        f"selfllm_request_latency_seconds_bucket{labels} {cumulative}"
                    )
                inf_labels = _fmt_labels({"endpoint": endpoint, "le": "+Inf"})
                total = self._latency_count[endpoint]
                lines.append(
                    f"selfllm_request_latency_seconds_bucket{inf_labels} {total}"
                )
                ep_labels = _fmt_labels({"endpoint": endpoint})
                lines.append(
                    f"selfllm_request_latency_seconds_sum{ep_labels} "
                    f"{self._latency_sum[endpoint]}"
                )
                lines.append(
                    f"selfllm_request_latency_seconds_count{ep_labels} {total}"
                )

            lines.append(
                "# HELP selfllm_requests_in_flight In-flight requests right now."
            )
            lines.append("# TYPE selfllm_requests_in_flight gauge")
            lines.append(f"selfllm_requests_in_flight {self._in_flight}")

        return "\n".join(lines) + "\n"

    def snapshot(self) -> Dict:
        """Structured snapshot (handy for tests and JSON debugging)."""
        with self._lock:
            return {
                "requests": {
                    "|".join(k): v for k, v in self._requests.items()
                },
                "in_flight": self._in_flight,
                "latency_count": dict(self._latency_count),
                "latency_sum": dict(self._latency_sum),
            }

    def reset(self) -> None:
        """Clear all series (used by tests)."""
        with self._lock:
            self._requests.clear()
            self._latency_buckets.clear()
            self._latency_sum.clear()
            self._latency_count.clear()
            self._in_flight = 0


# Process-global collector shared by the FastAPI middleware and /metrics.
METRICS = MetricsCollector()
