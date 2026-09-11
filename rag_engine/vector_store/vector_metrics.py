"""Vector Metrics Collector.

Records throughput, search latencies, upsert durations, and error rates.
"""

from __future__ import annotations

import threading
import time
from typing import Any


class VectorMetricsCollector:
    """Thread-safe collector for vector database operations telemetry."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._total_searches = 0
        self._total_upserts = 0
        self._total_deletes = 0
        self._search_durations: list[float] = []
        self._upsert_durations: list[float] = []

    def record_search(self, duration_ms: float) -> None:
        with self._lock:
            self._total_searches += 1
            self._search_durations.append(duration_ms)
            if len(self._search_durations) > 1000:
                self._search_durations.pop(0)

    def record_upsert(self, count: int, duration_ms: float) -> None:
        with self._lock:
            self._total_upserts += count
            self._upsert_durations.append(duration_ms)
            if len(self._upsert_durations) > 1000:
                self._upsert_durations.pop(0)

    def record_delete(self, count: int) -> None:
        with self._lock:
            self._total_deletes += count

    def get_summary(self) -> dict[str, Any]:
        with self._lock:
            avg_search = sum(self._search_durations) / len(self._search_durations) if self._search_durations else 0.0
            avg_upsert = sum(self._upsert_durations) / len(self._upsert_durations) if self._upsert_durations else 0.0
            return {
                "total_searches": self._total_searches,
                "total_upserts": self._total_upserts,
                "total_deletes": self._total_deletes,
                "avg_search_latency_ms": round(avg_search, 2),
                "avg_upsert_latency_ms": round(avg_upsert, 2),
            }


global_vector_metrics = VectorMetricsCollector()
