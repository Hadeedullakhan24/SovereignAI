"""Retrieval Telemetry & Metrics.

Tracks stage-by-stage latencies, throughput, score distributions, cache hit rates,
and system resource consumption.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import threading
import time
from typing import Optional

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


@dataclass
class RetrievalMetrics:
    """Telemetry report for a single retrieval execution."""

    query: str = ""
    strategy: str = "hybrid"
    total_duration_ms: float = 0.0
    dense_latency_ms: float = 0.0
    sparse_latency_ms: float = 0.0
    fusion_latency_ms: float = 0.0
    rerank_latency_ms: float = 0.0
    expansion_latency_ms: float = 0.0
    packing_latency_ms: float = 0.0
    
    dense_candidates_count: int = 0
    sparse_candidates_count: int = 0
    fused_candidates_count: int = 0
    returned_chunks_count: int = 0
    citations_count: int = 0
    expanded_neighbors_count: int = 0
    tokens_packed: int = 0
    
    top_score: float = 0.0
    average_score: float = 0.0
    cache_hit: bool = False
    
    memory_rss_mb: float = 0.0
    cpu_percent: float = 0.0


class RetrievalMetricsCollector:
    """Thread-safe collector aggregating system-wide retrieval performance metrics."""

    _instance: Optional[RetrievalMetricsCollector] = None
    _lock = threading.RLock()

    def __init__(self) -> None:
        self._metrics_history: list[RetrievalMetrics] = []
        self._history_lock = threading.RLock()
        self._total_queries: int = 0
        self._total_cache_hits: int = 0
        self._total_cache_misses: int = 0

    @classmethod
    def get_instance(cls) -> RetrievalMetricsCollector:
        """Singleton accessor with double-checked locking."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def record(self, metrics: RetrievalMetrics) -> None:
        """Record a completed retrieval operation's metrics."""
        with self._history_lock:
            # Measure current resource stats if psutil is available
            if PSUTIL_AVAILABLE:
                try:
                    proc = psutil.Process(os.getpid())
                    metrics.memory_rss_mb = proc.memory_info().rss / (1024 * 1024)
                    metrics.cpu_percent = proc.cpu_percent(interval=None)
                except Exception:
                    pass

            self._metrics_history.append(metrics)
            self._total_queries += 1
            if metrics.cache_hit:
                self._total_cache_hits += 1
            else:
                self._total_cache_misses += 1

    def get_cache_hit_ratio(self) -> float:
        """Return cache hit ratio between 0.0 and 1.0."""
        with self._history_lock:
            if self._total_queries == 0:
                return 0.0
            return self._total_cache_hits / self._total_queries

    def get_summary(self) -> dict[str, float]:
        """Compute aggregate performance statistics."""
        with self._history_lock:
            if not self._metrics_history:
                return {
                    "total_queries": 0.0,
                    "avg_total_latency_ms": 0.0,
                    "avg_dense_latency_ms": 0.0,
                    "avg_sparse_latency_ms": 0.0,
                    "avg_fusion_latency_ms": 0.0,
                    "avg_rerank_latency_ms": 0.0,
                    "cache_hit_ratio": 0.0,
                }

            n = len(self._metrics_history)
            avg_total = sum(m.total_duration_ms for m in self._metrics_history) / n
            avg_dense = sum(m.dense_latency_ms for m in self._metrics_history) / n
            avg_sparse = sum(m.sparse_latency_ms for m in self._metrics_history) / n
            avg_fusion = sum(m.fusion_latency_ms for m in self._metrics_history) / n
            avg_rerank = sum(m.rerank_latency_ms for m in self._metrics_history) / n

            return {
                "total_queries": float(self._total_queries),
                "avg_total_latency_ms": round(avg_total, 2),
                "avg_dense_latency_ms": round(avg_dense, 2),
                "avg_sparse_latency_ms": round(avg_sparse, 2),
                "avg_fusion_latency_ms": round(avg_fusion, 2),
                "avg_rerank_latency_ms": round(avg_rerank, 2),
                "cache_hit_ratio": round(self.get_cache_hit_ratio(), 4),
            }

    def reset(self) -> None:
        """Clear all metrics history."""
        with self._history_lock:
            self._metrics_history.clear()
            self._total_queries = 0
            self._total_cache_hits = 0
            self._total_cache_misses = 0
