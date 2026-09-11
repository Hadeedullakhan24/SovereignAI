"""Telemetry and performance metrics collector for the offline embedding pipeline."""

from __future__ import annotations

import os
import threading
import time
from typing import Optional

from rag_engine.schemas.embedding import EmbeddingMetrics


class EmbeddingMetricsCollector:
    """Thread-safe collector for embedding throughput, cache telemetry, and resource utilization."""

    def __init__(self, model_name: str = "", model_version: str = "1.0.0", device: str = "cpu") -> None:
        self._lock = threading.RLock()
        self.model_name = model_name
        self.model_version = model_version
        self.device = device

        self.total_chunks_processed = 0
        self.total_embeddings_generated = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_duration_seconds = 0.0
        self.model_load_time_seconds = 0.0

    def record_model_load_time(self, duration: float) -> None:
        with self._lock:
            self.model_load_time_seconds = duration

    def record_batch(
        self,
        num_chunks: int,
        num_generated: int,
        cache_hits: int,
        cache_misses: int,
        duration: float,
    ) -> None:
        with self._lock:
            self.total_chunks_processed += num_chunks
            self.total_embeddings_generated += num_generated
            self.cache_hits += cache_hits
            self.cache_misses += cache_misses
            self.total_duration_seconds += duration

    def get_metrics(self) -> EmbeddingMetrics:
        """Produce an immutable snapshot of current telemetry metrics."""
        with self._lock:
            total_req = self.cache_hits + self.cache_misses
            reuse_pct = (self.cache_hits / total_req * 100.0) if total_req > 0 else 0.0
            throughput = (
                (self.total_chunks_processed / self.total_duration_seconds)
                if self.total_duration_seconds > 0
                else 0.0
            )
            avg_latency = (
                (self.total_duration_seconds / self.total_chunks_processed * 1000.0)
                if self.total_chunks_processed > 0
                else 0.0
            )

            cpu_pct = None
            ram_mb = None
            try:
                import psutil

                process = psutil.Process(os.getpid())
                cpu_pct = process.cpu_percent()
                ram_mb = round(process.memory_info().rss / (1024 * 1024), 2)
            except Exception:
                pass

            return EmbeddingMetrics(
                model_name=self.model_name,
                model_version=self.model_version,
                device=self.device,
                total_chunks_processed=self.total_chunks_processed,
                total_embeddings_generated=self.total_embeddings_generated,
                cache_hits=self.cache_hits,
                cache_misses=self.cache_misses,
                cache_reuse_percentage=round(reuse_pct, 2),
                total_duration_seconds=round(self.total_duration_seconds, 4),
                average_latency_ms=round(avg_latency, 2),
                throughput_chunks_per_sec=round(throughput, 2),
                cpu_percent=cpu_pct,
                ram_mb=ram_mb,
            )

    def reset(self) -> None:
        with self._lock:
            self.total_chunks_processed = 0
            self.total_embeddings_generated = 0
            self.cache_hits = 0
            self.cache_misses = 0
            self.total_duration_seconds = 0.0
