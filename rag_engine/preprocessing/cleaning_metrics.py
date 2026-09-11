"""Observability metrics tracking and thread-safe metrics collector."""

from __future__ import annotations

import time
import threading
from typing import Any, Optional
from pydantic import BaseModel, Field

from rag_engine.schemas.parsed_document import CleaningStatistics


class CleaningMetrics(BaseModel):
    """Container for in-flight cleaning metrics of a single document."""

    document_id: str
    start_time: float = Field(default_factory=time.perf_counter)
    end_time: Optional[float] = None
    characters_removed: int = 0
    characters_normalized: int = 0
    headers_removed: int = 0
    footers_removed: int = 0
    whitespace_reductions: int = 0
    protected_tokens: set[str] = Field(default_factory=set)
    warnings: list[str] = Field(default_factory=list)
    failures: int = 0

    def to_statistics(self) -> CleaningStatistics:
        """Convert in-flight metrics to immutable CleaningStatistics schema."""
        elapsed_ms = (
            (self.end_time - self.start_time) * 1000.0
            if self.end_time is not None
            else (time.perf_counter() - self.start_time) * 1000.0
        )
        return CleaningStatistics(
            execution_time_ms=round(max(0.0, elapsed_ms), 3),
            characters_removed=self.characters_removed,
            characters_normalized=self.characters_normalized,
            headers_removed=self.headers_removed,
            footers_removed=self.footers_removed,
            whitespace_reductions=self.whitespace_reductions,
            protected_tokens_count=len(self.protected_tokens),
            warnings_count=len(self.warnings),
            failures_count=self.failures,
        )


class CleaningMetricsCollector:
    """Thread-safe collector for tracking cleaning engine metrics using RLock."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._metrics_map: dict[str, CleaningMetrics] = {}
        self._aggregate_stats = {
            "total_documents": 0,
            "total_characters_removed": 0,
            "total_characters_normalized": 0,
            "total_headers_removed": 0,
            "total_footers_removed": 0,
            "total_whitespace_reductions": 0,
            "total_protected_tokens": 0,
            "total_warnings": 0,
            "total_failures": 0,
            "total_execution_time_ms": 0.0,
        }

    def start_document(self, document_id: str) -> None:
        """Begin tracking metrics for a document."""
        with self._lock:
            self._metrics_map[document_id] = CleaningMetrics(document_id=document_id)

    def finish_document(self, document_id: str) -> CleaningStatistics:
        """Finish tracking metrics and update global aggregate counters."""
        with self._lock:
            metric = self._metrics_map.get(document_id)
            if not metric:
                metric = CleaningMetrics(document_id=document_id)
                self._metrics_map[document_id] = metric
            metric.end_time = time.perf_counter()
            stats = metric.to_statistics()

            # Update aggregates
            self._aggregate_stats["total_documents"] += 1
            self._aggregate_stats["total_characters_removed"] += stats.characters_removed
            self._aggregate_stats["total_characters_normalized"] += stats.characters_normalized
            self._aggregate_stats["total_headers_removed"] += stats.headers_removed
            self._aggregate_stats["total_footers_removed"] += stats.footers_removed
            self._aggregate_stats["total_whitespace_reductions"] += stats.whitespace_reductions
            self._aggregate_stats["total_protected_tokens"] += stats.protected_tokens_count
            self._aggregate_stats["total_warnings"] += stats.warnings_count
            self._aggregate_stats["total_failures"] += stats.failures_count
            self._aggregate_stats["total_execution_time_ms"] += stats.execution_time_ms
            return stats

    def record_characters_removed(self, document_id: str, count: int) -> None:
        """Record number of characters removed."""
        if count <= 0:
            return
        with self._lock:
            metric = self._get_or_create(document_id)
            metric.characters_removed += count

    def record_characters_normalized(self, document_id: str, count: int) -> None:
        """Record number of characters normalized/altered."""
        if count <= 0:
            return
        with self._lock:
            metric = self._get_or_create(document_id)
            metric.characters_normalized += count

    def record_header_removed(self, document_id: str, count: int = 1) -> None:
        """Record header instances removed."""
        with self._lock:
            metric = self._get_or_create(document_id)
            metric.headers_removed += count

    def record_footer_removed(self, document_id: str, count: int = 1) -> None:
        """Record footer instances removed."""
        with self._lock:
            metric = self._get_or_create(document_id)
            metric.footers_removed += count

    def record_whitespace_reduction(self, document_id: str, count: int) -> None:
        """Record whitespace characters eliminated."""
        if count <= 0:
            return
        with self._lock:
            metric = self._get_or_create(document_id)
            metric.whitespace_reductions += count

    def record_protected_tokens(self, document_id: str, tokens: list[str]) -> None:
        """Record protected tokens identified."""
        with self._lock:
            metric = self._get_or_create(document_id)
            metric.protected_tokens.update(tokens)

    def record_warning(self, document_id: str, warning_msg: str) -> None:
        """Record a non-fatal warning."""
        with self._lock:
            metric = self._get_or_create(document_id)
            metric.warnings.append(warning_msg)

    def record_failure(self, document_id: str) -> None:
        """Record a pipeline stage failure."""
        with self._lock:
            metric = self._get_or_create(document_id)
            metric.failures += 1

    def get_document_statistics(self, document_id: str) -> CleaningStatistics:
        """Retrieve current statistics for a document."""
        with self._lock:
            metric = self._metrics_map.get(document_id)
            if not metric:
                return CleaningStatistics()
            return metric.to_statistics()

    def get_aggregate_statistics(self) -> dict[str, Any]:
        """Return a copy of process-wide aggregate metrics."""
        with self._lock:
            return dict(self._aggregate_stats)

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._metrics_map.clear()
            for key in self._aggregate_stats:
                self._aggregate_stats[key] = 0.0 if "ms" in key else 0

    def _get_or_create(self, document_id: str) -> CleaningMetrics:
        if document_id not in self._metrics_map:
            self._metrics_map[document_id] = CleaningMetrics(document_id=document_id)
        return self._metrics_map[document_id]


# Global shared thread-safe metrics collector
_global_metrics_collector = CleaningMetricsCollector()


def get_cleaning_metrics_collector() -> CleaningMetricsCollector:
    """Get process-wide singleton cleaning metrics collector."""
    return _global_metrics_collector
