"""Observability telemetry and ChunkStatistics generator using RLock."""

from __future__ import annotations

import time
import threading
from typing import Any, Optional
from pydantic import BaseModel, Field

from rag_engine.schemas.chunk import Chunk, ChunkStatistics


class DocumentChunkingMetrics(BaseModel):
    """Container for in-flight chunking telemetry of a single document."""

    document_id: str
    start_time: float = Field(default_factory=time.perf_counter)
    end_time: Optional[float] = None
    chunks: list[Chunk] = Field(default_factory=list)
    rejections: list[str] = Field(default_factory=list)
    failures: int = 0

    def to_statistics(self) -> ChunkStatistics:
        """Convert accumulated in-flight telemetry into immutable ChunkStatistics."""
        elapsed_ms = (
            (self.end_time - self.start_time) * 1000.0
            if self.end_time is not None
            else (time.perf_counter() - self.start_time) * 1000.0
        )

        total = len(self.chunks)
        if total == 0:
            return ChunkStatistics(
                total_chunks=0,
                total_execution_time_ms=round(max(0.0, elapsed_ms), 3),
                rejected_chunks_count=len(self.rejections),
            )

        token_counts = [c.token_count for c in self.chunks]
        char_counts = [c.character_count for c in self.chunks]

        cat_dist: dict[str, int] = {}
        sec_dist: dict[str, int] = {}
        table_count = 0
        list_count = 0

        for c in self.chunks:
            cat = c.metadata.category or "Unknown"
            cat_dist[cat] = cat_dist.get(cat, 0) + 1

            sec = c.metadata.section_title or "Root"
            sec_dist[sec] = sec_dist.get(sec, 0) + 1

            if c.metadata.is_table_chunk:
                table_count += 1
            if c.metadata.is_list_chunk:
                list_count += 1

        return ChunkStatistics(
            total_chunks=total,
            average_tokens=round(sum(token_counts) / total, 2),
            min_tokens=min(token_counts),
            max_tokens=max(token_counts),
            average_characters=round(sum(char_counts) / total, 2),
            min_characters=min(char_counts),
            max_characters=max(char_counts),
            chunks_per_document={self.document_id: total},
            chunks_per_category=cat_dist,
            chunks_per_section=sec_dist,
            total_table_chunks=table_count,
            total_list_chunks=list_count,
            rejected_chunks_count=len(self.rejections),
            total_execution_time_ms=round(max(0.0, elapsed_ms), 3),
        )


class ChunkMetricsCollector:
    """Thread-safe collector for tracking chunk metrics and generating aggregate statistics using RLock."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._doc_metrics: dict[str, DocumentChunkingMetrics] = {}
        self._global_totals = {
            "total_documents_chunked": 0,
            "total_chunks_produced": 0,
            "total_tokens_produced": 0,
            "total_characters_produced": 0,
            "total_table_chunks": 0,
            "total_list_chunks": 0,
            "total_rejections": 0,
            "total_failures": 0,
            "total_time_ms": 0.0,
        }

    def start_document(self, document_id: str) -> None:
        """Initialize telemetry tracking for a document."""
        with self._lock:
            self._doc_metrics[document_id] = DocumentChunkingMetrics(document_id=document_id)

    def record_chunk(self, document_id: str, chunk: Chunk) -> None:
        """Record an individual produced chunk."""
        with self._lock:
            metric = self._doc_metrics.get(document_id)
            if not metric:
                metric = DocumentChunkingMetrics(document_id=document_id)
                self._doc_metrics[document_id] = metric
            metric.chunks.append(chunk)

    def record_rejection(self, document_id: str, reason: str) -> None:
        """Record a chunk rejected during validation."""
        with self._lock:
            metric = self._doc_metrics.get(document_id)
            if not metric:
                metric = DocumentChunkingMetrics(document_id=document_id)
                self._doc_metrics[document_id] = metric
            metric.rejections.append(reason)

    def record_failure(self, document_id: str) -> None:
        """Record an unhandled chunking failure."""
        with self._lock:
            metric = self._doc_metrics.get(document_id)
            if not metric:
                metric = DocumentChunkingMetrics(document_id=document_id)
                self._doc_metrics[document_id] = metric
            metric.failures += 1

    def finish_document(self, document_id: str) -> ChunkStatistics:
        """Conclude tracking for a document and update global counters."""
        with self._lock:
            metric = self._doc_metrics.get(document_id)
            if not metric:
                metric = DocumentChunkingMetrics(document_id=document_id)
                self._doc_metrics[document_id] = metric
            metric.end_time = time.perf_counter()
            stats = metric.to_statistics()

            # Update global totals
            self._global_totals["total_documents_chunked"] += 1
            self._global_totals["total_chunks_produced"] += stats.total_chunks
            self._global_totals["total_tokens_produced"] += int(stats.average_tokens * stats.total_chunks)
            self._global_totals["total_characters_produced"] += int(stats.average_characters * stats.total_chunks)
            self._global_totals["total_table_chunks"] += stats.total_table_chunks
            self._global_totals["total_list_chunks"] += stats.total_list_chunks
            self._global_totals["total_rejections"] += stats.rejected_chunks_count
            self._global_totals["total_failures"] += metric.failures
            self._global_totals["total_time_ms"] += stats.total_execution_time_ms

            return stats

    def get_document_statistics(self, document_id: str) -> ChunkStatistics:
        """Retrieve statistics for a specific document."""
        with self._lock:
            metric = self._doc_metrics.get(document_id)
            if not metric:
                return ChunkStatistics()
            return metric.to_statistics()

    get_stats = get_document_statistics

    def get_global_statistics(self) -> dict[str, Any]:
        """Return a copy of process-wide aggregate metrics."""
        with self._lock:
            return dict(self._global_totals)

    def reset(self) -> None:
        """Reset all metrics state."""
        with self._lock:
            self._doc_metrics.clear()
            for k in self._global_totals:
                self._global_totals[k] = 0.0 if "ms" in k else 0


# Global shared singleton metrics collector
_global_chunk_metrics_collector = ChunkMetricsCollector()


def get_chunk_metrics_collector() -> ChunkMetricsCollector:
    """Get process-wide singleton chunk metrics collector."""
    return _global_chunk_metrics_collector
