"""Thread-safe telemetry and metrics collection for document parsing."""

from __future__ import annotations

import threading
import time
from typing import Any, Optional
from pydantic import BaseModel, Field


class ParseMetrics(BaseModel):
    """Execution telemetry captured for a single parsing operation."""

    document_id: str = Field(description="Parsed document identifier")
    parser_name: str = Field(description="Parser class used")
    driver_used: str = Field(default="native", description="Underlying parsing driver or engine")
    duration_ms: float = Field(default=0.0, description="Total execution duration in milliseconds")
    memory_delta_mb: float = Field(default=0.0, description="Estimated memory delta in MB")
    pages_parsed: int = Field(default=0, ge=0)
    sections_extracted: int = Field(default=0, ge=0)
    tables_parsed: int = Field(default=0, ge=0)
    entities_extracted: int = Field(default=0, ge=0)
    warnings_count: int = Field(default=0, ge=0)
    errors_count: int = Field(default=0, ge=0)
    success: bool = Field(default=True)
    error_message: Optional[str] = Field(default=None)


class ParserMetricsCollector:
    """Thread-safe aggregator for system-wide parsing telemetry."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._metrics_log: list[ParseMetrics] = []
        self._total_parsed: int = 0
        self._total_failed: int = 0
        self._total_duration_ms: float = 0.0
        self._total_entities: int = 0
        self._total_tables: int = 0
        self._total_pages: int = 0

    def record(self, metrics: ParseMetrics) -> None:
        """Record a completed parsing operation metrics snapshot."""
        with self._lock:
            self._metrics_log.append(metrics)
            if metrics.success:
                self._total_parsed += 1
            else:
                self._total_failed += 1
            self._total_duration_ms += metrics.duration_ms
            self._total_entities += metrics.entities_extracted
            self._total_tables += metrics.tables_parsed
            self._total_pages += metrics.pages_parsed

    def get_summary(self) -> dict[str, Any]:
        """Return cumulative statistics across all parsed documents."""
        with self._lock:
            avg_duration = (
                self._total_duration_ms / self._total_parsed if self._total_parsed > 0 else 0.0
            )
            return {
                "total_documents_processed": self._total_parsed + self._total_failed,
                "total_successful": self._total_parsed,
                "total_failed": self._total_failed,
                "total_pages_parsed": self._total_pages,
                "total_entities_extracted": self._total_entities,
                "total_tables_parsed": self._total_tables,
                "average_duration_ms": round(avg_duration, 2),
                "total_duration_ms": round(self._total_duration_ms, 2),
            }

    def clear(self) -> None:
        """Reset all tracked metrics."""
        with self._lock:
            self._metrics_log.clear()
            self._total_parsed = 0
            self._total_failed = 0
            self._total_duration_ms = 0.0
            self._total_entities = 0
            self._total_tables = 0
            self._total_pages = 0
