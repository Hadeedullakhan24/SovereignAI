"""
Loader Metrics — Operational Performance & Diagnostics Tracking.

Sovereign On-Premise Agentic AI Workbench (SIH26117 / MRPL)
Thread-safe metrics collection for document loading operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import threading
import time
from typing import Any, Optional


@dataclass
class LoadMetrics:
    """Detailed telemetry for a single document loading operation."""

    file_path: str
    loader_name: str
    duration_ms: float = 0.0
    memory_delta_bytes: int = 0
    primary_driver_used: bool = True
    fallback_driver_used: bool = False
    driver_name: str = "primary"
    retries: int = 0
    success: bool = True
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    extra_stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "file_path": self.file_path,
            "loader_name": self.loader_name,
            "duration_ms": round(self.duration_ms, 2),
            "memory_delta_bytes": self.memory_delta_bytes,
            "primary_driver_used": self.primary_driver_used,
            "fallback_driver_used": self.fallback_driver_used,
            "driver_name": self.driver_name,
            "retries": self.retries,
            "success": self.success,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "extra_stats": dict(self.extra_stats),
        }


class MetricsCollector:
    """Thread-safe collector for aggregating load metrics across concurrent operations."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._history: list[LoadMetrics] = []

    def record(self, metrics: LoadMetrics) -> None:
        """Record a completed loading metric."""
        with self._lock:
            self._history.append(metrics)

    def get_summary(self) -> dict[str, Any]:
        """Generate aggregate performance summary."""
        with self._lock:
            total = len(self._history)
            if total == 0:
                return {
                    "total_operations": 0,
                    "success_rate_pct": 100.0,
                    "avg_duration_ms": 0.0,
                    "primary_driver_count": 0,
                    "fallback_driver_count": 0,
                }
            successes = sum(1 for m in self._history if m.success)
            primary = sum(1 for m in self._history if m.primary_driver_used)
            fallback = sum(1 for m in self._history if m.fallback_driver_used)
            avg_dur = sum(m.duration_ms for m in self._history) / total

            return {
                "total_operations": total,
                "successful_operations": successes,
                "success_rate_pct": round((successes / total) * 100.0, 2),
                "avg_duration_ms": round(avg_dur, 2),
                "primary_driver_count": primary,
                "fallback_driver_count": fallback,
            }

    def clear(self) -> None:
        """Clear recorded history."""
        with self._lock:
            self._history.clear()


# Global thread-safe collector instance
global_metrics = MetricsCollector()
