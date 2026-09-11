"""High-resolution telemetry for prompt synthesis, context packing, and token allocation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import threading
from typing import Any


@dataclass(frozen=True)
class PromptMetrics:
    """Telemetry data captured during a single prompt synthesis invocation."""

    query: str
    archetype: str
    total_tokens: int
    system_tokens: int
    context_tokens: int
    history_tokens: int
    query_tokens: int
    context_chunks_count: int
    citations_count: int
    assembly_latency_ms: float
    compression_ratio: float = 1.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PromptMetricsCollector:
    """Thread-safe ring buffer collecting prompt performance telemetry."""

    _instance: PromptMetricsCollector | None = None
    _lock = threading.Lock()

    def __init__(self, max_history: int = 1000) -> None:
        self.max_history = max_history
        self._history: list[PromptMetrics] = []
        self._r_lock = threading.RLock()

    @classmethod
    def get_instance(cls) -> PromptMetricsCollector:
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def record(self, metrics: PromptMetrics) -> None:
        with self._r_lock:
            self._history.append(metrics)
            if len(self._history) > self.max_history:
                self._history = self._history[-self.max_history :]

    def get_summary(self) -> dict[str, Any]:
        with self._r_lock:
            if not self._history:
                return {"total_prompts": 0}
            total = len(self._history)
            avg_lat = sum(m.assembly_latency_ms for m in self._history) / total
            avg_tokens = sum(m.total_tokens for m in self._history) / total
            return {
                "total_prompts": total,
                "avg_assembly_latency_ms": round(avg_lat, 3),
                "avg_tokens": round(avg_tokens, 1),
            }

    def clear(self) -> None:
        with self._r_lock:
            self._history.clear()
