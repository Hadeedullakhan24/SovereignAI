"""Retrieval Event Bus & Domain Events.

Pub/sub event bus decoupling observability, audit logging, and metrics
from core retrieval logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
import threading
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalEvent:
    """Base domain event emitted during retrieval pipeline execution."""

    query: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass(frozen=True)
class RetrievalStarted(RetrievalEvent):
    strategy: str = "hybrid"
    top_k: int = 10


@dataclass(frozen=True)
class QueryAnalyzed(RetrievalEvent):
    intent: str = "informational"
    extracted_equipment: list[str] = field(default_factory=list)
    extracted_standards: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DenseSearchCompleted(RetrievalEvent):
    candidates_count: int = 0
    duration_ms: float = 0.0


@dataclass(frozen=True)
class SparseSearchCompleted(RetrievalEvent):
    candidates_count: int = 0
    duration_ms: float = 0.0


@dataclass(frozen=True)
class FusionCompleted(RetrievalEvent):
    fused_candidates_count: int = 0
    duration_ms: float = 0.0


@dataclass(frozen=True)
class RerankCompleted(RetrievalEvent):
    reranked_count: int = 0
    duration_ms: float = 0.0


@dataclass(frozen=True)
class ContextExpanded(RetrievalEvent):
    original_chunks: int = 0
    expanded_chunks: int = 0


@dataclass(frozen=True)
class RetrievalCompleted(RetrievalEvent):
    total_chunks: int = 0
    total_citations: int = 0
    total_tokens: int = 0
    total_duration_ms: float = 0.0
    cache_hit: bool = False


@dataclass(frozen=True)
class RetrievalFailed(RetrievalEvent):
    error: str = ""


class RetrievalEventBus:
    """Thread-safe publish/subscribe bus for retrieval events."""

    _instance: Optional[RetrievalEventBus] = None
    _lock = threading.RLock()

    def __init__(self) -> None:
        self._subscribers: list[Callable[[RetrievalEvent], None]] = []
        self._sub_lock = threading.RLock()

    @classmethod
    def get_instance(cls) -> RetrievalEventBus:
        """Singleton accessor with double-checked locking."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def subscribe(self, callback: Callable[[RetrievalEvent], None]) -> None:
        """Register a callback for retrieval events."""
        with self._sub_lock:
            if callback not in self._subscribers:
                self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[RetrievalEvent], None]) -> None:
        """Unregister a callback."""
        with self._sub_lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

    def publish(self, event: RetrievalEvent) -> None:
        """Publish an event to all subscribers."""
        with self._sub_lock:
            subscribers = list(self._subscribers)

        for sub in subscribers:
            try:
                sub(event)
            except Exception as e:
                logger.warning(f"Error in retrieval event subscriber {sub}: {e}")

    def clear(self) -> None:
        """Clear all subscribers."""
        with self._sub_lock:
            self._subscribers.clear()
