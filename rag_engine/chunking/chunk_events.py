"""Thread-safe EventBus and event definitions for the Chunking Engine."""

from __future__ import annotations

from datetime import datetime, timezone
import threading
from typing import Callable, Optional, Type
from pydantic import BaseModel, ConfigDict, Field


class ChunkEvent(BaseModel):
    """Base model for all chunking lifecycle events."""

    model_config = ConfigDict(frozen=True)

    document_id: str = Field(description="Target document identifier")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp",
    )


class ChunkingStarted(ChunkEvent):
    """Published when chunking begins on a document."""

    strategy_name: str
    total_sections: int = 0
    total_tables: int = 0


class ChunkCreated(ChunkEvent):
    """Published when an individual chunk is successfully formed."""

    chunk_id: str
    chunk_index: int
    token_count: int
    strategy: str


class ChunkValidationFailed(ChunkEvent):
    """Published when a chunk is rejected by quality validation."""

    chunk_id: str
    reason: str


class ChunkingFinished(ChunkEvent):
    """Published when document chunking completes successfully."""

    strategy_name: str
    total_chunks: int
    average_tokens: float
    execution_time_ms: float


class ChunkingFailed(ChunkEvent):
    """Published when document chunking encounters an unhandled failure."""

    strategy_name: str
    error_message: str


ChunkEventListener = Callable[[ChunkEvent], None]


class ChunkEventBus:
    """Thread-safe event bus using RLock for publishing and subscribing to chunking events."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._subscribers: dict[Type[ChunkEvent], list[ChunkEventListener]] = {}
        self._history: list[ChunkEvent] = []

    def subscribe(self, event_type: Type[ChunkEvent], listener: ChunkEventListener) -> None:
        """Subscribe a callable listener to an event type."""
        with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            if listener not in self._subscribers[event_type]:
                self._subscribers[event_type].append(listener)

    def unsubscribe(self, event_type: Type[ChunkEvent], listener: ChunkEventListener) -> None:
        """Unsubscribe a listener from an event type."""
        with self._lock:
            if event_type in self._subscribers and listener in self._subscribers[event_type]:
                self._subscribers[event_type].remove(listener)

    def publish(self, event: ChunkEvent) -> None:
        """Thread-safely record event and dispatch to subscribers outside the lock."""
        with self._lock:
            self._history.append(event)
            listeners: list[ChunkEventListener] = []
            for event_type, type_listeners in self._subscribers.items():
                if isinstance(event, event_type):
                    listeners.extend(type_listeners)

        for listener in listeners:
            try:
                listener(event)
            except Exception:
                pass

    def get_history(self, document_id: Optional[str] = None) -> list[ChunkEvent]:
        """Return event history, optionally filtered by document_id."""
        with self._lock:
            if document_id is None:
                return list(self._history)
            return [e for e in self._history if e.document_id == document_id]

    def clear(self) -> None:
        """Clear all event history and subscribers."""
        with self._lock:
            self._history.clear()
            self._subscribers.clear()


# Global shared singleton event bus
_global_chunk_event_bus = ChunkEventBus()


def get_chunk_event_bus() -> ChunkEventBus:
    """Get process-wide singleton chunking event bus."""
    return _global_chunk_event_bus
