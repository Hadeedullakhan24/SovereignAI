"""Vector Lifecycle Events & Event Bus.

Publish-subscribe event bus for vector store events (collection created,
indexing finished, rollback triggered, snapshot created).
"""

from __future__ import annotations

from datetime import datetime, timezone
import threading
from typing import Any, Callable, Optional, Type


class VectorEvent:
    """Base event emitted during vector storage lifecycle."""

    def __init__(self, collection_name: str, payload: Optional[dict[str, Any]] = None) -> None:
        self.collection_name = collection_name
        self.payload = payload or {}
        self.timestamp = datetime.now(timezone.utc)


class CollectionCreatedEvent(VectorEvent):
    pass


class ChunksIndexedEvent(VectorEvent):
    pass


class VersionSwappedEvent(VectorEvent):
    pass


class SnapshotCreatedEvent(VectorEvent):
    pass


class VectorEventBus:
    """Thread-safe publish-subscribe event bus for vector lifecycle events."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._subscribers: dict[Type[VectorEvent], list[Callable[[VectorEvent], None]]] = {}

    def subscribe(self, event_cls: Type[VectorEvent], callback: Callable[[VectorEvent], None]) -> None:
        with self._lock:
            if event_cls not in self._subscribers:
                self._subscribers[event_cls] = []
            self._subscribers[event_cls].append(callback)

    def publish(self, event: VectorEvent) -> None:
        with self._lock:
            callbacks = list(self._subscribers.get(type(event), []))
        for cb in callbacks:
            try:
                cb(event)
            except Exception:
                pass


global_vector_event_bus = VectorEventBus()
