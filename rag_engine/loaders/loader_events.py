"""
Loader Events — Audit Trail & Event Notification System.

Sovereign On-Premise Agentic AI Workbench (SIH26117 / MRPL)
Provides structured events for loader lifecycle transitions and a thread-safe
in-memory event bus ready for future Audit Trail integration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import threading
from typing import Any, Callable, Type


@dataclass(frozen=True)
class LoaderEvent:
    """Base class for all loader events."""

    file_path: str
    loader_name: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DocumentLoaded(LoaderEvent):
    """Fired when a document is successfully loaded."""
    doc_id: str = ""
    driver_used: str = "primary"


@dataclass(frozen=True)
class DocumentFailed(LoaderEvent):
    """Fired when a document fails to load."""
    error: str = ""


@dataclass(frozen=True)
class FallbackActivated(LoaderEvent):
    """Fired when primary driver fails or is missing and fallback driver engages."""
    primary_driver: str = ""
    fallback_driver: str = ""
    reason: str = ""


@dataclass(frozen=True)
class ValidationFailed(LoaderEvent):
    """Fired when a document fails pre-load validation."""
    reason: str = ""


@dataclass(frozen=True)
class UnsupportedDocument(LoaderEvent):
    """Fired when an unsupported file format is encountered."""
    extension: str = ""


@dataclass(frozen=True)
class CorruptedDocument(LoaderEvent):
    """Fired when a document is detected as corrupted or malformed."""
    error_details: str = ""


class EventBus:
    """Thread-safe in-memory publish-subscribe event bus."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._subscribers: dict[Type[LoaderEvent], list[Callable[[LoaderEvent], None]]] = {}
        self._global_subscribers: list[Callable[[LoaderEvent], None]] = []
        self._event_log: list[LoaderEvent] = []

    def subscribe(
        self,
        event_type: Type[LoaderEvent],
        callback: Callable[[LoaderEvent], None],
    ) -> None:
        """Subscribe to a specific event type."""
        with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            self._subscribers[event_type].append(callback)

    def subscribe_all(self, callback: Callable[[LoaderEvent], None]) -> None:
        """Subscribe to all events emitted through the bus."""
        with self._lock:
            self._global_subscribers.append(callback)

    def publish(self, event: LoaderEvent) -> None:
        """Publish an event to all interested subscribers."""
        with self._lock:
            self._event_log.append(event)
            type_subs = self._subscribers.get(type(event), []).copy()
            global_subs = self._global_subscribers.copy()

        # Invoke callbacks outside the lock to prevent re-entrant deadlocks
        for sub in type_subs:
            try:
                sub(event)
            except Exception:
                pass

        for sub in global_subs:
            try:
                sub(event)
            except Exception:
                pass

    def get_events(self) -> list[LoaderEvent]:
        """Retrieve a copy of logged events."""
        with self._lock:
            return list(self._event_log)

    def clear(self) -> None:
        """Clear recorded events and subscribers."""
        with self._lock:
            self._event_log.clear()


# Global event bus singleton
global_event_bus = EventBus()
