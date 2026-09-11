"""Observability events and thread-safe event bus for parsing pipeline."""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List
from pydantic import BaseModel, Field


class ParserEvent(BaseModel):
    """Base event emitted during document parsing lifecycle."""

    event_type: str = Field(description="Name of event")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Event emission timestamp (ISO 8601)",
    )
    document_id: str = Field(description="Target document identifier")
    payload: dict[str, Any] = Field(default_factory=dict, description="Event contextual payload")


class DocumentParsed(ParserEvent):
    """Fired when document is completely parsed."""

    event_type: str = "DocumentParsed"


class SectionParsed(ParserEvent):
    """Fired when a document section is extracted."""

    event_type: str = "SectionParsed"


class EntityParsed(ParserEvent):
    """Fired when an equipment entity is extracted."""

    event_type: str = "EntityParsed"


class TableParsed(ParserEvent):
    """Fired when a tabular structure is parsed."""

    event_type: str = "TableParsed"


class MetadataParsed(ParserEvent):
    """Fired when document operational metadata is extracted."""

    event_type: str = "MetadataParsed"


class ValidationCompleted(ParserEvent):
    """Fired when parser validation report is completed."""

    event_type: str = "ValidationCompleted"


class ParsingFailed(ParserEvent):
    """Fired when document parsing encounters an error."""

    event_type: str = "ParsingFailed"


class ParserFallbackActivated(ParserEvent):
    """Fired when primary parser or driver fails and fallback driver executes."""

    event_type: str = "ParserFallbackActivated"


EventHandler = Callable[[ParserEvent], None]


class ParserEventBus:
    """Thread-safe publish-subscribe event bus for parsing engine."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: Dict[str, List[EventHandler]] = {}
        self._global_subscribers: List[EventHandler] = []

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """Subscribe handler to a specific event type."""
        with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            self._subscribers[event_type].append(handler)

    def subscribe_all(self, handler: EventHandler) -> None:
        """Subscribe handler to all events."""
        with self._lock:
            self._global_subscribers.append(handler)

    def publish(self, event: ParserEvent) -> None:
        """Publish event to all registered listeners."""
        with self._lock:
            type_handlers = list(self._subscribers.get(event.event_type, []))
            global_handlers = list(self._global_subscribers)

        for handler in type_handlers + global_handlers:
            try:
                handler(event)
            except Exception:
                # Event handlers must never crash the main parsing thread
                pass

    def clear(self) -> None:
        """Remove all subscriptions."""
        with self._lock:
            self._subscribers.clear()
            self._global_subscribers.clear()
