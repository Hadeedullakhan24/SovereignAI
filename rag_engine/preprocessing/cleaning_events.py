"""Event definitions and thread-safe EventBus for the Cleaning Engine."""

from __future__ import annotations

from datetime import datetime, timezone
import threading
from typing import Any, Callable, Optional, Type
from pydantic import BaseModel, ConfigDict, Field


class CleaningEvent(BaseModel):
    """Base model for all cleaning lifecycle events."""

    model_config = ConfigDict(frozen=True)

    document_id: str = Field(description="Target document identifier")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Event emission timestamp (ISO 8601)",
    )


class CleaningStarted(CleaningEvent):
    """Published when cleaning begins on a ParsedDocument."""

    total_sections: int = Field(default=0, ge=0)
    total_tables: int = Field(default=0, ge=0)


class CleaningFinished(CleaningEvent):
    """Published when cleaning finishes successfully."""

    execution_time_ms: float = Field(default=0.0, ge=0.0)
    characters_removed: int = Field(default=0, ge=0)
    characters_normalized: int = Field(default=0, ge=0)
    protected_tokens_count: int = Field(default=0, ge=0)


class CleaningFailed(CleaningEvent):
    """Published when an error occurs during cleaning."""

    stage_name: str = Field(default="UNKNOWN")
    error_message: str = Field(default="")


class HeaderRemoved(CleaningEvent):
    """Published when a recurring header is detected and pruned."""

    header_text: str = Field(description="Removed header text")
    occurrences: int = Field(default=1, ge=1)


class FooterRemoved(CleaningEvent):
    """Published when a recurring footer is detected and pruned."""

    footer_text: str = Field(description="Removed footer text")
    occurrences: int = Field(default=1, ge=1)


class ProtectedTokenDetected(CleaningEvent):
    """Published when an engineering token is identified and locked."""

    token: str = Field(description="Identified engineering token")
    token_type: str = Field(default="EQUIPMENT", description="Type of token: EQUIPMENT | STANDARD | UNIT | TAG")


class NormalizationApplied(CleaningEvent):
    """Published when a specific normalization stage modifies content."""

    stage_name: str = Field(description="Name of the pipeline stage")
    characters_modified: int = Field(default=0, ge=0)


EventListener = Callable[[CleaningEvent], None]


class CleaningEventBus:
    """Thread-safe event bus using RLock for publishing and subscribing to cleaning events."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._subscribers: dict[Type[CleaningEvent], list[EventListener]] = {}
        self._history: list[CleaningEvent] = []

    def subscribe(self, event_type: Type[CleaningEvent], listener: EventListener) -> None:
        """Subscribe a callable listener to an event type."""
        with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            if listener not in self._subscribers[event_type]:
                self._subscribers[event_type].append(listener)

    def unsubscribe(self, event_type: Type[CleaningEvent], listener: EventListener) -> None:
        """Unsubscribe a listener from an event type."""
        with self._lock:
            if event_type in self._subscribers and listener in self._subscribers[event_type]:
                self._subscribers[event_type].remove(listener)

    def publish(self, event: CleaningEvent) -> None:
        """Thread-safely append event to history and dispatch to subscribers."""
        with self._lock:
            self._history.append(event)
            listeners: list[EventListener] = []
            for event_type, type_listeners in self._subscribers.items():
                if isinstance(event, event_type):
                    listeners.extend(type_listeners)

        # Dispatch outside the lock to prevent re-entrancy deadlocks
        for listener in listeners:
            try:
                listener(event)
            except Exception:
                # Event dispatch should not crash pipeline execution
                pass

    def get_history(self, document_id: Optional[str] = None) -> list[CleaningEvent]:
        """Return a copy of the event history, optionally filtered by document_id."""
        with self._lock:
            if document_id is None:
                return list(self._history)
            return [e for e in self._history if e.document_id == document_id]

    def clear(self) -> None:
        """Clear all event history and subscribers."""
        with self._lock:
            self._history.clear()
            self._subscribers.clear()


# Default process-wide shared event bus
_global_cleaning_event_bus = CleaningEventBus()


def get_cleaning_event_bus() -> CleaningEventBus:
    """Get the singleton thread-safe cleaning event bus."""
    return _global_cleaning_event_bus
