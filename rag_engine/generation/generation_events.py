"""Generation Events — Pub/Sub Event Bus for Generation Lifecycle.

Provides thread-safe event dispatch for prompt assembly, generation lifecycle,
guardrail triggers, and streaming milestones.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import logging
import threading
from typing import Any, Callable, Dict, List

logger = logging.getLogger(__name__)


class GenerationEventType(str, Enum):
    """Enumeration of generation lifecycle event types."""

    PROMPT_BUILT = "generation:prompt_built"
    CACHE_HIT = "generation:cache_hit"
    CACHE_MISS = "generation:cache_miss"
    GENERATION_STARTED = "generation:started"
    TOKEN_STREAMED = "generation:token_streamed"
    GENERATION_COMPLETED = "generation:completed"
    CITATION_VALIDATED = "generation:citation_validated"
    HALLUCINATION_DETECTED = "generation:hallucination_detected"
    SAFETY_FLAGGED = "generation:safety_flagged"
    GENERATION_ERROR = "generation:error"


@dataclass(frozen=True)
class GenerationEvent:
    """Immutable event payload published across the generation pipeline."""

    event_type: GenerationEventType
    session_id: str
    query: str
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


GenerationEventHandler = Callable[[GenerationEvent], None]


class GenerationEventBus:
    """Thread-safe event bus for the Generation Engine."""

    _instance: GenerationEventBus | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._subscribers: Dict[GenerationEventType, List[GenerationEventHandler]] = {
            t: [] for t in GenerationEventType
        }
        self._global_subscribers: List[GenerationEventHandler] = []
        self._sub_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> GenerationEventBus:
        """Get or create singleton GenerationEventBus."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def subscribe(
        self,
        event_type: GenerationEventType,
        handler: GenerationEventHandler,
    ) -> None:
        """Subscribe a handler function to a specific event type."""
        with self._sub_lock:
            if handler not in self._subscribers[event_type]:
                self._subscribers[event_type].append(handler)

    def subscribe_all(self, handler: GenerationEventHandler) -> None:
        """Subscribe a handler function to all generation events."""
        with self._sub_lock:
            if handler not in self._global_subscribers:
                self._global_subscribers.append(handler)

    def publish(self, event: GenerationEvent) -> None:
        """Publish an event to all registered subscribers."""
        with self._sub_lock:
            handlers = list(self._subscribers.get(event.event_type, []))
            globals_ = list(self._global_subscribers)

        for handler in handlers + globals_:
            try:
                handler(event)
            except Exception as e:
                logger.error(
                    "Error executing generation event handler %s: %s",
                    handler,
                    e,
                    exc_info=True,
                )

    def clear(self) -> None:
        """Clear all registered event subscribers."""
        with self._sub_lock:
            for t in GenerationEventType:
                self._subscribers[t].clear()
            self._global_subscribers.clear()
