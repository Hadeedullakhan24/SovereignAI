"""Lifecycle events and thread-safe event bus for Prompt Engineering."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import threading
from typing import Any, Callable


class PromptEventType(str, Enum):
    """Enumeration of prompt pipeline lifecycle event types."""

    PROMPT_BUILD_START = "prompt_build_start"
    CONTEXT_PACKED = "context_packed"
    CONTEXT_COMPRESSED = "context_compressed"
    PROMPT_BUILT = "prompt_built"
    PROMPT_VALIDATED = "prompt_validated"
    PROMPT_VALIDATION_FAILED = "prompt_validation_failed"


@dataclass(frozen=True)
class PromptEvent:
    """Immutable event payload dispatched during prompt construction."""

    event_type: PromptEventType
    query: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    data: dict[str, Any] = field(default_factory=dict)


class PromptEventBus:
    """Thread-safe publish/subscribe event bus for prompt synthesis monitoring."""

    _instance: PromptEventBus | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._subscribers: dict[PromptEventType, list[Callable[[PromptEvent], None]]] = {
            t: [] for t in PromptEventType
        }
        self._sub_lock = threading.RLock()

    @classmethod
    def get_instance(cls) -> PromptEventBus:
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def subscribe(
        self,
        event_type: PromptEventType,
        handler: Callable[[PromptEvent], None],
    ) -> None:
        with self._sub_lock:
            self._subscribers[event_type].append(handler)

    def publish(self, event: PromptEvent) -> None:
        with self._sub_lock:
            handlers = list(self._subscribers.get(event.event_type, []))
        for h in handlers:
            try:
                h(event)
            except Exception:
                pass

    def clear(self) -> None:
        with self._sub_lock:
            for t in self._subscribers:
                self._subscribers[t].clear()
