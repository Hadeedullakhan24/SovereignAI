"""Thread-safe registry for cleaners and specialized cleaning pipelines."""

from __future__ import annotations

import threading
from typing import Callable, Optional, Type

from rag_engine.preprocessing.base_cleaner import BaseCleaner
from rag_engine.preprocessing.exceptions import CleaningError


class CleanerRegistry:
    """Thread-safe registry for BaseCleaner implementations using RLock."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._cleaners: dict[str, Type[BaseCleaner]] = {}

    def register(self, name: str, cleaner_cls: Type[BaseCleaner]) -> None:
        """Register a cleaner class under a name or category."""
        if not issubclass(cleaner_cls, BaseCleaner):
            raise CleaningError(f"Class {cleaner_cls} must inherit from BaseCleaner")

        with self._lock:
            self._cleaners[name.lower()] = cleaner_cls

    def get(self, name: str) -> Optional[Type[BaseCleaner]]:
        """Retrieve a registered cleaner class by name."""
        with self._lock:
            return self._cleaners.get(name.lower())

    def list_cleaners(self) -> dict[str, Type[BaseCleaner]]:
        """Return a copy of all registered cleaners."""
        with self._lock:
            return dict(self._cleaners)

    def contains(self, name: str) -> bool:
        """Check if cleaner is registered."""
        with self._lock:
            return name.lower() in self._cleaners

    def unregister(self, name: str) -> bool:
        """Unregister a cleaner by name."""
        with self._lock:
            if name.lower() in self._cleaners:
                del self._cleaners[name.lower()]
                return True
            return False

    def clear(self) -> None:
        """Clear all registered cleaners."""
        with self._lock:
            self._cleaners.clear()


# Global shared registry
_global_cleaner_registry = CleanerRegistry()


def get_cleaner_registry() -> CleanerRegistry:
    """Get singleton thread-safe cleaner registry."""
    return _global_cleaner_registry


def register_cleaner(name: str) -> Callable[[Type[BaseCleaner]], Type[BaseCleaner]]:
    """Decorator to register a BaseCleaner in the global cleaner registry."""

    def decorator(cls: Type[BaseCleaner]) -> Type[BaseCleaner]:
        get_cleaner_registry().register(name, cls)
        return cls

    return decorator
