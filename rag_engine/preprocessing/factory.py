"""Thread-safe Factory for creating cleaners and pipelines."""

from __future__ import annotations

import threading
from typing import Any, Optional

from rag_engine.preprocessing.base_cleaner import BaseCleaner
from rag_engine.preprocessing.cleaning_pipeline import CleaningPipeline
from rag_engine.preprocessing.registry import CleanerRegistry, get_cleaner_registry


class CleanerFactory:
    """Thread-safe Factory for resolving and instantiating cleaners using RLock."""

    def __init__(self, registry: Optional[CleanerRegistry] = None) -> None:
        self._lock = threading.RLock()
        self._registry = registry or get_cleaner_registry()

    def create_cleaner(
        self,
        category: Optional[str] = None,
        **kwargs: Any,
    ) -> BaseCleaner:
        """Resolve and instantiate appropriate cleaner or pipeline."""
        with self._lock:
            if category:
                cleaner_cls = self._registry.get(category)
                if cleaner_cls:
                    return cleaner_cls(**kwargs)

            # Default pipeline
            default_cls = self._registry.get("default")
            if default_cls:
                return default_cls(**kwargs)

            return CleaningPipeline(**kwargs)


# Global shared factory
_global_cleaner_factory = CleanerFactory()


def get_cleaner_factory() -> CleanerFactory:
    """Get the singleton thread-safe cleaner factory."""
    return _global_cleaner_factory
