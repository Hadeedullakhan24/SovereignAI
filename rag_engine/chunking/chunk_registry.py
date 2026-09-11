"""ChunkRegistry: Thread-safe registry mapping strategy identifiers to chunker classes."""

from __future__ import annotations

import threading
from typing import Callable, Optional, Type

from rag_engine.chunking.base_chunker import BaseChunker
from rag_engine.chunking.exceptions import StrategyNotFoundError


class ChunkRegistry:
    """Thread-safe registry for Chunking strategy classes.
    
    Guarantees thread-safe registration, lookup, and introspection of available
    chunking strategies in an air-gapped on-premise deployment.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._registry: dict[str, Type[BaseChunker]] = {}

    def register(self, strategy_name: str, chunker_cls: Type[BaseChunker], overwrite: bool = False) -> None:
        """Register a chunker strategy class with the registry."""
        if not strategy_name or not isinstance(strategy_name, str):
            raise ValueError("Strategy name must be a non-empty string")
        
        normalized = strategy_name.strip().lower()
        with self._lock:
            if normalized in self._registry and not overwrite:
                raise ValueError(
                    f"Strategy '{normalized}' is already registered to {self._registry[normalized].__name__}. "
                    "Set overwrite=True to replace."
                )
            self._registry[normalized] = chunker_cls

    def get(self, strategy_name: str) -> Type[BaseChunker]:
        """Retrieve the chunker class for a given strategy identifier."""
        normalized = (strategy_name or "").strip().lower()
        with self._lock:
            chunker_cls = self._registry.get(normalized)
            if chunker_cls is None:
                avail = list(self._registry.keys())
                raise StrategyNotFoundError(
                    f"Chunking strategy '{strategy_name}' not found. Available strategies: {avail}",
                    strategy_name=strategy_name,
                )
            return chunker_cls

    def has_strategy(self, strategy_name: str) -> bool:
        """Check if a strategy is registered."""
        normalized = (strategy_name or "").strip().lower()
        with self._lock:
            return normalized in self._registry

    def list_strategies(self) -> list[str]:
        """Return a sorted list of registered strategy identifiers."""
        with self._lock:
            return sorted(self._registry.keys())

    def clear(self) -> None:
        """Clear all registered strategies (primarily for testing)."""
        with self._lock:
            self._registry.clear()


# Global singleton registry instance
_GLOBAL_REGISTRY = ChunkRegistry()
_GLOBAL_LOCK = threading.RLock()


def get_chunk_registry() -> ChunkRegistry:
    """Get the global ChunkRegistry singleton."""
    return _GLOBAL_REGISTRY


def register_chunker(strategy_name: str) -> Callable[[Type[BaseChunker]], Type[BaseChunker]]:
    """Decorator to register a BaseChunker subclass into the global registry."""
    def decorator(cls: Type[BaseChunker]) -> Type[BaseChunker]:
        _GLOBAL_REGISTRY.register(strategy_name, cls, overwrite=True)
        return cls
    return decorator


def _initialize_default_strategies() -> None:
    """Register the standard enterprise chunking strategies."""
    from rag_engine.chunking.fixed_chunker import FixedChunker
    from rag_engine.chunking.list_chunker import ListChunker
    from rag_engine.chunking.recursive_chunker import RecursiveChunker
    from rag_engine.chunking.section_chunker import SectionChunker
    from rag_engine.chunking.table_chunker import TableChunker

    _GLOBAL_REGISTRY.register("fixed", FixedChunker, overwrite=True)
    _GLOBAL_REGISTRY.register("recursive", RecursiveChunker, overwrite=True)
    _GLOBAL_REGISTRY.register("section", SectionChunker, overwrite=True)
    _GLOBAL_REGISTRY.register("table", TableChunker, overwrite=True)
    _GLOBAL_REGISTRY.register("list", ListChunker, overwrite=True)


# Initialize default strategies on module import
_initialize_default_strategies()
