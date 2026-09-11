"""Retrieval Strategy Registry & Plugin Architecture.

Provides thread-safe registration and dynamic discovery for retrieval strategies
(Dense, BM25, Hybrid, Adaptive, Graph, SQL, etc.) without modifying core pipeline code.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, Optional, Type

from rag_engine.retrieval.base_retriever import BaseRetriever
from rag_engine.retrieval.retrieval_exceptions import StrategyNotFoundError

logger = logging.getLogger(__name__)


class RetrievalRegistry:
    """Thread-safe registry for retrieval strategies and plugin implementations."""

    _instance: Optional[RetrievalRegistry] = None
    _lock = threading.RLock()

    def __init__(self) -> None:
        self._registry: dict[str, Type[BaseRetriever] | Callable[..., BaseRetriever]] = {}
        self._descriptions: dict[str, str] = {}
        self._reg_lock = threading.RLock()
        self._register_core_defaults()

    def _register_core_defaults(self) -> None:
        """Register built-in strategies lazily."""
        from rag_engine.retrieval.bm25_retriever import BM25Retriever
        from rag_engine.retrieval.dense_retriever import DenseRetriever
        from rag_engine.retrieval.hybrid_retriever import HybridRetriever
        from rag_engine.retrieval.adaptive_retriever import AdaptiveRetriever

        self.register("dense", DenseRetriever, "Dense vector cosine similarity search")
        self.register("bm25", BM25Retriever, "Sparse Okapi BM25 lexical search")
        self.register("hybrid", HybridRetriever, "Dense + BM25 with Reciprocal Rank Fusion")
        self.register("adaptive", AdaptiveRetriever, "Query-adaptive hybrid retrieval strategy")

    @classmethod
    def get_instance(cls) -> RetrievalRegistry:
        """Singleton accessor with double-checked locking."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def register(
        self,
        name: str,
        retriever_cls: Type[BaseRetriever] | Callable[..., BaseRetriever],
        description: str = "",
        override: bool = False,
    ) -> None:
        """Register a retriever class or factory callable under a canonical name."""
        key = name.strip().lower()
        with self._reg_lock:
            if key in self._registry and not override:
                logger.debug(f"Retriever strategy '{key}' already registered. Skipping.")
                return
            self._registry[key] = retriever_cls
            self._descriptions[key] = description
            logger.info(f"Registered retrieval strategy: '{key}' ({description})")

    def get(self, name: str) -> Type[BaseRetriever] | Callable[..., BaseRetriever]:
        """Retrieve registered retriever constructor by name."""
        key = name.strip().lower()
        with self._reg_lock:
            if key not in self._registry:
                available = list(self._registry.keys())
                raise StrategyNotFoundError(
                    f"Retrieval strategy '{name}' not found. Available strategies: {available}"
                )
            return self._registry[key]

    def list_strategies(self) -> dict[str, str]:
        """List all registered retrieval strategies and their descriptions."""
        with self._reg_lock:
            return dict(self._descriptions)

    def contains(self, name: str) -> bool:
        """Check if a strategy is registered."""
        with self._reg_lock:
            return name.strip().lower() in self._registry


# Global decorator for registering custom or plugin retrievers
def register_retriever(name: str, description: str = ""):
    """Class decorator to register a retriever with RetrievalRegistry."""
    def decorator(cls: Type[BaseRetriever]):
        RetrievalRegistry.get_instance().register(name, cls, description=description)
        return cls
    return decorator
