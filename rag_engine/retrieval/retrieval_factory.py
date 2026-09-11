"""Retrieval Engine Factory.

Thread-safe factory creating, configuring, and caching retriever instances.
"""

from __future__ import annotations

import logging
from pathlib import Path
import threading
from typing import Any, Optional

from rag_engine.retrieval.base_retriever import BaseRetriever
from rag_engine.retrieval.retrieval_registry import RetrievalRegistry

logger = logging.getLogger(__name__)


class RetrievalFactory:
    """Factory instantiating and caching configured retriever implementations."""

    _instance: Optional[RetrievalFactory] = None
    _lock = threading.RLock()

    def __init__(self, registry: Optional[RetrievalRegistry] = None) -> None:
        self.registry = registry or RetrievalRegistry.get_instance()
        self._instances: dict[str, BaseRetriever] = {}
        self._instances_lock = threading.RLock()
        self._ensure_core_registrations()

    @classmethod
    def get_instance(cls) -> RetrievalFactory:
        """Singleton accessor with double-checked locking."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _ensure_core_registrations(self) -> None:
        """Register built-in strategies."""
        from rag_engine.retrieval.bm25_retriever import BM25Retriever
        from rag_engine.retrieval.dense_retriever import DenseRetriever
        from rag_engine.retrieval.hybrid_retriever import HybridRetriever
        from rag_engine.retrieval.adaptive_retriever import AdaptiveRetriever

        self.registry.register("dense", DenseRetriever, "Dense vector cosine similarity search")
        self.registry.register("bm25", BM25Retriever, "Sparse Okapi BM25 lexical search")
        self.registry.register("hybrid", HybridRetriever, "Dense + BM25 with Reciprocal Rank Fusion")
        self.registry.register("adaptive", AdaptiveRetriever, "Query-adaptive hybrid retrieval strategy")

    def create_retriever(
        self,
        strategy_name: str = "hybrid",
        use_cached: bool = True,
        **kwargs: Any,
    ) -> BaseRetriever:
        """Create or return cached instance of requested retrieval strategy."""
        key = strategy_name.strip().lower()

        with self._instances_lock:
            if use_cached and key in self._instances and not kwargs:
                return self._instances[key]

            creator = self.registry.get(key)
            try:
                instance = creator(**kwargs)
            except Exception as e:
                logger.error(f"Failed to instantiate retriever '{key}': {e}")
                raise

            if use_cached and not kwargs:
                self._instances[key] = instance

            return instance


# Global convenience accessor
def get_retriever(strategy_name: str = "hybrid", **kwargs: Any) -> BaseRetriever:
    """Convenience function retrieving a configured retriever."""
    return RetrievalFactory.get_instance().create_retriever(strategy_name, **kwargs)
