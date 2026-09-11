"""Vector Store Factory.

Dynamic instantiation and connection pooling for vector store backends.
"""

from __future__ import annotations

import threading
from typing import Optional

from rag_engine.interfaces.base_vector_store import BaseVectorStore
from rag_engine.vector_store.collection_config import VectorStoreConfig
from rag_engine.vector_store.vector_registry import VectorRegistry, global_vector_registry


class VectorFactory:
    """Factory for creating and caching vector store singletons."""

    def __init__(self, registry: Optional[VectorRegistry] = None) -> None:
        self.registry = registry or global_vector_registry
        self._lock = threading.RLock()
        self._instances: dict[str, BaseVectorStore] = {}

    def get_store(
        self,
        config: Optional[VectorStoreConfig] = None,
        reuse_instance: bool = True,
    ) -> BaseVectorStore:
        """Get or create a vector store instance configured for the requested backend."""
        cfg = config or VectorStoreConfig()
        backend_key = cfg.backend.lower().strip()
        cache_key = f"{backend_key}:{cfg.storage_path}:{cfg.url}"

        with self._lock:
            if reuse_instance and cache_key in self._instances:
                return self._instances[cache_key]

            store_cls = self.registry.get(backend_key)
            store_instance = store_cls(config=cfg)

            if reuse_instance:
                self._instances[cache_key] = store_instance

            return store_instance

    def clear_pool(self) -> None:
        """Close and clear all cached vector store instances."""
        with self._lock:
            for store in self._instances.values():
                try:
                    store.close()
                except Exception:
                    pass
            self._instances.clear()


# Global factory instance
global_vector_factory = VectorFactory()


def get_vector_store(config: Optional[VectorStoreConfig] = None) -> BaseVectorStore:
    """Convenience accessor for obtaining a vector store instance."""
    return global_vector_factory.get_store(config)
