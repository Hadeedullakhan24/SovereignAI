"""Vector Store Registry.

Thread-safe catalog of vector store backend implementations, enabling
dynamic driver discovery and hot-swapping without modifying application logic.
"""

from __future__ import annotations

import threading
from typing import Type

from rag_engine.interfaces.base_vector_store import BaseVectorStore
from rag_engine.vector_store.exceptions import BackendNotSupportedError


class VectorRegistry:
    """Thread-safe registry for vector database backend classes."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._registry: dict[str, Type[BaseVectorStore]] = {}

    def register(self, backend_name: str, store_cls: Type[BaseVectorStore]) -> None:
        """Register a vector store driver class."""
        key = backend_name.lower().strip()
        with self._lock:
            self._registry[key] = store_cls

    def get(self, backend_name: str) -> Type[BaseVectorStore]:
        """Retrieve a registered store class by backend name."""
        key = backend_name.lower().strip()
        with self._lock:
            if key not in self._registry:
                raise BackendNotSupportedError(
                    f"Vector backend '{backend_name}' is not registered. "
                    f"Available backends: {sorted(self._registry.keys())}"
                )
            return self._registry[key]

    def list_backends(self) -> list[str]:
        """List all currently registered backend names."""
        with self._lock:
            return sorted(self._registry.keys())

    def unregister(self, backend_name: str) -> bool:
        """Unregister a driver class."""
        key = backend_name.lower().strip()
        with self._lock:
            return self._registry.pop(key, None) is not None


# Global registry instance
global_vector_registry = VectorRegistry()

# Register default Qdrant driver
try:
    from rag_engine.vector_store.qdrant_store import QdrantVectorStore
    global_vector_registry.register("qdrant", QdrantVectorStore)
    global_vector_registry.register("qdrant_local", QdrantVectorStore)
except ImportError:
    pass
