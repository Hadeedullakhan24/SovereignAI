"""Factory for creating, configuring, and caching embedding model instances."""

from __future__ import annotations

import logging
from pathlib import Path
import threading
from typing import Optional

from rag_engine.embeddings.embedding_registry import EmbeddingRegistry
from rag_engine.embeddings.exceptions import ModelNotFoundError
from rag_engine.embeddings.local_embedder import (
    DeterministicTestEmbedder,
    LocalHuggingFaceEmbedder,
)
from rag_engine.interfaces.base_embedder import BaseEmbedder

logger = logging.getLogger(__name__)


class EmbeddingFactory:
    """Thread-safe factory creating and managing loaded embedder instances."""

    _instance: Optional[EmbeddingFactory] = None
    _lock = threading.RLock()

    def __init__(self, registry: Optional[EmbeddingRegistry] = None) -> None:
        self.registry = registry or EmbeddingRegistry.get_instance()
        self._instances: dict[str, BaseEmbedder] = {}
        self._instances_lock = threading.RLock()

    @classmethod
    def get_instance(cls) -> EmbeddingFactory:
        """Singleton accessor with double-checked locking."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def get_embedder(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        models_dir: str | Path = "models/embeddings",
        device: Optional[str] = None,
        use_test_fallback: bool = True,
    ) -> BaseEmbedder:
        """Retrieve an existing loaded embedder or create a new one.

        Args:
            model_name: Model identifier or alias.
            models_dir: Directory containing local model weights.
            device: Hardware device override (cpu / cuda).
            use_test_fallback: If True, falls back to DeterministicTestEmbedder
                when local weights are absent in air-gapped test environments.

        Returns:
            BaseEmbedder instance ready for inference.
        """
        canonical = self.registry.resolve(model_name)
        dev_str = device or "auto"
        cache_key = f"{canonical}:{dev_str}"

        with self._instances_lock:
            if cache_key in self._instances:
                return self._instances[cache_key]

            # If explicit test embedder requested
            if "test-embedder" in canonical:
                spec = self.registry.get_specification(canonical)
                embedder = DeterministicTestEmbedder(
                    model_name=canonical,
                    dimension=spec.get("dimension", 384),
                    version=spec.get("version", "1.0.0"),
                    device=device or "cpu",
                )
                self._instances[cache_key] = embedder
                return embedder

            # Attempt to instantiate LocalHuggingFaceEmbedder
            try:
                embedder = LocalHuggingFaceEmbedder(
                    model_name=canonical,
                    models_dir=models_dir,
                    device=device,
                )
                self._instances[cache_key] = embedder
                return embedder
            except (ModelNotFoundError, Exception) as exc:
                if use_test_fallback:
                    logger.warning(
                        "Local weights for %s not found. Falling back to DeterministicTestEmbedder for offline operation: %s",
                        canonical,
                        exc,
                    )
                    spec = self.registry.get_specification(canonical)
                    embedder = DeterministicTestEmbedder(
                        model_name=canonical,
                        dimension=spec.get("dimension", 384),
                        version=spec.get("version", "1.0.0"),
                        device=device or "cpu",
                    )
                    self._instances[cache_key] = embedder
                    return embedder
                raise

    def clear_cache(self) -> None:
        """Purge loaded model instances from memory."""
        with self._instances_lock:
            self._instances.clear()
