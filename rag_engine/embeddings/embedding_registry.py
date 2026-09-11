"""Thread-safe catalog and registry for offline embedding models."""

from __future__ import annotations

import threading
from typing import Any, Optional, Type

from rag_engine.embeddings.exceptions import ModelNotFoundError
from rag_engine.embeddings.local_embedder import (
    MODEL_ALIASES,
    MODEL_SPECIFICATIONS,
    DeterministicTestEmbedder,
    LocalHuggingFaceEmbedder,
    resolve_model_name,
)
from rag_engine.interfaces.base_embedder import BaseEmbedder


class EmbeddingRegistry:
    """Registry maintaining metadata and class bindings for embedding models."""

    _instance: Optional[EmbeddingRegistry] = None
    _lock = threading.RLock()

    def __init__(self) -> None:
        self._models: dict[str, dict[str, Any]] = dict(MODEL_SPECIFICATIONS)
        self._aliases: dict[str, str] = dict(MODEL_ALIASES)
        self._custom_classes: dict[str, Type[BaseEmbedder]] = {
            "test-embedder-384": DeterministicTestEmbedder,
            "test-embedder-768": lambda: DeterministicTestEmbedder(
                model_name="test-embedder-768", dimension=768
            ),
        }

    @classmethod
    def get_instance(cls) -> EmbeddingRegistry:
        """Singleton accessor with double-checked locking."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def register_model(
        self,
        model_name: str,
        dimension: int,
        local_folder: str = "",
        passage_prefix: str = "",
        query_prefix: str = "",
        version: str = "1.0.0",
        embedder_class: Optional[Type[BaseEmbedder]] = None,
        aliases: Optional[list[str]] = None,
    ) -> None:
        """Register a new embedding model specification."""
        with self._lock:
            self._models[model_name] = {
                "dimension": dimension,
                "local_folder": local_folder or model_name.replace("/", "_"),
                "passage_prefix": passage_prefix,
                "query_prefix": query_prefix,
                "version": version,
            }
            if embedder_class:
                self._custom_classes[model_name] = embedder_class
            if aliases:
                for a in aliases:
                    self._aliases[a] = model_name

    def resolve(self, name_or_alias: str) -> str:
        """Resolve an alias to its canonical model name."""
        with self._lock:
            return self._aliases.get(name_or_alias.strip(), name_or_alias.strip())

    def get_specification(self, model_name: str) -> dict[str, Any]:
        """Return model metadata dictionary."""
        canonical = self.resolve(model_name)
        with self._lock:
            if canonical in self._models:
                return dict(self._models[canonical])
            if "test-embedder" in canonical:
                return {
                    "dimension": 768 if "768" in canonical else 384,
                    "local_folder": "test",
                    "passage_prefix": "passage: ",
                    "query_prefix": "query: ",
                    "version": "1.0.0",
                }
            raise ModelNotFoundError(f"Model {model_name} is not registered.")

    def list_models(self) -> list[str]:
        """Return list of all registered model identifiers."""
        with self._lock:
            return sorted(self._models.keys())

    def get_embedder_class(self, model_name: str) -> Type[BaseEmbedder]:
        """Return custom embedder class if bound, otherwise LocalHuggingFaceEmbedder."""
        canonical = self.resolve(model_name)
        with self._lock:
            if canonical in self._custom_classes:
                return self._custom_classes[canonical]
            return LocalHuggingFaceEmbedder

