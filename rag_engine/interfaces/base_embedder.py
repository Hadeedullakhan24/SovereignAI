"""Interface: BaseEmbedder.

Abstract contract for embedding model wrappers. Implementations wrap local models
(sentence-transformers, Hugging Face transformers, ONNX runtime, etc.) operating completely
offline in air-gapped environments.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from rag_engine.schemas.embedding import EmbeddingVector


class BaseEmbedder(ABC):
    """Abstract base class for text embedding models in the sovereign RAG pipeline."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Generate a dense embedding vector for a single text.

        Args:
            text: Input text string to embed.

        Returns:
            List of float values representing the dense embedding vector.

        Raises:
            ValueError: If text is empty or invalid.
        """
        ...

    @abstractmethod
    def embed_batch(
        self,
        texts: list[str],
        batch_size: Optional[int] = None,
        show_progress_bar: Optional[bool] = None,
    ) -> list[list[float]]:
        """Generate embedding vectors for a batch of texts.

        Args:
            texts: List of input texts to embed.
            batch_size: Optional batch size override.
            show_progress_bar: Optional flag to toggle progress bar.

        Returns:
            List of embedding vectors in identical sequence to input texts.
        """
        ...

    def embed_vector(self, text: str, source_chunk_id: str = "") -> EmbeddingVector:
        """Convenience method returning a typed EmbeddingVector schema."""
        vec = self.embed(text)
        return EmbeddingVector(
            vector=vec,
            dimension=self.get_dimension(),
            model_name=self.get_model_name(),
            source_chunk_id=source_chunk_id,
        )

    def embed_passage(self, passage: str) -> list[float]:
        """Generate an embedding for a document chunk/passage with appropriate model prefixing."""
        return self.embed(passage)

    def embed_query(self, query: str) -> list[float]:
        """Generate an embedding for a user query with appropriate model prefixing."""
        return self.embed(query)

    @abstractmethod
    def get_dimension(self) -> int:
        """Return the dimensionality of the embedding vectors (e.g. 384, 768)."""
        ...

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the canonical name/identifier of the loaded model."""
        ...

    def get_model_version(self) -> str:
        """Return the semantic version of the model."""
        return "1.0.0"

    def is_normalized(self) -> bool:
        """Return whether the model outputs unit-length (L2-normalized) vectors."""
        return True

    def get_device(self) -> str:
        """Return device model is running on (e.g. 'cpu', 'cuda')."""
        return "cpu"

    def get_metadata(self) -> dict[str, Any]:
        """Return operational metadata about the loaded model."""
        return {
            "model_name": self.get_model_name(),
            "model_version": self.get_model_version(),
            "dimension": self.get_dimension(),
            "is_normalized": self.is_normalized(),
            "device": self.get_device(),
        }

