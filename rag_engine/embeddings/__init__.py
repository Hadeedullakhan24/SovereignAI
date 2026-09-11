"""Offline Embedding Pipeline package for Sovereign On-Premise Agentic AI Workbench (Milestone 6)."""

from rag_engine.embeddings.checkpoint_manager import CheckpointManager
from rag_engine.embeddings.embedding_cache import EmbeddingCache
from rag_engine.embeddings.embedding_factory import EmbeddingFactory
from rag_engine.embeddings.embedding_metrics import EmbeddingMetricsCollector
from rag_engine.embeddings.embedding_pipeline import EmbeddingPipeline
from rag_engine.embeddings.embedding_registry import EmbeddingRegistry
from rag_engine.embeddings.exceptions import (
    CacheError,
    CheckpointError,
    DeviceError,
    EmbeddingError,
    ModelIntegrityError,
    ModelNotFoundError,
    VectorValidationError,
)
from rag_engine.embeddings.local_embedder import (
    DeterministicTestEmbedder,
    LocalHuggingFaceEmbedder,
    detect_device,
    resolve_model_name,
)
from rag_engine.embeddings.vector_validator import VectorValidator
from rag_engine.interfaces.base_embedder import BaseEmbedder

__all__ = [
    "BaseEmbedder",
    "LocalHuggingFaceEmbedder",
    "DeterministicTestEmbedder",
    "EmbeddingRegistry",
    "EmbeddingFactory",
    "EmbeddingCache",
    "CheckpointManager",
    "VectorValidator",
    "EmbeddingPipeline",
    "EmbeddingMetricsCollector",
    "detect_device",
    "resolve_model_name",
    "EmbeddingError",
    "ModelNotFoundError",
    "ModelIntegrityError",
    "VectorValidationError",
    "CacheError",
    "CheckpointError",
    "DeviceError",
]
