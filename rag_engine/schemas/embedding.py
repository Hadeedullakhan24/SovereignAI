"""Schema: Embedding — Represents dense vector embeddings and embedded chunks.

Authoritative schemas for embedding vectors produced by Milestone 6 Offline Embedding
Pipeline, engineered for downstream vector database indexing (M7), hybrid retrieval (M8),
and cross-encoder reranking.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import struct
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from rag_engine.schemas.chunk import Chunk, ChunkMetadata


def compute_vector_checksum(vector: list[float]) -> str:
    """Compute deterministic SHA-256 checksum over float32 binary representation of vector."""
    packed = struct.pack(f"<{len(vector)}f", *vector)
    return hashlib.sha256(packed).hexdigest()


class EmbeddingVector(BaseModel):
    """A dense embedding vector for a chunk of text (backward-compatible)."""

    model_config = ConfigDict(frozen=True, extra="allow")

    vector: list[float] = Field(description="Dense vector representation")
    dimension: int = Field(description="Vector dimensionality")
    model_name: str = Field(default="", description="Embedding model used")
    source_chunk_id: str = Field(default="", description="ID of the source chunk")


class EmbeddedChunk(BaseModel):
    """A fully embedded chunk with dense vector representation and inherited lineage metadata.

    Produced by Milestone 6 Offline Embedding Pipeline as the atomic ingestion unit
    for Milestone 7 Vector Database.
    """

    model_config = ConfigDict(extra="allow")

    chunk_id: str = Field(description="Unique deterministic chunk ID from Milestone 5")
    chunk_hash: str = Field(description="SHA-256 hash of raw normalized chunk content")
    embedding: list[float] = Field(description="Dense floating point embedding vector")
    model_name: str = Field(description="Identifier of embedding model used (e.g. BAAI/bge-small-en-v1.5)")
    model_version: str = Field(default="1.0.0", description="Semantic version of model weights/config")
    embedding_dimension: int = Field(description="Dimensionality of dense vector (e.g. 384, 768)")
    embedding_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when embedding was generated or loaded from cache",
    )
    vector_checksum: str = Field(description="SHA-256 checksum of the vector for integrity verification")
    validation_status: str = Field(default="VALID", description="Validation status: VALID, INVALID, or CACHED")
    metadata: ChunkMetadata = Field(
        default_factory=ChunkMetadata,
        description="Lineage, equipment tags, safety standards, and coordinate metadata inherited from Chunk",
    )
    content: Optional[str] = Field(default=None, description="Full raw chunk content")
    text_preview: Optional[str] = Field(default=None, description="Short preview of embedded chunk text")

    @property
    def vector(self) -> list[float]:
        """Convenience alias for downstream vector store compatibility."""
        return self.embedding

    @property
    def dimension(self) -> int:
        """Convenience alias for embedding_dimension."""
        return self.embedding_dimension

    @field_validator("vector_checksum", mode="before")
    @classmethod
    def ensure_vector_checksum(cls, v: Optional[str], info: Any) -> str:
        """Auto-calculate vector checksum if omitted."""
        if v and len(v) == 64:
            return v
        data = info.data if hasattr(info, "data") else {}
        vec = data.get("embedding", [])
        if vec:
            return compute_vector_checksum(vec)
        return v or ""


class EmbeddingRequest(BaseModel):
    """Request to generate embeddings for one or more texts."""

    model_config = ConfigDict(frozen=True, extra="allow")

    texts: list[str] = Field(description="Texts to embed")
    model_name: str = Field(default="", description="Model to use for embedding")
    normalize: bool = Field(default=True, description="Whether to L2-normalize vectors")


class EmbeddingBatchRequest(BaseModel):
    """Batch embedding request containing chunk objects."""

    model_config = ConfigDict(extra="allow")

    chunks: list[Chunk] = Field(description="List of Chunk objects to embed")
    model_name: str = Field(default="BAAI/bge-small-en-v1.5", description="Target model")
    batch_size: Optional[int] = Field(default=None, description="Override batch size")
    normalize: bool = Field(default=True, description="Enforce L2 normalization")


class EmbeddingBatchResponse(BaseModel):
    """Response containing embedded chunks and execution statistics."""

    model_config = ConfigDict(extra="allow")

    embedded_chunks: list[EmbeddedChunk] = Field(default_factory=list)
    model_name: str = Field(description="Model used")
    embedding_dimension: int = Field(description="Vector dimension")
    total_chunks: int = Field(default=0)
    cache_hits: int = Field(default=0)
    cache_misses: int = Field(default=0)
    duration_seconds: float = Field(default=0.0)


class EmbeddingMetrics(BaseModel):
    """Telemetry and performance metrics for the embedding pipeline."""

    model_config = ConfigDict(extra="allow")

    model_name: str = Field(default="")
    model_version: str = Field(default="1.0.0")
    device: str = Field(default="cpu")
    total_chunks_processed: int = Field(default=0)
    total_embeddings_generated: int = Field(default=0)
    cache_hits: int = Field(default=0)
    cache_misses: int = Field(default=0)
    cache_reuse_percentage: float = Field(default=0.0)
    total_duration_seconds: float = Field(default=0.0)
    average_latency_ms: float = Field(default=0.0)
    throughput_chunks_per_sec: float = Field(default=0.0)
    cpu_percent: Optional[float] = Field(default=None)
    ram_mb: Optional[float] = Field(default=None)

