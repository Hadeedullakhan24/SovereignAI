"""Interface: BaseVectorStore.

Abstract contract for enterprise vector storage and indexing backends.
Maintains 100% backward compatibility with early Milestone 1 methods while
providing the full enterprise API required for Milestone 7.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from rag_engine.schemas.chunk import Chunk
    from rag_engine.schemas.embedding import EmbeddedChunk, EmbeddingVector
    from rag_engine.schemas.retrieved_document import ScoredChunk
    from rag_engine.schemas.vector_store import (
        CollectionStats,
        IndexingResult,
        MetadataFilter,
        OptimizationMetrics,
        PayloadSchemaType,
        ScoredVectorChunk,
        VectorDBHealthReport,
    )
    from rag_engine.vector_store.collection_config import CollectionConfig, VectorStoreConfig


class BaseVectorStore(ABC):
    """Universal abstract base class for enterprise vector storage platforms."""

    # -------------------------------------------------------------------------
    # Lifecycle & Initialization
    # -------------------------------------------------------------------------

    @abstractmethod
    def initialize(self, config: Optional[VectorStoreConfig] = None) -> None:
        """Initialize the storage engine, directory paths, and client connection."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Gracefully release client resources, file locks, and flush pending writes."""
        ...

    # -------------------------------------------------------------------------
    # Collection Governance
    # -------------------------------------------------------------------------

    @abstractmethod
    def create_collection(self, config: CollectionConfig) -> bool:
        """Create a new collection with specified dimensions, distance, and HNSW parameters."""
        ...

    @abstractmethod
    def collection_exists(self, name: str) -> bool:
        """Check whether a collection exists in the vector store."""
        ...

    @abstractmethod
    def delete_collection(self, name: str) -> bool:
        """Delete a collection and all associated vector segments and payload indexes."""
        ...

    @abstractmethod
    def list_collections(self) -> list[str]:
        """List all collection names registered in the vector store."""
        ...

    @abstractmethod
    def get_collection_stats(self, name: str) -> CollectionStats:
        """Retrieve point counts, segment counts, disk usage, and health status for a collection."""
        ...

    # -------------------------------------------------------------------------
    # Enterprise Ingestion & Upsert
    # -------------------------------------------------------------------------

    @abstractmethod
    def upsert_chunks(self, collection_name: str, chunks: list[EmbeddedChunk]) -> IndexingResult:
        """Upsert a list of EmbeddedChunk entities into the designated collection."""
        ...

    # -------------------------------------------------------------------------
    # Retrieval & Search
    # -------------------------------------------------------------------------

    @abstractmethod
    def search_vectors(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 10,
        filters: Optional[MetadataFilter] = None,
    ) -> list[ScoredVectorChunk]:
        """Execute approximate nearest neighbor (ANN) search with optional payload filters."""
        ...

    @abstractmethod
    def get_chunk(self, collection_name: str, chunk_id: str) -> Optional[EmbeddedChunk]:
        """Retrieve a specific chunk by its deterministic chunk identifier."""
        ...

    @abstractmethod
    def get_neighbors(
        self,
        collection_name: str,
        chunk_id: str,
        window: int = 2,
    ) -> list[EmbeddedChunk]:
        """Retrieve adjacent chunks (prev/next) for context expansion using lineage pointers."""
        ...

    # -------------------------------------------------------------------------
    # Deletion API
    # -------------------------------------------------------------------------

    @abstractmethod
    def delete_chunks(self, collection_name: str, chunk_ids: list[str]) -> int:
        """Delete chunks by their deterministic IDs. Returns count of deleted points."""
        ...

    @abstractmethod
    def delete_by_document(self, collection_name: str, document_id: str) -> int:
        """Delete all chunks belonging to a document ID. Returns count of deleted points."""
        ...

    # -------------------------------------------------------------------------
    # Payload Indexing & Optimization
    # -------------------------------------------------------------------------

    @abstractmethod
    def create_payload_index(
        self,
        collection_name: str,
        field_name: str,
        field_type: PayloadSchemaType,
    ) -> bool:
        """Register a payload index on a specific metadata field for fast filtered search."""
        ...

    @abstractmethod
    def optimize_collection(self, collection_name: str) -> OptimizationMetrics:
        """Trigger segment merging, payload compaction, and vacuuming on a collection."""
        ...

    # -------------------------------------------------------------------------
    # Snapshots & Diagnostics
    # -------------------------------------------------------------------------

    @abstractmethod
    def create_snapshot(self, collection_name: str, target_path: Path) -> Path:
        """Create a point-in-time snapshot archive of the collection."""
        ...

    @abstractmethod
    def restore_snapshot(self, collection_name: str, snapshot_path: Path) -> bool:
        """Restore a collection from a point-in-time snapshot archive."""
        ...

    @abstractmethod
    def health_check(self) -> VectorDBHealthReport:
        """Perform diagnostic probes (latency, disk, connectivity) and report health."""
        ...

    # =========================================================================
    # Backward Compatibility Layer (Milestone 1 Contract)
    # =========================================================================

    DEFAULT_COLLECTION: str = "default_collection"

    def add(self, chunks: list[Chunk], embeddings: list[EmbeddingVector]) -> int:
        """Backward-compatible add method from Milestone 1."""
        if len(chunks) != len(embeddings):
            raise ValueError(f"Lengths differ: {len(chunks)} chunks vs {len(embeddings)} embeddings")
        from rag_engine.schemas.embedding import EmbeddedChunk, compute_vector_checksum
        from rag_engine.vector_store.collection_config import CollectionConfig
        
        if not self.collection_exists(self.DEFAULT_COLLECTION):
            dim = embeddings[0].dimension if embeddings else 384
            self.create_collection(CollectionConfig(name=self.DEFAULT_COLLECTION, vector_size=dim))
        
        embedded_list = [
            EmbeddedChunk(
                chunk_id=chk.chunk_id,
                chunk_hash=chk.metadata.sha256 if hasattr(chk.metadata, "sha256") else "",
                embedding=emb.vector,
                model_name=emb.model_name or "unknown",
                embedding_dimension=emb.dimension,
                vector_checksum=compute_vector_checksum(emb.vector),
                metadata=chk.metadata,
                text_preview=chk.content[:200] if chk.content else None,
            )
            for chk, emb in zip(chunks, embeddings)
        ]
        res = self.upsert_chunks(self.DEFAULT_COLLECTION, embedded_list)
        return res.inserted + res.updated

    def search(self, query_embedding: EmbeddingVector, top_k: int = 5) -> list[ScoredChunk]:
        """Backward-compatible search method from Milestone 1."""
        from rag_engine.schemas.chunk import Chunk
        from rag_engine.schemas.retrieved_document import ScoredChunk
        
        if not self.collection_exists(self.DEFAULT_COLLECTION):
            return []
        
        scored_vecs = self.search_vectors(
            collection_name=self.DEFAULT_COLLECTION,
            query_vector=query_embedding.vector,
            limit=top_k,
        )
        results = []
        for i, sv in enumerate(scored_vecs):
            chk = Chunk(
                chunk_id=sv.chunk_id,
                content=sv.text_preview or "",
                token_count=0,
                metadata=sv.metadata,
            )
            results.append(ScoredChunk(chunk=chk, score=sv.score, rank=i))
        return results

    def delete(self, chunk_ids: list[str]) -> int:
        """Backward-compatible delete method from Milestone 1."""
        if not self.collection_exists(self.DEFAULT_COLLECTION):
            return 0
        return self.delete_chunks(self.DEFAULT_COLLECTION, chunk_ids)

    def count(self) -> int:
        """Backward-compatible count method from Milestone 1."""
        if not self.collection_exists(self.DEFAULT_COLLECTION):
            return 0
        return self.get_collection_stats(self.DEFAULT_COLLECTION).vector_count

    def clear(self) -> None:
        """Backward-compatible clear method from Milestone 1."""
        if self.collection_exists(self.DEFAULT_COLLECTION):
            self.delete_collection(self.DEFAULT_COLLECTION)
