"""Vector Repository.

Universal repository layer abstracting vector storage operations from concrete backends.
All pipeline indexing, querying, deleting, filtering, and updating passes exclusively
through this repository.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from rag_engine.interfaces.base_vector_store import BaseVectorStore
from rag_engine.schemas.embedding import EmbeddedChunk
from rag_engine.schemas.vector_store import (
    CompleteOptimizationReport,
    FieldFilter,
    FilterOperator,
    IndexingResult,
    MetadataFilter,
    ScoredVectorChunk,
    StorageStatsReport,
    VectorDBHealthReport,
)
from rag_engine.vector_store.collection_manager import CollectionManager
from rag_engine.vector_store.collection_router import CollectionRouter
from rag_engine.vector_store.collection_version_manager import CollectionVersionManager
from rag_engine.vector_store.payload_validator import PayloadValidator
from rag_engine.vector_store.schema_validator import SchemaValidator
from rag_engine.vector_store.vector_factory import get_vector_store
from rag_engine.vector_store.vector_lifecycle_manager import VectorLifecycleManager
from rag_engine.vector_store.vector_optimizer import VectorOptimizer

logger = logging.getLogger(__name__)


class VectorRepository:
    """Universal repository layer providing a clean facade over the vector platform."""

    def __init__(
        self,
        store: Optional[BaseVectorStore] = None,
        router: Optional[CollectionRouter] = None,
        version_manager: Optional[CollectionVersionManager] = None,
        lifecycle_manager: Optional[VectorLifecycleManager] = None,
        schema_validator: Optional[SchemaValidator] = None,
        payload_validator: Optional[PayloadValidator] = None,
        optimizer: Optional[VectorOptimizer] = None,
    ) -> None:
        self.store = store or get_vector_store()
        self.router = router or CollectionRouter()
        self.version_manager = version_manager or CollectionVersionManager(self.store)
        self.lifecycle_manager = lifecycle_manager or VectorLifecycleManager(self.store)
        self.schema_validator = schema_validator or SchemaValidator()
        self.payload_validator = payload_validator or PayloadValidator()
        self.optimizer = optimizer or VectorOptimizer(self.store)

    # -------------------------------------------------------------------------
    # Indexing API
    # -------------------------------------------------------------------------

    def save_chunk(
        self,
        chunk: EmbeddedChunk,
        collection_name: Optional[str] = None,
    ) -> str:
        """Save a single chunk, routing automatically if collection_name is omitted."""
        res = self.save_chunks([chunk], collection_name=collection_name)
        return chunk.chunk_id

    def save_chunks(
        self,
        chunks: list[EmbeddedChunk],
        collection_name: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> IndexingResult:
        """Validate, route, and save a batch of embedded chunks."""
        if not chunks:
            return IndexingResult(collection_name=collection_name or "empty", total_chunks=0)

        # 1. Strict pre-validation
        self.schema_validator.validate_batch(chunks)
        self.payload_validator.validate_batch(chunks)

        # 2. Partition by target collection
        if collection_name:
            routed_map = {collection_name: chunks}
        else:
            routed_map = self.router.route_chunks(chunks)

        total_inserted = 0
        total_duration = 0.0
        last_col = collection_name or "multi_collection"

        # 3. Ingest into each destination collection
        for col_base, col_chunks in routed_map.items():
            active_col = self.version_manager.get_active_collection(col_base)

            # Ensure collection exists
            if not self.store.collection_exists(active_col):
                from rag_engine.vector_store.collection_config import CollectionConfig
                dim = col_chunks[0].embedding_dimension if col_chunks else 384
                self.store.create_collection(CollectionConfig(name=active_col, vector_size=dim))

            res = self.store.upsert_chunks(active_col, col_chunks)
            total_inserted += res.inserted
            total_duration += res.duration_ms
            last_col = active_col

        return IndexingResult(
            collection_name=last_col,
            total_chunks=len(chunks),
            inserted=total_inserted,
            duration_ms=total_duration,
        )

    # -------------------------------------------------------------------------
    # Retrieval API
    # -------------------------------------------------------------------------

    def find_by_vector(
        self,
        query_vector: Optional[list[float]] = None,
        collection_name: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 10,
        filters: Optional[MetadataFilter] = None,
    ) -> list[ScoredVectorChunk]:
        """Search for similar vectors or filter metadata across a specific collection or resolved category."""
        target_col = collection_name
        if not target_col and category:
            # Map category to default collection
            cat_lower = category.lower()
            if "manual" in cat_lower:
                target_col = "mrpl_manuals_v1"
            elif "safety" in cat_lower or "oisd" in cat_lower:
                target_col = "mrpl_safety_v1"
            elif "inspect" in cat_lower or "ndt" in cat_lower:
                target_col = "mrpl_inspection_v1"
            elif "maint" in cat_lower:
                target_col = "mrpl_maintenance_v1"
            elif "pid" in cat_lower:
                target_col = "mrpl_pids_v1"
            elif "drawing" in cat_lower:
                target_col = "mrpl_drawings_v1"
            elif "sop" in cat_lower:
                target_col = "mrpl_sops_v1"
            elif "email" in cat_lower:
                target_col = "mrpl_emails_v1"
            else:
                target_col = "mrpl_docs_v1"

        cols = self.store.list_collections()

        # If target collection does not exist, fall back to mrpl_docs_v1 or first available
        if target_col and not self.store.collection_exists(self.version_manager.get_active_collection(target_col)):
            if "mrpl_docs_v1" in cols:
                target_col = "mrpl_docs_v1"
            elif "mrpl_general_v1" in cols:
                target_col = "mrpl_general_v1"
            elif cols:
                target_col = cols[0]

        if not target_col:
            # Search active collection or first available collection
            if "mrpl_docs_v1" in cols:
                target_col = "mrpl_docs_v1"
            elif "mrpl_general_v1" in cols:
                target_col = "mrpl_general_v1"
            elif cols:
                target_col = cols[0]
            else:
                target_col = "mrpl_docs_v1"

        active_col = self.version_manager.get_active_collection(target_col)
        if not self.store.collection_exists(active_col):
            return []

        return self.store.search_vectors(
            collection_name=active_col,
            query_vector=query_vector,
            limit=limit,
            filters=filters,
        )

    def find_by_id(
        self,
        chunk_id: str,
        collection_name: Optional[str] = None,
    ) -> Optional[EmbeddedChunk]:
        """Find a chunk by its deterministic ID across collections."""
        collections = [collection_name] if collection_name else self.store.list_collections()
        for col in collections:
            act_col = self.version_manager.get_active_collection(col)
            chk = self.store.get_chunk(act_col, chunk_id)
            if chk:
                return chk
        return None

    def find_by_equipment(
        self,
        equipment_tag: str,
        collection_name: Optional[str] = None,
        limit: int = 20,
    ) -> list[ScoredVectorChunk]:
        """Find chunks tagged with a specific refinery equipment identifier."""
        flt = MetadataFilter(
            must=[
                FieldFilter(
                    field="equipment_entities",
                    operator=FilterOperator.CONTAINS,
                    value=equipment_tag,
                )
            ]
        )
        return self.find_by_vector(
            query_vector=None,
            collection_name=collection_name,
            limit=limit,
            filters=flt,
        )

    def find_by_document(
        self,
        document_id: str,
        collection_name: Optional[str] = None,
        limit: int = 100,
    ) -> list[ScoredVectorChunk]:
        """Find all chunks belonging to a document."""
        flt = MetadataFilter(
            must=[
                FieldFilter(
                    field="document_id",
                    operator=FilterOperator.EQUALS,
                    value=document_id,
                )
            ]
        )
        return self.find_by_vector(
            query_vector=None,
            collection_name=collection_name,
            limit=limit,
            filters=flt,
        )

    def find_neighbors(
        self,
        chunk_id: str,
        collection_name: Optional[str] = None,
        window: int = 2,
    ) -> list[EmbeddedChunk]:
        """Expand retrieval window by fetching adjacent chunks (prev/next)."""
        collections = [collection_name] if collection_name else self.store.list_collections()
        for col in collections:
            act_col = self.version_manager.get_active_collection(col)
            neighbors = self.store.get_neighbors(act_col, chunk_id, window=window)
            if neighbors:
                return neighbors
        return []

    # -------------------------------------------------------------------------
    # Deletion API
    # -------------------------------------------------------------------------

    def delete_by_id(self, chunk_id: str, collection_name: Optional[str] = None) -> bool:
        """Delete a chunk by ID across collections."""
        collections = [collection_name] if collection_name else self.store.list_collections()
        deleted = False
        for col in collections:
            act_col = self.version_manager.get_active_collection(col)
            cnt = self.store.delete_chunks(act_col, [chunk_id])
            if cnt > 0:
                deleted = True
        return deleted

    def delete_by_document(self, document_id: str, collection_name: Optional[str] = None) -> int:
        """Delete all chunks belonging to a document ID."""
        collections = [collection_name] if collection_name else self.store.list_collections()
        total_deleted = 0
        for col in collections:
            act_col = self.version_manager.get_active_collection(col)
            cnt = self.store.delete_by_document(act_col, document_id)
            total_deleted += cnt
        return total_deleted

    # -------------------------------------------------------------------------
    # Maintenance API
    # -------------------------------------------------------------------------

    def optimize(self, collection_name: Optional[str] = None) -> CompleteOptimizationReport:
        """Trigger segment optimization and vacuuming."""
        return self.optimizer.optimize_all(collection_name)

    def health(self) -> VectorDBHealthReport:
        """Check overall vector database health."""
        return self.store.health_check()

    def stats(self) -> StorageStatsReport:
        """Gather storage statistics and telemetry."""
        from rag_engine.vector_store.storage_stats import StorageStatsCollector
        collector = StorageStatsCollector(self.store)
        return collector.collect_stats()


# Global repository singleton
_global_repository: Optional[VectorRepository] = None


def get_vector_repository() -> VectorRepository:
    """Convenience accessor for global vector repository singleton."""
    global _global_repository
    if _global_repository is None:
        _global_repository = VectorRepository()
    return _global_repository
