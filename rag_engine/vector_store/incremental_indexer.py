"""Incremental Indexer.

Implements 3-way diff reconciliation (New, Modified, Unchanged, Deleted)
to ensure deterministic updates and zero redundant vector writes.
"""

from __future__ import annotations

import logging
from typing import Optional

from rag_engine.interfaces.base_vector_store import BaseVectorStore
from rag_engine.schemas.embedding import EmbeddedChunk
from rag_engine.schemas.vector_store import DiffPlan, FieldFilter, FilterOperator, MetadataFilter

logger = logging.getLogger(__name__)


class IncrementalIndexer:
    """Computes incremental synchronization diffs against existing vector store records."""

    def __init__(self, store: BaseVectorStore) -> None:
        self.store = store

    def calculate_diff(
        self,
        collection_name: str,
        incoming_chunks: list[EmbeddedChunk],
        document_id: Optional[str] = None,
    ) -> DiffPlan:
        """Calculate reconciliation diff plan between incoming chunks and existing records."""
        if not self.store.collection_exists(collection_name):
            # If collection does not exist, all chunks are new
            new_ids = [c.chunk_id for c in incoming_chunks]
            return DiffPlan(new_chunk_ids=new_ids)

        new_ids: list[str] = []
        modified_ids: list[str] = []
        unchanged_ids: list[str] = []

        # Check existing chunks individually
        incoming_map = {c.chunk_id: c for c in incoming_chunks}
        for chunk_id, chunk in incoming_map.items():
            existing = self.store.get_chunk(collection_name, chunk_id)
            if existing is None:
                new_ids.append(chunk_id)
            else:
                # Compare content hash
                if existing.chunk_hash == chunk.chunk_hash:
                    unchanged_ids.append(chunk_id)
                else:
                    modified_ids.append(chunk_id)

        # Detect deleted chunks for the document if document_id provided
        deleted_ids: list[str] = []
        if document_id:
            flt = MetadataFilter(
                must=[
                    FieldFilter(
                        field="document_id",
                        operator=FilterOperator.EQUALS,
                        value=document_id,
                    )
                ]
            )
            # Find existing points for document
            dim = incoming_chunks[0].embedding_dimension if incoming_chunks else 384
            existing_doc_chunks = self.store.search_vectors(
                collection_name=collection_name,
                query_vector=[0.0] * dim,
                limit=1000,
                filters=flt,
            )
            for ex in existing_doc_chunks:
                if ex.chunk_id not in incoming_map:
                    deleted_ids.append(ex.chunk_id)

        return DiffPlan(
            new_chunk_ids=new_ids,
            modified_chunk_ids=modified_ids,
            unchanged_chunk_ids=unchanged_ids,
            deleted_chunk_ids=deleted_ids,
        )
