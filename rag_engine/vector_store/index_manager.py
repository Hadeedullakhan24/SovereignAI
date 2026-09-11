"""Index Manager.

Orchestrates 8-stage batch ingestion, schema validation, 3-way diff reconciliation,
atomic transaction logging, chunk slicing, and index manifest persistence.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time
from typing import Optional

from rag_engine.interfaces.base_vector_store import BaseVectorStore
from rag_engine.schemas.embedding import EmbeddedChunk
from rag_engine.schemas.vector_store import IndexingResult
from rag_engine.vector_store.collection_router import CollectionRouter
from rag_engine.vector_store.incremental_indexer import IncrementalIndexer
from rag_engine.vector_store.payload_validator import PayloadValidator
from rag_engine.vector_store.schema_validator import SchemaValidator
from rag_engine.vector_store.transaction_manager import TransactionManager

logger = logging.getLogger(__name__)


class IndexManager:
    """Master orchestrator for batch chunk ingestion and index maintenance."""

    def __init__(
        self,
        store: BaseVectorStore,
        batch_size: int = 500,
        router: Optional[CollectionRouter] = None,
        schema_validator: Optional[SchemaValidator] = None,
        payload_validator: Optional[PayloadValidator] = None,
        tx_manager: Optional[TransactionManager] = None,
        reconciler: Optional[IncrementalIndexer] = None,
        manifest_path: Path = Path("vector_db/index_manifest.json"),
    ) -> None:
        self.store = store
        self.batch_size = batch_size
        self.router = router or CollectionRouter()
        self.schema_validator = schema_validator or SchemaValidator()
        self.payload_validator = payload_validator or PayloadValidator()
        self.tx_manager = tx_manager or TransactionManager(store)
        self.reconciler = reconciler or IncrementalIndexer(store)
        self.manifest_path = manifest_path

    def index_document_chunks(
        self,
        chunks: list[EmbeddedChunk],
        collection_name: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> IndexingResult:
        """Execute the full 8-stage ingestion pipeline on a list of EmbeddedChunk entities."""
        if not chunks:
            return IndexingResult(collection_name=collection_name or "none", total_chunks=0)

        start_time = time.perf_counter()

        # Stage 1: Vector quality & dimension validation
        self.schema_validator.validate_batch(chunks)

        # Stage 2: Payload schema validation
        self.payload_validator.validate_batch(chunks)

        # Stage 3: Multi-collection routing
        if collection_name:
            routed_batches = {collection_name: chunks}
        else:
            routed_batches = self.router.route_chunks(chunks)

        total_inserted = 0
        total_updated = 0
        total_skipped = 0
        total_deleted = 0
        last_col = collection_name or "multi"

        for col, col_chunks in routed_batches.items():
            last_col = col

            # Ensure destination collection exists
            if not self.store.collection_exists(col):
                from rag_engine.vector_store.collection_config import CollectionConfig
                dim = col_chunks[0].embedding_dimension if col_chunks else 384
                self.store.create_collection(CollectionConfig(name=col, vector_size=dim))

            # Stage 4: 3-Way diff reconciliation
            diff = self.reconciler.calculate_diff(col, col_chunks, document_id=document_id)
            total_skipped += len(diff.unchanged_chunk_ids)

            chunks_to_write_ids = set(diff.new_chunk_ids) | set(diff.modified_chunk_ids)
            chunks_to_write = [c for c in col_chunks if c.chunk_id in chunks_to_write_ids]

            # Stage 5: Begin atomic transaction
            tx_id = self.tx_manager.begin_transaction(
                collection_name=col,
                chunk_ids=[c.chunk_id for c in chunks_to_write],
            )

            try:
                # Stage 6: Purge deleted chunks
                if diff.deleted_chunk_ids:
                    self.store.delete_chunks(col, diff.deleted_chunk_ids)
                    total_deleted += len(diff.deleted_chunk_ids)

                # Stage 7: Slice in batches and upsert
                for i in range(0, len(chunks_to_write), self.batch_size):
                    slice_batch = chunks_to_write[i : i + self.batch_size]
                    self.store.upsert_chunks(col, slice_batch)

                total_inserted += len(diff.new_chunk_ids)
                total_updated += len(diff.modified_chunk_ids)

                # Commit transaction
                self.tx_manager.commit_transaction(tx_id)
            except Exception as e:
                logger.error("Transaction '%s' failed during ingestion: %s", tx_id, e)
                self.tx_manager.rollback_transaction(tx_id)
                raise

        # Stage 8: Update authoritative index manifest
        self._update_manifest(last_col, total_inserted + total_updated)

        duration_ms = (time.perf_counter() - start_time) * 1000.0
        return IndexingResult(
            collection_name=last_col,
            total_chunks=len(chunks),
            inserted=total_inserted,
            updated=total_updated,
            deleted=total_deleted,
            skipped=total_skipped,
            duration_ms=duration_ms,
        )

    def _update_manifest(self, collection_name: str, written_count: int) -> None:
        """Update persistent index manifest."""
        try:
            self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest = {}
            if self.manifest_path.exists():
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)

            cols = manifest.get("collections", {})
            col_info = cols.get(collection_name, {"total_points": 0})
            col_info["total_points"] = col_info.get("total_points", 0) + written_count
            col_info["last_updated"] = datetime.now(timezone.utc).isoformat()
            cols[collection_name] = col_info

            manifest["collections"] = cols
            manifest["updated_at"] = datetime.now(timezone.utc).isoformat()

            with open(self.manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
        except Exception as e:
            logger.warning("Error updating index manifest: %s", e)
