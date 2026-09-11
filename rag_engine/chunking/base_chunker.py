"""BaseChunker: Abstract base class implementing the template method pattern for chunking."""

from __future__ import annotations

from abc import abstractmethod
from typing import Optional, Union

from rag_engine.chunking.chunk_context import ChunkContext
from rag_engine.chunking.chunk_events import (
    ChunkCreated,
    ChunkEventBus,
    ChunkValidationFailed,
    ChunkingFailed,
    ChunkingFinished,
    ChunkingStarted,
    get_chunk_event_bus,
)
from rag_engine.chunking.chunk_metrics import (
    ChunkMetricsCollector,
    get_chunk_metrics_collector,
)
from rag_engine.chunking.chunk_validator import ChunkValidator
from rag_engine.chunking.exceptions import ChunkingError, EmptyDocumentError
from rag_engine.chunking.hierarchy_builder import HierarchyBuilder
from rag_engine.interfaces.base_chunker import BaseChunker as IBaseChunker
from rag_engine.schemas.chunk import Chunk
from rag_engine.schemas.document import Document
from rag_engine.schemas.parsed_document import CleanParsedDocument, ParsedDocument


class BaseChunker(IBaseChunker):
    """Template method executor for text chunking strategies."""

    def __init__(
        self,
        context: Optional[ChunkContext] = None,
        event_bus: Optional[ChunkEventBus] = None,
        metrics: Optional[ChunkMetricsCollector] = None,
        validator: Optional[ChunkValidator] = None,
    ) -> None:
        self.context = context or ChunkContext()
        self.event_bus = event_bus or get_chunk_event_bus()
        self.metrics = metrics or get_chunk_metrics_collector()
        self.validator = validator or ChunkValidator(self.context)

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Return the unique chunking strategy identifier."""
        ...

    def get_strategy_name(self) -> str:
        """Fulfill interface contract."""
        return self.strategy_name

    def chunk(self, document: Union[Document, ParsedDocument, CleanParsedDocument]) -> list[Chunk]:
        """Execute the chunking template: validation, extraction, filtering, linking, and metrics."""
        if document is None:
            raise EmptyDocumentError("Input document cannot be None", strategy_name=self.strategy_name)

        doc_id = getattr(document, "document_id", getattr(document, "doc_id", "unknown_doc"))
        self.metrics.start_document(doc_id)
        self.validator.reset()

        total_secs = len(getattr(document, "sections", []))
        total_tbls = len(getattr(document, "tables", []))

        self.event_bus.publish(
            ChunkingStarted(
                document_id=doc_id,
                strategy_name=self.strategy_name,
                total_sections=total_secs,
                total_tables=total_tbls,
            )
        )

        try:
            # 1. Execute concrete strategy extraction
            raw_chunks = self._chunk_document(document)

            if not raw_chunks:
                # Check if document truly had no text
                full_text = getattr(document, "content", "")
                if hasattr(document, "get_full_text"):
                    full_text = document.get_full_text()
                if not full_text.strip():
                    raise EmptyDocumentError(
                        f"Document {doc_id} contains no text content to chunk",
                        strategy_name=self.strategy_name,
                    )

            # 2. Quality validation and filtering
            accepted_chunks, rejected = self.validator.filter_chunks(raw_chunks)

            for rej_chunk, reason in rejected:
                self.metrics.record_rejection(doc_id, reason)
                self.event_bus.publish(
                    ChunkValidationFailed(
                        document_id=doc_id,
                        chunk_id=rej_chunk.chunk_id,
                        reason=reason,
                    )
                )

            # 3. Establish sequential previous/next and parent-child hierarchy
            linked_chunks = HierarchyBuilder.link_chunks(accepted_chunks)

            # 4. Record telemetry and events for accepted chunks
            for chk in linked_chunks:
                self.metrics.record_chunk(doc_id, chk)
                self.event_bus.publish(
                    ChunkCreated(
                        document_id=doc_id,
                        chunk_id=chk.chunk_id,
                        chunk_index=chk.metadata.chunk_index,
                        token_count=chk.token_count,
                        strategy=self.strategy_name,
                    )
                )

            # 5. Complete metrics tracking
            stats = self.metrics.finish_document(doc_id)

            self.event_bus.publish(
                ChunkingFinished(
                    document_id=doc_id,
                    strategy_name=self.strategy_name,
                    total_chunks=len(linked_chunks),
                    average_tokens=stats.average_tokens,
                    execution_time_ms=stats.total_execution_time_ms,
                )
            )

            return linked_chunks

        except EmptyDocumentError:
            raise
        except Exception as exc:
            self.metrics.record_failure(doc_id)
            self.event_bus.publish(
                ChunkingFailed(
                    document_id=doc_id,
                    strategy_name=self.strategy_name,
                    error_message=str(exc),
                )
            )
            raise ChunkingError(
                f"Chunking failed on document {doc_id}: {exc}",
                strategy_name=self.strategy_name,
            ) from exc

    @abstractmethod
    def _chunk_document(self, document: Union[Document, ParsedDocument, CleanParsedDocument]) -> list[Chunk]:
        """Concrete chunking implementation to be provided by subclasses."""
        ...
