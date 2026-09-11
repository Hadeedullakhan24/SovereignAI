"""Dense Vector Retriever.

Interfaces strictly with Milestone 7 VectorRepository to execute dense cosine similarity
search, payload filtering, and neighbor retrieval in local air-gapped Qdrant storage.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from rag_engine.embeddings.embedding_factory import EmbeddingFactory
from rag_engine.interfaces.base_embedder import BaseEmbedder
from rag_engine.retrieval.base_retriever import (
    BaseRetriever,
    CitationBundle,
    RetrievalResult,
    ScoredRetrievalChunk,
)
from rag_engine.retrieval.retrieval_exceptions import DenseRetrievalError
from rag_engine.retrieval.retrieval_metrics import RetrievalMetrics
from rag_engine.schemas.chunk import Chunk, ChunkHierarchy, ChunkMetadata
from rag_engine.schemas.vector_store import MetadataFilter, ScoredVectorChunk
from rag_engine.vector_store.vector_repository import (
    VectorRepository,
    get_vector_repository,
)

logger = logging.getLogger(__name__)


class DenseRetriever(BaseRetriever):
    """Dense vector retriever using local Qdrant via VectorRepository."""

    def __init__(
        self,
        repository: Optional[VectorRepository] = None,
        embedder: Optional[BaseEmbedder] = None,
        model_name: str = "BAAI/bge-small-en-v1.5",
        default_collection: Optional[str] = None,
    ) -> None:
        self.repository = repository or get_vector_repository()
        if embedder:
            self.embedder = embedder
        else:
            factory = EmbeddingFactory.get_instance()
            self.embedder = factory.get_embedder(model_name=model_name, use_test_fallback=True)
        self.default_collection = default_collection

    def get_strategy_name(self) -> str:
        return "dense"

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[MetadataFilter | dict[str, Any]] = None,
        collection_name: Optional[str] = None,
        category: Optional[str] = None,
        **kwargs: Any,
    ) -> RetrievalResult:
        """Execute dense cosine search via VectorRepository."""
        t0 = time.perf_counter()

        if not query or not query.strip():
            raise DenseRetrievalError("Query string cannot be empty for dense retrieval.")

        # 1. Generate query embedding vector
        try:
            query_vector = self.embedder.embed_query(query)
        except Exception as e:
            raise DenseRetrievalError(f"Query vectorization failed: {e}") from e

        # 2. Format MetadataFilter if a dict was passed
        q_filters: Optional[MetadataFilter] = None
        if isinstance(filters, MetadataFilter):
            q_filters = filters
        elif isinstance(filters, dict):
            from rag_engine.retrieval.metadata_filter import MetadataFilterPlanner
            from rag_engine.retrieval.query_analyzer import QueryAnalyzer
            intent = QueryAnalyzer().analyze(query)
            q_filters = MetadataFilterPlanner.plan_filters(intent, explicit_filters=filters)

        target_col = collection_name or self.default_collection

        # 3. Query VectorRepository
        try:
            hits: list[ScoredVectorChunk] = self.repository.find_by_vector(
                query_vector=query_vector,
                collection_name=target_col,
                category=category,
                limit=top_k,
                filters=q_filters,
            )
            # If strict inferred filter produced zero hits, retry unfiltered semantic search
            if not hits and q_filters is not None and not (isinstance(filters, MetadataFilter) or (isinstance(filters, dict) and filters)):
                hits = self.repository.find_by_vector(
                    query_vector=query_vector,
                    collection_name=target_col,
                    category=category,
                    limit=top_k,
                    filters=None,
                )
        except Exception as e:
            raise DenseRetrievalError(f"Vector search failed in VectorRepository: {e}") from e

        dur_ms = (time.perf_counter() - t0) * 1000.0

        # 4. Map ScoredVectorChunk to ScoredRetrievalChunk
        scored_chunks: list[ScoredRetrievalChunk] = []
        citations: list[CitationBundle] = []

        for rank, hit in enumerate(hits):
            meta = hit.metadata
            payload = hit.payload or {}
            content = payload.get("content") or payload.get("text_preview") or hit.text_preview or ""

            # Build hierarchy
            prev_id = payload.get("prev_chunk_id")
            next_id = payload.get("next_chunk_id")
            hierarchy = None
            if prev_id or next_id:
                hierarchy = ChunkHierarchy(
                    document_id=hit.document_id or meta.document_id or "",
                    prev_chunk_id=prev_id,
                    next_chunk_id=next_id,
                )

            chunk = Chunk(
                chunk_id=hit.chunk_id,
                content=content,
                metadata=meta,
                hierarchy=hierarchy,
            )

            explain = f"Dense vector cosine similarity score={hit.score:.4f}."
            sc = ScoredRetrievalChunk(
                chunk=chunk,
                score=hit.score,
                rank=rank,
                dense_score=hit.score,
                explainability=explain,
            )
            scored_chunks.append(sc)

            cit = CitationBundle(
                citation_id=f"[{rank + 1}]",
                document_id=meta.document_id,
                document_name=meta.document_name,
                source_path=meta.source_path,
                page_number=meta.page_number,
                section_title=meta.section_title,
                chunk_id=hit.chunk_id,
                verbatim_quote=content[:250],
                score=hit.score,
                equipment_tags=meta.equipment_entities or [],
                safety_tags=meta.safety_entities or [],
                revision=getattr(meta, "revision", None),
                sha256=meta.sha256,
            )
            citations.append(cit)

        metrics = RetrievalMetrics(
            query=query,
            strategy=self.get_strategy_name(),
            total_duration_ms=dur_ms,
            dense_latency_ms=dur_ms,
            dense_candidates_count=len(hits),
            returned_chunks_count=len(scored_chunks),
            citations_count=len(citations),
            top_score=scored_chunks[0].score if scored_chunks else 0.0,
            average_score=sum(s.score for s in scored_chunks) / len(scored_chunks) if scored_chunks else 0.0,
        )

        return RetrievalResult(
            query=query,
            original_query=query,
            strategy_name=self.get_strategy_name(),
            scored_chunks=scored_chunks,
            citations=citations,
            total_candidates=len(hits),
            execution_time_ms=dur_ms,
            metrics=metrics,
        )

    def health(self) -> dict[str, Any]:
        """Perform health check on underlying VectorRepository and embedder."""
        repo_health = self.repository.health()
        return {
            "strategy": self.get_strategy_name(),
            "status": "HEALTHY" if repo_health.status == "healthy" else "DEGRADED",
            "vector_store_health": repo_health.model_dump(),
            "embedder_model": self.embedder.get_model_name(),
        }
