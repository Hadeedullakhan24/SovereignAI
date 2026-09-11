"""Base Retriever Interface & Result Schemas.

Abstract contract for all retrieval components in Milestone 8, defining uniform
interfaces for dense, sparse (BM25), hybrid, and adaptive retrieval pipelines.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from rag_engine.interfaces.base_retriever import BaseRetriever as IBaseRetriever
from rag_engine.retrieval.retrieval_metrics import RetrievalMetrics
from rag_engine.schemas.chunk import Chunk
from rag_engine.schemas.citation import Citation
from rag_engine.schemas.retrieved_document import RetrievedDocument, ScoredChunk


class ScoredRetrievalChunk(BaseModel):
    """Rich retrieval candidate with multi-stage scoring telemetry and explainability."""

    model_config = ConfigDict(extra="allow")

    chunk: Chunk = Field(description="Underlying retrieved chunk")
    score: float = Field(description="Final composite or reranked relevance score")
    rank: int = Field(default=0, ge=0, description="0-indexed rank position")
    dense_score: Optional[float] = Field(default=None, description="Original dense cosine similarity")
    bm25_score: Optional[float] = Field(default=None, description="Original sparse BM25 score")
    fusion_score: Optional[float] = Field(default=None, description="Score after RRF fusion")
    rerank_score: Optional[float] = Field(default=None, description="Score from neural cross-encoder")
    boost_applied: float = Field(default=0.0, description="Score increment from metadata boosting")
    explainability: str = Field(default="", description="Human-readable explanation of why retrieved")

    def to_scored_chunk(self) -> ScoredChunk:
        """Convert to base ScoredChunk schema for backward compatibility."""
        # Normalize score into [0.0, 1.0]
        norm_score = max(0.0, min(1.0, self.score))
        return ScoredChunk(chunk=self.chunk, score=norm_score, rank=self.rank)


class CitationBundle(BaseModel):
    """Provenance citation bundle linking retrieved content directly to source document coordinates."""

    model_config = ConfigDict(frozen=True)

    citation_id: str = Field(description="Anchor identifier e.g. '[1]', '[2]'")
    document_id: str = Field(description="Parent document unique identifier")
    document_name: str = Field(default="", description="Source file name")
    source_path: str = Field(default="", description="Filesystem path to source file")
    page_number: Optional[int] = Field(default=None, description="1-indexed source page")
    section_title: Optional[str] = Field(default=None, description="Section heading")
    chunk_id: str = Field(description="Deterministic chunk identifier")
    verbatim_quote: str = Field(default="", description="Verbatim text excerpt")
    score: float = Field(default=0.0, description="Relevance score")
    equipment_tags: list[str] = Field(default_factory=list, description="Associated equipment tags")
    safety_tags: list[str] = Field(default_factory=list, description="Associated safety warnings/standards")
    revision: Optional[str] = Field(default=None, description="Document revision identifier")
    sha256: str = Field(default="", description="SHA-256 hash of source or chunk")

    @property
    def equipment_tag(self) -> str:
        """Primary equipment tag for backward compatibility."""
        return self.equipment_tags[0] if self.equipment_tags else ""

    def to_citation(self) -> Citation:
        """Convert to base Citation schema."""
        from rag_engine.schemas.citation import Source, SourceType
        st = SourceType.PDF if (self.document_name or self.document_id).lower().endswith(".pdf") else SourceType.TEXT
        return Citation(
            citation_id=self.citation_id,
            source=Source(
                file_name=self.document_name or self.document_id,
                file_path=self.source_path,
                source_type=st,
                page_number=self.page_number,
                section=self.section_title,
            ),
            chunk_id=self.chunk_id,
            excerpt=self.verbatim_quote,
            confidence=max(0.0, min(1.0, self.score)),
        )


class RetrievalResult(BaseModel):
    """Authoritative retrieval output consumed by downstream LLM reasoning modules."""

    model_config = ConfigDict(extra="allow")

    query: str = Field(description="Normalized input query string")
    original_query: str = Field(default="", description="Raw input query prior to rewriting")
    strategy_name: str = Field(default="hybrid", description="Identifier of retrieval strategy executed")
    scored_chunks: list[ScoredRetrievalChunk] = Field(default_factory=list, description="Retrieved scored chunks")
    citations: list[CitationBundle] = Field(default_factory=list, description="Deterministic source citations")
    packed_context: str = Field(default="", description="Token-budgeted context string for LLM injection")
    total_candidates: int = Field(default=0, description="Total candidates retrieved before top-k filtering")
    tokens_packed: int = Field(default=0, description="Token count in packed_context")
    execution_time_ms: float = Field(default=0.0, description="Total retrieval duration in milliseconds")
    metrics: Optional[RetrievalMetrics] = Field(default=None, description="Detailed stage-by-stage telemetry")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Operational and diagnostic metadata")

    def __init__(self, **data: Any) -> None:
        if "retrieval_strategy" in data and "strategy_name" not in data:
            data["strategy_name"] = data["retrieval_strategy"]
        if "candidates" in data and "scored_chunks" not in data:
            data["scored_chunks"] = data["candidates"]
        if "formatted_context" in data and "packed_context" not in data:
            data["packed_context"] = data["formatted_context"]
        super().__init__(**data)

    @property
    def candidates(self) -> list[ScoredRetrievalChunk]:
        """Convenience alias for scored_chunks."""
        return self.scored_chunks

    @property
    def formatted_context(self) -> str:
        """Convenience alias for packed_context."""
        return self.packed_context

    def to_retrieved_document(self) -> RetrievedDocument:
        """Export to base RetrievedDocument schema for Milestone 1-7 backward compatibility."""
        return RetrievedDocument(
            query=self.query,
            scored_chunks=[c.to_scored_chunk() for c in self.scored_chunks],
            citations=[c.to_citation() for c in self.citations],
            retrieval_strategy=self.strategy_name,
            total_candidates=self.total_candidates,
        )


class BaseRetriever(IBaseRetriever, ABC):
    """Abstract Base Class for all retrieval components in the Sovereign RAG engine."""

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Any] = None,
        **kwargs: Any,
    ) -> RetrievalResult:
        """Retrieve top-K relevant chunks for query.
        
        Args:
            query: Query string.
            top_k: Maximum chunks to return.
            filters: Optional metadata filters.
            
        Returns:
            RetrievalResult containing scored chunks, citations, and metrics.
        """
        ...

    def batch_retrieve(
        self,
        queries: list[str],
        top_k: int = 10,
        filters: Optional[Any] = None,
        **kwargs: Any,
    ) -> list[RetrievalResult]:
        """Batch retrieve results for multiple queries sequentially or in parallel."""
        return [self.retrieve(q, top_k=top_k, filters=filters, **kwargs) for q in queries]

    def health(self) -> dict[str, Any]:
        """Health check for this retriever component."""
        return {
            "strategy": self.get_strategy_name(),
            "status": "HEALTHY",
        }
