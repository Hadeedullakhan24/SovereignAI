"""Schema: RetrievedDocument — Represents retrieval results."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from rag_engine.schemas.chunk import Chunk
from rag_engine.schemas.citation import Citation


class ScoredChunk(BaseModel):
    """A chunk with its similarity score from retrieval."""

    model_config = ConfigDict(frozen=True)

    chunk: Chunk = Field(description="The retrieved chunk")
    score: float = Field(ge=0.0, le=1.0, description="Similarity score")
    rank: int = Field(default=0, ge=0, description="Rank position (0-indexed)")


class RetrievedDocument(BaseModel):
    """Complete retrieval result for a query."""

    model_config = ConfigDict(frozen=True)

    query: str = Field(description="Original query string")
    scored_chunks: list[ScoredChunk] = Field(
        default_factory=list, description="Retrieved chunks with scores"
    )
    citations: list[Citation] = Field(
        default_factory=list, description="Source citations"
    )
    retrieval_strategy: str = Field(
        default="", description="Strategy used (dense, hybrid, etc.)"
    )
    total_candidates: int = Field(
        default=0, description="Total candidates before top-k filter"
    )
