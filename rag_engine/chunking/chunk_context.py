"""Configuration context for the Chunking Engine."""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ChunkContext(BaseModel):
    """Immutable configuration and parameter context governing chunking execution."""

    model_config = ConfigDict(frozen=True, extra="allow")

    # Token and sizing constraints
    target_chunk_tokens: int = Field(
        default=512, ge=16, le=4096, description="Target token count for a chunk"
    )
    chunk_overlap_tokens: int = Field(
        default=64, ge=0, le=1024, description="Overlap token count between sequential chunks"
    )
    min_chunk_tokens: int = Field(
        default=10, ge=1, description="Minimum tokens required for a valid chunk (avoids tiny noise)"
    )
    max_chunk_tokens: int = Field(
        default=1024, ge=32, description="Hard ceiling for maximum allowable tokens per chunk"
    )

    # Strategy and model attributes
    strategy_name: str = Field(
        default="recursive",
        description="Chunking strategy: 'fixed' | 'recursive' | 'section' | 'table' | 'list' | 'auto'",
    )
    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="Target open-weight embedding model for downstream vectorization",
    )

    # Structural preservation toggles
    preserve_tables: bool = Field(
        default=True,
        description="If True, extracts tables into dedicated, cohesive tabular chunks",
    )
    preserve_lists: bool = Field(
        default=True,
        description="If True, attempts to keep sequential procedures and checklists intact",
    )
    preserve_sections: bool = Field(
        default=True,
        description="If True, respects H1-H6 section boundaries without cross-section bleeding",
    )
    include_heading_context: bool = Field(
        default=True,
        description="If True, prepends or embeds hierarchical heading breadcrumbs into chunk context",
    )

    # Quality control
    deduplicate_chunks: bool = Field(
        default=True,
        description="If True, automatically rejects duplicate chunks based on content hash",
    )
    strict_token_limits: bool = Field(
        default=False,
        description="If True, raises error if any chunk exceeds max_chunk_tokens",
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Map common alias names to internal fields
            if "target_tokens" in data and "target_chunk_tokens" not in data:
                data["target_chunk_tokens"] = data.pop("target_tokens")
            if "overlap_tokens" in data and "chunk_overlap_tokens" not in data:
                data["chunk_overlap_tokens"] = data.pop("overlap_tokens")
            if "overlap" in data and "chunk_overlap_tokens" not in data:
                data["chunk_overlap_tokens"] = data.pop("overlap")
            if "min_tokens" in data and "min_chunk_tokens" not in data:
                data["min_chunk_tokens"] = data.pop("min_tokens")
            if "max_tokens" in data and "max_chunk_tokens" not in data:
                data["max_chunk_tokens"] = data.pop("max_tokens")
        return data

    @property
    def target_tokens(self) -> int:
        """Alias for target_chunk_tokens."""
        return self.target_chunk_tokens

    @property
    def overlap_tokens(self) -> int:
        """Alias for chunk_overlap_tokens."""
        return self.chunk_overlap_tokens

    @property
    def min_tokens(self) -> int:
        """Alias for min_chunk_tokens."""
        return self.min_chunk_tokens

    @property
    def max_tokens(self) -> int:
        """Alias for max_chunk_tokens."""
        return self.max_chunk_tokens
