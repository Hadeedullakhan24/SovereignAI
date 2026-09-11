"""Schema: Document — Represents a loaded and processed source document."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class DocumentLifecycleState(StrEnum):
    """Lifecycle states through the RAG pipeline."""

    LOADED = "LOADED"
    VALIDATED = "VALIDATED"
    PARSED = "PARSED"
    CLEANED = "CLEANED"
    CHUNKED = "CHUNKED"
    EMBEDDED = "EMBEDDED"
    INDEXED = "INDEXED"
    FAILED = "FAILED"



class DocumentMetadata(BaseModel):
    """Metadata associated with a source document."""

    model_config = ConfigDict(frozen=True)

    source_path: str = Field(description="Absolute path to the source file")
    file_name: str = Field(description="Original file name with extension")
    file_format: str = Field(description="File extension, e.g. '.pdf'")
    mime_type: str = Field(default="application/octet-stream", description="MIME type")
    file_size_bytes: int = Field(default=0, ge=0, description="File size in bytes")
    checksum_sha256: str = Field(default="", description="SHA-256 hex digest of file")

    category: Optional[str] = Field(
        default=None, description="Dataset category, e.g. 'manuals', 'safety_docs'"
    )
    subcategory: Optional[str] = Field(
        default="", description="Refinery subcategory, e.g. 'pumps', 'compressors'"
    )

    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Creation or ingestion timestamp (ISO 8601)",
    )
    modified_at: Optional[str] = Field(
        default=None, description="Source file modification timestamp"
    )
    indexed_at: Optional[str] = Field(
        default=None, description="Pipeline indexing timestamp"
    )

    loader_name: str = Field(default="", description="Loader class used")
    parser_name: Optional[str] = Field(default=None, description="Downstream parser name")
    chunk_strategy: Optional[str] = Field(default=None, description="Downstream chunking strategy")
    embedding_model: Optional[str] = Field(default=None, description="Downstream embedding model")
    vector_store_id: Optional[str] = Field(default=None, description="Vector store document identifier")

    language: str = Field(default="en", description="Document language code")
    page_count: Optional[int] = Field(default=None, description="Number of pages if applicable")
    word_count: int = Field(default=0, ge=0, description="Estimated word count")
    character_count: int = Field(default=0, ge=0, description="Total character count")
    estimated_tokens: int = Field(default=0, ge=0, description="Heuristic token estimation")

    image_reference: Optional[str] = Field(
        default=None, description="Reference path or URI for visual asset documents"
    )
    loading_status: str = Field(default="SUCCESS", description="Loading outcome: SUCCESS | PARTIAL | FAILED")
    lifecycle_state: DocumentLifecycleState = Field(
        default=DocumentLifecycleState.LOADED, description="Current lifecycle state"
    )

    processing_history: list[dict[str, Any]] = Field(
        default_factory=list, description="Audit log of processing stages and timestamps"
    )
    extra_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Format-specific extracted properties"
    )


class Document(BaseModel):
    """A loaded document with full content, metadata, and lifecycle attributes."""

    model_config = ConfigDict(frozen=True)

    doc_id: str = Field(description="Unique document identifier (UUID)")
    content: str = Field(default="", description="Raw text content of the document")
    metadata: DocumentMetadata = Field(description="Document metadata")

    @property
    def uuid(self) -> str:
        """Alias for doc_id."""
        return self.doc_id

    @property
    def raw_content(self) -> str:
        """Alias for content."""
        return self.content

    @property
    def image_reference(self) -> Optional[str]:
        """Convenience property for image reference."""
        return self.metadata.image_reference

    @property
    def loading_status(self) -> str:
        """Convenience property for loading status."""
        return self.metadata.loading_status
