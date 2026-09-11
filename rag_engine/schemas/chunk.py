"""Schema: Chunk, ChunkMetadata, ChunkHierarchy, and ChunkStatistics.

Authoritative schemas for atomic text chunks produced by Milestone 5 Chunking Engine,
engineered for downstream vectorization (M6), vector DB storage (M7), hybrid retrieval (M8),
and cross-encoder reranking.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ChunkHierarchy(BaseModel):
    """Parent-child hierarchy and sequential relational linkage for a chunk."""

    model_config = ConfigDict(frozen=True, extra="allow")

    document_id: str = Field(description="Unique identifier of parent document")
    parent_doc_id: Optional[str] = Field(
        default=None, description="Alias for document_id"
    )
    parent_section_id: Optional[str] = Field(
        default=None, description="Identifier of direct parent Section"
    )
    heading_path: list[str] = Field(
        default_factory=list,
        description="Breadcrumb trail of ancestor section titles (e.g. ['Chapter 3', '3.1 Pumps'])",
    )
    prev_chunk_id: Optional[str] = Field(
        default=None, description="Deterministic ID of immediately preceding sequential chunk"
    )
    next_chunk_id: Optional[str] = Field(
        default=None, description="Deterministic ID of immediately succeeding sequential chunk"
    )
    hierarchy_depth: int = Field(
        default=1, ge=1, le=8, description="Depth level in document tree hierarchy"
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_doc_ids(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "parent_doc_id" in data and "document_id" not in data:
                data["document_id"] = data["parent_doc_id"]
            elif "document_id" in data and "parent_doc_id" not in data:
                data["parent_doc_id"] = data["document_id"]
        return data


class ChunkMetadata(BaseModel):
    """Enriched operational, provenance, and domain metadata for an atomic chunk."""

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    # Document Lineage & Provenance
    document_id: str = Field(
        default="", description="Unique UUID/ID of parent source Document"
    )
    source_doc_id: str = Field(
        default="", description="Backward-compatible alias for document_id"
    )
    document_name: str = Field(
        default="", description="Original source file name with extension"
    )
    source_file: str = Field(
        default="", description="Backward-compatible alias for document_name"
    )
    source_path: str = Field(
        default="", description="Filesystem path to the source file"
    )
    sha256: str = Field(
        default="", description="SHA-256 hex digest of this chunk's content"
    )
    parent_doc_sha256: str = Field(
        default="", description="SHA-256 hex digest of parent source document"
    )

    # Document Geometry & Location
    page_number: Optional[int] = Field(
        default=None, description="1-indexed source page where this chunk appears"
    )
    section_title: Optional[str] = Field(
        default=None, description="Heading of the section containing this chunk"
    )
    section: Optional[str] = Field(
        default=None, description="Backward-compatible alias for section_title"
    )
    section_id: Optional[str] = Field(
        default=None, description="Unique identifier of parent section"
    )
    heading_path: list[str] = Field(
        default_factory=list, description="Breadcrumb path of ancestor headings"
    )
    chunk_index: int = Field(
        default=0, ge=0, description="Sequential position index within document"
    )
    char_start: Optional[int] = Field(
        default=None, ge=0, description="Start character offset in source/clean text"
    )
    char_end: Optional[int] = Field(
        default=None, ge=0, description="End character offset in source/clean text"
    )

    # Refinery Domain Classification
    category: str = Field(
        default="Manual", description="Document operational category"
    )
    subcategory: str = Field(
        default="general", description="Refinery subcategory or plant unit"
    )
    plant_unit: Optional[str] = Field(
        default=None, description="Refinery unit tag (e.g. 'CDU-1', 'VDU', 'DCU')"
    )

    # Structured Domain Intelligence
    equipment_entities: list[str] = Field(
        default_factory=list,
        description="Equipment tags appearing in or associated with this chunk",
    )
    equipment_id: Optional[str] = Field(
        default=None, description="Primary equipment tag (backward compatible)"
    )
    safety_entities: list[str] = Field(
        default_factory=list,
        description="Safety warnings, standards, or severities (e.g. ['DANGER', 'OISD-105'])",
    )
    operating_parameters: dict[str, str] = Field(
        default_factory=dict,
        description="Operating limits (e.g. {'pressure': '10 bar', 'temperature': '250°C'})",
    )

    # Structural Type & Modality
    is_table_chunk: bool = Field(
        default=False, description="True if chunk represents structured tabular data"
    )
    table_id: Optional[str] = Field(
        default=None, description="Parent table identifier if is_table_chunk is True"
    )
    is_list_chunk: bool = Field(
        default=False, description="True if chunk represents list/procedure items"
    )
    image_reference: Optional[str] = Field(
        default=None, description="Associated drawing raster or schematic reference"
    )

    # Operational & Strategy Attributes
    chunk_strategy: str = Field(
        default="recursive", description="Strategy used: 'fixed' | 'recursive' | 'section' | 'table' | 'list'"
    )
    language: str = Field(default="en", description="ISO 639-1 language code")
    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5", description="Target embedding model"
    )
    version: str = Field(default="1.0.0", description="Chunk schema version")
    processing_timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC creation timestamp",
    )

    # Downstream Retrieval & Reranking Placeholders
    retrieval_score: Optional[float] = Field(
        default=None, description="Search similarity score populated at retrieval time"
    )
    rerank_score: Optional[float] = Field(
        default=None, description="Cross-encoder relevance score populated at rerank time"
    )

    @model_validator(mode="before")
    @classmethod
    def sync_aliases(cls, data: Any) -> Any:
        """Ensure backward compatibility by synchronizing field aliases."""
        if not isinstance(data, dict):
            return data

        # document_id <-> source_doc_id
        doc_id = data.get("document_id") or data.get("source_doc_id") or ""
        data["document_id"] = doc_id
        data["source_doc_id"] = doc_id

        # document_name <-> source_file
        doc_name = data.get("document_name") or data.get("source_file") or ""
        data["document_name"] = doc_name
        data["source_file"] = doc_name

        # section_title <-> section
        sec = data.get("section_title") or data.get("section")
        data["section_title"] = sec
        data["section"] = sec

        # equipment_id <-> equipment_entities
        eq_entities = data.get("equipment_entities")
        if eq_entities and not data.get("equipment_id"):
            data["equipment_id"] = eq_entities[0] if isinstance(eq_entities, list) and eq_entities else None
        elif data.get("equipment_id") and not eq_entities:
            data["equipment_entities"] = [data["equipment_id"]]

        return data


class Chunk(BaseModel):
    """Atomic text chunk with enriched provenance, hierarchy, and tokens."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(
        description="Deterministic, stable chunk identifier (e.g. 'chk_a1b2c3d4_p42_0001_8f3b61a9')"
    )
    content: str = Field(description="Normalized, sanitized text content of the chunk")
    token_count: int = Field(default=0, ge=0, description="Estimated token count")
    word_count: int = Field(default=0, ge=0, description="Word count")
    character_count: int = Field(default=0, ge=0, description="Character count")
    metadata: ChunkMetadata = Field(
        default_factory=ChunkMetadata,
        description="Operational and provenance metadata",
    )
    hierarchy: Optional[ChunkHierarchy] = Field(
        default=None, description="Parent-child and relational hierarchy"
    )

    @property
    def chunk_hash(self) -> str:
        """SHA-256 digest of normalized chunk content for caching and embedding."""
        if hasattr(self, "metadata") and getattr(self.metadata, "sha256", None):
            return self.metadata.sha256
        return hashlib.sha256(self.content.strip().encode("utf-8")).hexdigest()

    @property
    def sha256(self) -> str:
        """Alias for chunk_hash."""
        return self.chunk_hash

    @classmethod
    def create(
        cls,
        document_id: str,
        content: str,
        chunk_index: int,
        document_name: str = "",
        source_path: str = "",
        page_number: Optional[int] = None,
        section_title: Optional[str] = None,
        section_id: Optional[str] = None,
        heading_path: Optional[list[str]] = None,
        category: str = "Manual",
        subcategory: str = "general",
        plant_unit: Optional[str] = None,
        equipment_entities: Optional[list[str]] = None,
        safety_entities: Optional[list[str]] = None,
        operating_parameters: Optional[dict[str, str]] = None,
        is_table_chunk: bool = False,
        table_id: Optional[str] = None,
        is_list_chunk: bool = False,
        chunk_strategy: str = "recursive",
        char_start: Optional[int] = None,
        char_end: Optional[int] = None,
        parent_doc_sha256: str = "",
        token_count: int = 0,
        hierarchy: Optional[ChunkHierarchy] = None,
        **kwargs: Any,
    ) -> Chunk:
        """Factory constructor generating deterministic chunk IDs and SHA-256 digests."""
        clean_text = content.strip()
        sha = hashlib.sha256(clean_text.encode("utf-8")).hexdigest()

        # Deterministic stable Chunk ID: never UUID
        # Uses document_id prefix, page number, chunk_index, and content hash prefix
        short_doc = hashlib.sha256(document_id.encode("utf-8")).hexdigest()[:8]
        page_str = f"p{page_number}" if page_number is not None else "p0"
        chunk_id = f"chk_{short_doc}_{page_str}_{chunk_index:04d}_{sha[:8]}"

        words = len(clean_text.split())
        chars = len(clean_text)
        # Approximate tokens if not provided: ~4 characters per token for technical English
        est_tokens = token_count if token_count > 0 else max(1, (chars + 3) // 4) if clean_text else 0

        meta = ChunkMetadata(
            document_id=document_id,
            document_name=document_name,
            source_path=source_path,
            sha256=sha,
            parent_doc_sha256=parent_doc_sha256,
            page_number=page_number,
            section_title=section_title,
            section_id=section_id,
            heading_path=heading_path or [],
            chunk_index=chunk_index,
            char_start=char_start,
            char_end=char_end,
            category=category,
            subcategory=subcategory,
            plant_unit=plant_unit,
            equipment_entities=equipment_entities or [],
            safety_entities=safety_entities or [],
            operating_parameters=operating_parameters or {},
            is_table_chunk=is_table_chunk,
            table_id=table_id,
            is_list_chunk=is_list_chunk,
            chunk_strategy=chunk_strategy,
            **kwargs,
        )

        return cls(
            chunk_id=chunk_id,
            content=clean_text,
            token_count=est_tokens,
            word_count=words,
            character_count=chars,
            metadata=meta,
            hierarchy=hierarchy,
        )


class ChunkStatistics(BaseModel):
    """Aggregate statistics for chunks produced from a document or dataset."""

    model_config = ConfigDict(frozen=True)

    total_chunks: int = Field(default=0, ge=0)
    average_tokens: float = Field(default=0.0, ge=0.0)
    min_tokens: int = Field(default=0, ge=0)
    max_tokens: int = Field(default=0, ge=0)
    average_characters: float = Field(default=0.0, ge=0.0)
    min_characters: int = Field(default=0, ge=0)
    max_characters: int = Field(default=0, ge=0)
    chunks_per_document: dict[str, int] = Field(default_factory=dict)
    chunks_per_category: dict[str, int] = Field(default_factory=dict)
    chunks_per_section: dict[str, int] = Field(default_factory=dict)
    total_table_chunks: int = Field(default=0, ge=0)
    total_list_chunks: int = Field(default=0, ge=0)
    rejected_chunks_count: int = Field(default=0, ge=0)
    total_execution_time_ms: float = Field(default=0.0, ge=0.0)

    @property
    def smallest_chunk_tokens(self) -> int:
        return self.min_tokens

    @property
    def largest_chunk_tokens(self) -> int:
        return self.max_tokens

    @property
    def smallest_chunk_characters(self) -> int:
        return self.min_characters

    @property
    def largest_chunk_characters(self) -> int:
        return self.max_characters
