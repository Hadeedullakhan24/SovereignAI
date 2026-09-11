# Master Chunk Schema Specification
## Sovereign On-Premise Agentic AI Workbench (SIH26117 / MRPL)

---

## 1. Executive Summary & Design Rationale

In an enterprise on-premise RAG system for refinery critical infrastructure, chunks cannot be simple disjoint strings of text. A chunk must serve as a **self-contained atomic unit of knowledge**, carrying its complete lineage, geographical location within the source document, and rich operational tags.

If a chunk retrieved by an agent lacks its parent page number or equipment tag, the agent cannot provide verifiable source attribution to the refinery engineer. 

This specification defines the authoritative `Chunk` and `ChunkMetadata` schema required for **Milestone 5 (Chunking Engine)** through **Milestone 9 (Complete RAG Pipeline)**.

---

## 2. Complete Pydantic v2 Schema Specification

```python
"""Authoritative Pydantic v2 schema for Chunk and ChunkMetadata."""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class ChunkMetadata(BaseModel):
    """Enriched operational and provenance metadata associated with an atomic text chunk."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Document Lineage & Provenance
    document_id: str = Field(
        description="Unique UUID of parent source Document",
    )
    document_name: str = Field(
        description="Original source file name with extension (e.g. 'Emerson_Control_Valve_Handbook.pdf')",
    )
    source_path: str = Field(
        description="Absolute POSIX filesystem path to source file",
    )
    sha256: str = Field(
        description="SHA-256 hex digest of the chunk's text content (for deduplication and caching)",
    )
    parent_doc_sha256: str = Field(
        default="",
        description="SHA-256 hex digest of the entire parent source document",
    )

    # Document Geometry & Location
    page_number: Optional[int] = Field(
        default=None,
        description="1-indexed source page where this chunk appears",
    )
    section_title: Optional[str] = Field(
        default=None,
        description="Hierarchical section heading (e.g. '1.2 Pump Specifications > Operational Limits')",
    )
    chunk_index: int = Field(
        default=0,
        ge=0,
        description="Sequential 0-indexed position of this chunk within the document",
    )
    char_start: Optional[int] = Field(
        default=None,
        ge=0,
        description="Start character offset in the cleaned document text stream",
    )
    char_end: Optional[int] = Field(
        default=None,
        ge=0,
        description="End character offset in the cleaned document text stream",
    )

    # Refinery Domain Classification
    category: str = Field(
        default="Manual",
        description="Operational category (e.g. 'Manual', 'Safety Document', 'Inspection Report')",
    )
    subcategory: str = Field(
        default="general",
        description="Refinery subcategory or plant unit (e.g. 'pumps', 'compressors', 'CDU-1')",
    )
    plant_unit: Optional[str] = Field(
        default=None,
        description="Specific refinery unit if identified (e.g. 'CDU-1', 'VDU-2', 'DCU', 'PFCC')",
    )

    # Structured Domain Intelligence
    equipment_entities: list[str] = Field(
        default_factory=list,
        description="Equipment tags appearing in or associated with this chunk (e.g. ['P-203', 'MOV-101', 'HX-01'])",
    )
    safety_entities: list[str] = Field(
        default_factory=list,
        description="Safety standards, warning severities, or clauses (e.g. ['OISD-105', 'DANGER', 'API 610'])",
    )
    operating_parameters: dict[str, str] = Field(
        default_factory=dict,
        description="Extracted physical limits (e.g. {'pressure': '12.5 bar', 'temperature': '140 °C'})",
    )

    # Structural Type & Modality
    is_table_chunk: bool = Field(
        default=False,
        description="True if chunk represents serialized tabular data",
    )
    table_id: Optional[str] = Field(
        default=None,
        description="Parent table identifier if is_table_chunk is True",
    )
    image_reference: Optional[str] = Field(
        default=None,
        description="Associated drawing raster path or diagram reference",
    )

    # Operational & Model Attributes
    language: str = Field(
        default="en",
        description="ISO 639-1 language code of chunk content",
    )
    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="Embedding model name used or intended for vectorization",
    )
    version: str = Field(
        default="1.0.0",
        description="Chunk schema specification version",
    )
    processing_timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp of chunk creation",
    )

    # Downstream Retrieval & Reranking Placeholders
    retrieval_score: Optional[float] = Field(
        default=None,
        description="Similarity or dense retrieval score populated at search time",
    )
    rerank_score: Optional[float] = Field(
        default=None,
        description="Cross-encoder relevance score populated at reranking time",
    )


class Chunk(BaseModel):
    """Atomic text chunk with complete metadata, ready for vectorization and retrieval."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(
        description="Deterministic chunk identifier (e.g. 'chk_a1b2c3d4_p42_003')",
    )
    content: str = Field(
        description="Sanitized, normalized text content of the chunk",
    )
    token_count: int = Field(
        default=0,
        ge=0,
        description="Precise or estimated token count according to active model tokenizer",
    )
    character_count: int = Field(
        default=0,
        ge=0,
        description="Length of content string in characters",
    )
    metadata: ChunkMetadata = Field(
        description="Enriched metadata container",
    )

    @classmethod
    def create(
        cls,
        document_id: str,
        document_name: str,
        source_path: str,
        content: str,
        chunk_index: int,
        page_number: Optional[int] = None,
        section_title: Optional[str] = None,
        category: str = "Manual",
        subcategory: str = "general",
        equipment_entities: Optional[list[str]] = None,
        safety_entities: Optional[list[str]] = None,
        token_count: int = 0,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        **kwargs: Any,
    ) -> Chunk:
        """Factory constructor computing deterministic IDs and content SHA-256."""
        clean_text = content.strip()
        sha = hashlib.sha256(clean_text.encode("utf-8")).hexdigest()
        
        # Deterministic chunk ID
        short_doc = document_id.replace("-", "")[:8]
        page_str = f"p{page_number}" if page_number is not None else "p0"
        chunk_id = f"chk_{short_doc}_{page_str}_{chunk_index:04d}"

        meta = ChunkMetadata(
            document_id=document_id,
            document_name=document_name,
            source_path=source_path,
            sha256=sha,
            page_number=page_number,
            section_title=section_title,
            chunk_index=chunk_index,
            category=category,
            subcategory=subcategory,
            equipment_entities=equipment_entities or [],
            safety_entities=safety_entities or [],
            embedding_model=embedding_model,
            **kwargs,
        )

        return cls(
            chunk_id=chunk_id,
            content=clean_text,
            token_count=token_count if token_count > 0 else len(clean_text.split()),
            character_count=len(clean_text),
            metadata=meta,
        )
```

---

## 3. JSON Serialization Representation

Below is an authentic serialization example of a `Chunk` derived from an MRPL engineering manual:

```json
{
  "chunk_id": "chk_271b1917_p42_0008",
  "content": "Centrifugal feed pump P-203 discharges crude feed directly to shell-and-tube heat exchanger HX-01 at an operating pressure of 14.5 bar and an operating temperature of 135 °C. Bypass valve MOV-101 must remain locked closed during normal continuous operations. DANGER: Ensure compliance with OISD-105 isolation procedures prior to servicing line LINE-101-CS.",
  "token_count": 68,
  "character_count": 348,
  "metadata": {
    "document_id": "271b1917-d571-5db6-acca-d21859e0c37a",
    "document_name": "Emerson_Control_Valve_Handbook.pdf",
    "source_path": "D:/SovereignAI/datasets/manuals/Emerson_Control_Valve_Handbook.pdf",
    "sha256": "8f3b61a9332e18d6bc70d99ef824a73e6d8a4f0012589e4c19fa39cb01d32a4e",
    "parent_doc_sha256": "4b68e92f58a36c1e1948ba28178dca9638c4b260e0a5c0b118b05cf398188151",
    "page_number": 42,
    "section_title": "Section 3.1 Crude Feed Pumping Circuit",
    "chunk_index": 8,
    "char_start": 14200,
    "char_end": 14548,
    "category": "Manual",
    "subcategory": "pumps",
    "plant_unit": "CDU-1",
    "equipment_entities": [
      "P-203",
      "HX-01",
      "MOV-101",
      "LINE-101-CS"
    ],
    "safety_entities": [
      "DANGER",
      "OISD-105"
    ],
    "operating_parameters": {
      "operating_pressure": "14.5 bar",
      "operating_temperature": "135 °C"
    },
    "is_table_chunk": false,
    "table_id": null,
    "image_reference": null,
    "language": "en",
    "embedding_model": "BAAI/bge-small-en-v1.5",
    "version": "1.0.0",
    "processing_timestamp": "2026-09-06T18:30:00+00:00",
    "retrieval_score": null,
    "rerank_score": null
  }
}
```

---

## 4. Downstream Module Dependency Matrix

Every field in `ChunkMetadata` serves a specific downstream module:

| Field | Consuming Milestone | Purpose in Subsystem |
|:---|:---|:---|
| `chunk_id` | M7 (Vector DB), M8 (Retriever) | Primary key in ChromaDB / FAISS; anchor for vector deduplication. |
| `sha256` | M6 (Embedding Cache) | Key in disk cache: if chunk text hasn't changed, vector is loaded instantly. |
| `page_number` | M8 (Citation Engine), Member 3 (LLM) | Grounded citation: *"Refer to Page 42 of Emerson Handbook"*. |
| `section_title` | M8 (Context Builder) | Heading breadcrumbs in prompt: `[Section 3.1 Crude Feed Pumping Circuit]`. |
| `equipment_entities` | M7 (Vector DB Filters), M8 (Retriever) | Enables metadata filtering: query for `P-203` can filter `equipment_entities.contains("P-203")`. |
| `safety_entities` | M8 (Retriever Guardrails) | Enables safety-critical queries to prioritize safety notices (`DANGER`, `OISD-105`). |
| `retrieval_score` | M8 (Retriever) | Dense similarity distance recorded during nearest neighbor search. |
| `rerank_score` | M8 (Cross-Encoder) | Final relevance logit assigned by `bge-reranker-base`. |
