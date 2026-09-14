"""Metadata Serializer.

Provides lossless serialization and deserialization between EmbeddedChunk
lineage metadata and Qdrant JSON-compatible payload dictionaries.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from rag_engine.schemas.chunk import ChunkMetadata
from rag_engine.schemas.embedding import EmbeddedChunk


class MetadataSerializer:
    """Serializes EmbeddedChunk metadata into flattened Qdrant payloads and vice versa."""

    @staticmethod
    def to_payload(chunk: EmbeddedChunk) -> dict[str, Any]:
        """Convert EmbeddedChunk into a structured, queryable Qdrant payload dictionary."""
        meta = chunk.metadata
        
        # Primary attributes for payload indexing
        doc_id = getattr(meta, "document_id", "") or ""
        doc_name = getattr(meta, "document_name", "") or ""
        doc_type = getattr(meta, "document_type", "") or ""
        cat = getattr(meta, "category", "") or ""
        subcat = getattr(meta, "subcategory", "") or ""
        plant = getattr(meta, "plant_unit", "") or ""
        
        # Lists of entities (keyword array matching in Qdrant)
        equip = list(getattr(meta, "equipment_entities", [])) if getattr(meta, "equipment_entities", None) else []
        safety = list(getattr(meta, "safety_entities", [])) if getattr(meta, "safety_entities", None) else []
        
        page = getattr(meta, "page_number", None)
        chunk_idx = getattr(meta, "chunk_index", 0)
        is_table = bool(getattr(meta, "is_table_chunk", False))
        is_list = bool(getattr(meta, "is_list_chunk", False))
        section = getattr(meta, "section_title", "") or getattr(meta, "heading", "") or ""
        rev = getattr(meta, "revision", "") or ""
        ver = getattr(meta, "version", "") or ""
        src_file = getattr(meta, "source_file", "") or getattr(meta, "source_path", "") or ""
        lang = getattr(meta, "language", "en") or "en"
        
        # Relational and hierarchy pointers
        prev_id = (
            getattr(chunk, "prev_chunk_id", None)
            or getattr(meta, "prev_chunk_id", None)
            or (getattr(chunk, "hierarchy", None) and getattr(chunk.hierarchy, "prev_chunk_id", None))
        )
        next_id = (
            getattr(chunk, "next_chunk_id", None)
            or getattr(meta, "next_chunk_id", None)
            or (getattr(chunk, "hierarchy", None) and getattr(chunk.hierarchy, "next_chunk_id", None))
        )
        parent_sec = getattr(meta, "parent_section_id", None)
        
        payload: dict[str, Any] = {
            # Identification & Hashing
            "chunk_id": chunk.chunk_id,
            "chunk_hash": chunk.chunk_hash,
            "vector_checksum": chunk.vector_checksum,
            "model_name": chunk.model_name,
            "embedding_dimension": chunk.embedding_dimension,
            "content": getattr(chunk, "content", None) or chunk.text_preview or "",
            "text_preview": chunk.text_preview or "",
            
            # Core Categorization & Ingestion Linage
            "document_id": doc_id,
            "document_name": doc_name,
            "document_type": doc_type,
            "category": cat,
            "subcategory": subcat,
            "plant_unit": plant,
            "source_file": src_file,
            "language": lang,
            "revision": rev,
            "version": ver,
            
            # Domain Entities (for Exact Tag Filtering)
            "equipment_entities": equip,
            "safety_entities": safety,
            
            # Document Layout & Structure
            "page_number": page,
            "chunk_index": chunk_idx,
            "is_table_chunk": is_table,
            "is_list_chunk": is_list,
            "section_title": section,
            
            # Sequential / Context Expansion Links
            "prev_chunk_id": prev_id,
            "next_chunk_id": next_id,
            "parent_section_id": parent_sec,
        }
        
        # Preserve extra metadata attributes under a nested dict
        if hasattr(meta, "extra") and meta.extra:
            payload["extra"] = meta.extra
        
        return payload

    @staticmethod
    def to_metadata(payload: dict[str, Any]) -> ChunkMetadata:
        """Reconstruct a ChunkMetadata object from a stored Qdrant payload dictionary."""
        raw_kwargs: dict[str, Any] = {
            "document_id": payload.get("document_id", ""),
            "document_name": payload.get("document_name", ""),
            "document_type": payload.get("document_type", ""),
            "category": payload.get("category", ""),
            "subcategory": payload.get("subcategory", ""),
            "plant_unit": payload.get("plant_unit", ""),
            "equipment_entities": payload.get("equipment_entities", []),
            "safety_entities": payload.get("safety_entities", []),
            "page_number": payload.get("page_number"),
            "chunk_index": payload.get("chunk_index", 0),
            "is_table_chunk": payload.get("is_table_chunk", False),
            "is_list_chunk": payload.get("is_list_chunk", False),
            "section_title": payload.get("section_title", ""),
            "revision": payload.get("revision", ""),
            "version": payload.get("version", ""),
            "source_file": payload.get("source_file", ""),
            "language": payload.get("language", "en"),
            "prev_chunk_id": payload.get("prev_chunk_id"),
            "next_chunk_id": payload.get("next_chunk_id"),
            "parent_section_id": payload.get("parent_section_id"),
            "sha256": payload.get("chunk_hash", ""),
        }
        if "extra" in payload and isinstance(payload["extra"], dict):
            raw_kwargs.update(payload["extra"])
            
        return ChunkMetadata(**raw_kwargs)
