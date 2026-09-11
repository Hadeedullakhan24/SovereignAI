"""Metadata inheritance engine cascading document, section, and entity tags to chunks."""

from __future__ import annotations

import re
from typing import Any, Optional, Union

from rag_engine.chunking.chunk_context import ChunkContext
from rag_engine.chunking.chunk_utils import compute_sha256
from rag_engine.schemas.chunk import ChunkMetadata
from rag_engine.schemas.document import Document
from rag_engine.schemas.parsed_document import CleanParsedDocument, ParsedDocument, Section, Table


class MetadataInheritor:
    """Propagates lineage, domain entities, safety notices, and coordinates to chunks."""

    @classmethod
    def inherit(
        cls,
        document: Union[Document, ParsedDocument, CleanParsedDocument],
        chunk_content: str,
        chunk_index: int,
        section: Optional[Section] = None,
        table: Optional[Table] = None,
        page_number: Optional[int] = None,
        heading_path: Optional[list[str]] = None,
        char_start: Optional[int] = None,
        char_end: Optional[int] = None,
        context: Optional[ChunkContext] = None,
        is_list: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChunkMetadata:
        """Construct an enriched ChunkMetadata inheriting all upstream context."""
        ctx = context or ChunkContext()

        # 1. Document Level Lineage
        doc_id = getattr(document, "document_id", getattr(document, "doc_id", "unknown_doc"))
        doc_meta = getattr(document, "metadata", None)

        doc_name = getattr(document, "title", "")
        source_path = ""
        doc_sha = ""
        category = getattr(document, "category", "Manual")
        subcategory = "general"

        if doc_meta:
            if hasattr(doc_meta, "file_name"):
                doc_name = doc_meta.file_name or doc_name
            if hasattr(doc_meta, "source_path"):
                source_path = doc_meta.source_path or ""
            if hasattr(doc_meta, "checksum_sha256"):
                doc_sha = doc_meta.checksum_sha256 or ""
            if hasattr(doc_meta, "category") and doc_meta.category:
                category = doc_meta.category
            if hasattr(doc_meta, "subcategory") and doc_meta.subcategory:
                subcategory = doc_meta.subcategory

        # 2. Section & Geometry Coordinates
        sec_title = section.title if section else None
        sec_id = section.section_id if section else None
        resolved_page = page_number
        if resolved_page is None and section and section.page_number:
            resolved_page = section.page_number
        if resolved_page is None and table and table.page_number:
            resolved_page = table.page_number

        # 3. Equipment Entities Extraction & Inheritance
        equipment_tags: set[str] = set()
        operating_params: dict[str, str] = {}

        # Check known document equipment entities
        all_equipment = getattr(document, "equipment", [])
        for eq in all_equipment:
            tag = eq.tag
            if tag in chunk_content:
                equipment_tags.add(tag)
                if eq.operating_pressure:
                    operating_params[f"{tag}_pressure"] = eq.operating_pressure
                if eq.operating_temperature:
                    operating_params[f"{tag}_temperature"] = eq.operating_temperature

        # Also check protected tokens from CleanParsedDocument
        protected_tokens = getattr(document, "protected_tokens", [])
        for tok in protected_tokens:
            if tok in chunk_content and len(tok) > 2:
                # Add if matches equipment or tag pattern
                if re.match(r"^[A-Z]{1,4}[-_][0-9]{2,4}[A-Z]?$", tok) or "Pump" in tok or "Valve" in tok:
                    equipment_tags.add(tok)

        # 4. Safety Entities Extraction & Inheritance
        safety_entities: set[str] = set()
        all_warnings = getattr(document, "warnings", [])
        for w in all_warnings:
            if w.severity:
                if str(w.severity) in chunk_content or (w.text and w.text[:30] in chunk_content):
                    safety_entities.add(str(w.severity))
            for std in getattr(w, "standards", []):
                if std in chunk_content:
                    safety_entities.add(std)

        # Common standards cited in chunk text
        for std_keyword in ["OISD-105", "OISD-116", "OISD-117", "API-610", "ASME", "PNGRB", "ISO-9001"]:
            if std_keyword in chunk_content:
                safety_entities.add(std_keyword)

        # 5. Tabular or List metadata
        is_table = table is not None
        table_id = table.table_id if table else None
        if is_list is None:
            is_list = False
            if not is_table:
                first_lines = [l.strip() for l in chunk_content.splitlines() if l.strip()][:3]
                for fl in first_lines:
                    if re.match(r"^(?:\d+[\.\)]|[\*\-\•\◦\–]|Step\s+\d+|Procedure\s+\d+)", fl, re.IGNORECASE):
                        is_list = True
                        break

        content_sha = compute_sha256(chunk_content)

        return ChunkMetadata(
            document_id=doc_id,
            source_doc_id=doc_id,
            document_name=doc_name,
            source_file=doc_name,
            source_path=source_path,
            sha256=content_sha,
            parent_doc_sha256=doc_sha,
            page_number=resolved_page,
            section_title=sec_title,
            section=sec_title,
            section_id=sec_id,
            heading_path=heading_path or ([] if not sec_title else [sec_title]),
            chunk_index=chunk_index,
            char_start=char_start,
            char_end=char_end,
            category=str(category),
            subcategory=str(subcategory),
            plant_unit=(
                getattr(doc_meta, "plant_unit", None)
                or getattr(document, "plant_unit", None)
                or (getattr(doc_meta, "extra_metadata", {}).get("plant_unit") if doc_meta else None)
            ),
            equipment_entities=sorted(equipment_tags),
            safety_entities=sorted(safety_entities),
            operating_parameters=operating_params,
            is_table_chunk=is_table,
            table_id=table_id,
            is_list_chunk=is_list,
            chunk_strategy=ctx.strategy_name,
            embedding_model=ctx.embedding_model,
        )

    inherit_metadata = inherit
