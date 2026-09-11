"""Generic text parser for standard text, logs, specifications, and fallback documents."""

from __future__ import annotations

import uuid
from typing import Optional

from rag_engine.parsers.base_parser import BaseParser
from rag_engine.parsers.parser_registry import register_parser
from rag_engine.parsers.parser_utils import ParserUtils
from rag_engine.parsers.parsing_context import ParsingContext
from rag_engine.schemas.document import Document
from rag_engine.schemas.parsed_document import (
    DocumentStatistics,
    ParsedDocument,
    ParsedMetadata,
)


@register_parser(extensions=[".txt", ".text", ".log", ".dat", ".conf", ".cfg", ".ini", ".env"])
class GenericTextParser(BaseParser):
    """Universal parser for plain text, configuration files, and unspecialized documents."""

    def can_parse(self, document: Document) -> bool:
        """Can parse any document with text content or supported extension."""
        return True

    def supported_formats(self) -> list[str]:
        return [".txt", ".text", ".log", ".dat", ".conf", ".cfg", ".ini", ".env"]

    def supported_categories(self) -> list[str]:
        return ["Unknown", "Text", "General"]

    def driver_name(self) -> str:
        return "native_text_tokenizer"

    def _parse_document(self, document: Document, context: ParsingContext) -> ParsedDocument:
        content = document.content or ""
        profile = context.profile

        # 1. Sections
        sections = (
            ParserUtils.extract_sections(content, default_page=1)
            if context.extract_sections
            else []
        )

        # 2. Tables
        tables = (
            ParserUtils.parse_tables_from_text(content, page_number=1)
            if context.extract_tables
            else []
        )

        # 3. Equipment & Entity Graph
        equipment = (
            ParserUtils.extract_equipment(content, profile=profile, page_number=1)
            if context.extract_equipment
            else []
        )
        entity_graph = (
            ParserUtils.build_entity_graph(content, equipment)
            if context.extract_relationships
            else None
        )

        # 4. Warnings & Safety
        warnings = (
            ParserUtils.extract_safety_warnings(content, profile=profile, page_number=1)
            if context.extract_safety
            else []
        )

        # 5. Cross References
        cross_refs = (
            ParserUtils.extract_cross_references(content, page_number=1)
            if context.extract_cross_references
            else []
        )

        # 6. Operational Metadata
        op_meta = (
            ParserUtils.extract_operational_metadata(content, profile=profile)
            if context.extract_metadata
            else {}
        )

        parsed_meta = ParsedMetadata(
            title=document.metadata.file_name.rsplit(".", 1)[0].replace("_", " ").title(),
            category=document.metadata.category or "Text",
            revision_numbers=op_meta.get("revisions", []),
            engineer_names=op_meta.get("engineers", []),
            inspection_dates=op_meta.get("dates", []),
            standards_referenced=op_meta.get("standards", []),
            extra_metadata=dict(document.metadata.extra_metadata),
        )

        return ParsedDocument(
            document_id=f"parsed_{uuid.uuid4().hex[:12]}",
            raw_document_id=document.doc_id,
            title=parsed_meta.title,
            category=parsed_meta.category,
            sections=sections,
            tables=tables,
            equipment=equipment,
            entity_graph=entity_graph or None,
            warnings=warnings,
            cross_references=cross_refs,
            metadata=parsed_meta,
            statistics=DocumentStatistics(),
            processing_history=[
                {"stage": "GenericTextParser", "status": "COMPLETED", "driver": self.driver_name()}
            ],
        )
