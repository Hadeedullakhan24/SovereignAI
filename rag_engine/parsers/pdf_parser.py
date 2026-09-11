"""PDF document parser extracting page-aware sections, tables, and refinery entities."""

from __future__ import annotations

import hashlib
import re
from typing import Optional
import uuid

from rag_engine.parsers.base_parser import BaseParser
from rag_engine.parsers.parser_registry import register_parser
from rag_engine.parsers.parser_utils import ParserUtils
from rag_engine.parsers.parsing_context import ParsingContext
from rag_engine.schemas.document import Document
from rag_engine.schemas.parsed_document import (
    CitationCoordinates,
    CrossReference,
    DocumentStatistics,
    EntityGraph,
    EquipmentEntity,
    ParsedDocument,
    ParsedMetadata,
    SafetyWarning,
    Section,
    Table,
)


@register_parser(extensions=[".pdf"], categories=["manual", "report", "pdf"])
class PDFParser(BaseParser):
    """Deep parser for PDF documents with page structure preservation."""

    PAGE_SPLIT_PATTERN = re.compile(r"^---\s*Page\s*(\d+)\s*---", re.MULTILINE)

    def can_parse(self, document: Document) -> bool:
        ext = (document.metadata.file_format or "").lower()
        return ext == ".pdf" or bool(self.PAGE_SPLIT_PATTERN.search(document.content or ""))

    def supported_formats(self) -> list[str]:
        return [".pdf"]

    def supported_categories(self) -> list[str]:
        return ["manual", "report", "pdf", "specification"]

    def driver_name(self) -> str:
        return "native_page_pdf_parser"

    def _parse_document(self, document: Document, context: ParsingContext) -> ParsedDocument:
        content = document.content or ""
        profile = context.profile

        # Split content into pages if markers exist
        page_chunks = self._split_into_pages(content)

        all_sections: list[Section] = []
        all_tables: list[Table] = []
        all_equipment: list[EquipmentEntity] = []
        all_warnings: list[SafetyWarning] = []
        all_cross_refs: list[CrossReference] = []

        for page_num, page_text in page_chunks:
            if not page_text.strip():
                continue

            # Sections for this page
            if context.extract_sections:
                secs = ParserUtils.extract_sections(page_text, default_page=page_num)
                all_sections.extend(secs)

            # Tables for this page
            if context.extract_tables:
                tbls = ParserUtils.parse_tables_from_text(page_text, page_number=page_num)
                all_tables.extend(tbls)

            # Equipment entities for this page
            if context.extract_equipment:
                eqs = ParserUtils.extract_equipment(
                    page_text, profile=profile, page_number=page_num
                )
                all_equipment.extend(eqs)

            # Warnings for this page
            if context.extract_safety:
                warns = ParserUtils.extract_safety_warnings(
                    page_text, profile=profile, page_number=page_num
                )
                all_warnings.extend(warns)

            # Cross references for this page
            if context.extract_cross_references:
                refs = ParserUtils.extract_cross_references(page_text, page_number=page_num)
                all_cross_refs.extend(refs)

        # Entity graph
        entity_graph = (
            ParserUtils.build_entity_graph(content, all_equipment)
            if context.extract_relationships
            else EntityGraph()
        )

        # Operational metadata
        op_meta = (
            ParserUtils.extract_operational_metadata(content, profile=profile)
            if context.extract_metadata
            else {}
        )

        detected_title = document.metadata.file_name.rsplit(".", 1)[0].replace("_", " ").title()
        if all_sections:
            detected_title = all_sections[0].title

        parsed_meta = ParsedMetadata(
            title=detected_title,
            category=document.metadata.category or "Manual",
            revision_numbers=op_meta.get("revisions", []),
            engineer_names=op_meta.get("engineers", []),
            inspection_dates=op_meta.get("dates", []),
            standards_referenced=op_meta.get("standards", []),
            extra_metadata=dict(document.metadata.extra_metadata),
        )

        doc_hash = hashlib.sha256(document.doc_id.encode("utf-8")).hexdigest()[:12]
        return ParsedDocument(
            document_id=f"parsed_{doc_hash}",
            raw_document_id=document.doc_id,
            title=parsed_meta.title,
            category=parsed_meta.category,
            sections=all_sections,
            tables=all_tables,
            equipment=all_equipment,
            entity_graph=entity_graph,
            warnings=all_warnings,
            cross_references=all_cross_refs,
            metadata=parsed_meta,
            statistics=DocumentStatistics(total_pages=len(page_chunks)),
            processing_history=[
                {"stage": "PDFParser", "status": "COMPLETED", "driver": self.driver_name()}
            ],
        )

    def _split_into_pages(self, content: str) -> list[tuple[int, str]]:
        """Split text by '--- Page N ---' markers or treat as single page."""
        matches = list(self.PAGE_SPLIT_PATTERN.finditer(content))
        if not matches:
            return [(1, content)]

        pages: list[tuple[int, str]] = []
        for i in range(len(matches)):
            page_num = int(matches[i].group(1))
            start_pos = matches[i].end()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            page_text = content[start_pos:end_pos].strip()
            pages.append((page_num, page_text))

        return pages
