"""PowerPoint presentation parser extracting slides, tables, and equipment mentions."""

from __future__ import annotations

import hashlib
import re
import uuid
from typing import Optional

from rag_engine.parsers.base_parser import BaseParser
from rag_engine.parsers.parser_registry import register_parser
from rag_engine.parsers.parser_utils import ParserUtils
from rag_engine.parsers.parsing_context import ParsingContext
from rag_engine.schemas.document import Document
from rag_engine.schemas.parsed_document import (
    DocumentStatistics,
    EntityGraph,
    ParsedDocument,
    ParsedMetadata,
    Section,
    Table,
)


@register_parser(extensions=[".pptx", ".ppt"], categories=["presentation", "training"])
class PPTXParser(BaseParser):
    """Deep parser for PowerPoint (.pptx) presentation decks."""

    SLIDE_PATTERN = re.compile(r"^---\s*Slide\s*(\d+)\s*---", re.MULTILINE)

    def can_parse(self, document: Document) -> bool:
        ext = (document.metadata.file_format or "").lower()
        return ext in [".pptx", ".ppt"] or bool(self.SLIDE_PATTERN.search(document.content or ""))

    def supported_formats(self) -> list[str]:
        return [".pptx", ".ppt"]

    def supported_categories(self) -> list[str]:
        return ["presentation", "training", "briefing"]

    def driver_name(self) -> str:
        return "native_pptx_slide_parser"

    def _parse_document(self, document: Document, context: ParsingContext) -> ParsedDocument:
        content = document.content or ""
        profile = context.profile

        slide_matches = list(self.SLIDE_PATTERN.finditer(content))
        sections: list[Section] = []
        tables: list[Table] = []

        if slide_matches:
            for i in range(len(slide_matches)):
                slide_num = int(slide_matches[i].group(1))
                start_pos = slide_matches[i].end()
                end_pos = slide_matches[i + 1].start() if i + 1 < len(slide_matches) else len(content)
                slide_body = content[start_pos:end_pos].strip()

                lines = [line.strip() for line in slide_body.split("\n") if line.strip()]
                slide_title = lines[0] if lines else f"Slide {slide_num}"
                bullet_points = [
                    l.lstrip("-*•– ").strip()
                    for l in lines[1:]
                    if l.startswith(("-", "*", "•", "–"))
                ]

                sec_id = f"sec_slide_{slide_num}"
                sections.append(
                    Section(
                        section_id=sec_id,
                        title=f"Slide {slide_num}: {slide_title}",
                        level=2,
                        content=slide_body,
                        raw_text=slide_body,
                        normalized_text=ParserUtils.normalize_text(slide_body),
                        paragraphs=lines[1:],
                        bullet_points=bullet_points,
                        page_number=slide_num,
                        confidence=0.96,
                    )
                )
                # Check for tables within slide
                slide_tables = ParserUtils.parse_tables_from_text(slide_body, page_number=slide_num)
                tables.extend(slide_tables)
        else:
            sections = ParserUtils.extract_sections(content, default_page=1)
            tables = ParserUtils.parse_tables_from_text(content, page_number=1)

        # Equipment extraction
        equipment = (
            ParserUtils.extract_equipment(content, profile=profile, page_number=1)
            if context.extract_equipment
            else []
        )
        entity_graph = (
            ParserUtils.build_entity_graph(content, equipment)
            if context.extract_relationships
            else EntityGraph()
        )
        warnings = (
            ParserUtils.extract_safety_warnings(content, profile=profile, page_number=1)
            if context.extract_safety
            else []
        )
        cross_refs = (
            ParserUtils.extract_cross_references(content, page_number=1)
            if context.extract_cross_references
            else []
        )
        op_meta = (
            ParserUtils.extract_operational_metadata(content, profile=profile)
            if context.extract_metadata
            else {}
        )

        title = document.metadata.file_name.rsplit(".", 1)[0].replace("_", " ").title()
        if sections:
            title = sections[0].title

        parsed_meta = ParsedMetadata(
            title=title,
            category=document.metadata.category or "Presentation",
            revision_numbers=op_meta.get("revisions", []),
            engineer_names=op_meta.get("engineers", []),
            inspection_dates=op_meta.get("dates", []),
            standards_referenced=op_meta.get("standards", []),
            extra_metadata=dict(document.metadata.extra_metadata),
        )

        total_slides = len(slide_matches) if slide_matches else 1
        return ParsedDocument(
            document_id=f"parsed_{hashlib.sha256(document.doc_id.encode('utf-8')).hexdigest()[:12]}",
            raw_document_id=document.doc_id,
            title=title,
            category=parsed_meta.category,
            sections=sections,
            tables=tables,
            equipment=equipment,
            entity_graph=entity_graph,
            warnings=warnings,
            cross_references=cross_refs,
            metadata=parsed_meta,
            statistics=DocumentStatistics(total_pages=total_slides),
            processing_history=[
                {"stage": "PPTXParser", "status": "COMPLETED", "driver": self.driver_name()}
            ],
        )
