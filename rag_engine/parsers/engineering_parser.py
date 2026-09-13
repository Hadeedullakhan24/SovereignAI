"""Engineering manual and technical specification parser."""

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
)


@register_parser(extensions=[], categories=["manual", "engineering", "specification", "technical manual"])
class EngineeringParser(BaseParser):
    """Deep parser specialized for refinery engineering manuals and equipment data sheets."""

    DESIGN_LIMITS_PATTERN = re.compile(
        r"(?:Design\s+Pressure|Operating\s+Pressure|Design\s+Temperature|Operating\s+Temperature|"
        r"Flow\s+Rate|Design\s+Capacity|Head|Power\s+Rating)[\s.:-]+([^\n;]+)",
        re.IGNORECASE,
    )

    def can_parse(self, document: Document) -> bool:
        return True

    def supported_formats(self) -> list[str]:
        return []


    def supported_categories(self) -> list[str]:
        return ["manual", "engineering", "specification", "technical manual", "data sheet"]

    def driver_name(self) -> str:
        return "refinery_engineering_grammar_parser"

    def _parse_document(self, document: Document, context: ParsingContext) -> ParsedDocument:
        content = document.content or ""
        profile = context.profile

        sections = (
            ParserUtils.extract_sections(content, default_page=1)
            if context.extract_sections
            else []
        )
        tables = (
            ParserUtils.parse_tables_from_text(content, page_number=1)
            if context.extract_tables
            else []
        )
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

        # Extract specific design limit annotations
        design_limits = [
            m.group(0).strip() for m in self.DESIGN_LIMITS_PATTERN.finditer(content)
        ]

        title = document.metadata.file_name.rsplit(".", 1)[0].replace("_", " ").title()
        if sections:
            title = sections[0].title

        parsed_meta = ParsedMetadata(
            title=title,
            category="Manual",
            revision_numbers=op_meta.get("revisions", []),
            engineer_names=op_meta.get("engineers", []),
            inspection_dates=op_meta.get("dates", []),
            standards_referenced=op_meta.get("standards", []),
            extra_metadata={
                "design_limits_detected": len(design_limits),
                "plant_units": [u for u in profile.plant_units if u in content.upper()],
            },
        )

        return ParsedDocument(
            document_id=f"parsed_{hashlib.sha256(document.doc_id.encode('utf-8')).hexdigest()[:12]}",
            raw_document_id=document.doc_id,
            title=title,
            category="Manual",
            sections=sections,
            tables=tables,
            equipment=equipment,
            entity_graph=entity_graph,
            warnings=warnings,
            cross_references=cross_refs,
            metadata=parsed_meta,
            statistics=DocumentStatistics(),
            processing_history=[
                {"stage": "EngineeringParser", "status": "COMPLETED", "driver": self.driver_name()}
            ],
        )
