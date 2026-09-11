"""Safety manual, regulatory standard, and SOP parser."""

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
    DocumentStatistics,
    EntityGraph,
    ParsedDocument,
    ParsedMetadata,
    Section,
)


@register_parser(extensions=[], categories=["safety document", "safety", "safety standard", "sop", "oisd", "hse"])
class SafetyParser(BaseParser):
    """Deep parser specialized for refinery safety standards, OISD/PNGRB compliance, and SOPs."""

    PPE_PATTERN = re.compile(
        r"\b(?:PPE|Protective\s+Equipment|Helmet|Safety\s+Shoes|Safety\s+Goggles|Ear\s+Plugs|"
        r"SCBA|Breathing\s+Apparatus|Fire\s+Suit|Face\s+Shield)\b",
        re.IGNORECASE,
    )
    PERMIT_PATTERN = re.compile(
        r"\b(?:Cold\s+Work\s+Permit|Hot\s+Work\s+Permit|Confined\s+Space\s+Entry|Working\s+at\s+Height|"
        r"Electrical\s+Isolation|PTW)\b",
        re.IGNORECASE,
    )

    def can_parse(self, document: Document) -> bool:
        return True

    def supported_formats(self) -> list[str]:
        return []


    def supported_categories(self) -> list[str]:
        return ["safety document", "safety", "safety standard", "sop", "oisd", "hse", "hazard"]

    def driver_name(self) -> str:
        return "refinery_safety_compliance_parser"

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

        # Extract PPE and Work Permit requirements
        ppe_matches = sorted(list(set(m.group(0).title() for m in self.PPE_PATTERN.finditer(content))))
        permit_matches = sorted(list(set(m.group(0).title() for m in self.PERMIT_PATTERN.finditer(content))))

        title = document.metadata.file_name.rsplit(".", 1)[0].replace("_", " ").title()
        if sections:
            title = sections[0].title

        parsed_meta = ParsedMetadata(
            title=title,
            category="Safety Document",
            revision_numbers=op_meta.get("revisions", []),
            engineer_names=op_meta.get("engineers", []),
            inspection_dates=op_meta.get("dates", []),
            standards_referenced=op_meta.get("standards", []),
            extra_metadata={
                "ppe_required": ppe_matches,
                "work_permits_applicable": permit_matches,
                "safety_notices_count": len(warnings),
            },
        )

        doc_hash = hashlib.sha256(document.doc_id.encode("utf-8")).hexdigest()[:12]
        return ParsedDocument(
            document_id=f"parsed_{doc_hash}",
            raw_document_id=document.doc_id,
            title=title,
            category="Safety Document",
            sections=sections,
            tables=tables,
            equipment=equipment,
            entity_graph=entity_graph,
            warnings=warnings,
            cross_references=cross_refs,
            metadata=parsed_meta,
            statistics=DocumentStatistics(),
            processing_history=[
                {"stage": "SafetyParser", "status": "COMPLETED", "driver": self.driver_name()}
            ],
        )
