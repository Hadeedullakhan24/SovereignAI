"""Markdown document parser extracting H1-H6 headings, tables, and equipment entities."""

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
    Table,
)


@register_parser(extensions=[".md", ".markdown"], categories=["documentation", "notes", "manual"])
class MarkdownParser(BaseParser):
    """Deep parser for CommonMark and GitHub Flavored Markdown documents."""

    FRONTMATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)

    def can_parse(self, document: Document) -> bool:
        ext = (document.metadata.file_format or "").lower()
        return ext in [".md", ".markdown"]

    def supported_formats(self) -> list[str]:
        return [".md", ".markdown"]

    def supported_categories(self) -> list[str]:
        return ["documentation", "notes", "manual", "standards"]

    def driver_name(self) -> str:
        return "native_gfm_parser"

    def _parse_document(self, document: Document, context: ParsingContext) -> ParsedDocument:
        content = document.content or ""
        profile = context.profile
        extra_meta = dict(document.metadata.extra_metadata)

        # Parse YAML frontmatter if present
        body_text = content
        fm_match = self.FRONTMATTER_PATTERN.match(content)
        if fm_match:
            frontmatter_raw = fm_match.group(1)
            body_text = content[fm_match.end():]
            for line in frontmatter_raw.split("\n"):
                if ":" in line:
                    k, v = line.split(":", 1)
                    extra_meta[k.strip()] = v.strip().strip('"\'')

        sections = (
            ParserUtils.extract_sections(body_text, default_page=1)
            if context.extract_sections
            else []
        )
        tables = (
            ParserUtils.parse_tables_from_text(body_text, page_number=1)
            if context.extract_tables
            else []
        )
        equipment = (
            ParserUtils.extract_equipment(body_text, profile=profile, page_number=1)
            if context.extract_equipment
            else []
        )
        entity_graph = (
            ParserUtils.build_entity_graph(body_text, equipment)
            if context.extract_relationships
            else EntityGraph()
        )
        warnings = (
            ParserUtils.extract_safety_warnings(body_text, profile=profile, page_number=1)
            if context.extract_safety
            else []
        )
        cross_refs = (
            ParserUtils.extract_cross_references(body_text, page_number=1)
            if context.extract_cross_references
            else []
        )
        op_meta = (
            ParserUtils.extract_operational_metadata(body_text, profile=profile)
            if context.extract_metadata
            else {}
        )

        title = extra_meta.get("title") or document.metadata.file_name.rsplit(".", 1)[0].replace("_", " ").title()
        if sections and not extra_meta.get("title"):
            title = sections[0].title

        parsed_meta = ParsedMetadata(
            title=title,
            author=extra_meta.get("author"),
            category=document.metadata.category or "Documentation",
            revision_numbers=op_meta.get("revisions", []),
            engineer_names=op_meta.get("engineers", []),
            inspection_dates=op_meta.get("dates", []),
            standards_referenced=op_meta.get("standards", []),
            extra_metadata=extra_meta,
        )

        doc_hash = hashlib.sha256(document.doc_id.encode("utf-8")).hexdigest()[:12]
        return ParsedDocument(
            document_id=f"parsed_{doc_hash}",
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
            statistics=DocumentStatistics(),
            processing_history=[
                {"stage": "MarkdownParser", "status": "COMPLETED", "driver": self.driver_name()}
            ],
        )
