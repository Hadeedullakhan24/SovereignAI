"""Email and operational correspondence parser."""

from __future__ import annotations

import email
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


@register_parser(extensions=[".eml", ".msg"], categories=["email", "correspondence", "memo"])
class EmailParser(BaseParser):
    """Deterministic parser for operational emails and memoranda."""

    HEADER_PATTERN = re.compile(r"^(From|To|Subject|Date|Cc|Bcc):\s*(.*)$", re.IGNORECASE | re.MULTILINE)
    ACTION_PATTERN = re.compile(
        r"(?:Action(?:\s*Item)?|Required\s*Action|Please\s*ensure|Next\s*steps?)[\s.:-]+([^\n]+)",
        re.IGNORECASE,
    )

    def can_parse(self, document: Document) -> bool:
        ext = (document.metadata.file_format or "").lower()
        return ext in [".eml", ".msg"] or "email" in (document.metadata.category or "").lower()

    def supported_formats(self) -> list[str]:
        return [".eml", ".msg"]

    def supported_categories(self) -> list[str]:
        return ["email", "correspondence", "memo", "communication"]

    def driver_name(self) -> str:
        return "native_email_parser"

    def _parse_document(self, document: Document, context: ParsingContext) -> ParsedDocument:
        content = document.content or ""
        profile = context.profile

        # Parse standard RFC 822 email headers
        headers: dict[str, str] = {}
        body = content

        for match in self.HEADER_PATTERN.finditer(content):
            headers[match.group(1).title()] = match.group(2).strip()

        # Separate headers block from body
        parts = re.split(r"\n\s*\n", content, maxsplit=1)
        if len(parts) > 1 and headers:
            body = parts[1]

        subject = headers.get("Subject") or document.metadata.file_name
        sender = headers.get("From") or "Unknown Sender"
        recipient = headers.get("To") or "Unknown Recipient"
        sent_date = headers.get("Date") or ""

        # Extract action items
        action_items = [
            m.group(1).strip() for m in self.ACTION_PATTERN.finditer(body)
        ]

        sections: list[Section] = [
            Section(
                section_id=f"sec_email_hdr_{uuid.uuid4().hex[:8]}",
                title=f"Email: {subject}",
                level=1,
                content=f"From: {sender}\nTo: {recipient}\nDate: {sent_date}\nSubject: {subject}",
                paragraphs=[
                    f"Sender: {sender}",
                    f"Recipient: {recipient}",
                    f"Date: {sent_date}",
                    f"Subject: {subject}",
                ],
                page_number=1,
                confidence=1.0,
            ),
        ]

        # Extract body sections
        body_sections = ParserUtils.extract_sections(body, default_page=1)
        sections.extend(body_sections)

        # If action items detected, add an Action Items section
        if action_items:
            sections.append(
                Section(
                    section_id=f"sec_email_act_{uuid.uuid4().hex[:8]}",
                    title="Action Items & Tasks",
                    level=2,
                    content="\n".join(f"- {act}" for act in action_items),
                    bullet_points=action_items,
                    page_number=1,
                    confidence=0.95,
                )
            )

        tables = (
            ParserUtils.parse_tables_from_text(body, page_number=1)
            if context.extract_tables
            else []
        )
        equipment = (
            ParserUtils.extract_equipment(body, profile=profile, page_number=1)
            if context.extract_equipment
            else []
        )
        entity_graph = (
            ParserUtils.build_entity_graph(body, equipment)
            if context.extract_relationships
            else EntityGraph()
        )
        warnings = (
            ParserUtils.extract_safety_warnings(body, profile=profile, page_number=1)
            if context.extract_safety
            else []
        )
        cross_refs = (
            ParserUtils.extract_cross_references(body, page_number=1)
            if context.extract_cross_references
            else []
        )
        op_meta = (
            ParserUtils.extract_operational_metadata(body, profile=profile)
            if context.extract_metadata
            else {}
        )

        all_dates = list(set(([sent_date] if sent_date else []) + op_meta.get("dates", [])))

        parsed_meta = ParsedMetadata(
            title=subject,
            author=sender,
            category="Email",
            revision_numbers=op_meta.get("revisions", []),
            engineer_names=op_meta.get("engineers", []),
            inspection_dates=all_dates,
            standards_referenced=op_meta.get("standards", []),
            extra_metadata={
                "from": sender,
                "to": recipient,
                "date": sent_date,
                "subject": subject,
                "action_items_count": len(action_items),
            },
        )

        return ParsedDocument(
            document_id=f"parsed_{hashlib.sha256(document.doc_id.encode('utf-8')).hexdigest()[:12]}",
            raw_document_id=document.doc_id,
            title=subject,
            category="Email",
            sections=sections,
            tables=tables,
            equipment=equipment,
            entity_graph=entity_graph,
            warnings=warnings,
            cross_references=cross_refs,
            metadata=parsed_meta,
            statistics=DocumentStatistics(),
            processing_history=[
                {"stage": "EmailParser", "status": "COMPLETED", "driver": self.driver_name()}
            ],
        )
