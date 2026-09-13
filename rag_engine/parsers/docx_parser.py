"""DOCX document parser extracting headings, XML tables, and refinery entities."""

from __future__ import annotations

import hashlib
import uuid
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
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
    TableCell,
)


@register_parser(extensions=[".docx", ".doc"], categories=["word", "word document", "docx"])
class DOCXParser(BaseParser):

    """Deep parser for Microsoft Word (.docx) documents."""

    W_NAMESPACE = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

    def can_parse(self, document: Document) -> bool:
        ext = (document.metadata.file_format or "").lower()
        return ext in [".docx", ".doc"]

    def supported_formats(self) -> list[str]:
        return [".docx", ".doc"]

    def supported_categories(self) -> list[str]:
        return ["manual", "report", "word", "procedure"]

    def driver_name(self) -> str:
        return "native_openxml_parser"

    def _parse_document(self, document: Document, context: ParsingContext) -> ParsedDocument:
        content = document.content or ""
        profile = context.profile
        source_path = Path(document.metadata.source_path)

        tables: list[Table] = []

        # Attempt high-fidelity XML table extraction if original docx zip archive is accessible
        if source_path.exists() and source_path.suffix.lower() == ".docx":
            try:
                tables = self._extract_tables_from_docx(source_path)
            except Exception:
                # Fallback to text table parsing
                tables = ParserUtils.parse_tables_from_text(content, page_number=1)
        else:
            tables = ParserUtils.parse_tables_from_text(content, page_number=1)

        # Extract hierarchical sections
        sections = (
            ParserUtils.extract_sections(content, default_page=1)
            if context.extract_sections
            else []
        )

        # Extract equipment
        equipment = (
            ParserUtils.extract_equipment(content, profile=profile, page_number=1)
            if context.extract_equipment
            else []
        )

        # Build entity graph
        entity_graph = (
            ParserUtils.build_entity_graph(content, equipment)
            if context.extract_relationships
            else EntityGraph()
        )

        # Extract warnings
        warnings = (
            ParserUtils.extract_safety_warnings(content, profile=profile, page_number=1)
            if context.extract_safety
            else []
        )

        # Extract cross references
        cross_refs = (
            ParserUtils.extract_cross_references(content, page_number=1)
            if context.extract_cross_references
            else []
        )

        # Extract operational metadata
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
            category=document.metadata.category or "Manual",
            revision_numbers=op_meta.get("revisions", []),
            engineer_names=op_meta.get("engineers", []),
            inspection_dates=op_meta.get("dates", []),
            standards_referenced=op_meta.get("standards", []),
            extra_metadata=dict(document.metadata.extra_metadata),
        )

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
            statistics=DocumentStatistics(),
            processing_history=[
                {"stage": "DOCXParser", "status": "COMPLETED", "driver": self.driver_name()}
            ],
        )

    def _extract_tables_from_docx(self, docx_path: Path) -> list[Table]:
        """Directly parse Word OpenXML tables to preserve exact cells."""
        tables: list[Table] = []
        with zipfile.ZipFile(docx_path, "r") as zf:
            if "word/document.xml" not in zf.namelist():
                return []
            xml_bytes = zf.read("word/document.xml")
            root = ET.fromstring(xml_bytes)

            tbl_elements = root.findall(f".//{self.W_NAMESPACE}tbl")
            for t_idx, tbl_elem in enumerate(tbl_elements):
                row_elements = tbl_elem.findall(f".//{self.W_NAMESPACE}tr")
                if not row_elements:
                    continue

                raw_rows: list[list[str]] = []
                for row_elem in row_elements:
                    cell_elements = row_elem.findall(f".//{self.W_NAMESPACE}tc")
                    row_vals: list[str] = []
                    for c_elem in cell_elements:
                        texts = [
                            t.text
                            for t in c_elem.findall(f".//{self.W_NAMESPACE}t")
                            if t.text
                        ]
                        row_vals.append(" ".join(texts).strip())
                    raw_rows.append(row_vals)

                if not raw_rows:
                    continue

                headers = raw_rows[0]
                data_rows = raw_rows[1:] if len(raw_rows) > 1 else []
                col_count = len(headers)

                cells: list[TableCell] = []
                for c_idx, h in enumerate(headers):
                    cells.append(
                        TableCell(row_idx=0, col_idx=c_idx, value=h, raw_value=h, is_header=True)
                    )
                for r_idx, r in enumerate(data_rows):
                    for c_idx, val in enumerate(r):
                        cells.append(
                            TableCell(row_idx=r_idx + 1, col_idx=c_idx, value=val, raw_value=val, is_header=False)
                        )

                table_id = f"tbl_docx_{t_idx + 1}"
                dataframe_dict = {
                    headers[c]: [row[c] for row in data_rows if c < len(row)]
                    for c in range(col_count)
                }

                tables.append(
                    Table(
                        table_id=table_id,
                        caption=f"Word Table {t_idx + 1}",
                        headers=headers,
                        rows=data_rows,
                        row_count=len(data_rows),
                        col_count=col_count,
                        page_number=1,
                        dataframe_dict=dataframe_dict,
                        cells=cells,
                        confidence=0.99,
                    )
                )

        return tables
