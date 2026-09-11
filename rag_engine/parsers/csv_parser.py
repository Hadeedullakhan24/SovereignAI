"""CSV and spreadsheet parser converting tabular data into rich structured Tables."""

from __future__ import annotations

import csv
import io
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
    TableCell,
)


@register_parser(extensions=[".csv", ".tsv"], categories=["tabular", "data", "sheet"])
class CSVParser(BaseParser):
    """Structured parser for comma-separated and tab-separated tabular data."""

    def can_parse(self, document: Document) -> bool:
        ext = (document.metadata.file_format or "").lower()
        return ext in [".csv", ".tsv"]

    def supported_formats(self) -> list[str]:
        return [".csv", ".tsv"]

    def supported_categories(self) -> list[str]:
        return ["tabular", "data", "sheet", "record"]

    def driver_name(self) -> str:
        return "native_csv_parser"

    def _parse_document(self, document: Document, context: ParsingContext) -> ParsedDocument:
        content = document.content or ""
        profile = context.profile
        delimiter = "\t" if (document.metadata.file_format or "").lower() == ".tsv" else ","

        tables: list[Table] = []
        sections: list[Section] = []

        if content.strip():
            reader = csv.reader(io.StringIO(content), delimiter=delimiter)
            rows = [row for row in reader if any(cell.strip() for cell in row)]

            if rows:
                headers = [h.strip() for h in rows[0]]
                data_rows = rows[1:] if len(rows) > 1 else []
                col_count = len(headers)

                cells: list[TableCell] = []
                for c_idx, h in enumerate(headers):
                    cells.append(
                        TableCell(row_idx=0, col_idx=c_idx, value=h, raw_value=h, is_header=True)
                    )
                for r_idx, r in enumerate(data_rows):
                    for c_idx, val in enumerate(r):
                        if c_idx < col_count:
                            cells.append(
                                TableCell(
                                    row_idx=r_idx + 1,
                                    col_idx=c_idx,
                                    value=val.strip(),
                                    raw_value=val,
                                    is_header=False,
                                )
                            )

                dataframe_dict = {
                    headers[c]: [row[c] if c < len(row) else "" for row in data_rows]
                    for c in range(col_count)
                }

                tbl = Table(
                    table_id=f"tbl_csv_{uuid.uuid4().hex[:8]}",
                    caption=f"Tabular Dataset: {document.metadata.file_name}",
                    headers=headers,
                    rows=data_rows,
                    row_count=len(data_rows),
                    col_count=col_count,
                    page_number=1,
                    dataframe_dict=dataframe_dict,
                    raw_text=content[:2000],
                    cells=cells,
                    confidence=1.0,
                )
                tables.append(tbl)

                # Create section representation
                sections.append(
                    Section(
                        section_id=f"sec_csv_{uuid.uuid4().hex[:8]}",
                        title=document.metadata.file_name,
                        level=1,
                        content=f"Table containing {len(data_rows)} rows and {col_count} columns with headers: {', '.join(headers)}",
                        paragraphs=[f"Headers: {', '.join(headers)}"],
                        page_number=1,
                        confidence=1.0,
                    )
                )

        # Extract equipment from CSV cell contents
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
        parsed_meta = ParsedMetadata(
            title=title,
            category=document.metadata.category or "Tabular Data",
            revision_numbers=op_meta.get("revisions", []),
            engineer_names=op_meta.get("engineers", []),
            inspection_dates=op_meta.get("dates", []),
            standards_referenced=op_meta.get("standards", []),
            extra_metadata=dict(document.metadata.extra_metadata),
        )

        return ParsedDocument(
            document_id=f"parsed_{uuid.uuid4().hex[:12]}",
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
                {"stage": "CSVParser", "status": "COMPLETED", "driver": self.driver_name()}
            ],
        )
