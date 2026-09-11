"""Table whitespace and structure normalization engine."""

from __future__ import annotations

import re
from rag_engine.schemas.parsed_document import Table, TableCell


class TableCleaner:
    """Stage 9: Cleans table whitespace while strictly preserving table grid structure and coordinates."""

    WHITESPACE_COLLAPSE = re.compile(r"[^\S\n\r]+")

    @classmethod
    def clean_table(cls, table: Table) -> Table:
        """Clean cell contents and rebuild normalized Markdown representation while preserving grid."""
        # 1. Clean headers
        cleaned_headers = [cls._clean_cell_str(h) for h in table.headers]

        # 2. Clean rows
        cleaned_rows: list[list[str]] = []
        for row in table.rows:
            cleaned_row = [cls._clean_cell_str(cell) for cell in row]
            # Ensure row length matches col_count or headers if possible
            cleaned_rows.append(cleaned_row)

        # 3. Clean coordinate-indexed cells (reuse unmodified cells to avoid Pydantic copy overhead)
        cleaned_cells: list[TableCell] = []
        for cell in table.cells:
            new_val = cls._clean_cell_str(cell.value)
            if new_val == cell.value:
                cleaned_cells.append(cell)
            else:
                cleaned_cells.append(cell.model_copy(update={"value": new_val}))

        # 4. Reconstruct DataFrame dictionary using fast C-level zip
        col_count = len(cleaned_headers) if cleaned_headers else (len(cleaned_rows[0]) if cleaned_rows else 0)
        df_dict: dict[str, list[str]] = {}
        if cleaned_rows and col_count > 0:
            cols = list(zip(*cleaned_rows))
            for col_idx in range(min(col_count, len(cols))):
                header_name = cleaned_headers[col_idx] if col_idx < len(cleaned_headers) else f"Column_{col_idx}"
                df_dict[header_name] = list(cols[col_idx])

        # 5. Build clean markdown table string for normalized_text
        markdown_lines: list[str] = []
        if cleaned_headers:
            markdown_lines.append("| " + " | ".join(cleaned_headers) + " |")
            markdown_lines.append("| " + " | ".join(["---"] * len(cleaned_headers)) + " |")
        # For huge tables (>500 rows), include top 500 rows in normalized_text to preserve memory/speed
        preview_rows = cleaned_rows[:500] if len(cleaned_rows) > 500 else cleaned_rows
        for r in preview_rows:
            markdown_lines.append("| " + " | ".join(r) + " |")
        if len(cleaned_rows) > 500:
            markdown_lines.append(f"| ... ({len(cleaned_rows) - 500} additional rows truncated for preview) |")

        normalized_repr = "\n".join(markdown_lines)

        return table.model_copy(
            update={
                "headers": cleaned_headers,
                "rows": cleaned_rows,
                "cells": cleaned_cells,
                "dataframe_dict": df_dict,
                "normalized_text": normalized_repr,
            }
        )

    @classmethod
    def clean_tables(cls, tables: list[Table]) -> list[Table]:
        """Clean all tables in a document."""
        return [cls.clean_table(t) for t in tables]

    @classmethod
    def _clean_cell_str(cls, text: str) -> str:
        """Collapse multiple spaces and clean cell string."""
        if not text:
            return ""
        # Replace line breaks inside cells with space or semicolon to keep tabular markdown valid
        cleaned = text.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
        cleaned = cls.WHITESPACE_COLLAPSE.sub(" ", cleaned).strip()
        # Escape pipe character inside markdown table cells
        cleaned = cleaned.replace("|", "\\|")
        return cleaned
