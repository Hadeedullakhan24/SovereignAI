"""
CSV Loader — Tabular Delimited Data Loading with Dialect & Encoding Detection.

Sovereign On-Premise Agentic AI Workbench (SIH26117 / MRPL)
Primary driver: Standard library csv.reader with csv.Sniffer dialect detection
Fallback driver: Multi-encoding raw line delimiter splitter (air-gapped)
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

from rag_engine.loaders.base_loader import BaseLoader
from rag_engine.loaders.loader_registry import register_loader


@register_loader(".csv")
class CSVLoader(BaseLoader):
    """Universal loader for CSV and delimited tabular text files."""

    def supported_formats(self) -> list[str]:
        return [".csv"]

    def is_primary_driver_available(self) -> bool:
        return True  # Built into standard library

    def dependencies(self) -> dict[str, bool]:
        return {"csv": True}

    def _read_with_encoding(self, file_path: Path) -> tuple[str, str]:
        """Attempt reading with common encodings."""
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
        for enc in encodings:
            try:
                with open(file_path, "r", encoding=enc) as fh:
                    return fh.read(), enc
            except (UnicodeDecodeError, LookupError):
                continue
        with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read(), "utf-8-replace"

    def _load_primary(self, file_path: Path) -> tuple[str, dict[str, Any]]:
        """Load CSV with sniffer dialect detection."""
        raw_text, encoding = self._read_with_encoding(file_path)

        # Detect delimiter using sniffer if possible
        sample = raw_text[:4096]
        delimiter = ","
        has_header = False
        try:
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(sample)
            delimiter = dialect.delimiter
            has_header = sniffer.has_header(sample)
        except Exception:
            # Default to comma
            delimiter = ","

        reader = csv.reader(io.StringIO(raw_text), delimiter=delimiter)
        rows_text: list[str] = []
        row_count = 0
        max_cols = 0

        for row in reader:
            clean_cells = [c.strip() for c in row if c.strip()]
            if clean_cells:
                rows_text.append(" | ".join(clean_cells))
                row_count += 1
                max_cols = max(max_cols, len(clean_cells))

        full_content = "\n".join(rows_text)
        meta: dict[str, Any] = {
            "row_count": row_count,
            "column_count": max_cols,
            "delimiter": delimiter,
            "has_header": has_header,
            "encoding": encoding,
            "driver": "csv_sniffer",
        }
        return full_content, meta

    def _load_fallback(self, file_path: Path) -> tuple[str, dict[str, Any]]:
        """Air-gapped fallback: simple line-by-line split."""
        raw_text, encoding = self._read_with_encoding(file_path)
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

        meta: dict[str, Any] = {
            "row_count": len(lines),
            "encoding": encoding,
            "driver": "stdlib_line_fallback",
        }
        return "\n".join(lines), meta
