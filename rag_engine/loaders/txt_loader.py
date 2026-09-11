"""
TXT Loader — Plain Text Document Loading with Robust Multi-Encoding Support.

Sovereign On-Premise Agentic AI Workbench (SIH26117 / MRPL)
Primary driver: UTF-8 reader with structure normalization
Fallback driver: Multi-encoding decode (Latin-1 / CP1252 / errors=replace)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rag_engine.loaders.base_loader import BaseLoader
from rag_engine.loaders.loader_registry import register_loader


@register_loader(".txt")
class TXTLoader(BaseLoader):
    """Universal loader for plain text documents, logs, and configuration dumps."""

    def supported_formats(self) -> list[str]:
        return [".txt"]

    def is_primary_driver_available(self) -> bool:
        return True

    def dependencies(self) -> dict[str, bool]:
        return {"builtins": True}

    def _load_primary(self, file_path: Path) -> tuple[str, dict[str, Any]]:
        """Load text using standard UTF-8 decoder."""
        with open(file_path, "r", encoding="utf-8") as fh:
            content = fh.read()

        lines = content.splitlines()
        blank_lines = sum(1 for line in lines if not line.strip())

        meta: dict[str, Any] = {
            "line_count": len(lines),
            "blank_lines_count": blank_lines,
            "encoding": "utf-8",
            "driver": "utf8_reader",
        }
        return content, meta

    def _load_fallback(self, file_path: Path) -> tuple[str, dict[str, Any]]:
        """Fallback for non-UTF8 text files (Latin-1 / CP1252 / replace)."""
        encodings = ["latin-1", "cp1252", "ascii"]
        raw_bytes = file_path.read_bytes()

        for enc in encodings:
            try:
                content = raw_bytes.decode(enc)
                lines = content.splitlines()
                return content, {
                    "line_count": len(lines),
                    "encoding": enc,
                    "driver": "fallback_multi_encoding",
                }
            except UnicodeDecodeError:
                continue

        content = raw_bytes.decode("utf-8", errors="replace")
        lines = content.splitlines()
        return content, {
            "line_count": len(lines),
            "encoding": "utf-8-replace",
            "driver": "fallback_replace_encoding",
        }
