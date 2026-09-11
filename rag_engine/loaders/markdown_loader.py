"""
Markdown Loader — Markdown Documentation Loading & Structural Metadata Extraction.

Sovereign On-Premise Agentic AI Workbench (SIH26117 / MRPL)
Primary driver: Frontmatter + heading structure extractor
Fallback driver: Raw line reader (air-gapped)
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any
import yaml

from rag_engine.loaders.base_loader import BaseLoader
from rag_engine.loaders.loader_registry import register_loader


@register_loader(".md")
class MarkdownLoader(BaseLoader):
    """Universal loader for Markdown technical documentation, SOPs, and system manuals."""

    def supported_formats(self) -> list[str]:
        return [".md"]

    def is_primary_driver_available(self) -> bool:
        return True

    def dependencies(self) -> dict[str, bool]:
        return {"builtins": True}

    def _load_primary(self, file_path: Path) -> tuple[str, dict[str, Any]]:
        """Load markdown and extract structural metadata."""
        content = file_path.read_text(encoding="utf-8", errors="replace")

        # 1. Frontmatter extraction
        frontmatter_meta: dict[str, Any] = {}
        has_frontmatter = False
        body_content = content

        fm_match = re.match(r"^---\r?\n(.*?)\r?\n---\r?\n(.*)$", content, re.DOTALL)
        if fm_match:
            raw_fm = fm_match.group(1)
            body_content = fm_match.group(2)
            try:
                parsed = yaml.safe_load(raw_fm)
                if isinstance(parsed, dict):
                    frontmatter_meta = parsed
                    has_frontmatter = True
            except Exception:
                pass

        # 2. Structural elements count
        headings = re.findall(r"^(#{1,6})\s+(.+)$", body_content, re.MULTILINE)
        code_blocks = len(re.findall(r"^```", body_content, re.MULTILINE)) // 2

        meta: dict[str, Any] = {
            "has_frontmatter": has_frontmatter,
            "frontmatter": frontmatter_meta,
            "heading_count": len(headings),
            "code_block_count": code_blocks,
            "driver": "markdown_structural",
        }
        return body_content, meta

    def _load_fallback(self, file_path: Path) -> tuple[str, dict[str, Any]]:
        """Air-gapped fallback: raw read without structure analysis."""
        content = file_path.read_text(encoding="utf-8", errors="replace")
        return content, {"driver": "stdlib_text_fallback"}
