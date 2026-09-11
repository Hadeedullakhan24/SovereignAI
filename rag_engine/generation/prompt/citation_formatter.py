"""Citation Formatter for In-Prompt Anchors and Bibliographic Tables."""

from __future__ import annotations

from typing import Any, Sequence

from rag_engine.interfaces.base_prompt import BaseCitationFormatter


class CitationFormatter(BaseCitationFormatter):
    """Renders citation evidence and anchors into verified bibliographic formats."""

    def __init__(self, default_style: str = "inline_anchors") -> None:
        self.default_style = default_style

    def format_citations(
        self,
        citations: Sequence[Any],
        format_style: str = "inline_anchors",
    ) -> str:
        """Format citation bundles into designated style."""
        if not citations:
            return ""

        style = format_style or self.default_style

        if style == "tabular":
            lines = [
                "| Ref | Document ID | Section / Page | Equipment | Verbatim Excerpt |",
                "|---|---|---|---|---|",
            ]
            for idx, c in enumerate(citations, start=1):
                doc_id = getattr(c, "document_id", "N/A")
                sec = getattr(c, "section_title", f"Page {getattr(c, 'page_number', 'N/A')}")
                
                # Support both equipment_tag and equipment_tags
                eq_tag = getattr(c, "equipment_tag", None)
                eq_tags = getattr(c, "equipment_tags", None)
                if eq_tag:
                    eq = str(eq_tag)
                elif eq_tags and isinstance(eq_tags, (list, tuple)):
                    eq = ", ".join(eq_tags)
                else:
                    eq = "N/A"

                quote = getattr(c, "verbatim_quote", "")[:60].replace("\n", " ")
                lines.append(f"| [{idx}] | {doc_id} | {sec} | {eq} | {quote}... |")
            return "\n".join(lines)

        elif style == "footnotes":
            lines = []
            for idx, c in enumerate(citations, start=1):
                doc_id = getattr(c, "document_id", "Document")
                sec = getattr(c, "section_title", "")
                page = getattr(c, "page_number", None)
                loc = f", Page {page}" if page else ""
                lines.append(f"[^{idx}]: {doc_id} — {sec}{loc}")
            return "\n".join(lines)

        else:  # inline_anchors
            lines = []
            for idx, c in enumerate(citations, start=1):
                doc_id = getattr(c, "document_id", "N/A")
                sec = getattr(c, "section_title", "General")
                lines.append(f"[{idx}] {doc_id} ({sec})")
            return "\n".join(lines)
