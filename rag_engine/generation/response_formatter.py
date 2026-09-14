"""Response Formatter — Verifiable Bibliographic Provenance Assembly.

Provides structured formatting utilities for generated answers and auditable
provenance sections mapping [n] anchors to document filenames, page numbers,
section titles, equipment tags, and verbatim source quotes.
"""

from __future__ import annotations

import re
from typing import Sequence

from rag_engine.retrieval.base_retriever import CitationBundle


class ResponseFormatter:
    """Formats final generated text and provides auditable source sections."""

    @staticmethod
    def format_sources_section(
        citations: Sequence[CitationBundle],
        include_verbatim_quotes: bool = True,
    ) -> str:
        """Format clean, structured SOURCES / EVIDENCE section from citation bundles."""
        if not citations:
            return "  (No citations referenced or required)"

        lines: list[str] = []
        for idx, bundle in enumerate(citations, start=1):
            anchor = getattr(bundle, "citation_id", f"[{idx}]")
            doc = getattr(bundle, "document_name", None) or getattr(bundle, "document_id", "Document")
            page = f"Page {bundle.page_number}" if bundle.page_number else None
            section = f"Section: {bundle.section_title}" if bundle.section_title else None
            tag = f"Tag: {bundle.equipment_tag}" if bundle.equipment_tag else None

            meta_parts = [p for p in [page, section, tag] if p]
            meta_str = f" ({' | '.join(meta_parts)})" if meta_parts else ""

            lines.append(f"{anchor} {doc}{meta_str}")
            if include_verbatim_quotes and bundle.verbatim_quote:
                quote = bundle.verbatim_quote.strip().replace("\n", " ")
                if len(quote) > 200:
                    quote = quote[:197] + "..."
                lines.append(f'    "{quote}"')

        return "\n".join(lines)

    @staticmethod
    def format_concise_sources(
        citations: Sequence[CitationBundle],
        query: str = "",
        answer: str = "",
        max_sources: int = 5,
    ) -> str:
        """Format clean, human-readable sources list (e.g., '### Sources\\n[1] Approval Note — Page 1')."""
        if not citations:
            return ""

        # Check if citations [1], [2] are referenced in the answer text
        referenced_ids = set(re.findall(r"\[(\d+)\]", answer))

        selected: list[CitationBundle] = []
        if referenced_ids:
            for c in citations:
                cit_id = getattr(c, "citation_id", "").strip("[]")
                if cit_id in referenced_ids:
                    selected.append(c)

        if not selected:
            q_lower = query.lower() if query else ""
            for c in citations:
                doc = (getattr(c, "document_name", None) or getattr(c, "document_id", "")).lower()
                if "approval note" in q_lower and "approval" not in doc:
                    continue
                selected.append(c)

        if not selected:
            selected = list(citations[:max_sources])
        else:
            selected = selected[:max_sources]

        lines: list[str] = ["### Sources"]
        for idx, bundle in enumerate(selected, start=1):
            anchor = getattr(bundle, "citation_id", f"[{idx}]")
            if not anchor.startswith("["):
                anchor = f"[{anchor}]"
            raw_doc = getattr(bundle, "document_name", None) or getattr(bundle, "document_id", "Document")

            # Clean document title
            clean_doc = raw_doc
            if clean_doc.endswith((".pdf", ".md", ".docx", ".txt")):
                clean_doc = clean_doc.rsplit(".", 1)[0].replace("_", " ").title()
            elif clean_doc.isupper():
                clean_doc = clean_doc.title()

            page = bundle.page_number if getattr(bundle, "page_number", None) else 1
            lines.append(f"{anchor} {clean_doc} — Page {page}")

        return "\n".join(lines)

    @staticmethod
    def strip_provenance(text: str) -> str:
        """Strip existing reference or provenance headers from raw answer text."""
        cleaned = re.split(
            r"\n+(?:###?\s*(?:References|Provenance|Sources|Supporting Citations)|(?:References|Sources|Supporting Citations):)",
            text,
            flags=re.IGNORECASE,
        )[0]
        return cleaned.strip()

    @staticmethod
    def format_with_provenance(
        answer_text: str,
        citations: Sequence[CitationBundle],
        include_verbatim_quotes: bool = True,
    ) -> str:
        """Append a clean bibliographic references section to generated text."""
        cleaned_answer = ResponseFormatter.strip_provenance(answer_text)
        if not citations:
            return cleaned_answer

        lines: list[str] = [cleaned_answer, "\n\n### References & Provenance"]

        for idx, bundle in enumerate(citations, start=1):
            anchor = getattr(bundle, "citation_id", f"[{idx}]")
            doc = getattr(bundle, "document_name", None) or getattr(bundle, "document_id", "Document")
            page = f"Page {bundle.page_number}" if bundle.page_number else "Page N/A"
            section = f"Section: {bundle.section_title}" if bundle.section_title else ""
            tag = f"Tag: {bundle.equipment_tag}" if bundle.equipment_tag else ""

            meta_parts = [p for p in [doc, page, section, tag] if p]
            meta_str = " | ".join(meta_parts)

            lines.append(f"- **{anchor}** {meta_str}")
            if include_verbatim_quotes and bundle.verbatim_quote:
                quote = bundle.verbatim_quote.strip().replace("\n", " ")
                if len(quote) > 160:
                    quote = quote[:157] + "..."
                lines.append(f'  > "{quote}"')

        return "\n".join(lines)
