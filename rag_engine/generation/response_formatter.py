"""Response Formatter — Verifiable Bibliographic Provenance Assembly.

Appends structured reference blocks to generated answers mapping [n] anchors
to document filenames, page numbers, section titles, and verbatim source quotes.
"""

from __future__ import annotations

from typing import Sequence

from rag_engine.retrieval.base_retriever import CitationBundle


class ResponseFormatter:
    """Formats final generated text and attaches auditable references."""

    @staticmethod
    def format_with_provenance(
        answer_text: str,
        citations: Sequence[CitationBundle],
        include_verbatim_quotes: bool = True,
    ) -> str:
        """Append a clean bibliographic references section to generated text."""
        cleaned_answer = answer_text.strip()
        if not citations:
            return cleaned_answer

        lines: list[str] = [cleaned_answer, "\n\n### References & Provenance"]

        for idx, bundle in enumerate(citations, start=1):
            anchor = f"[{idx}]"
            doc = bundle.document_id
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
