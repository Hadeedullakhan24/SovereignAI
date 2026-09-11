"""Citation & Provenance Builder.

Constructs auditable CitationBundle objects from retrieved candidate chunks, linking
verbatim quotes directly to source documents, page numbers, section headers, and equipment tags.
"""

from __future__ import annotations

from typing import Sequence

from rag_engine.retrieval.base_retriever import CitationBundle, ScoredRetrievalChunk


class CitationBuilder:
    """Enterprise citation builder establishing 100% auditable source provenance."""

    def __init__(self, max_quote_length: int = 300) -> None:
        self.max_quote_length = max_quote_length

    def build_citations(
        self, candidates: Sequence[ScoredRetrievalChunk]
    ) -> list[CitationBundle]:
        """Convert scored candidate chunks into sequential CitationBundles."""
        citations: list[CitationBundle] = []

        for idx, item in enumerate(candidates, start=1):
            chk = item.chunk
            meta = chk.metadata

            # Extract clean verbatim quote
            quote = chk.content.strip()
            if len(quote) > self.max_quote_length:
                quote = quote[: self.max_quote_length].rstrip() + "..."

            cit = CitationBundle(
                citation_id=f"[{idx}]",
                document_id=meta.document_id,
                document_name=meta.document_name,
                source_path=meta.source_path,
                page_number=meta.page_number,
                section_title=meta.section_title,
                chunk_id=chk.chunk_id,
                verbatim_quote=quote,
                score=round(item.score, 4),
                equipment_tags=list(meta.equipment_entities or []),
                safety_tags=list(meta.safety_entities or []),
                revision=getattr(meta, "revision", None),
                sha256=meta.sha256,
            )
            citations.append(cit)

        return citations
