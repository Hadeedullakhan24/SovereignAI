"""Duplicate Removal & Content Deduplication.

Eliminates duplicate or heavily overlapping candidate chunks, duplicate citations,
and redundant tables or procedure lists from final retrieval sets.
"""

from __future__ import annotations

import difflib
import logging
from typing import Sequence

from rag_engine.retrieval.base_retriever import CitationBundle, ScoredRetrievalChunk

logger = logging.getLogger(__name__)


class DuplicateRemover:
    """Detects and prunes semantic and exact textual duplicates."""

    def __init__(self, content_similarity_threshold: float = 0.85) -> None:
        self.threshold = content_similarity_threshold

    def deduplicate_chunks(
        self, candidates: list[ScoredRetrievalChunk]
    ) -> list[ScoredRetrievalChunk]:
        """Prune duplicate chunks while retaining highest-scoring instances."""
        if not candidates:
            return []

        unique_candidates: list[ScoredRetrievalChunk] = []
        seen_chunk_ids: set[str] = set()
        seen_texts: list[str] = []

        for item in candidates:
            cid = item.chunk.chunk_id
            if cid in seen_chunk_ids:
                continue

            content = item.chunk.content.strip()
            if not content:
                continue

            # Check exact or high near-duplicate similarity
            is_dup = False
            for existing in seen_texts:
                if content == existing:
                    is_dup = True
                    break
                # Fast length heuristic before difflib
                len_ratio = len(content) / len(existing) if len(existing) > 0 else 0
                if 0.8 <= len_ratio <= 1.25:
                    ratio = difflib.SequenceMatcher(None, content, existing).quick_ratio()
                    if ratio >= self.threshold:
                        is_dup = True
                        break

            if not is_dup:
                seen_chunk_ids.add(cid)
                seen_texts.append(content)
                unique_candidates.append(item)

        # Re-index ranks
        for rank, item in enumerate(unique_candidates):
            item.rank = rank

        return unique_candidates

    def deduplicate_citations(
        self, citations: list[CitationBundle]
    ) -> list[CitationBundle]:
        """Prune duplicate citation anchors pointing to identical document coordinates."""
        unique_citations: list[CitationBundle] = []
        seen_coords: set[tuple[str, Optional[int], Optional[str]]] = set()

        for cit in citations:
            coord = (cit.document_id, cit.page_number, cit.section_title)
            if coord not in seen_coords:
                seen_coords.add(coord)
                unique_citations.append(cit)

        # Renumber citation IDs deterministically: [1], [2], ...
        renumbered: list[CitationBundle] = []
        for idx, cit in enumerate(unique_citations, start=1):
            renumbered.append(
                CitationBundle(
                    citation_id=f"[{idx}]",
                    document_id=cit.document_id,
                    document_name=cit.document_name,
                    source_path=cit.source_path,
                    page_number=cit.page_number,
                    section_title=cit.section_title,
                    chunk_id=cit.chunk_id,
                    verbatim_quote=cit.verbatim_quote,
                    score=cit.score,
                    equipment_tags=cit.equipment_tags,
                    safety_tags=cit.safety_tags,
                    revision=cit.revision,
                    sha256=cit.sha256,
                )
            )

        return renumbered
