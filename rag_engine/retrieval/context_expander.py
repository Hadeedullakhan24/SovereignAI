"""Context Expansion Engine.

Expands top-ranked retrieval candidates by traversing relational ChunkHierarchy pointers
(prev_chunk_id, next_chunk_id, parent_section_id) via VectorRepository to supply
surrounding context for downstream LLM reasoning without duplicates.
"""

from __future__ import annotations

import logging
from typing import Optional

from rag_engine.retrieval.base_retriever import ScoredRetrievalChunk
from rag_engine.schemas.chunk import Chunk, ChunkHierarchy
from rag_engine.vector_store.vector_repository import (
    VectorRepository,
    get_vector_repository,
)

logger = logging.getLogger(__name__)


class ContextExpander:
    """Relational neighbor expander navigating chunk hierarchies."""

    def __init__(self, repository: Optional[VectorRepository] = None) -> None:
        self.repository = repository or get_vector_repository()

    def expand(
        self,
        candidates: list[ScoredRetrievalChunk],
        radius: int = 1,
        collection_name: Optional[str] = None,
    ) -> list[ScoredRetrievalChunk]:
        """Expand retrieval candidates with adjacent sequential chunks within radius.
        
        Args:
            candidates: Top-ranked candidate chunks.
            radius: Number of preceding and succeeding chunks to retrieve.
            collection_name: Optional target collection name.
            
        Returns:
            Deduplicated list of chunks including expanded neighbors.
        """
        if not candidates or radius <= 0:
            return candidates

        seen_chunk_ids: set[str] = {c.chunk.chunk_id for c in candidates}
        expanded_list: list[ScoredRetrievalChunk] = list(candidates)

        for item in candidates:
            chk = item.chunk
            cid = chk.chunk_id

            # 1. Expand via VectorRepository.find_neighbors
            try:
                neighbors = self.repository.find_neighbors(
                    chunk_id=cid,
                    collection_name=collection_name,
                    window=radius,
                )
            except Exception as e:
                logger.warning(f"Neighbor expansion failed for chunk '{cid}': {e}")
                neighbors = []

            for n_chk in neighbors:
                if n_chk.chunk_id in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(n_chk.chunk_id)

                # Assign slightly decayed score of parent candidate
                decayed_score = item.score * 0.85
                exp_chunk = ScoredRetrievalChunk(
                    chunk=Chunk(
                        chunk_id=n_chk.chunk_id,
                        content=n_chk.text_preview or "",
                        metadata=n_chk.metadata,
                        hierarchy=ChunkHierarchy(
                            document_id=n_chk.metadata.document_id,
                            prev_chunk_id=getattr(n_chk, "prev_chunk_id", None),
                            next_chunk_id=getattr(n_chk, "next_chunk_id", None),
                        ),
                    ),
                    score=decayed_score,
                    rank=len(expanded_list),
                    dense_score=item.dense_score,
                    bm25_score=item.bm25_score,
                    fusion_score=item.fusion_score,
                    rerank_score=decayed_score,
                    boost_applied=0.0,
                    explainability=f"Context neighbor of {cid} (radius={radius}).",
                )
                expanded_list.append(exp_chunk)

        return expanded_list
