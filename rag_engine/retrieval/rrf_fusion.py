"""Reciprocal Rank Fusion (RRF).

Combines ranked result lists from disparate retrieval mechanisms (dense vector and sparse BM25)
using weighted reciprocal rank fusion with configurable smoothing parameters.
"""

from __future__ import annotations

from typing import Optional

from rag_engine.retrieval.base_retriever import ScoredRetrievalChunk
from rag_engine.retrieval.retrieval_exceptions import RRFError


class ReciprocalRankFusion:
    """Configurable weighted Reciprocal Rank Fusion algorithm."""

    def __init__(
        self,
        k: int = 60,
        dense_weight: float = 0.5,
        bm25_weight: float = 0.5,
    ) -> None:
        """Initialize RRF with smoothing constant and channel weights.
        
        Args:
            k: Ranking smoothing factor (standard 60).
            dense_weight: Multiplier weight for dense vector rankings.
            bm25_weight: Multiplier weight for sparse BM25 rankings.
        """
        if k < 1:
            raise RRFError("RRF smoothing factor 'k' must be >= 1.")
        self.k = k
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight

    def fuse(
        self,
        dense_results: list[ScoredRetrievalChunk],
        bm25_results: list[ScoredRetrievalChunk],
        top_k: Optional[int] = None,
    ) -> list[ScoredRetrievalChunk]:
        """Merge dense and BM25 ranked candidate lists using weighted RRF formula.
        
        RRF(d) = w_dense * (1 / (k + rank_dense(d))) + w_bm25 * (1 / (k + rank_bm25(d)))
        """
        fused_scores: dict[str, float] = {}
        chunk_lookup: dict[str, ScoredRetrievalChunk] = {}
        dense_scores_map: dict[str, float] = {}
        bm25_scores_map: dict[str, float] = {}

        # 1. Process dense rankings
        for rank, item in enumerate(dense_results, start=1):
            cid = item.chunk.chunk_id
            chunk_lookup[cid] = item
            dense_scores_map[cid] = item.score
            rrf_contrib = self.dense_weight * (1.0 / (self.k + rank))
            fused_scores[cid] = fused_scores.get(cid, 0.0) + rrf_contrib

        # 2. Process BM25 rankings
        for rank, item in enumerate(bm25_results, start=1):
            cid = item.chunk.chunk_id
            if cid not in chunk_lookup:
                chunk_lookup[cid] = item
            bm25_scores_map[cid] = item.score
            rrf_contrib = self.bm25_weight * (1.0 / (self.k + rank))
            fused_scores[cid] = fused_scores.get(cid, 0.0) + rrf_contrib

        # 3. Sort merged candidates by composite RRF score descending
        sorted_cids = sorted(fused_scores.keys(), key=lambda cid: fused_scores[cid], reverse=True)

        # 4. Construct unified output candidates
        fused_results: list[ScoredRetrievalChunk] = []
        for rank, cid in enumerate(sorted_cids):
            base_item = chunk_lookup[cid]
            fusion_score = fused_scores[cid]
            
            # Form explanation
            in_dense = cid in dense_scores_map
            in_bm25 = cid in bm25_scores_map
            channels = []
            if in_dense:
                channels.append(f"dense (score={dense_scores_map[cid]:.4f})")
            if in_bm25:
                channels.append(f"bm25 (score={bm25_scores_map[cid]:.4f})")
            explain = f"Fused from {', '.join(channels)} via weighted RRF."

            fused_chunk = ScoredRetrievalChunk(
                chunk=base_item.chunk,
                score=fusion_score,
                rank=rank,
                dense_score=dense_scores_map.get(cid),
                bm25_score=bm25_scores_map.get(cid),
                fusion_score=fusion_score,
                boost_applied=base_item.boost_applied,
                explainability=explain,
            )
            fused_results.append(fused_chunk)

        if top_k is not None and top_k > 0:
            return fused_results[:top_k]
        return fused_results
