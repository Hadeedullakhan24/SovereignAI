"""Hybrid Retrieval Engine.

Executes parallel dual-channel candidate search (Dense Vector via Qdrant + Sparse BM25),
calibrates scores, merges candidates via weighted Reciprocal Rank Fusion, applies domain
metadata boosting, performs local cross-encoder neural reranking, expands relational
chunk context, deduplicates results, builds verified citations, and packs token-budgeted prompts.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from rag_engine.retrieval.base_retriever import (
    BaseRetriever,
    RetrievalResult,
    ScoredRetrievalChunk,
)
from rag_engine.retrieval.bm25_retriever import BM25Retriever
from rag_engine.retrieval.citation_builder import CitationBuilder
from rag_engine.retrieval.context_expander import ContextExpander
from rag_engine.retrieval.context_packer import ContextPacker
from rag_engine.retrieval.dense_retriever import DenseRetriever
from rag_engine.retrieval.duplicate_remover import DuplicateRemover
from rag_engine.retrieval.metadata_booster import MetadataBooster
from rag_engine.retrieval.metadata_filter import MetadataFilterPlanner
from rag_engine.retrieval.query_analyzer import QueryAnalyzer
from rag_engine.retrieval.query_cache import QueryCache
from rag_engine.retrieval.query_normalizer import QueryNormalizer
from rag_engine.retrieval.reranker import BaseReranker, LocalCrossEncoderReranker
from rag_engine.retrieval.retrieval_events import (
    DenseSearchCompleted,
    FusionCompleted,
    RerankCompleted,
    RetrievalCompleted,
    RetrievalEventBus,
    RetrievalStarted,
    SparseSearchCompleted,
)
from rag_engine.retrieval.retrieval_metrics import (
    RetrievalMetrics,
    RetrievalMetricsCollector,
)
from rag_engine.retrieval.retrieval_utils import format_cache_key
from rag_engine.retrieval.retrieval_validator import RetrievalValidator
from rag_engine.retrieval.rrf_fusion import ReciprocalRankFusion
from rag_engine.retrieval.score_normalizer import NormalizationMethod, ScoreNormalizer

logger = logging.getLogger(__name__)


class HybridRetriever(BaseRetriever):
    """Production-grade enterprise hybrid retriever combining dense vector and lexical BM25 search."""

    def __init__(
        self,
        dense_retriever: Optional[DenseRetriever] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
        reranker: Optional[BaseReranker] = None,
        fusion: Optional[ReciprocalRankFusion] = None,
        booster: Optional[MetadataBooster] = None,
        expander: Optional[ContextExpander] = None,
        duplicate_remover: Optional[DuplicateRemover] = None,
        citation_builder: Optional[CitationBuilder] = None,
        context_packer: Optional[ContextPacker] = None,
        query_cache: Optional[QueryCache] = None,
        validator: Optional[RetrievalValidator] = None,
        use_cache: bool = True,
        use_reranker: bool = True,
        expansion_radius: int = 1,
        token_budget: int = 3000,
    ) -> None:
        self.dense_retriever = dense_retriever or DenseRetriever()
        self.bm25_retriever = bm25_retriever or BM25Retriever()
        self.reranker = reranker or LocalCrossEncoderReranker(use_fallback_if_missing=True)
        self.fusion = fusion or ReciprocalRankFusion(k=60, dense_weight=0.5, bm25_weight=0.5)
        self.booster = booster or MetadataBooster()
        self.expander = expander or ContextExpander(self.dense_retriever.repository)
        self.duplicate_remover = duplicate_remover or DuplicateRemover()
        self.citation_builder = citation_builder or CitationBuilder()
        self.context_packer = context_packer or ContextPacker(default_token_budget=token_budget)
        self.query_cache = query_cache or QueryCache(enabled=use_cache)
        self.validator = validator or RetrievalValidator()
        self.use_cache = use_cache
        self.use_reranker = use_reranker
        self.expansion_radius = expansion_radius
        self.token_budget = token_budget
        
        self.event_bus = RetrievalEventBus.get_instance()
        self.metrics_collector = RetrievalMetricsCollector.get_instance()

    def get_strategy_name(self) -> str:
        return "hybrid"

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Any] = None,
        collection_name: Optional[str] = None,
        category: Optional[str] = None,
        expansion_radius: Optional[int] = None,
        token_budget: Optional[int] = None,
        use_reranker: Optional[bool] = None,
        score_normalization: NormalizationMethod = NormalizationMethod.MIN_MAX,
        **kwargs: Any,
    ) -> RetrievalResult:
        """Execute complete hybrid retrieval pipeline."""
        t_start = time.perf_counter()
        self.event_bus.publish(RetrievalStarted(query=query, strategy=self.get_strategy_name(), top_k=top_k))

        # 1. Query Analysis & Normalization
        norm_query = QueryNormalizer.normalize(query)
        analyzer = QueryAnalyzer()
        intent = analyzer.analyze(norm_query)

        # 2. Check Query Cache
        cache_key = format_cache_key(norm_query, self.get_strategy_name(), top_k, filters)
        if self.use_cache:
            cached_result = self.query_cache.get(cache_key)
            if cached_result:
                dur_ms = (time.perf_counter() - t_start) * 1000.0
                if cached_result.metrics:
                    cached_result.metrics.cache_hit = True
                    cached_result.metrics.total_duration_ms = dur_ms
                    self.metrics_collector.record(cached_result.metrics)
                self.event_bus.publish(
                    RetrievalCompleted(
                        query=query,
                        total_chunks=len(cached_result.scored_chunks),
                        total_citations=len(cached_result.citations),
                        total_tokens=cached_result.tokens_packed,
                        total_duration_ms=dur_ms,
                        cache_hit=True,
                    )
                )
                return cached_result

        # 3. Plan Metadata Filters
        planned_filters = MetadataFilterPlanner.plan_filters(intent, explicit_filters=filters if isinstance(filters, dict) else None)

        # Candidate count: gather more candidates for fusion than final top-k (e.g. 2.5x)
        candidate_k = max(20, top_k * 2)

        target_category = category

        # 4. Dense Retrieval
        t_dense_start = time.perf_counter()
        try:
            dense_res = self.dense_retriever.retrieve(
                query=norm_query,
                top_k=candidate_k,
                filters=planned_filters,
                collection_name=collection_name,
                category=target_category,
            )
            dense_candidates = dense_res.scored_chunks
        except Exception as e:
            logger.warning(f"Dense retrieval error: {e}. Proceeding with sparse candidates only.")
            dense_candidates = []
        dense_dur = (time.perf_counter() - t_dense_start) * 1000.0
        self.event_bus.publish(DenseSearchCompleted(query=query, candidates_count=len(dense_candidates), duration_ms=dense_dur))

        # 5. Sparse BM25 Retrieval
        t_sparse_start = time.perf_counter()
        try:
            # BM25 filter dict
            bm25_filters = filters if isinstance(filters, dict) else None
            sparse_res = self.bm25_retriever.retrieve(
                query=norm_query,
                top_k=candidate_k,
                filters=bm25_filters,
            )
            sparse_candidates = sparse_res.scored_chunks
        except Exception as e:
            logger.warning(f"BM25 retrieval error: {e}. Proceeding with dense candidates only.")
            sparse_candidates = []
        sparse_dur = (time.perf_counter() - t_sparse_start) * 1000.0
        self.event_bus.publish(SparseSearchCompleted(query=query, candidates_count=len(sparse_candidates), duration_ms=sparse_dur))

        # 6. Score Normalization (if scores are non-empty)
        if dense_candidates and score_normalization != NormalizationMethod.NONE:
            d_scores = [c.score for c in dense_candidates]
            norm_d = ScoreNormalizer.normalize(d_scores, method=score_normalization)
            for c, ns in zip(dense_candidates, norm_d):
                c.score = ns

        if sparse_candidates and score_normalization != NormalizationMethod.NONE:
            s_scores = [c.score for c in sparse_candidates]
            norm_s = ScoreNormalizer.normalize(s_scores, method=score_normalization)
            for c, ns in zip(sparse_candidates, norm_s):
                c.score = ns

        # 7. Reciprocal Rank Fusion (RRF)
        t_fusion_start = time.perf_counter()
        fused_candidates = self.fusion.fuse(dense_candidates, sparse_candidates, top_k=candidate_k)
        fusion_dur = (time.perf_counter() - t_fusion_start) * 1000.0
        self.event_bus.publish(FusionCompleted(query=query, fused_candidates_count=len(fused_candidates), duration_ms=fusion_dur))

        # 8. Metadata Boosting
        boosted_candidates = self.booster.boost(fused_candidates, intent=intent)

        # 9. Neural Cross-Encoder Reranking
        t_rerank_start = time.perf_counter()
        should_rerank = self.use_reranker if use_reranker is None else use_reranker
        if should_rerank and boosted_candidates:
            # Rerank top candidate subset
            rerank_k = min(len(boosted_candidates), max(15, top_k))
            reranked_candidates = self.reranker.rerank(norm_query, boosted_candidates[:rerank_k], top_k=top_k)
        else:
            reranked_candidates = boosted_candidates[:top_k]
        rerank_dur = (time.perf_counter() - t_rerank_start) * 1000.0
        self.event_bus.publish(RerankCompleted(query=query, reranked_count=len(reranked_candidates), duration_ms=rerank_dur))

        # 10. Context Expansion (Neighbors)
        radius = expansion_radius if expansion_radius is not None else self.expansion_radius
        if radius > 0 and (intent.requires_neighbor_expansion or radius > 0):
            t_exp_start = time.perf_counter()
            expanded_candidates = self.expander.expand(
                reranked_candidates, radius=radius, collection_name=collection_name
            )
            exp_dur = (time.perf_counter() - t_exp_start) * 1000.0
        else:
            expanded_candidates = reranked_candidates
            exp_dur = 0.0

        # 11. Duplicate Removal
        unique_candidates = self.duplicate_remover.deduplicate_chunks(expanded_candidates)

        # 12. Citation Builder
        raw_citations = self.citation_builder.build_citations(unique_candidates)
        deduped_citations = self.duplicate_remover.deduplicate_citations(raw_citations)

        # 13. Context Packing
        t_pack_start = time.perf_counter()
        effective_budget = token_budget if token_budget is not None else self.token_budget
        packed_context, tokens_packed = self.context_packer.pack(
            unique_candidates, deduped_citations, token_budget=effective_budget
        )
        pack_dur = (time.perf_counter() - t_pack_start) * 1000.0

        total_dur = (time.perf_counter() - t_start) * 1000.0

        # Telemetry
        metrics = RetrievalMetrics(
            query=query,
            strategy=self.get_strategy_name(),
            total_duration_ms=round(total_dur, 2),
            dense_latency_ms=round(dense_dur, 2),
            sparse_latency_ms=round(sparse_dur, 2),
            fusion_latency_ms=round(fusion_dur, 2),
            rerank_latency_ms=round(rerank_dur, 2),
            expansion_latency_ms=round(exp_dur, 2),
            packing_latency_ms=round(pack_dur, 2),
            dense_candidates_count=len(dense_candidates),
            sparse_candidates_count=len(sparse_candidates),
            fused_candidates_count=len(fused_candidates),
            returned_chunks_count=len(unique_candidates),
            citations_count=len(deduped_citations),
            expanded_neighbors_count=max(0, len(expanded_candidates) - len(reranked_candidates)),
            tokens_packed=tokens_packed,
            top_score=unique_candidates[0].score if unique_candidates else 0.0,
            average_score=sum(c.score for c in unique_candidates) / len(unique_candidates) if unique_candidates else 0.0,
            cache_hit=False,
        )
        self.metrics_collector.record(metrics)

        result = RetrievalResult(
            query=norm_query,
            original_query=query,
            strategy_name=self.get_strategy_name(),
            scored_chunks=unique_candidates,
            citations=deduped_citations,
            packed_context=packed_context,
            total_candidates=len(dense_candidates) + len(sparse_candidates),
            tokens_packed=tokens_packed,
            execution_time_ms=round(total_dur, 2),
            metrics=metrics,
            metadata={
                "intent": intent.primary_intent.value,
                "secondary_intents": [s.value for s in intent.secondary_intents],
                "equipment_tags": intent.entities.equipment_tags,
                "standards": intent.entities.standards,
                "cached": False,
            },
        )

        # 14. Validation
        self.validator.validate(result)

        # 15. Store into Query Cache
        if self.use_cache and unique_candidates:
            self.query_cache.set(cache_key, result)

        self.event_bus.publish(
            RetrievalCompleted(
                query=query,
                total_chunks=len(unique_candidates),
                total_citations=len(deduped_citations),
                total_tokens=tokens_packed,
                total_duration_ms=round(total_dur, 2),
                cache_hit=False,
            )
        )

        return result
