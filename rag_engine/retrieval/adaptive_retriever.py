"""Adaptive Retrieval Engine.

Automatically classifies user queries into refinery operational archetypes and selects optimal
retrieval configurations (Top-K, channel fusion weights, neighbor expansion depth,
payload metadata filters, and token budgets) dynamically without user intervention.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from rag_engine.retrieval.base_retriever import (
    BaseRetriever,
    RetrievalResult,
)
from rag_engine.retrieval.hybrid_retriever import HybridRetriever
from rag_engine.retrieval.query_analyzer import (
    IntentType,
    QueryAnalyzer,
    QueryIntent,
)
from rag_engine.retrieval.rrf_fusion import ReciprocalRankFusion

logger = logging.getLogger(__name__)


class AdaptiveRetriever(BaseRetriever):
    """Adaptive retriever dynamically tailoring multi-channel search based on query intent."""

    def __init__(
        self,
        hybrid_retriever: Optional[HybridRetriever] = None,
        query_analyzer: Optional[QueryAnalyzer] = None,
    ) -> None:
        self.hybrid = hybrid_retriever or HybridRetriever()
        self.analyzer = query_analyzer or QueryAnalyzer()

    def get_strategy_name(self) -> str:
        return "adaptive"

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Any] = None,
        collection_name: Optional[str] = None,
        category: Optional[str] = None,
        **kwargs: Any,
    ) -> RetrievalResult:
        """Analyze query and execute customized retrieval pipeline."""
        intent: QueryIntent = self.analyzer.analyze(query)

        # Determine adaptive execution parameters
        adaptive_top_k = top_k or intent.suggested_top_k
        adaptive_radius = 1
        adaptive_budget = 3000
        dense_wt = 0.5
        bm25_wt = 0.5
        target_category = category or intent.preferred_category

        if intent.primary_intent == IntentType.EQUIPMENT_LOOKUP:
            # Heavy sparse weight for exact equipment tag lookup (P-203, MOV-101)
            adaptive_top_k = top_k or 5
            adaptive_radius = 1
            dense_wt = 0.3
            bm25_wt = 0.7
            adaptive_budget = 2500

        elif intent.primary_intent == IntentType.SAFETY_LOOKUP:
            # High recall for safety requirements, balanced channels
            adaptive_top_k = top_k or 15
            adaptive_radius = 2
            dense_wt = 0.5
            bm25_wt = 0.5
            adaptive_budget = 4000

        elif intent.primary_intent == IntentType.MAINTENANCE_LOOKUP:
            adaptive_top_k = top_k or 15
            adaptive_radius = 2
            dense_wt = 0.4
            bm25_wt = 0.6
            adaptive_budget = 3500

        elif intent.primary_intent == IntentType.SPECIFICATION_LOOKUP:
            adaptive_top_k = top_k or 8
            adaptive_radius = 1
            dense_wt = 0.4
            bm25_wt = 0.6
            adaptive_budget = 2500

        elif intent.primary_intent == IntentType.TABLE_LOOKUP:
            adaptive_top_k = top_k or 8
            adaptive_radius = 2
            dense_wt = 0.3
            bm25_wt = 0.7
            adaptive_budget = 3500

        elif intent.primary_intent == IntentType.TROUBLESHOOTING:
            # Complex troubleshooting requires broad context
            adaptive_top_k = top_k or 30
            adaptive_radius = 3
            dense_wt = 0.6
            bm25_wt = 0.4
            adaptive_budget = 6000

        elif intent.primary_intent == IntentType.COMPARISON:
            adaptive_top_k = top_k or 20
            adaptive_radius = 2
            dense_wt = 0.6
            bm25_wt = 0.4
            adaptive_budget = 4500

        else:
            # General semantic informational query
            adaptive_top_k = top_k or 10
            adaptive_radius = 1
            dense_wt = 0.6
            bm25_wt = 0.4
            adaptive_budget = 3000

        # Configure dynamic fusion weights on hybrid retriever
        self.hybrid.fusion = ReciprocalRankFusion(k=60, dense_weight=dense_wt, bm25_weight=bm25_wt)

        # Delegate execution to configured hybrid retriever
        res = self.hybrid.retrieve(
            query=query,
            top_k=adaptive_top_k,
            filters=filters,
            collection_name=collection_name,
            category=category,
            expansion_radius=adaptive_radius,
            token_budget=adaptive_budget,
            **kwargs,
        )

        res.strategy_name = self.get_strategy_name()
        res.metadata["adaptive_intent"] = intent.primary_intent.value
        res.metadata["adaptive_dense_weight"] = dense_wt
        res.metadata["adaptive_bm25_weight"] = bm25_wt
        res.metadata["adaptive_top_k"] = adaptive_top_k
        res.metadata["adaptive_radius"] = adaptive_radius
        res.metadata["adaptive_budget"] = adaptive_budget

        return res
