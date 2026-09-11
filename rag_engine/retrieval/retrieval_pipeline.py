"""Master Retrieval Pipeline Orchestrator.

Coordinates the complete 12-stage sequential retrieval pipeline:
Query Analysis -> Normalization -> Metadata Filter Planning -> Parallel Dual-Channel
Search -> Score Calibration -> RRF Fusion -> Metadata Boosting -> Cross-Encoder Reranking ->
Hierarchy Context Expansion -> Duplicate Removal -> Citation Building -> Context Packing ->
Validation -> Telemetry & Diagnostics.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from rag_engine.retrieval.adaptive_retriever import AdaptiveRetriever
from rag_engine.retrieval.base_retriever import (
    BaseRetriever,
    RetrievalResult,
)
from rag_engine.retrieval.bm25_retriever import BM25Index, BM25Retriever
from rag_engine.retrieval.dense_retriever import DenseRetriever
from rag_engine.retrieval.hybrid_retriever import HybridRetriever
from rag_engine.retrieval.retrieval_factory import RetrievalFactory
from rag_engine.retrieval.retrieval_health import (
    RetrievalHealthChecker,
    RetrievalHealthReport,
)
from rag_engine.retrieval.retrieval_metrics import (
    RetrievalMetricsCollector,
)
from rag_engine.schemas.chunk import Chunk
from rag_engine.vector_store.vector_repository import (
    VectorRepository,
    get_vector_repository,
)

logger = logging.getLogger(__name__)


class RetrievalPipeline:
    """Master production-grade retrieval pipeline facade for SIH26117 / MRPL."""

    def __init__(
        self,
        strategy_name: str = "adaptive",
        repository: Optional[VectorRepository] = None,
        bm25_index: Optional[BM25Index] = None,
        use_cache: bool = True,
        use_reranker: bool = True,
        token_budget: int = 3000,
    ) -> None:
        self.strategy_name = strategy_name
        self.repository = repository or get_vector_repository()
        self.bm25_index = bm25_index
        self.use_cache = use_cache
        self.use_reranker = use_reranker
        self.token_budget = token_budget

        self.factory = RetrievalFactory.get_instance()
        self.health_checker = RetrievalHealthChecker(self.repository)
        self.metrics_collector = RetrievalMetricsCollector.get_instance()

        # Build default underlying retriever
        self._retriever: Optional[BaseRetriever] = None
        self._init_retriever()

    def _init_retriever(self) -> None:
        """Initialize or configure active retrieval strategy."""
        dense = DenseRetriever(repository=self.repository)
        bm25 = BM25Retriever(index=self.bm25_index)
        hybrid = HybridRetriever(
            dense_retriever=dense,
            bm25_retriever=bm25,
            use_cache=self.use_cache,
            use_reranker=self.use_reranker,
            token_budget=self.token_budget,
        )

        if self.strategy_name == "dense":
            self._retriever = dense
        elif self.strategy_name == "bm25":
            self._retriever = bm25
        elif self.strategy_name == "hybrid":
            self._retriever = hybrid
        else:
            # Default: adaptive
            self._retriever = AdaptiveRetriever(hybrid_retriever=hybrid)

    def execute(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Any] = None,
        collection_name: Optional[str] = None,
        category: Optional[str] = None,
        **kwargs: Any,
    ) -> RetrievalResult:
        """Execute end-to-end retrieval for a user or agent query."""
        if not self._retriever:
            self._init_retriever()

        assert self._retriever is not None
        return self._retriever.retrieve(
            query=query,
            top_k=top_k,
            filters=filters,
            collection_name=collection_name,
            category=category,
            **kwargs,
        )

    # Standard interface alias
    retrieve = execute

    def index_chunks(self, chunks: list[Chunk]) -> int:
        """Synchronously index chunks into the sparse BM25 engine for hybrid search."""
        if not chunks:
            return 0
        idx = None
        if isinstance(self._retriever, (HybridRetriever, AdaptiveRetriever)):
            hybrid = self._retriever if isinstance(self._retriever, HybridRetriever) else self._retriever.hybrid
            idx = hybrid.bm25_retriever.index
        elif isinstance(self._retriever, BM25Retriever):
            idx = self._retriever.index
        elif self.bm25_index:
            idx = self.bm25_index

        if idx:
            idx.add_chunks(chunks)
            if idx.storage_path:
                idx.storage_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    idx.save()
                except Exception as e:
                    logger.warning(f"Failed to persist BM25 index to {idx.storage_path}: {e}")
        return len(chunks)

    def check_health(self) -> RetrievalHealthReport:
        """Execute system health diagnostics."""
        return self.health_checker.check_health()

    def get_metrics_summary(self) -> dict[str, float]:
        """Return aggregated retrieval telemetry."""
        return self.metrics_collector.get_summary()
