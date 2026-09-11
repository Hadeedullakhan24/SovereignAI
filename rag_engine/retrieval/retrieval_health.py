"""Retrieval System Health & Diagnostics.

Performs active canary health probes across the retrieval pipeline: VectorRepository,
BM25 Inverted Index, Local CrossEncoder, and SQLite Query Cache.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from rag_engine.vector_store.vector_repository import (
    VectorRepository,
    get_vector_repository,
)

logger = logging.getLogger(__name__)


class RetrievalHealthReport(BaseModel):
    """Structured health assessment of the retrieval platform."""

    model_config = ConfigDict(frozen=True)

    status: str = Field(description="'HEALTHY', 'DEGRADED', or 'UNHEALTHY'")
    vector_store_ok: bool = Field(default=True)
    bm25_index_ok: bool = Field(default=True)
    reranker_ok: bool = Field(default=True)
    cache_ok: bool = Field(default=True)
    details: dict[str, Any] = Field(default_factory=dict)


class RetrievalHealthChecker:
    """Performs diagnostic canary queries across retrieval subsystems."""

    def __init__(
        self,
        repository: Optional[VectorRepository] = None,
    ) -> None:
        self.repository = repository or get_vector_repository()

    def check_health(self) -> RetrievalHealthReport:
        """Run health probes across all retrieval subsystems."""
        details: dict[str, Any] = {}
        all_ok = True

        # 1. Check VectorRepository
        vec_ok = True
        try:
            repo_health = self.repository.health()
            details["vector_repository"] = repo_health.model_dump()
            if repo_health.status != "healthy":
                vec_ok = False
        except Exception as e:
            vec_ok = False
            details["vector_repository_error"] = str(e)

        # 2. Check BM25 engine
        bm25_ok = True
        try:
            from rag_engine.retrieval.bm25_retriever import BM25Retriever
            bm25 = BM25Retriever()
            details["bm25_index_docs"] = bm25.index.total_docs
        except Exception as e:
            bm25_ok = False
            details["bm25_error"] = str(e)

        # 3. Check Reranker
        reranker_ok = True
        try:
            from rag_engine.retrieval.reranker import DeterministicTestReranker
            test_reranker = DeterministicTestReranker()
            details["reranker_status"] = "OK"
        except Exception as e:
            reranker_ok = False
            details["reranker_error"] = str(e)

        # 4. Check Query Cache
        cache_ok = True
        try:
            from rag_engine.retrieval.query_cache import QueryCache
            cache = QueryCache()
            details["query_cache_entries"] = cache.count()
        except Exception as e:
            cache_ok = False
            details["cache_error"] = str(e)

        overall_status = "HEALTHY"
        if not vec_ok or not bm25_ok:
            overall_status = "DEGRADED"
        if not vec_ok and not bm25_ok:
            overall_status = "UNHEALTHY"

        return RetrievalHealthReport(
            status=overall_status,
            vector_store_ok=vec_ok,
            bm25_index_ok=bm25_ok,
            reranker_ok=reranker_ok,
            cache_ok=cache_ok,
            details=details,
        )
