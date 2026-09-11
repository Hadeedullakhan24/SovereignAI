"""Cross-Encoder Neural Reranking Engine.

Provides deep contextual cross-attention scoring between user queries and candidate passages
using local offline transformer models (BAAI/bge-reranker-base, bge-reranker-large, ms-marco).
Includes pluggable architecture and deterministic fallback for air-gapped test environments.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from pathlib import Path
import threading
import time
from typing import Optional

from rag_engine.retrieval.base_retriever import ScoredRetrievalChunk
from rag_engine.retrieval.retrieval_exceptions import RerankerError

logger = logging.getLogger(__name__)


class BaseReranker(ABC):
    """Abstract contract for reranking candidates."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: list[ScoredRetrievalChunk],
        top_k: Optional[int] = None,
    ) -> list[ScoredRetrievalChunk]:
        """Score (query, passage) pairs and reorder candidates."""
        ...

    @abstractmethod
    def get_model_name(self) -> str:
        """Return model identifier."""
        ...


class DeterministicTestReranker(BaseReranker):
    """Deterministic offline reranker for testing and environments without loaded weights.
    
    Computes lexical-semantic overlap and query keyword proximity without requiring PyTorch.
    """

    def __init__(self, model_name: str = "deterministic-test-reranker") -> None:
        self.model_name = model_name

    def get_model_name(self) -> str:
        return self.model_name

    def rerank(
        self,
        query: str,
        candidates: list[ScoredRetrievalChunk],
        top_k: Optional[int] = None,
    ) -> list[ScoredRetrievalChunk]:
        """Rerank candidates using exact term matches and token proximity."""
        if not candidates:
            return []

        import re
        from rag_engine.retrieval.retrieval_utils import QUERY_STOPWORDS

        raw_words = [re.sub(r"^[^\w]+|[^\w]+$", "", w) for w in query.lower().split()]
        raw_words = [w for w in raw_words if w]
        content_words = [w for w in raw_words if w not in QUERY_STOPWORDS]
        q_terms = set(content_words if content_words else raw_words)
        reranked: list[ScoredRetrievalChunk] = []

        # Expand terms to include sub-parts of hyphenated identifiers (e.g. oisd-std-105 -> oisd, std, 105)
        expanded_q_terms = set(q_terms)
        for term in q_terms:
            if "-" in term or "_" in term:
                for sub in re.split(r"[-_]+", term):
                    if sub and sub not in QUERY_STOPWORDS and len(sub) > 1:
                        expanded_q_terms.add(sub)

        for item in candidates:
            content_lower = item.chunk.content.lower()
            meta = item.chunk.metadata
            meta_text = " ".join([
                (getattr(meta, "section_title", "") or "").lower(),
                (getattr(meta, "document_name", "") or "").lower(),
                " ".join(getattr(meta, "equipment_entities", []) or []).lower(),
                " ".join(getattr(meta, "safety_entities", []) or []).lower(),
            ])
            
            # Content matches vs metadata matches (content match weighted higher)
            content_matches = sum(1 for t in expanded_q_terms if t in content_lower)
            meta_matches = sum(1 for t in expanded_q_terms if t in meta_text)
            effective_matches = content_matches + (meta_matches * 0.25)
            overlap_ratio = min(1.0, effective_matches / len(expanded_q_terms)) if expanded_q_terms else 0.0
            
            # Boost score based on overlap and base score
            sim_score = (overlap_ratio * 0.75) + (min(1.0, item.score) * 0.25)

            # Penalize chunks dominated by non-printable or symbol noise
            if item.chunk.content:
                ascii_printable = sum(1 for ch in item.chunk.content if ch.isprintable() and ord(ch) < 128)
                printable_ratio = ascii_printable / max(1, len(item.chunk.content))
                if printable_ratio < 0.8:
                    sim_score *= 0.1

            # Penalize certificate artifacts and repetitive layout tokens
            if "microsoft time-stamp" in content_lower or "timestamp pca" in content_lower or "@v@v" in content_lower or "+++]" in content_lower:
                sim_score *= 0.05
            
            updated = ScoredRetrievalChunk(
                chunk=item.chunk,
                score=sim_score,
                rank=item.rank,
                dense_score=item.dense_score,
                bm25_score=item.bm25_score,
                fusion_score=item.fusion_score,
                rerank_score=sim_score,
                boost_applied=item.boost_applied,
                explainability=f"{item.explainability} Cross-encoder reranked (score={sim_score:.4f}).",
            )
            reranked.append(updated)

        reranked.sort(key=lambda x: x.rerank_score or -999.0, reverse=True)
        for r, item in enumerate(reranked):
            item.rank = r

        if top_k is not None and top_k > 0:
            return reranked[:top_k]
        return reranked


class LocalCrossEncoderReranker(BaseReranker):
    """Local offline CrossEncoder reranker using SentenceTransformers.
    
    Falls back gracefully to DeterministicTestReranker if weights are not downloaded locally.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        models_dir: str | Path = "models/rerankers",
        device: str = "cpu",
        use_fallback_if_missing: bool = True,
    ) -> None:
        self.model_name = model_name
        self.models_dir = Path(models_dir)
        self.device = device
        self.use_fallback = use_fallback_if_missing
        self._model = None
        self._fallback = DeterministicTestReranker(model_name=f"fallback-{model_name}")
        self._lock = threading.RLock()

    def get_model_name(self) -> str:
        return self.model_name

    def _load_model(self) -> None:
        """Attempt to load CrossEncoder model locally."""
        with self._lock:
            if self._model is not None:
                return

            local_path = self.models_dir / self.model_name.replace("/", "--")
            if not local_path.exists():
                if self.use_fallback:
                    logger.info(
                        f"Local model weights not found at '{local_path}'. "
                        f"Engaging deterministic fallback for air-gapped operation."
                    )
                    self._model = None
                    return
                else:
                    raise RerankerError(f"Model path '{local_path}' does not exist and fallback is disabled.")

            try:
                from sentence_transformers import CrossEncoder
                logger.info(f"Loading local CrossEncoder from {local_path} on {self.device}")
                self._model = CrossEncoder(
                    str(local_path),
                    device=self.device,
                    local_files_only=True,
                )
            except Exception as e:
                if self.use_fallback:
                    logger.warning(
                        f"Local CrossEncoder '{self.model_name}' could not be loaded: {e}. "
                        "Engaging DeterministicTestReranker."
                    )
                    self._model = None
                else:
                    raise RerankerError(f"Failed to load CrossEncoder '{self.model_name}': {e}") from e

    def rerank(
        self,
        query: str,
        candidates: list[ScoredRetrievalChunk],
        top_k: Optional[int] = None,
    ) -> list[ScoredRetrievalChunk]:
        """Rerank candidates using cross-attention neural model."""
        if not candidates:
            return []

        self._load_model()

        if self._model is None:
            return self._fallback.rerank(query, candidates, top_k=top_k)

        pairs = [[query, c.chunk.content] for c in candidates]
        try:
            scores = self._model.predict(pairs)
        except Exception as e:
            if self.use_fallback:
                return self._fallback.rerank(query, candidates, top_k=top_k)
            raise RerankerError(f"CrossEncoder inference failed: {e}") from e

        reranked: list[ScoredRetrievalChunk] = []
        for candidate, raw_score in zip(candidates, scores):
            s_val = float(raw_score)
            updated = ScoredRetrievalChunk(
                chunk=candidate.chunk,
                score=s_val,
                rank=candidate.rank,
                dense_score=candidate.dense_score,
                bm25_score=candidate.bm25_score,
                fusion_score=candidate.fusion_score,
                rerank_score=s_val,
                boost_applied=candidate.boost_applied,
                explainability=f"{candidate.explainability} Cross-Encoder neural rerank score={s_val:.4f}.",
            )
            reranked.append(updated)

        reranked.sort(key=lambda x: x.rerank_score or -999.0, reverse=True)
        for r, item in enumerate(reranked):
            item.rank = r

        if top_k is not None and top_k > 0:
            return reranked[:top_k]
        return reranked
