"""Sparse BM25 Retrieval Engine.

Implements a pure-Python, persistent, and incremental Okapi BM25 indexing engine
optimized for petroleum refinery technical vocabularies (equipment tags, OISD standards,
engineering units, P&IDs).
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
import threading
import time
from typing import Any, Optional

from rag_engine.retrieval.base_retriever import (
    BaseRetriever,
    CitationBundle,
    RetrievalResult,
    ScoredRetrievalChunk,
)
from rag_engine.retrieval.retrieval_exceptions import SparseRetrievalError
from rag_engine.retrieval.retrieval_metrics import RetrievalMetrics
from rag_engine.retrieval.retrieval_utils import (
    estimate_token_count,
    tokenize_refinery_text,
)
from rag_engine.schemas.chunk import Chunk

logger = logging.getLogger(__name__)


class BM25Index:
    """Thread-safe persistent Okapi BM25 inverted index supporting incremental updates."""

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        storage_path: Optional[str | Path] = None,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.storage_path = Path(storage_path) if storage_path else None
        
        self._lock = threading.RLock()
        
        # State
        self.chunks: dict[str, Chunk] = {}
        self.doc_lengths: dict[str, int] = {}
        self.avg_doc_len: float = 0.0
        self.total_docs: int = 0
        
        # Inverted index: term -> {chunk_id: frequency}
        self.inverted_index: dict[str, dict[str, int]] = {}
        # Document frequency: term -> number of documents containing term
        self.doc_freq: dict[str, int] = {}

    def add_chunk(self, chunk: Chunk) -> None:
        """Add a single chunk to index."""
        self.add_chunks([chunk])

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Incrementally index a batch of chunks."""
        with self._lock:
            for chk in chunks:
                cid = chk.chunk_id
                if cid in self.chunks:
                    # Remove existing first to handle updates
                    self._remove_chunk_internal(cid)

                # Tokenize content enriched with metadata tags
                meta = chk.metadata
                enriched_text = f"{chk.content} {getattr(meta, 'section_title', '') or ''} {' '.join(getattr(meta, 'equipment_entities', []) or [])} {' '.join(getattr(meta, 'safety_entities', []) or [])}"
                tokens = tokenize_refinery_text(enriched_text)
                
                doc_len = len(tokens)
                self.chunks[cid] = chk
                self.doc_lengths[cid] = doc_len
                
                # Count frequencies
                term_counts: dict[str, int] = {}
                for t in tokens:
                    term_counts[t] = term_counts.get(t, 0) + 1

                for t, count in term_counts.items():
                    if t not in self.inverted_index:
                        self.inverted_index[t] = {}
                        self.doc_freq[t] = 0
                    self.inverted_index[t][cid] = count
                    self.doc_freq[t] += 1

            self.total_docs = len(self.chunks)
            total_len = sum(self.doc_lengths.values())
            self.avg_doc_len = (total_len / self.total_docs) if self.total_docs > 0 else 0.0

            if self.storage_path:
                self.save()

    def remove_chunk(self, chunk_id: str) -> bool:
        """Remove a single chunk from index."""
        with self._lock:
            removed = self._remove_chunk_internal(chunk_id)
            if removed:
                self.total_docs = len(self.chunks)
                total_len = sum(self.doc_lengths.values())
                self.avg_doc_len = (total_len / self.total_docs) if self.total_docs > 0 else 0.0
                if self.storage_path:
                    self.save()
            return removed

    def remove_document(self, document_id: str) -> int:
        """Remove all chunks associated with a parent document ID."""
        with self._lock:
            matching = [
                cid for cid, chk in self.chunks.items()
                if chk.metadata.document_id == document_id
            ]
            for cid in matching:
                self._remove_chunk_internal(cid)
            
            self.total_docs = len(self.chunks)
            total_len = sum(self.doc_lengths.values())
            self.avg_doc_len = (total_len / self.total_docs) if self.total_docs > 0 else 0.0
            
            if self.storage_path and matching:
                self.save()
            return len(matching)

    def _remove_chunk_internal(self, chunk_id: str) -> bool:
        if chunk_id not in self.chunks:
            return False

        del self.chunks[chunk_id]
        if chunk_id in self.doc_lengths:
            del self.doc_lengths[chunk_id]

        # Clean from inverted index
        terms_to_delete = []
        for term, postings in self.inverted_index.items():
            if chunk_id in postings:
                del postings[chunk_id]
                self.doc_freq[term] -= 1
                if self.doc_freq[term] <= 0:
                    terms_to_delete.append(term)

        for term in terms_to_delete:
            if term in self.inverted_index and not self.inverted_index[term]:
                del self.inverted_index[term]
            if term in self.doc_freq and self.doc_freq[term] <= 0:
                del self.doc_freq[term]

        return True

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[tuple[Chunk, float]]:
        """Search index and calculate Okapi BM25 scores for matching chunks."""
        with self._lock:
            if not query or self.total_docs == 0:
                return []

            query_tokens = tokenize_refinery_text(query)
            if not query_tokens:
                return []

            from rag_engine.retrieval.retrieval_utils import QUERY_STOPWORDS
            content_tokens = [t for t in query_tokens if t not in QUERY_STOPWORDS]
            search_tokens = content_tokens if content_tokens else query_tokens

            scores: dict[str, float] = {}

            for q_term in search_tokens:
                if q_term not in self.inverted_index:
                    continue

                n_q = self.doc_freq.get(q_term, 0)
                # Standard Robertson-Spärck Jones IDF
                idf = math.log(((self.total_docs - n_q + 0.5) / (n_q + 0.5)) + 1.0)
                if idf <= 0:
                    idf = 1e-4

                postings = self.inverted_index[q_term]
                for cid, f_qd in postings.items():
                    doc_len = self.doc_lengths.get(cid, 1)
                    denom = f_qd + self.k1 * (1.0 - self.b + self.b * (doc_len / (self.avg_doc_len or 1.0)))
                    term_score = idf * ((f_qd * (self.k1 + 1.0)) / (denom or 1.0))
                    scores[cid] = scores.get(cid, 0.0) + term_score

            # Apply metadata filters if provided
            filtered_cids = list(scores.keys())
            if filters:
                matching_cids = []
                for cid in filtered_cids:
                    chk = self.chunks[cid]
                    meta = chk.metadata
                    match = True
                    for k, v in filters.items():
                        chunk_v = getattr(meta, k, None)
                        if isinstance(chunk_v, list):
                            if v not in chunk_v:
                                match = False
                                break
                        elif chunk_v != v:
                            match = False
                            break
                    if match:
                        matching_cids.append(cid)
                filtered_cids = matching_cids

            # Sort descending
            filtered_cids.sort(key=lambda cid: scores[cid], reverse=True)
            results = [(self.chunks[cid], scores[cid]) for cid in filtered_cids[:top_k]]
            return results

    def save(self, path: Optional[str | Path] = None) -> None:
        """Persist index state and metadata to disk."""
        target = Path(path) if path else self.storage_path
        if not target:
            return

        target.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            data = {
                "k1": self.k1,
                "b": self.b,
                "total_docs": self.total_docs,
                "avg_doc_len": self.avg_doc_len,
                "doc_lengths": self.doc_lengths,
                "inverted_index": self.inverted_index,
                "doc_freq": self.doc_freq,
                "chunks": {cid: chk.model_dump() for cid, chk in self.chunks.items()},
            }
            with open(target, "w", encoding="utf-8") as f:
                json.dump(data, f)

    def load(self, path: Optional[str | Path] = None) -> None:
        """Load index state from disk."""
        target = Path(path) if path else self.storage_path
        if not target or not target.exists():
            return

        with self._lock:
            try:
                with open(target, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.k1 = data.get("k1", 1.5)
                self.b = data.get("b", 0.75)
                self.total_docs = data.get("total_docs", 0)
                self.avg_doc_len = data.get("avg_doc_len", 0.0)
                self.doc_lengths = data.get("doc_lengths", {})
                self.inverted_index = data.get("inverted_index", {})
                self.doc_freq = data.get("doc_freq", {})
                self.chunks = {
                    cid: Chunk.model_validate(chk_dict)
                    for cid, chk_dict in data.get("chunks", {}).items()
                }
            except Exception as e:
                raise SparseRetrievalError(f"Failed to load BM25 index from {target}: {e}") from e


class BM25Retriever(BaseRetriever):
    """BM25 sparse retriever implementing BaseRetriever interface."""

    def __init__(
        self,
        index: Optional[BM25Index] = None,
        storage_path: Optional[str | Path] = None,
    ) -> None:
        sp = storage_path or Path("cache/retrieval/bm25_index.json")
        self.index = index or BM25Index(storage_path=sp)
        if sp and Path(sp).exists() and not self.index.chunks:
            try:
                self.index.load(sp)
            except Exception as e:
                logger.warning(f"Failed to auto-load BM25 index from {sp}: {e}")

    def get_strategy_name(self) -> str:
        return "bm25"

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Any] = None,
        **kwargs: Any,
    ) -> RetrievalResult:
        """Execute BM25 lexical search."""
        t0 = time.perf_counter()
        
        # Convert MetadataFilter or dict to simple filter dict if needed
        filter_dict = None
        if isinstance(filters, dict):
            filter_dict = filters

        search_hits = self.index.search(query, top_k=top_k, filters=filter_dict)
        dur_ms = (time.perf_counter() - t0) * 1000.0

        scored_chunks: list[ScoredRetrievalChunk] = []
        citations: list[CitationBundle] = []

        for rank, (chk, score) in enumerate(search_hits):
            meta = chk.metadata
            explain = f"BM25 lexical match score={score:.4f} on terms."
            
            sc = ScoredRetrievalChunk(
                chunk=chk,
                score=score,
                rank=rank,
                bm25_score=score,
                explainability=explain,
            )
            scored_chunks.append(sc)

            cit = CitationBundle(
                citation_id=f"[{rank + 1}]",
                document_id=meta.document_id,
                document_name=meta.document_name,
                source_path=meta.source_path,
                page_number=meta.page_number,
                section_title=meta.section_title,
                chunk_id=chk.chunk_id,
                verbatim_quote=chk.content[:250],
                score=score,
                equipment_tags=meta.equipment_entities or [],
                safety_tags=meta.safety_entities or [],
                revision=getattr(meta, "revision", None),
                sha256=meta.sha256,
            )
            citations.append(cit)

        metrics = RetrievalMetrics(
            query=query,
            strategy=self.get_strategy_name(),
            total_duration_ms=dur_ms,
            sparse_latency_ms=dur_ms,
            sparse_candidates_count=len(search_hits),
            returned_chunks_count=len(scored_chunks),
            citations_count=len(citations),
            top_score=scored_chunks[0].score if scored_chunks else 0.0,
            average_score=sum(s.score for s in scored_chunks) / len(scored_chunks) if scored_chunks else 0.0,
        )

        return RetrievalResult(
            query=query,
            original_query=query,
            strategy_name=self.get_strategy_name(),
            scored_chunks=scored_chunks,
            citations=citations,
            total_candidates=len(search_hits),
            execution_time_ms=dur_ms,
            metrics=metrics,
        )
