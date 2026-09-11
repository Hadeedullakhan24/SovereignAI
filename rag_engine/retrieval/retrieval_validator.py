"""Retrieval Result Validator.

Enforces strict quality controls and validation invariant checks on all retrieval outputs:
guaranteeing duplicate elimination, non-empty contexts, ranking integrity, citation completeness,
and token budget compliance.
"""

from __future__ import annotations

import logging
from typing import Optional

from rag_engine.retrieval.base_retriever import RetrievalResult
from rag_engine.retrieval.retrieval_exceptions import RetrievalValidationError
from rag_engine.retrieval.retrieval_utils import estimate_token_count

logger = logging.getLogger(__name__)


class RetrievalValidator:
    """Validates structural and semantic integrity of retrieval results."""

    def __init__(
        self,
        allow_empty_context: bool = True,
        max_allowed_tokens: int = 16000,
        enforce_citation_completeness: bool = True,
    ) -> None:
        self.allow_empty_context = allow_empty_context
        self.max_allowed_tokens = max_allowed_tokens
        self.enforce_citation_completeness = enforce_citation_completeness

    def validate(self, result: RetrievalResult) -> list[str]:
        """Validate a RetrievalResult instance against all architectural invariants.
        
        Returns:
            List of warning messages (if any). Raises RetrievalValidationError on critical failure.
        """
        warnings: list[str] = []

        if not result.query or not result.query.strip():
            raise RetrievalValidationError("RetrievalResult has an empty query string.")

        # 1. Check empty context
        if not result.scored_chunks:
            if not self.allow_empty_context:
                raise RetrievalValidationError(f"Retrieval yielded empty chunks for query: '{result.query}'")
            else:
                warnings.append("No chunks retrieved matching the query criteria.")
                return warnings

        # 2. Check duplicate chunks
        seen_cids: set[str] = set()
        for idx, item in enumerate(result.scored_chunks):
            cid = item.chunk.chunk_id
            if cid in seen_cids:
                raise RetrievalValidationError(f"Duplicate chunk '{cid}' detected at index {idx}.")
            seen_cids.add(cid)

        # 3. Check ranking consistency (monotonically non-increasing scores)
        prev_score = float("inf")
        for idx, item in enumerate(result.scored_chunks):
            if item.score > prev_score + 1e-6:
                warnings.append(
                    f"Ranking score inconsistency at rank {idx}: score {item.score:.4f} > previous {prev_score:.4f}."
                )
            prev_score = item.score

        # 4. Check citation completeness
        if self.enforce_citation_completeness and result.scored_chunks:
            chunk_ids = {item.chunk.chunk_id for item in result.scored_chunks}
            citation_chunk_ids = {cit.chunk_id for cit in result.citations}
            
            missing_cits = chunk_ids - citation_chunk_ids
            if missing_cits:
                warnings.append(f"{len(missing_cits)} chunks lack a corresponding CitationBundle.")

        # 5. Check token limits
        token_count = result.tokens_packed or estimate_token_count(result.packed_context)
        if token_count > self.max_allowed_tokens:
            raise RetrievalValidationError(
                f"Packed context exceeds maximum allowed token limit ({token_count} > {self.max_allowed_tokens})."
            )

        # 6. Check metadata integrity
        for item in result.scored_chunks:
            meta = item.chunk.metadata
            if not meta.document_id:
                warnings.append(f"Chunk '{item.chunk.chunk_id}' lacks parent document_id.")

        return warnings
