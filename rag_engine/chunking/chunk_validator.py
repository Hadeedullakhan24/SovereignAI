"""Quality validation engine for filtering and rejecting malformed chunks."""

from __future__ import annotations

import re
from typing import Optional, Set
from rag_engine.chunking.chunk_context import ChunkContext
from rag_engine.schemas.chunk import Chunk


class ChunkValidator:
    """Validates chunk quality and rejects empty, duplicate, tiny, or oversized chunks."""

    def __init__(self, context: Optional[ChunkContext] = None) -> None:
        self.context = context or ChunkContext()
        self._seen_hashes: Set[str] = set()

    def reset(self) -> None:
        """Reset deduplication state for a new document."""
        self._seen_hashes.clear()

    def validate_chunk(self, chunk: Chunk) -> tuple[bool, Optional[str]]:
        """Validate an individual chunk against quality and sizing criteria.

        Returns (is_valid, rejection_reason).
        """
        raw_text = chunk.content
        clean_text = raw_text.strip()

        # 1. Reject empty or whitespace-only chunks
        if not clean_text:
            return False, "EMPTY_WHITESPACE"

        # 2. Reject non-informative noise (pure punctuation or special symbols)
        alnum_chars = sum(1 for c in clean_text if c.isalnum())
        if alnum_chars == 0:
            return False, "NO_ALPHANUMERIC_CONTENT"

        # 3. Reject duplicate chunks if deduplication is enabled
        if self.context.deduplicate_chunks:
            chunk_hash = chunk.metadata.sha256
            if chunk_hash in self._seen_hashes:
                return False, "DUPLICATE_CONTENT"

        # 4. Reject tiny chunks (unless marked as table chunk or structural warning)
        if not chunk.metadata.is_table_chunk and not chunk.metadata.safety_entities:
            if chunk.token_count < self.context.min_chunk_tokens:
                return False, f"TINY_CHUNK_{chunk.token_count}_TOKENS"

        # 5. Reject oversized chunks if strict limits enabled
        if chunk.token_count > self.context.max_chunk_tokens:
            if self.context.strict_token_limits:
                return False, f"OVERSIZED_CHUNK_{chunk.token_count}_TOKENS"

        # Record hash if accepted
        if self.context.deduplicate_chunks:
            self._seen_hashes.add(chunk.metadata.sha256)

        return True, None

    def filter_chunks(self, chunks: list[Chunk]) -> tuple[list[Chunk], list[tuple[Chunk, str]]]:
        """Filter a list of candidate chunks, returning (accepted_chunks, rejected_with_reasons)."""
        accepted: list[Chunk] = []
        rejected: list[tuple[Chunk, str]] = []

        for chk in chunks:
            valid, reason = self.validate_chunk(chk)
            if valid:
                accepted.append(chk)
            else:
                rejected.append((chk, reason or "UNKNOWN"))

        return accepted, rejected
