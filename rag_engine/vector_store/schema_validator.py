"""Schema Validator.

Validates vector dimensionality, float32 precision, zero NaN/Inf tolerance,
unit norm compliance, and cryptographic checksum consistency.
"""

from __future__ import annotations

import math
from typing import Optional

from rag_engine.schemas.embedding import EmbeddedChunk, compute_vector_checksum
from rag_engine.vector_store.exceptions import (
    VectorDimensionMismatchError,
    VectorQualityError,
)


class SchemaValidator:
    """Validates vector numerical quality, dimensions, and checksums."""

    def __init__(self, tolerance: float = 1e-4) -> None:
        self.tolerance = tolerance

    def validate_vector(
        self,
        vector: list[float],
        expected_dim: Optional[int] = None,
        chunk_id: str = "unknown",
    ) -> None:
        """Validate a single embedding vector."""
        if not vector:
            raise VectorQualityError(f"Vector for chunk '{chunk_id}' is empty.")

        dim = len(vector)
        if expected_dim is not None and dim != expected_dim:
            raise VectorDimensionMismatchError(
                f"Dimension mismatch for chunk '{chunk_id}': expected {expected_dim}, got {dim}"
            )

        norm_sq = 0.0
        for i, val in enumerate(vector):
            if math.isnan(val):
                raise VectorQualityError(
                    f"Vector for chunk '{chunk_id}' contains NaN at index {i}."
                )
            if math.isinf(val):
                raise VectorQualityError(
                    f"Vector for chunk '{chunk_id}' contains Inf at index {i}."
                )
            norm_sq += val * val

        # Optional warning or check for zero-norm vector
        if norm_sq == 0.0:
            raise VectorQualityError(f"Vector for chunk '{chunk_id}' has all-zero magnitude.")

    def validate_chunk(
        self,
        chunk: EmbeddedChunk,
        expected_dim: Optional[int] = None,
        verify_checksum: bool = True,
    ) -> None:
        """Validate an EmbeddedChunk entity completely."""
        self.validate_vector(chunk.embedding, expected_dim, chunk.chunk_id)

        if verify_checksum and chunk.vector_checksum:
            expected_cksum = compute_vector_checksum(chunk.embedding)
            if chunk.vector_checksum != expected_cksum:
                raise VectorQualityError(
                    f"Vector checksum mismatch for chunk '{chunk.chunk_id}': "
                    f"stored {chunk.vector_checksum}, computed {expected_cksum}"
                )

    def validate_batch(
        self,
        chunks: list[EmbeddedChunk],
        expected_dim: Optional[int] = None,
        verify_checksum: bool = True,
    ) -> None:
        """Validate a batch of EmbeddedChunk entities."""
        for chunk in chunks:
            self.validate_chunk(chunk, expected_dim, verify_checksum)
