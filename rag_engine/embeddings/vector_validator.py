"""Vector quality and numerical integrity validator.

Ensures dense vectors comply with strict mathematical and architectural constraints:
- Correct dimensionality
- Float32 precision
- Zero NaN or Infinite values
- L2 unit normalization (if required)
- Cryptographic checksum verification
"""

from __future__ import annotations

import math
from typing import Optional

from rag_engine.embeddings.exceptions import VectorValidationError
from rag_engine.schemas.embedding import compute_vector_checksum


class VectorValidator:
    """Validates dense embedding vectors for numerical sanity and model compliance."""

    def __init__(
        self,
        default_tolerance: float = 0.02,
        strict: bool = True,
    ) -> None:
        self.default_tolerance = default_tolerance
        self.strict = strict

    def validate_vector(
        self,
        vector: list[float],
        expected_dimension: int,
        check_normalized: bool = True,
        tolerance: Optional[float] = None,
    ) -> bool:
        """Validate a single embedding vector.

        Args:
            vector: List of float values representing the embedding.
            expected_dimension: Expected length of the vector.
            check_normalized: Whether to enforce L2 unit length (norm ~= 1.0).
            tolerance: Permitted delta from unit length (defaults to 0.02).

        Returns:
            True if vector passes all validation checks.

        Raises:
            VectorValidationError: If strict=True and any check fails.
        """
        tol = tolerance if tolerance is not None else self.default_tolerance

        # 1. Non-empty check
        if not vector:
            msg = "Embedding vector is empty."
            if self.strict:
                raise VectorValidationError(msg)
            return False

        # 2. Dimension check
        if len(vector) != expected_dimension:
            msg = (
                f"Dimension mismatch: expected {expected_dimension}, "
                f"got {len(vector)}."
            )
            if self.strict:
                raise VectorValidationError(msg)
            return False

        # 3. Finite numerical check (no NaN, no Inf)
        sum_sq = 0.0
        for idx, val in enumerate(vector):
            if not isinstance(val, (int, float)):
                msg = f"Element at index {idx} is non-numeric: {type(val)}."
                if self.strict:
                    raise VectorValidationError(msg)
                return False
            if math.isnan(val):
                msg = f"Embedding vector contains NaN at index {idx}."
                if self.strict:
                    raise VectorValidationError(msg)
                return False
            if math.isinf(val):
                msg = f"Embedding vector contains Inf at index {idx}."
                if self.strict:
                    raise VectorValidationError(msg)
                return False
            sum_sq += val * val

        # 4. Normalization check (L2 norm)
        if check_normalized:
            l2_norm = math.sqrt(sum_sq)
            if abs(l2_norm - 1.0) > tol:
                msg = (
                    f"Vector is not L2-normalized: ||v|| = {l2_norm:.4f} "
                    f"(expected 1.0 +/- {tol})."
                )
                if self.strict:
                    raise VectorValidationError(msg)
                return False

        return True

    def validate_batch(
        self,
        vectors: list[list[float]],
        expected_dimension: int,
        check_normalized: bool = True,
    ) -> list[bool]:
        """Validate a batch of embedding vectors.

        Returns:
            List of boolean validation outcomes.
        """
        results = []
        for vec in vectors:
            try:
                valid = self.validate_vector(
                    vec,
                    expected_dimension=expected_dimension,
                    check_normalized=check_normalized,
                )
                results.append(valid)
            except VectorValidationError:
                if self.strict:
                    raise
                results.append(False)
        return results

    def verify_checksum(self, vector: list[float], expected_checksum: str) -> bool:
        """Verify vector checksum matches float32 binary representation."""
        actual = compute_vector_checksum(vector)
        if actual != expected_checksum:
            if self.strict:
                raise VectorValidationError(
                    f"Checksum mismatch: expected {expected_checksum}, got {actual}."
                )
            return False
        return True

