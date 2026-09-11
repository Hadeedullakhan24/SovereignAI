"""Score Calibration & Normalizer.

Calibrates disparate score distributions between unbounded BM25 lexical scores
and dense vector cosine similarities using Min-Max, Softmax, Z-score, or Percentile scaling.
"""

from __future__ import annotations

from enum import StrEnum
import math
from typing import Sequence

from rag_engine.retrieval.retrieval_exceptions import ScoreNormalizationError


class NormalizationMethod(StrEnum):
    """Supported score calibration algorithms."""

    MIN_MAX = "min_max"
    SOFTMAX = "softmax"
    Z_SCORE = "z_score"
    PERCENTILE = "percentile"
    NONE = "none"


class ScoreNormalizer:
    """Configurable score normalizer for hybrid retrieval fusion."""

    @classmethod
    def normalize(
        cls,
        scores: Sequence[float],
        method: NormalizationMethod | str = NormalizationMethod.MIN_MAX,
    ) -> list[float]:
        """Normalize a sequence of float scores into calibrated values.
        
        Args:
            scores: Sequence of raw numeric scores.
            method: Calibration algorithm to apply.
            
        Returns:
            Calibrated list of floats.
        """
        if not scores:
            return []

        if len(scores) == 1:
            return [1.0]

        method_enum = NormalizationMethod(str(method).lower())

        if method_enum == NormalizationMethod.NONE:
            return [float(s) for s in scores]

        if method_enum == NormalizationMethod.MIN_MAX:
            return cls._min_max(scores)
        elif method_enum == NormalizationMethod.SOFTMAX:
            return cls._softmax(scores)
        elif method_enum == NormalizationMethod.Z_SCORE:
            return cls._z_score(scores)
        elif method_enum == NormalizationMethod.PERCENTILE:
            return cls._percentile(scores)
        else:
            raise ScoreNormalizationError(f"Unknown normalization method '{method}'")

    @staticmethod
    def _min_max(scores: Sequence[float]) -> list[float]:
        min_v = min(scores)
        max_v = max(scores)
        spread = max_v - min_v
        if spread == 0.0:
            return [1.0 for _ in scores]
        return [(float(s) - min_v) / spread for s in scores]

    @staticmethod
    def _softmax(scores: Sequence[float], temperature: float = 1.0) -> list[float]:
        if temperature <= 0:
            temperature = 1.0
        # Prevent numerical overflow by subtracting max
        max_v = max(scores)
        exp_vals = [math.exp((float(s) - max_v) / temperature) for s in scores]
        total = sum(exp_vals)
        if total == 0.0:
            return [1.0 / len(scores) for _ in scores]
        return [ev / total for ev in exp_vals]

    @staticmethod
    def _z_score(scores: Sequence[float]) -> list[float]:
        n = len(scores)
        mean = sum(scores) / n
        var = sum((s - mean) ** 2 for s in scores) / n
        std = math.sqrt(var)
        if std == 0.0:
            return [0.5 for _ in scores]
        # Map z-scores through logistic sigmoid to bound in [0, 1]
        z_scores = [(s - mean) / std for s in scores]
        return [1.0 / (1.0 + math.exp(-z)) for z in z_scores]

    @staticmethod
    def _percentile(scores: Sequence[float]) -> list[float]:
        # Rank scores and assign percentile rank in [0, 1]
        indexed = sorted(enumerate(scores), key=lambda x: x[1])
        n = len(scores)
        ranks = [0.0] * n
        for rank_idx, (orig_idx, _) in enumerate(indexed):
            ranks[orig_idx] = rank_idx / (n - 1) if n > 1 else 1.0
        return ranks
