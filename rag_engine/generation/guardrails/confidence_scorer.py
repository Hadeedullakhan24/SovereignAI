"""Confidence Scorer — Composite Factuality and Grounding Score."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConfidenceBreakdown:
    """Detailed breakdown of generation confidence components."""

    composite_score: float
    retrieval_confidence: float
    citation_precision: float
    grounding_score: float
    is_high_confidence: bool


class ConfidenceScorer:
    """Calculates overall composite confidence score for generated responses."""

    def __init__(
        self,
        retrieval_weight: float = 0.35,
        citation_weight: float = 0.35,
        grounding_weight: float = 0.30,
        high_confidence_threshold: float = 0.80,
    ) -> None:
        self.retrieval_weight = retrieval_weight
        self.citation_weight = citation_weight
        self.grounding_weight = grounding_weight
        self.threshold = high_confidence_threshold

    def calculate(
        self,
        retrieval_confidence: float,
        citation_precision: float,
        grounding_score: float,
    ) -> ConfidenceBreakdown:
        """Calculate weighted composite confidence score."""
        retrieval_clamped = max(0.0, min(1.0, retrieval_confidence))
        citation_clamped = max(0.0, min(1.0, citation_precision))
        grounding_clamped = max(0.0, min(1.0, grounding_score))

        composite = (
            self.retrieval_weight * retrieval_clamped
            + self.citation_weight * citation_clamped
            + self.grounding_weight * grounding_clamped
        )
        composite = round(composite, 4)

        return ConfidenceBreakdown(
            composite_score=composite,
            retrieval_confidence=round(retrieval_clamped, 4),
            citation_precision=round(citation_clamped, 4),
            grounding_score=round(grounding_clamped, 4),
            is_high_confidence=composite >= self.threshold,
        )
