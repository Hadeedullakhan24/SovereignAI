"""Hallucination Guard re-export for top-level generation access."""

from rag_engine.generation.guardrails.hallucination_guard import (
    GroundingVerificationReport,
    HallucinationGuard,
)

__all__ = [
    "HallucinationGuard",
    "GroundingVerificationReport",
]
