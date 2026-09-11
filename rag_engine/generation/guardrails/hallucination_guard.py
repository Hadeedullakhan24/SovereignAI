"""Hallucination Guard — Technical Entity Cross-Verification.

Cross-checks numerical parameters, engineering units, and equipment tags asserted
in the LLM response against the retrieved source context to ensure strict grounding.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from typing import List, Set

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GroundingVerificationReport:
    """Audit report detailing technical entity verification across context."""

    grounding_score: float
    verified_entities: List[str]
    unverified_entities: List[str]
    is_grounded: bool


class HallucinationGuard:
    """Detects ungrounded technical facts, tags, or numbers in generated responses."""

    def __init__(self, tolerance_threshold: float = 0.85) -> None:
        self.threshold = tolerance_threshold

        # Regex for equipment tags, units, numbers
        self._tag_regex = re.compile(r"\b([A-Z]{1,4}-[0-9]{3,5}[A-Z]?)\b")
        self._param_regex = re.compile(
            r"\b([0-9]+(?:\.[0-9]+)?\s*(?:bar|psi|kpa|mpa|°c|degc|°f|degf|m3/h|rpm|kw|mw|v|hz|gpm))\b",
            re.IGNORECASE,
        )

    def verify(
        self,
        generated_text: str,
        source_context: str,
    ) -> GroundingVerificationReport:
        """Verify that technical entities asserted in response appear in source context."""
        ctx_lower = source_context.lower()

        # 1. Extract equipment tags
        resp_tags = set(self._tag_regex.findall(generated_text))
        # 2. Extract technical parameters with units
        resp_params = set(self._param_regex.findall(generated_text))

        all_entities = resp_tags.union(resp_params)
        if not all_entities:
            # If no technical entities exist, text is qualitative; pass grounding
            return GroundingVerificationReport(
                grounding_score=1.0,
                verified_entities=[],
                unverified_entities=[],
                is_grounded=True,
            )

        verified: list[str] = []
        unverified: list[str] = []

        for entity in all_entities:
            # Normalize for search
            ent_clean = entity.lower().strip()
            ent_no_space = ent_clean.replace(" ", "")
            if ent_clean in ctx_lower or ent_no_space in ctx_lower.replace(" ", ""):
                verified.append(entity)
            else:
                unverified.append(entity)

        total = len(verified) + len(unverified)
        score = len(verified) / total if total > 0 else 1.0
        is_grounded = score >= self.threshold

        if not is_grounded:
            logger.warning(
                "Hallucination Guard: Score %.2f below threshold %.2f. Unverified entities: %s",
                score,
                self.threshold,
                unverified,
            )

        return GroundingVerificationReport(
            grounding_score=round(score, 4),
            verified_entities=sorted(verified),
            unverified_entities=sorted(unverified),
            is_grounded=is_grounded,
        )
