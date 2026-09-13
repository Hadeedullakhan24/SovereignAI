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
    cleaned_text: str = ""


class HallucinationGuard:
    """Detects ungrounded technical facts, tags, or numbers in generated responses."""

    def __init__(self, tolerance_threshold: float = 0.85) -> None:
        self.threshold = tolerance_threshold

        # Regex for equipment tags, units, numbers
        self._tag_regex = re.compile(r"\b([A-Z]{1,4}-[0-9]{3,5}[A-Z]?)\b")
        self._param_regex = re.compile(
            r"\b((?:[0-9]+(?:\.[0-9]+)?|one|two|three|four|five|six|seven|eight|nine|ten)\s*(?:bar|psi|kpa|mpa|°c|degc|°f|degf|m3/h|rpm|kw|mw|v|hz|gpm|years?|months?|days?|mm|cm|%|percent))\b",
            re.IGNORECASE,
        )

    @staticmethod
    def strip_reference_section(text: str) -> tuple[str, bool]:
        """Strip self-generated References/Bibliography section if present."""
        ref_pattern = re.compile(
            r"(?:\n|\A|[.!?]\s+)\s*(?:[-*]{3,}\s*\n)?\s*(?:#{1,4}\s*|\*{1,2})?(?:Reference(?:s)?|Reference\(s\)|Bibliography|Sources?|Source\(s\))(?:\*{1,2})?(?:\s*:|\n)[\s\S]*",
            re.IGNORECASE,
        )
        match = ref_pattern.search(text)
        if match:
            logger.warning(
                "Prompt-compliance violation: Model generated an unauthorized References/Bibliography section. Stripping section."
            )
            return text[:match.start()].strip(), True
        return text.strip(), False

    def verify(
        self,
        generated_text: str,
        source_context: str,
    ) -> GroundingVerificationReport:
        """Verify that technical entities asserted in response appear in source context."""
        cleaned_text, had_ref_section = self.strip_reference_section(generated_text)
        ctx_lower = source_context.lower()

        # 1. Extract equipment tags
        resp_tags = set(self._tag_regex.findall(cleaned_text))
        # 2. Extract technical parameters with units
        resp_params = set(self._param_regex.findall(cleaned_text))

        all_entities = resp_tags.union(resp_params)
        if not all_entities:
            # If no technical entities exist, text is qualitative; pass grounding
            return GroundingVerificationReport(
                grounding_score=1.0,
                verified_entities=[],
                unverified_entities=[],
                is_grounded=True,
                cleaned_text=cleaned_text,
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

        if unverified:
            # Defense-in-depth: sanitize sentences asserting unverified parameters
            sanitized_lines = []
            for line in cleaned_text.split("\n"):
                matching_unverified = [
                    u for u in unverified
                    if re.search(r"\b" + re.escape(u) + r"\b", line, re.IGNORECASE)
                ]
                if matching_unverified:
                    subbed = line
                    for u in matching_unverified:
                        def _replace_unverified(m: re.Match) -> str:
                            pfx = (m.group(1) or "").lower()
                            if "are" in pfx:
                                return "are not specified in the available documentation"
                            elif "is" in pfx:
                                return "is not specified in the available documentation"
                            return "not specified in the available documentation"

                        subbed = re.sub(
                            r"\b(are\s+|is\s+)?(?:every\s+)?" + re.escape(u) + r"\b",
                            _replace_unverified,
                            subbed,
                            flags=re.IGNORECASE,
                        )
                    subbed = re.sub(r"\s+", " ", subbed).replace(" .", ".").strip()
                    sanitized_lines.append(subbed)
                else:
                    sanitized_lines.append(line)
            cleaned_text = "\n".join(sanitized_lines)

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
            cleaned_text=cleaned_text,
        )
