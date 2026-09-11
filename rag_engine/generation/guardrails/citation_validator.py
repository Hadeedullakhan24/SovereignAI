"""Citation Validator — Strict Provenance Verification and Phantom Citation Pruning.

Validates that every citation anchor ([1], [2]) appearing in the generated response
strictly corresponds to an authentic candidate chunk provided in the prompt context.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from typing import Dict, List, Set

from rag_engine.generation.generation_exceptions import CitationValidationError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CitationValidationReport:
    """Detailed audit report on citation integrity within generated text."""

    cleaned_text: str
    total_citations_found: int
    valid_citations: List[str]
    phantom_citations: List[str]
    citation_precision: float
    is_valid: bool


class CitationValidator:
    """Verifies citation anchors and prunes hallucinated or unanchored citations."""

    def __init__(self, strip_phantom_citations: bool = True) -> None:
        self.strip_phantom = strip_phantom_citations
        self._cit_regex = re.compile(r"\[([0-9]+)\]")

    def validate(
        self,
        generated_text: str,
        valid_anchors: Set[str] | Dict[str, str],
    ) -> CitationValidationReport:
        """Validate all citations in generated text against valid anchor set.
        
        Args:
            generated_text: Raw LLM response string.
            valid_anchors: Set of valid anchors (e.g. {"[1]", "[2]"}) or chunk-to-anchor dict.
            
        Returns:
            CitationValidationReport with cleaned text and audit metrics.
        """
        if isinstance(valid_anchors, dict):
            allowed = set(valid_anchors.values())
        else:
            allowed = set(valid_anchors)

        # Normalize allowed set to include both "1" and "[1]"
        allowed_nums = {a.strip("[]") for a in allowed}

        found_matches = self._cit_regex.findall(generated_text)
        total_found = len(found_matches)

        valid_cits: list[str] = []
        phantom_cits: list[str] = []

        for num in found_matches:
            anchor = f"[{num}]"
            if num in allowed_nums:
                valid_cits.append(anchor)
            else:
                phantom_cits.append(anchor)

        # Deduplicate
        valid_cits = list(dict.fromkeys(valid_cits))
        phantom_cits = list(dict.fromkeys(phantom_cits))

        # Precision
        precision = (
            len(valid_cits) / (len(valid_cits) + len(phantom_cits))
            if (len(valid_cits) + len(phantom_cits)) > 0
            else 1.0
        )

        # Strip phantom citations from text if enabled
        cleaned = generated_text
        if self.strip_phantom and phantom_cits:
            for p in phantom_cits:
                cleaned = cleaned.replace(p, "")
            # Clean up double spaces created by stripping
            cleaned = re.sub(r" +", " ", cleaned)

        is_valid = len(phantom_cits) == 0

        if not is_valid:
            logger.warning(
                "Detected %d phantom citations in generated text: %s",
                len(phantom_cits),
                phantom_cits,
            )

        return CitationValidationReport(
            cleaned_text=cleaned.strip(),
            total_citations_found=total_found,
            valid_citations=valid_cits,
            phantom_citations=phantom_cits,
            citation_precision=round(precision, 4),
            is_valid=is_valid,
        )
