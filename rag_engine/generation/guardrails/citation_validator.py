"""Citation Validator — Strict Provenance Verification and Phantom Citation Pruning.

Validates that every citation anchor ([1], [2]) appearing in the generated response
strictly corresponds to an authentic candidate chunk provided in the prompt context.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from typing import Dict, List, Sequence, Set

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
        self._ref_section_regex = re.compile(
            r"(?:\n|\A|[.!?]\s+)\s*(?:[-*]{3,}\s*\n)?\s*(?:#{1,4}\s*|\*{1,2})?(?:Reference(?:s)?|Reference\(s\)|Bibliography|Sources?|Source\(s\))(?:\*{1,2})?(?:\s*:|\n)[\s\S]*",
            re.IGNORECASE,
        )

    def validate(
        self,
        generated_text: str,
        valid_anchors: Set[str] | Dict[str, str],
        valid_sources: Sequence[str] | None = None,
    ) -> CitationValidationReport:
        """Validate all citations in generated text against valid anchor set.
        
        Args:
            generated_text: Raw LLM response string.
            valid_anchors: Set of valid anchors (e.g. {"[1]", "[2]"}) or chunk-to-anchor dict.
            valid_sources: Optional sequence of valid source filenames/document IDs.
            
        Returns:
            CitationValidationReport with cleaned text and audit metrics.
        """
        if isinstance(valid_anchors, dict):
            allowed = set(valid_anchors.values())
        else:
            allowed = set(valid_anchors)

        # Normalize allowed set to include both "1" and "[1]"
        allowed_nums = {a.strip("[]") for a in allowed}

        # Check for self-generated bibliography / references section
        bib_match = self._ref_section_regex.search(generated_text)
        has_fabricated_source = False
        if bib_match:
            bib_text = bib_match.group(0)
            if valid_sources:
                valid_sources_lower = {s.lower() for s in valid_sources if s}
                # Check for filenames or document identifiers like "api_510.pdf", "oisd_130.pdf", etc.
                found_filenames = set(re.findall(r"\b[\w\-]+(?:\.(?:pdf|md|docx|txt|csv))\b", bib_text, re.IGNORECASE))
                # Also match document names right after anchor tags, e.g. [1] api_510
                anchor_docs = set(re.findall(r"\[\d+\]\s*([a-zA-Z0-9_\-\.]+)", bib_text))
                candidate_sources = found_filenames.union(anchor_docs)
                for src in candidate_sources:
                    src_clean = src.lower().strip(".,;:|")
                    if not src_clean or src_clean.isdigit():
                        continue
                    if src_clean not in valid_sources_lower and not any(src_clean in s for s in valid_sources_lower):
                        has_fabricated_source = True
                        logger.warning(
                            "Citation Validator: Model bibliography contains fabricated document name '%s' not matching real provenance sources %s",
                            src,
                            valid_sources_lower,
                        )

            # Strip self-generated bibliography section from cleaned text
            cleaned = generated_text[:bib_match.start()].strip()
        else:
            cleaned = generated_text

        # Clean inline "Reference(s): [1], [2]" labels if left in prose
        cleaned = re.sub(
            r"\b(?:Reference(?:s)?|Reference\(s\)|Source(?:s)?|Source\(s\))\s*:\s*(\[[0-9,\s]+\])",
            r"\1",
            cleaned,
            flags=re.IGNORECASE,
        )

        found_matches = self._cit_regex.findall(cleaned)
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
        if self.strip_phantom and phantom_cits:
            for p in phantom_cits:
                cleaned = cleaned.replace(p, "")
            # Clean up double spaces created by stripping
            cleaned = re.sub(r" +", " ", cleaned)

        if has_fabricated_source:
            is_valid = False
            precision = 0.0
        else:
            is_valid = len(phantom_cits) == 0

        if not is_valid:
            logger.warning(
                "Detected citation validity failure (phantom citations: %s, fabricated sources: %s)",
                phantom_cits,
                has_fabricated_source,
            )

        return CitationValidationReport(
            cleaned_text=cleaned.strip(),
            total_citations_found=total_found,
            valid_citations=valid_cits,
            phantom_citations=phantom_cits,
            citation_precision=round(precision, 4),
            is_valid=is_valid,
        )
