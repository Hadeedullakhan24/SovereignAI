"""Citation Validator — Strict Provenance Verification and Phantom Citation Pruning.

Validates that every citation anchor ([1], [2]) appearing in the generated response
strictly corresponds to an authentic candidate chunk provided in the prompt context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Set

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
    unsupported_claims: List[str] = field(default_factory=list)
    supported_claims: List[str] = field(default_factory=list)


class CitationValidator:
    """Verifies citation anchors and validates that cited claims are factually supported by citation content."""

    def __init__(self, strip_phantom_citations: bool = True) -> None:
        self.strip_phantom = strip_phantom_citations
        self._cit_regex = re.compile(r"\[([0-9]+)\]")
        self._ref_section_regex = re.compile(
            r"(?:\n|\A|[.!?]\s+)\s*(?:[-*]{3,}\s*\n)?\s*(?:#{1,4}\s*|\*{1,2})?(?:Reference(?:s)?|Reference\(s\)|Bibliography|Sources?|Source\(s\))(?:\*{1,2})?(?:\s*:|\n)[\s\S]*",
            re.IGNORECASE,
        )
        self._param_regex = re.compile(
            r"\b((?:[0-9]+(?:\.[0-9]+)?|one|two|three|four|five|six|seven|eight|nine|ten)\s*(?:bar|barg|psi|kpa|mpa|kg/cm2g?|°c|degc|°f|degf|m3/h|rpm|kw|mw|v|hz|gpm|years?|months?|days?|hours?|hrs?|minutes?|mins?|mm|cm|meters?|m|%|percent))\b",
            re.IGNORECASE,
        )

    def validate(
        self,
        generated_text: str,
        valid_anchors: Set[str] | Dict[str, str],
        valid_sources: Sequence[str] | None = None,
        citation_context: Optional[Dict[str, str] | Sequence[Any]] = None,
    ) -> CitationValidationReport:
        """Validate all citations in generated text against valid anchor set and citation content.
        
        Args:
            generated_text: Raw LLM response string.
            valid_anchors: Set of valid anchors (e.g. {"[1]", "[2]"}) or chunk-to-anchor dict.
            valid_sources: Optional sequence of valid source filenames/document IDs.
            citation_context: Optional mapping of anchor (e.g. "[1]" or "1") to citation chunk text, or sequence of CitationBundle.
            
        Returns:
            CitationValidationReport with cleaned text and audit metrics.
        """
        if isinstance(valid_anchors, dict):
            allowed = set(valid_anchors.values())
        else:
            allowed = set(valid_anchors)

        # Normalize allowed set to include both "1" and "[1]"
        allowed_nums = {a.strip("[]") for a in allowed}

        # Normalize citation_context map
        context_map: dict[str, str] = {}
        if citation_context:
            if isinstance(citation_context, dict):
                for k, v in citation_context.items():
                    context_map[k.strip("[]")] = v if isinstance(v, str) else str(v)
            elif isinstance(citation_context, (list, tuple)):
                for idx, c in enumerate(citation_context, 1):
                    cid = getattr(c, "citation_id", f"[{idx}]").strip("[]")
                    txt = getattr(c, "verbatim_quote", "") or getattr(c, "content", "") or str(c)
                    context_map[cid] = txt
                    context_map[str(idx)] = txt

        # Check for self-generated bibliography / references section
        bib_match = self._ref_section_regex.search(generated_text)
        has_fabricated_source = False
        if bib_match:
            bib_text = bib_match.group(0)
            if valid_sources:
                valid_sources_lower = {s.lower() for s in valid_sources if s}
                found_filenames = set(re.findall(r"\b[\w\-]+(?:\.(?:pdf|md|docx|txt|csv))\b", bib_text, re.IGNORECASE))
                anchor_docs = set(re.findall(r"\[\d+\]\s*([a-zA-Z0-9_\-\.]+)", bib_text))
                candidate_sources = found_filenames.union(anchor_docs)
                for src in candidate_sources:
                    src_clean = src.lower().strip(".,;:|")
                    if not src_clean or src_clean.isdigit():
                        continue
                    if src_clean not in valid_sources_lower and not any(src_clean in s for s in valid_sources_lower):
                        has_fabricated_source = True
                        logger.warning(
                            "Citation Validator: Model bibliography contains fabricated document name '%s'",
                            src,
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

        # Content support validation for cited sentences
        supported_claims: list[str] = []
        unsupported_claims: list[str] = []

        if context_map:
            # Split into sentence/line units
            sentences = re.split(r"(?<=[.!?\n])\s+", cleaned)
            for sent in sentences:
                cit_in_sent = self._cit_regex.findall(sent)
                if not cit_in_sent:
                    continue
                for c_num in cit_in_sent:
                    c_text = context_map.get(c_num, "").lower()
                    if not c_text:
                        continue
                    # Check technical entities in this specific sentence
                    sent_params = self._param_regex.findall(sent)
                    sent_unsupported = False
                    for param in sent_params:
                        p_clean = param.lower().strip()
                        p_nospace = p_clean.replace(" ", "")
                        if p_clean not in c_text and p_nospace not in c_text.replace(" ", ""):
                            unsupported_claims.append(f"[{c_num}] '{param}' in '{sent.strip()}' not found in citation evidence")
                            sent_unsupported = True
                    if not sent_unsupported:
                        supported_claims.append(f"[{c_num}] {sent.strip()}")

        # Deduplicate
        valid_cits = list(dict.fromkeys(valid_cits))
        phantom_cits = list(dict.fromkeys(phantom_cits))

        # Precision calculation incorporating content grounding
        total_anchor_checks = len(valid_cits) + len(phantom_cits)
        anchor_prec = len(valid_cits) / total_anchor_checks if total_anchor_checks > 0 else 1.0

        if unsupported_claims:
            total_claim_checks = len(supported_claims) + len(unsupported_claims)
            content_prec = len(supported_claims) / total_claim_checks if total_claim_checks > 0 else 0.0
            precision = min(anchor_prec, content_prec)
        else:
            precision = anchor_prec

        # Strip phantom citations from text if enabled
        if self.strip_phantom and phantom_cits:
            for p in phantom_cits:
                cleaned = cleaned.replace(p, "")
            cleaned = re.sub(r" +", " ", cleaned)

        is_valid = (len(phantom_cits) == 0) and not has_fabricated_source and (len(unsupported_claims) == 0)

        if not is_valid:
            logger.warning(
                "Detected citation validity failure (phantom: %s, fabricated sources: %s, unsupported claims: %s)",
                phantom_cits,
                has_fabricated_source,
                unsupported_claims,
            )

        return CitationValidationReport(
            cleaned_text=cleaned.strip(),
            total_citations_found=total_found,
            valid_citations=valid_cits,
            phantom_citations=phantom_cits,
            citation_precision=round(precision, 4),
            is_valid=is_valid,
            unsupported_claims=unsupported_claims,
            supported_claims=supported_claims,
        )
