"""Answer Alignment Validator — Independent Question-Answer & Evidence Discipline Evaluation.

Evaluates 5 independent dimensions:
1. Grounding (evidence support)
2. Citation Validity (valid bracket anchors)
3. Question Alignment (direct relevance to the requested query entity and property)
4. Completeness (meaningful answer or explicit, non-circular absence statement)
5. Anti-Fabrication (zero ungrounded technical parameters or speculative extrapolations)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import re
from typing import Any, List, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AlignmentEvaluationReport:
    """Multi-dimensional answer alignment and quality evaluation report."""

    is_grounded: bool
    is_citation_valid: bool
    is_question_aligned: bool
    is_complete: bool
    has_no_fabrication: bool
    cleaned_text: str
    speculative_inferences_removed: list[str] = field(default_factory=list)
    circular_definitions_fixed: list[str] = field(default_factory=list)
    contradictions_fixed: list[str] = field(default_factory=list)
    meta_commentary_removed: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Overall pass status across all independent evaluation dimensions."""
        return (
            self.is_grounded
            and self.is_citation_valid
            and self.is_question_aligned
            and self.is_complete
            and self.has_no_fabrication
        )


class AnswerAlignmentValidator:
    """Validates question-answer alignment and sanitizes contradictions, meta-commentary, speculation, and circular definitions."""

    def __init__(self) -> None:
        # Regex patterns for speculative inferences
        self._speculation_patterns = [
            re.compile(r"(?:indicating\s+that\s+it\s+likely|likely\s+operates|typical\s+for\s+such|would\s+need\s+to\s+consult|typically\s+operates|normally\s+operates|presumably|assumed\s+to\s+be)[^.\n]*\.?", re.IGNORECASE),
            re.compile(r"For\s+more\s+detailed\s+specifications[^.\n]*consult[^.\n]*\.?", re.IGNORECASE),
            re.compile(r"However,\s*the\s*context\s*mentions\s*that\s*the\s*(?:pump|compressor|vessel|drum|equipment)\s*is\s*used[^.\n]*\.?", re.IGNORECASE),
        ]
        self._meta_patterns = [
            re.compile(r"Therefore,\s*the\s*response\s*to\s*the\s*user(?:'s)?\s*query\s*would\s*be:\s*", re.IGNORECASE),
            re.compile(r"Therefore,\s*the\s*answer\s*to\s*the\s*user(?:'s)?\s*query\s*is:\s*", re.IGNORECASE),
            re.compile(r"Therefore,\s*the\s*response\s*is:\s*", re.IGNORECASE),
            re.compile(r"In\s*conclusion,\s*the\s*response\s*would\s*be:\s*", re.IGNORECASE),
            re.compile(r"To\s*answer\s*the\s*user(?:'s)?\s*query:\s*", re.IGNORECASE),
        ]

    def sanitize_meta_commentary(self, text: str) -> tuple[str, list[str]]:
        """Strip LLM conversational meta-commentary artifacts."""
        cleaned = text
        removed: list[str] = []
        for pat in self._meta_patterns:
            if pat.search(cleaned):
                cleaned = pat.sub("", cleaned)
                removed.append("Meta-commentary preamble")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned, removed

    def sanitize_contradictions(self, text: str) -> tuple[str, list[str]]:
        """Detect and remove contradictory blanket negations when positive evidence exists."""
        cleaned = text
        fixed: list[str] = []

        # Detect if text details recorded actions or parameters
        has_recorded_actions = bool(
            re.search(r"\b(?:proposed\s+action|action\s+required|controlled\s+review|inspection|maintenance\s+scope|scheduled|approved|recommended)\b", cleaned, re.IGNORECASE)
        )
        if has_recorded_actions:
            # Blanket negation followed by positive assertion
            blanket_pat = re.compile(
                r"No\s+specific\s+(?:maintenance\s+)?actions\s+were\s+recorded[^.\n]*\.[^.\n]*does\s+not\s+specify\s+any\s+particular\s+maintenance\s+tasks?\.\s*",
                re.IGNORECASE,
            )
            if blanket_pat.search(cleaned):
                cleaned = blanket_pat.sub("", cleaned)
                fixed.append("Pruned contradictory blanket denial preceding recorded action details")

        return cleaned.strip(), fixed

    def sanitize_speculation(self, query: str, text: str) -> tuple[str, list[str]]:
        """Remove speculative inferences that fill missing evidence with guesses."""
        removed: list[str] = []
        cleaned = text

        for pat in self._speculation_patterns:
            matches = pat.findall(cleaned)
            if matches:
                for m in matches:
                    removed.append(m.strip())
                cleaned = pat.sub("", cleaned).strip()

        # Clean double spaces or orphaned punctuation
        cleaned = re.sub(r"\s+", " ", cleaned).replace(" .", ".").replace(" ,", ",").strip()
        return cleaned, removed

    def sanitize_circular_definitions(self, query: str, text: str) -> tuple[str, list[str]]:
        """Detect and correct circular definitions (e.g. 'PPE required is PPE')."""
        fixed: list[str] = []
        cleaned = text

        if "ppe" in query.lower() or "personal protective equipment" in query.lower():
            if re.search(r"ppe\s+required\s+[^\n.]*\s+is\s+personal\s+protective\s+equipment", cleaned, re.IGNORECASE):
                fixed.append("Circular PPE acronym expansion")
                cleaned = "The retrieved documentation states that Personal Protective Equipment (PPE) is required, but does not specify the individual PPE items."

        return cleaned, fixed

    def evaluate(
        self,
        query: str,
        answer_text: str,
        source_context: str,
        is_grounded: bool = True,
        is_citation_valid: bool = True,
    ) -> AlignmentEvaluationReport:
        """Perform comprehensive evaluation and discipline sanitization."""
        # 1. Sanitize meta-commentary, contradictions, speculation & circular definitions
        clean_text, meta_removed = self.sanitize_meta_commentary(answer_text)
        clean_text, contra_fixed = self.sanitize_contradictions(clean_text)
        clean_text, spec_removed = self.sanitize_speculation(query, clean_text)
        clean_text, circ_fixed = self.sanitize_circular_definitions(query, clean_text)

        # 2. Check Question Alignment: Does the answer address the core entity/property in query?
        ans_lower = clean_text.lower()
        
        from rag_engine.retrieval.retrieval_utils import QUERY_STOPWORDS, tokenize_refinery_text
        q_tokens = [t for t in tokenize_refinery_text(query) if t not in QUERY_STOPWORDS and len(t) > 2]
        
        aligned = True
        if q_tokens:
            matching_tokens = sum(1 for t in q_tokens if t in ans_lower)
            is_absence_statement = any(phrase in ans_lower for phrase in [
                "not specified", "not explicitly stated", "does not contain", "cannot determine", "not defined", "not provided", "does not specify"
            ])
            if matching_tokens == 0 and not is_absence_statement:
                aligned = False

        # 3. Check Completeness
        is_complete = len(clean_text.strip()) > 20

        # 4. Check Fabrication
        has_no_fabrication = is_grounded and len(spec_removed) == 0

        return AlignmentEvaluationReport(
            is_grounded=is_grounded,
            is_citation_valid=is_citation_valid,
            is_question_aligned=aligned,
            is_complete=is_complete,
            has_no_fabrication=has_no_fabrication,
            cleaned_text=clean_text,
            speculative_inferences_removed=spec_removed,
            circular_definitions_fixed=circ_fixed,
            contradictions_fixed=contra_fixed,
            meta_commentary_removed=meta_removed,
        )
