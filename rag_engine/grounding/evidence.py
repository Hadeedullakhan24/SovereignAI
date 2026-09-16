"""LLM-independent evidence selection, dynamic scoping, and canonical structured report construction."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any, Iterable, Optional, Sequence

from rag_engine.retrieval.base_retriever import CitationBundle, ScoredRetrievalChunk


class EvidenceRelation(str, Enum):
    DIRECT = "direct"
    RELATED = "related"
    GENERIC_REQUIREMENT = "generic_requirement"
    UNRELATED = "unrelated"
    AMBIGUOUS_OR_CONFLICTING = "ambiguous_or_conflicting"


_STOP = frozenset(
    "a an the and or of for to in on with from by is are was were what which who when where "
    "does do did show tell give list create generate report document documents documentation "
    "docx pdf xlsx pptx available evidence actual "
    "prepare export produce build make draft write download artifact file format using use "
    "containing contain only based provide according".split()
)
_ID_PATTERN = re.compile(r"(?<!\w)[A-Za-z][A-Za-z0-9]*(?:[-_/][A-Za-z0-9]+)+(?!\w)")
_WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[-_/][A-Za-z0-9]+)*")
_PLACEHOLDER_PATTERN = re.compile(
    r"\b(?:20XX|2XXX|YYYY|\d{4}-XX-XX|XX/XX/XXXX|UNKNOWN_TAG|SYNTHETIC_REPORT|DUMMY_TAG)\b",
    re.IGNORECASE,
)

# Words indicating general normative requirements rather than specific inspection findings
_REGULATORY_NORMATIVE_WORDS = frozenset(
    "shall should requirement requirements guidelines standard standards schedule regulation "
    "regulations procedure procedures policy general provisions code oisd api pngrb statutory "
    "mandatory compliance".split()
)

# Words indicating specific recorded findings or inspection observations
_SPECIFIC_FINDING_WORDS = frozenset(
    "inspected measured thickness observed reading corrosion result results tag serial "
    "defect repair ultrasonic mpt dpt test date inspector finding findings record log".split()
)


def _terms(value: str) -> set[str]:
    return {
        m.group(0).casefold()
        for m in _WORD_PATTERN.finditer(value)
        if len(m.group(0)) > 2 and m.group(0).casefold() not in _STOP
    }


def _term_overlap(q_terms: set[str], source_terms: set[str]) -> float:
    if not q_terms or not source_terms:
        return 0.0
    matches = 0
    for q in q_terms:
        if q in source_terms or any(
            q.startswith(s) or s.startswith(q) or (len(q) > 3 and len(s) > 3 and q[:4] == s[:4])
            for s in source_terms
        ):
            matches += 1
    return matches / max(1, len(q_terms))


def _identifiers(value: str) -> set[str]:
    return {m.group(0).casefold() for m in _ID_PATTERN.finditer(value)}


@dataclass(frozen=True)
class EvidenceItem:
    relation: EvidenceRelation
    chunk: ScoredRetrievalChunk
    citation: CitationBundle
    entities: tuple[str, ...]
    reason: str
    is_general_norm: bool = False

    @property
    def text(self) -> str:
        return self.chunk.chunk.content


@dataclass(frozen=True)
class ReportClaim:
    text: str
    citation_id: str
    document_id: str
    document_name: str
    page_number: Optional[int]
    chunk_id: str
    entities: tuple[str, ...]
    relation: EvidenceRelation
    source_quote: str = ""

    def has_placeholders(self) -> bool:
        return bool(_PLACEHOLDER_PATTERN.search(self.text))


@dataclass(frozen=True)
class StructuredReport:
    title: str
    summary: str
    sections: tuple[dict, ...]
    citations: tuple[CitationBundle, ...]
    claims: tuple[ReportClaim, ...]
    is_grounded: bool = True
    missing_information: bool = False
    scope_established: bool = True
    conclusion: Optional[str] = None

    def validate(self) -> tuple[bool, list[str]]:
        """Validate canonical StructuredReport integrity before rendering."""
        errors: list[str] = []
        if self.missing_information:
            errors.append("Report marked as missing required evidence.")
        if not self.scope_established:
            errors.append("Report scope could not be established from query or evidence.")
        if not self.claims and not self.missing_information:
            errors.append("Report contains zero factual claims.")

        for idx, claim in enumerate(self.claims, 1):
            if claim.has_placeholders():
                errors.append(f"Claim [{idx}] contains forbidden placeholder tokens: '{claim.text[:60]}'")
            if not claim.document_id and not claim.document_name:
                errors.append(f"Claim [{idx}] lacks document provenance.")
            if not claim.citation_id:
                errors.append(f"Claim [{idx}] lacks citation ID.")

        is_valid = len(errors) == 0
        return is_valid, errors

    def renderer_payload(self) -> dict:
        return {
            "title": self.title,
            "summary": self.summary,
            "sections": list(self.sections),
            "citations": list(self.citations),
            "conclusion": self.conclusion,
            "structured_report": self,
        }


@dataclass(frozen=True)
class EvidencePackage:
    query: str
    requested_entities: tuple[str, ...]
    items: tuple[EvidenceItem, ...]
    conflicts: tuple[str, ...] = ()
    scope_topic: str = ""
    is_scope_established: bool = True

    @property
    def direct_and_related(self) -> tuple[EvidenceItem, ...]:
        return tuple(i for i in self.items if i.relation in (EvidenceRelation.DIRECT, EvidenceRelation.RELATED))

    @property
    def generic_requirements(self) -> tuple[EvidenceItem, ...]:
        return tuple(i for i in self.items if i.relation is EvidenceRelation.GENERIC_REQUIREMENT)

    @property
    def selected(self) -> tuple[EvidenceItem, ...]:
        return tuple(
            i for i in self.items
            if i.relation in (
                EvidenceRelation.DIRECT,
                EvidenceRelation.RELATED,
                EvidenceRelation.GENERIC_REQUIREMENT,
                EvidenceRelation.AMBIGUOUS_OR_CONFLICTING,
            )
        )

    @property
    def citations(self) -> tuple[CitationBundle, ...]:
        # Deduplicate citations by (document_name, page_number) while preserving distinct sources
        seen: set[tuple[str, int | None]] = set()
        cits: list[CitationBundle] = []
        for i in self.selected:
            c = i.citation
            key = (c.document_name or c.document_id, c.page_number)
            if key not in seen:
                seen.add(key)
                cits.append(c)
        if not cits and self.selected:
            cits = [i.citation for i in self.selected]
        return tuple(cits)

    def context(self) -> str:
        return "\n\n".join(f"{i.citation.citation_id} {i.text}" for i in self.selected)

    def report(self, title: str = "Grounded Evidence Report") -> StructuredReport:
        if not self.is_scope_established or not self.selected:
            return StructuredReport(
                title=title,
                summary="The available documentation is insufficient to establish the scope for the requested report.",
                sections=(),
                citations=(),
                claims=(),
                is_grounded=True,
                missing_information=True,
                scope_established=False,
            )

        claims = tuple(
            ReportClaim(
                text=i.text,
                citation_id=i.citation.citation_id,
                document_id=i.citation.document_id,
                document_name=i.citation.document_name or i.citation.document_id,
                page_number=i.citation.page_number,
                chunk_id=i.citation.chunk_id,
                entities=i.entities,
                relation=i.relation,
                source_quote=i.citation.verbatim_quote,
            )
            for i in self.selected
        )

        sections: list[dict] = []

        # 1. Direct & Related Specific Findings
        direct_claims = [c for c in claims if c.relation in (EvidenceRelation.DIRECT, EvidenceRelation.RELATED)]
        if direct_claims:
            grouped_direct: dict[str, list[str]] = {}
            for claim in direct_claims:
                label = claim.document_name or "Documented Findings"
                page = f" (page {claim.page_number})" if claim.page_number else ""
                grouped_direct.setdefault(label, []).append(f"{claim.text} {claim.citation_id}{page}")
            for doc_name, values in grouped_direct.items():
                sections.append({"heading": f"Documented Findings — {doc_name}", "bullets": values})

        # 2. General Regulatory Requirements & Standards (strictly separated from entity findings)
        generic_claims = [c for c in claims if c.relation is EvidenceRelation.GENERIC_REQUIREMENT]
        if generic_claims:
            grouped_generic: dict[str, list[str]] = {}
            for claim in generic_claims:
                label = claim.document_name or "Standard Requirements"
                page = f" (page {claim.page_number})" if claim.page_number else ""
                grouped_generic.setdefault(label, []).append(f"{claim.text} {claim.citation_id}{page}")
            for doc_name, values in grouped_generic.items():
                sections.append({"heading": f"Documented General Requirements — {doc_name}", "bullets": values})

        # 3. Documentation Conflicts (if any)
        if self.conflicts:
            sections.append({"heading": "Documentation Inconsistencies & Conflicts", "bullets": list(self.conflicts)})

        # 4. Scope and Missing Information Notice
        sections.append({
            "heading": "Documented Scope & Verification Limitations",
            "bullets": [
                "Only facts explicitly stated in the listed source documents are presented.",
                "Parameters, conditions, or dates not explicitly present in the retrieved evidence are not documented.",
                "No speculative conclusions, unrecorded maintenance actions, or synthetic dates have been added.",
            ],
        })

        summary = (
            "Supported evidence is structured verbatim from verified sources. "
            "All documented findings and general requirements are distinguished with full provenance."
        )

        return StructuredReport(
            title=title,
            summary=summary,
            sections=tuple(sections),
            citations=self.citations,
            claims=claims,
            is_grounded=True,
            missing_information=False,
            scope_established=True,
            conclusion="All reported data reflects the verified contents of the available source documentation without extrapolation.",
        )


class EvidenceSelector:
    """Enforces explicit entity/document scope before any model sees evidence."""

    def select(
        self,
        query: str,
        candidates: Sequence[ScoredRetrievalChunk],
        citations: Sequence[CitationBundle],
        is_artifact_request: bool = False,
        exact_entity_only: bool = False,
        requested_source_names: Sequence[str] = (),
    ) -> EvidencePackage:
        by_chunk = {c.chunk_id: c for c in citations}
        requested = _identifiers(query)
        requested_sources = {
            re.sub(r"\s+", " ", name).strip().casefold()
            for name in requested_source_names if name and "." in name
        }
        q_terms = _terms(query)
        seen: set[tuple[str, str, int | None]] = set()
        items: list[EvidenceItem] = []

        # If query is an artifact request without any domain terms or identifiers, scope cannot be established
        if is_artifact_request and not requested and not q_terms:
            return EvidencePackage(
                query=query,
                requested_entities=(),
                items=(),
                conflicts=(),
                is_scope_established=False,
            )

        for candidate in candidates:
            meta = candidate.chunk.metadata
            source = candidate.chunk.content.strip()
            if not source:
                continue

            citation = by_chunk.get(candidate.chunk.chunk_id) or CitationBundle(
                citation_id=f"[{len(items)+1}]",
                document_id=meta.document_id,
                document_name=meta.document_name,
                source_path=meta.source_path,
                page_number=meta.page_number,
                section_title=meta.section_title,
                chunk_id=candidate.chunk.chunk_id,
                verbatim_quote=source,
                score=candidate.score,
                equipment_tags=list(meta.equipment_entities),
                sha256=meta.sha256,
            )

            # An explicitly named file is an exact provenance constraint.  A
            # semantically similar report must never stand in for it.
            document_name = (citation.document_name or meta.document_name or "").casefold()
            if requested_sources and document_name not in requested_sources:
                items.append(EvidenceItem(
                    EvidenceRelation.UNRELATED, candidate, citation, (),
                    "candidate does not match explicitly requested source filename",
                ))
                continue

            key = (citation.document_id, re.sub(r"\s+", " ", source).casefold(), citation.page_number)
            if key in seen:
                continue
            seen.add(key)

            entities = set(_identifiers(source)) | {e.casefold() for e in meta.equipment_entities}
            source_terms = _terms(source)

            has_normative_keywords = bool(source_terms.intersection(_REGULATORY_NORMATIVE_WORDS)) or bool(re.search(r"\b(shall|must|shall be|mandatory|provisions?|guidelines?|standards?|requirements?)\b", source, re.IGNORECASE))
            has_specific_record = bool(re.search(r"\b(inspected on \d|\d+\.\d+\s*(?:mm|psi|bar|c|kg/cm2)|zero defects|thickness measured|observed reading)\b", source, re.IGNORECASE))
            is_normative = has_normative_keywords and not has_specific_record

            overlap = _term_overlap(q_terms, source_terms)

            foreign = bool(requested and entities and not requested.intersection(entities))

            if foreign:
                relation = EvidenceRelation.UNRELATED
                reason = "explicit query identifier does not match candidate entity provenance"
            elif requested and requested.intersection(entities):
                if is_normative:
                    relation = EvidenceRelation.GENERIC_REQUIREMENT
                    reason = "explicit query identifier matches candidate standard/rule"
                elif overlap >= 0.20 or candidate.score >= 0.70:
                    relation = EvidenceRelation.DIRECT
                    reason = "explicit query identifier matches candidate entity"
                else:
                    relation = EvidenceRelation.RELATED
                    reason = "explicit query identifier matches entity with low semantic overlap"
            elif requested and not entities:
                if exact_entity_only:
                    relation = EvidenceRelation.UNRELATED
                    reason = "exact entity mode rejects evidence without matching entity provenance"
                elif is_normative:
                    if overlap >= 0.20 or candidate.score >= 0.65:
                        relation = EvidenceRelation.GENERIC_REQUIREMENT
                        reason = "general normative standard matches query topic"
                    else:
                        relation = EvidenceRelation.UNRELATED
                        reason = "insufficient semantic overlap for general standard"
                else:
                    if overlap >= 0.30 or candidate.score >= 0.70:
                        relation = EvidenceRelation.RELATED
                        reason = "general candidate findings match query topic"
                    else:
                        relation = EvidenceRelation.UNRELATED
                        reason = "insufficient semantic and provenance overlap"
            elif not requested and q_terms:
                if is_normative:
                    if overlap >= 0.30 or candidate.score >= 0.65:
                        relation = EvidenceRelation.GENERIC_REQUIREMENT
                        reason = "general normative standard matches query topic"
                    else:
                        relation = EvidenceRelation.UNRELATED
                        reason = "insufficient semantic overlap for general standard"
                else:
                    if overlap >= 0.35 or candidate.score >= 0.70:
                        relation = EvidenceRelation.DIRECT
                        reason = "substantive query terms match candidate findings"
                    elif overlap >= 0.20 or candidate.score >= 0.50:
                        relation = EvidenceRelation.RELATED
                        reason = "partial query terms match candidate findings"
                    else:
                        relation = EvidenceRelation.UNRELATED
                        reason = "insufficient semantic and provenance overlap"
            else:
                relation = EvidenceRelation.UNRELATED
                reason = "unscoped query with no matching entity or substantive terms"

            items.append(EvidenceItem(relation, candidate, citation, tuple(sorted(entities)), reason, is_normative))

        # Check if scope was established
        has_direct_or_related = any(i.relation in (EvidenceRelation.DIRECT, EvidenceRelation.RELATED) for i in items)
        has_generic = any(i.relation is EvidenceRelation.GENERIC_REQUIREMENT for i in items)

        # For an artifact generation request without explicit entity: if no direct/related findings exist
        # and no matching generic standards/requirements exist, reject scope creation to prevent hallucinated reports.
        is_scope_ok = True
        if is_artifact_request and not requested and not has_direct_or_related and not has_generic:
            is_scope_ok = False

        # If the request named a file, successful evidence selection requires
        # evidence from that file.  This turns "refer to X.pdf" into a hard
        # execution requirement rather than a prompt-only suggestion.
        if requested_sources and not has_direct_or_related and not has_generic:
            is_scope_ok = False

        return EvidencePackage(
            query=query,
            requested_entities=tuple(sorted(requested)),
            items=tuple(items),
            conflicts=tuple(self._conflicts(items)),
            is_scope_established=is_scope_ok and (has_direct_or_related or has_generic),
        )

    @staticmethod
    def _conflicts(items: Iterable[EvidenceItem]) -> list[str]:
        values: dict[tuple[str, str], set[tuple[str, str]]] = {}
        for item in items:
            if item.relation not in (EvidenceRelation.DIRECT, EvidenceRelation.RELATED):
                continue
            for line in re.split(r"(?<=[.!?])\s+|\n+", item.text):
                match = re.match(r"\s*([A-Za-z][A-Za-z /_-]{2,50})\s*[:=-]\s*(.+)", line)
                if match:
                    entity = ",".join(item.entities) or item.citation.document_id
                    field_name = match.group(1).casefold().strip()
                    value = match.group(2).strip()
                    values.setdefault((entity, field_name), set()).add((value, item.citation.citation_id))
        return [
            f"Available documentation contains conflicting values for '{field_name}': "
            + "; ".join(f"{v} ({c})" for v, c in sorted(found))
            for (_, field_name), found in values.items()
            if len({v.casefold() for v, _ in found}) > 1
        ]
