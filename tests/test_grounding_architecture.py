"""Generic regression tests for evidence scope, strict grounding boundary, and canonical artifacts.

All tests use synthetic fixtures and verify architecture-level guarantees.
"""
from pathlib import Path
import pytest

from agent.tool_executor import (
    DocumentGenerator,
    NativePDFGenerator,
    PresentationGenerator,
    SandboxedFileManager,
    SpreadsheetGenerator,
    ToolExecutor,
    validate_artifact,
)
from rag_engine.grounding.evidence import (
    EvidencePackage,
    EvidenceRelation,
    EvidenceSelector,
    ReportClaim,
    StructuredReport,
)
from rag_engine.retrieval.base_retriever import CitationBundle, ScoredRetrievalChunk
from rag_engine.schemas.chunk import Chunk, ChunkMetadata


def _candidate(identifier: str, text: str, document: str, page: int = 1, score: float = 0.9) -> ScoredRetrievalChunk:
    chunk = Chunk(
        chunk_id=f"chunk-{identifier}-{document}-{page}",
        content=text,
        token_count=len(text.split()),
        word_count=len(text.split()),
        character_count=len(text),
        metadata=ChunkMetadata(
            document_id=document,
            document_name=document,
            page_number=page,
            equipment_entities=[identifier] if identifier else [],
        ),
    )
    return ScoredRetrievalChunk(chunk=chunk, score=score, rank=0)


def _citation(candidate: ScoredRetrievalChunk) -> CitationBundle:
    m = candidate.chunk.metadata
    return CitationBundle(
        citation_id="[1]",
        document_id=m.document_id,
        document_name=m.document_name,
        page_number=m.page_number,
        chunk_id=candidate.chunk.chunk_id,
        verbatim_quote=candidate.chunk.content,
        score=candidate.score,
        equipment_tags=m.equipment_entities,
    )


# 1. Unsupported generated facts cannot enter StructuredReport
def test_unsupported_facts_cannot_enter_structured_report():
    unsupported_claim = ReportClaim(
        text="The pump operates at 450 PSI.",
        citation_id="",  # Missing citation ID
        document_id="",
        document_name="",  # Missing provenance
        page_number=None,
        chunk_id="",
        entities=("ASSET-101",),
        relation=EvidenceRelation.DIRECT,
    )
    report = StructuredReport(
        title="Asset Report",
        summary="Summary of assets",
        sections=(),
        citations=(),
        claims=(unsupported_claim,),
        scope_established=True,
    )
    is_valid, errors = report.validate()
    assert is_valid is False
    assert any("provenance" in err.lower() or "citation" in err.lower() for err in errors)


# 2. Generic requirements remain generic and separate from findings
def test_generic_requirements_remain_generic_and_separate():
    generic_chunk = _candidate("", "All pressure safety valves shall be inspected annually per standard procedure.", "safety_standard_doc.md")
    direct_chunk = _candidate("VALVE-44", "VALVE-44: inspected on 2026-03-01 with zero defects.", "inspection_log.md")

    package = EvidenceSelector().select(
        "Report findings and standards for VALVE-44",
        [generic_chunk, direct_chunk],
        [_citation(generic_chunk), _citation(direct_chunk)],
    )

    report = package.report("Valve Inspection Report")
    payload = report.renderer_payload()

    # Generic requirements must not be represented as direct equipment inspection records
    sections = {s["heading"]: s for s in payload["sections"]}
    assert any("Documented Findings" in h for h in sections)
    assert any("General Requirements" in h for h in sections)

    findings_sec = next(s for h, s in sections.items() if "Documented Findings" in h)
    reqs_sec = next(s for h, s in sections.items() if "General Requirements" in h)

    assert "VALVE-44" in str(findings_sec)
    assert "zero defects" in str(findings_sec)
    assert "inspected annually per standard procedure" in str(reqs_sec)


# 3. Missing values remain missing
def test_missing_values_remain_missing():
    # Empty evidence report
    empty_report = StructuredReport(
        title="Unscoped Report",
        summary="Not documented in the available evidence.",
        sections=(),
        citations=(),
        claims=(),
        missing_information=True,
        scope_established=False,
    )
    is_valid, errors = empty_report.validate()
    assert is_valid is False
    payload = empty_report.renderer_payload()
    assert "Not documented in the available evidence." in payload["summary"]


# 4. Placeholder dates and values are rejected
def test_placeholder_dates_and_values_are_rejected():
    claim_with_placeholder = ReportClaim(
        text="Inspection date: 20XX-XX-XX for equipment ASSET-55",
        citation_id="[1]",
        document_id="doc.md",
        document_name="doc.md",
        page_number=1,
        chunk_id="chk-1",
        entities=("ASSET-55",),
        relation=EvidenceRelation.DIRECT,
        source_quote="Inspection date: 20XX-XX-XX for equipment ASSET-55",
    )
    assert claim_with_placeholder.has_placeholders() is True

    report = StructuredReport(
        title="Report",
        summary="Summary",
        sections=(),
        citations=(),
        claims=(claim_with_placeholder,),
        scope_established=True,
    )
    is_valid, errors = report.validate()
    assert is_valid is False
    assert any("placeholder" in err.lower() for err in errors)


# 5. Source citation without claim-level support is rejected
def test_source_citation_without_claim_support_is_rejected():
    claim_without_support = ReportClaim(
        text="Fabricated finding with missing citation ID.",
        citation_id="",  # Missing citation ID
        document_id="",
        document_name="",
        page_number=1,
        chunk_id="chk-valid",
        entities=("PUMP-01",),
        relation=EvidenceRelation.DIRECT,
        source_quote="",
    )
    report = StructuredReport(
        title="Report",
        summary="Summary",
        sections=(),
        citations=(),
        claims=(claim_without_support,),
        scope_established=True,
    )
    is_valid, errors = report.validate()
    assert is_valid is False


# 6. Artifact cannot be generated from ungrounded report
def test_artifact_cannot_be_generated_from_ungrounded_report(tmp_path: Path):
    executor = ToolExecutor(sandbox_dir=tmp_path)

    invalid_report = StructuredReport(
        title="Invalid Report",
        summary="Invalid Summary",
        sections=(),
        citations=(),
        claims=(
            ReportClaim(
                text="Unsupported claim",
                citation_id="",
                document_id="",
                document_name="",
                page_number=None,
                chunk_id="",
                entities=(),
                relation=EvidenceRelation.UNRELATED,
            ),
        ),
        scope_established=False,
    )

    res = executor.execute(
        "pdf_generator",
        task="Create PDF report",
        structured_report=invalid_report,
    )
    assert res.status == "insufficient_evidence"
    assert res.is_verified is False
    assert res.output is None


# 7. Agent document generation requests cannot bypass grounding
def test_agent_document_generation_cannot_bypass_grounding(tmp_path: Path):
    executor = ToolExecutor(sandbox_dir=tmp_path)
    res = executor.execute(
        "document_generator",
        task="Generate an actual DOCX report from the available documentation",
        use_rag_context=True,
    )
    assert res.status == "insufficient_evidence"
    assert res.is_verified is False


# 8. All four artifact formats receive identical canonical factual content
def test_all_four_artifact_formats_receive_identical_canonical_content(tmp_path: Path):
    candidate = _candidate("VESSEL-88", "VESSEL-88: wall thickness measured at 14.2 mm.", "integrity_log.md", page=3)
    package = EvidenceSelector().select(
        "Report integrity for VESSEL-88",
        [candidate],
        [_citation(candidate)],
        is_artifact_request=True,
    )
    report = package.report("Vessel Integrity Report")
    is_valid, errors = report.validate()
    assert is_valid is True, f"Validation errors: {errors}"

    payload = report.renderer_payload()
    manager = SandboxedFileManager(tmp_path)

    docx_out = DocumentGenerator(manager).generate_report(filename="report.docx", **payload)
    xlsx_out = SpreadsheetGenerator(manager).generate(filename="report.xlsx", **payload)
    pptx_out = PresentationGenerator(manager).generate(filename="report.pptx", **payload)
    pdf_out = NativePDFGenerator(manager).generate(filename="report.pdf", **payload)

    for out, fmt in [(docx_out, "docx"), (xlsx_out, "xlsx"), (pptx_out, "pptx"), (pdf_out, "pdf")]:
        val = validate_artifact(out["path"], fmt, required_sources=payload["citations"])
        assert val["validated"] is True


# 9. Arbitrary entities are handled dynamically
def test_arbitrary_entities_are_handled_dynamically():
    entity_a = _candidate("GAMMA-900", "GAMMA-900: status operational.", "doc_a.md")
    entity_b = _candidate("ZETA-100", "ZETA-100: status standby.", "doc_b.md")

    # Query specifically for GAMMA-900
    pkg_a = EvidenceSelector().select("Inspect GAMMA-900", [entity_a, entity_b], [_citation(entity_a), _citation(entity_b)])
    assert len(pkg_a.selected) == 1
    assert "gamma-900" in pkg_a.selected[0].entities

    # Query specifically for ZETA-100
    pkg_b = EvidenceSelector().select("Inspect ZETA-100", [entity_a, entity_b], [_citation(entity_a), _citation(entity_b)])
    assert len(pkg_b.selected) == 1
    assert "zeta-100" in pkg_b.selected[0].entities


# 10. Unrelated retrieved documents cannot contaminate the report
def test_unrelated_documents_cannot_contaminate_report():
    relevant = _candidate("COOLER-12", "COOLER-12: cooling water outlet temperature 38 C.", "unit_ops.md")
    unrelated = _candidate("HEATER-99", "HEATER-99: fuel gas pressure 4.2 kg/cm2.", "unrelated_furnace.md")

    package = EvidenceSelector().select(
        "What is the cooling water temperature for COOLER-12?",
        [relevant, unrelated],
        [_citation(relevant), _citation(unrelated)],
    )

    assert len(package.selected) == 1
    assert package.selected[0].chunk.chunk.content == relevant.chunk.content

    report = package.report("Cooler Operational Report")
    payload = report.renderer_payload()
    rendered_text = str(payload)
    assert "COOLER-12" in rendered_text
    assert "HEATER-99" not in rendered_text
