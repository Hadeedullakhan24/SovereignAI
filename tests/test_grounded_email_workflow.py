"""Regression tests for grounded, source-aware email drafting."""

from rag_engine.generation.prompt.task_intent import TaskIntentClassifier
from rag_engine.grounding.evidence import EvidenceSelector
from rag_engine.pipeline.rag_pipeline import RAGPipeline
from rag_engine.retrieval.base_retriever import CitationBundle, ScoredRetrievalChunk
from rag_engine.schemas.chunk import Chunk


def _candidate(tag: str, text: str, document_name: str) -> ScoredRetrievalChunk:
    chunk = Chunk.create(
        document_id=document_name,
        content=text,
        chunk_index=0,
        document_name=document_name,
        equipment_entities=[tag],
    )
    return ScoredRetrievalChunk(chunk=chunk, score=0.95, rank=0)


def _citation(candidate: ScoredRetrievalChunk) -> CitationBundle:
    meta = candidate.chunk.metadata
    return CitationBundle(
        citation_id="[1]", document_id=meta.document_id, document_name=meta.document_name,
        source_path=meta.source_path, page_number=meta.page_number,
        section_title=meta.section_title, chunk_id=candidate.chunk.chunk_id,
        verbatim_quote=candidate.chunk.content, score=candidate.score,
        equipment_tags=meta.equipment_entities,
    )


def test_email_intent_marks_requested_pdf_and_user_point() -> None:
    intent = TaskIntentClassifier.classify(
        "Draft an email to the maintenance team using inspection.pdf as the reference "
        "and add the specific point I mentioned about following up with the maintenance team."
    )
    assert intent.requires_source_evidence
    assert intent.requested_source_names == ["inspection.pdf"]
    assert intent.explicit_email_instructions == ["following up with the maintenance team"]


def test_exact_email_entity_scope_rejects_similar_equipment() -> None:
    target = _candidate("H-051", "H-051 has a documented shell observation.", "inspection.pdf")
    similar = _candidate("H-005", "H-005 requires a repair.", "inspection.pdf")
    package = EvidenceSelector().select(
        "Draft an email about H-051 using inspection documentation.",
        [target, similar], [_citation(target), _citation(similar)], exact_entity_only=True,
    )
    assert [item.chunk.chunk.content for item in package.selected] == [target.chunk.content]


def test_named_document_is_a_hard_provenance_constraint() -> None:
    requested = _candidate("H-051", "H-051 observation from the requested report.", "inspection.pdf")
    other = _candidate("H-051", "H-051 observation from another report.", "other.pdf")
    package = EvidenceSelector().select(
        "Draft an email about H-051 using inspection.pdf.",
        [requested, other], [_citation(requested), _citation(other)],
        exact_entity_only=True, requested_source_names=["inspection.pdf"],
    )
    assert [item.citation.document_name for item in package.selected] == ["inspection.pdf"]


def test_user_email_point_is_preserved_without_claiming_document_support() -> None:
    intent = TaskIntentClassifier.classify(
        "Draft an email using the inspection PDF and add the specific point about following up with maintenance."
    )
    rendered = RAGPipeline._preserve_user_email_points(
        "Subject: Inspection Update\n\nDear Maintenance Team,\n\nDocumented finding.\n\nBest regards,\n[Engineering Team]",
        intent,
    )
    assert "Additionally, following up with maintenance." in rendered
    assert rendered.index("Additionally") < rendered.index("Best regards")
