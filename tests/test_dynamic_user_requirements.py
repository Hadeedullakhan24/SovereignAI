"""Generic user-requirement propagation tests."""

from rag_engine.generation.prompt.task_intent import TaskIntentClassifier
from rag_engine.pipeline.rag_pipeline import RAGPipeline


def test_email_keeps_arbitrary_follow_on_clauses() -> None:
    intent = TaskIntentClassifier.classify(
        "Draft an email for Friday. State that I have arranged alternate coverage. Keep it concise."
    )
    assert intent.user_requirements == [
        "State that I have arranged alternate coverage.", "Keep it concise."
    ]
    assert "USER_PROVIDED_INFORMATION" in intent.get_directive_instructions()


def test_report_keeps_non_template_requirements() -> None:
    intent = TaskIntentClassifier.classify(
        "Generate a report on the work order. Include a section for assumptions and use a one-paragraph executive summary."
    )
    assert len(intent.user_requirements) == 1
    assert "assumptions" in intent.user_requirements[0]
    assert "executive summary" in intent.user_requirements[0]


def test_user_context_is_not_a_citable_document_source() -> None:
    intent = TaskIntentClassifier.classify(
        "Draft a maintenance request. The operator has isolated the area."
    )
    candidates, context = RAGPipeline._user_provided_context(intent)
    assert candidates and "USER_PROVIDED_INFORMATION" in context
    assert "SOURCE_GROUNDED_INFORMATION" in context
    assert candidates[0].chunk.metadata.document_name == "USER_PROVIDED_INFORMATION"


def test_multiple_requirements_and_generic_conflict_detection() -> None:
    intent = TaskIntentClassifier.classify(
        "Draft an email. Include a table. Do not include a table."
    )
    assert len(intent.user_requirements) == 2
    assert intent.has_conflicting_requirements
