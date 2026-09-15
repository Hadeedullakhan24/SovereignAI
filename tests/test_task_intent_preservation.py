"""Comprehensive Test Suite for User Intent, Output Type, and Task-Purpose Preservation.

Verifies:
1. Classification of User Intent before generation (summarize, answer, compare, extract, explain,
   list, calculate, draft email, draft approval request, draft notification, draft action request,
   generate report, generate procedure, generate checklist).
2. Email purpose preservation (summary email vs approval request vs confirmation request vs
   notification vs action request).
3. Salutation and Recipient preservation ("for the site safety team" -> "Dear Site Safety Team,").
4. Entity Scope preservation (general question does not become entity-specific, no unrequested E-330).
5. Subject line matches user intent and operation.
6. Strict grounding mode detection ("use only provided documentation").
7. AnswerAlignmentValidator sanitization of hijacked subjects/salutations/tones.
8. DeterministicTestLLM and RAGPipeline execution on live email draft test queries.
"""

from __future__ import annotations

import unittest

from rag_engine.generation.guardrails.answer_alignment_validator import AnswerAlignmentValidator
from rag_engine.generation.models.deterministic_test_llm import DeterministicTestLLM
from rag_engine.generation.prompt.prompt_builder import PromptBuilder
from rag_engine.generation.prompt.prompt_templates import PromptArchetype
from rag_engine.generation.prompt.task_intent import (
    EmailPurpose,
    OutputFormat,
    TaskIntentClassifier,
    TaskOperation,
)
from rag_engine.retrieval.base_retriever import CitationBundle, ScoredRetrievalChunk
from rag_engine.schemas.chunk import Chunk, ChunkHierarchy, ChunkMetadata


class TestTaskIntentPreservation(unittest.TestCase):
    """Test suite for TaskIntentClassifier and Intent Preservation across generation."""

    def test_classify_summary_email_with_recipient_and_strict_grounding(self) -> None:
        """User: 'Draft an email summarizing the safety requirements for hot work for the site safety team. Use only the provided documentation.'"""
        q = "Draft an email summarizing the safety requirements for hot work for the site safety team. Use only the provided documentation."
        intent = TaskIntentClassifier.classify(q)

        self.assertEqual(intent.operation, TaskOperation.SUMMARIZE)
        self.assertEqual(intent.output_format, OutputFormat.EMAIL)
        self.assertEqual(intent.email_purpose, EmailPurpose.SUMMARY)
        self.assertEqual(intent.recipient, "Site Safety Team")
        self.assertTrue(intent.is_general_query)
        self.assertEqual(len(intent.explicit_entities), 0)
        self.assertTrue(intent.strict_grounding)
        self.assertEqual(intent.archetype, PromptArchetype.EMAIL)

        directives = intent.get_directive_instructions()
        self.assertIn("TASK OBJECTIVE: DRAFT A PROFESSIONAL EMAIL", directives)
        self.assertIn("Dear Site Safety Team,", directives)
        self.assertIn("Summary", directives)
        self.assertIn("STRICT GROUNDING MODE ACTIVE", directives)
        self.assertIn("SCOPE DISCIPLINE: The user asked a general question", directives)

    def test_classify_approval_request_email(self) -> None:
        """User: 'Draft an email requesting approval for hot work in Storage Area to the Plant Manager.'"""
        q = "Draft an email requesting approval for hot work in Storage Area to the Plant Manager."
        intent = TaskIntentClassifier.classify(q)

        self.assertEqual(intent.operation, TaskOperation.DRAFT_APPROVAL_REQUEST)
        self.assertEqual(intent.output_format, OutputFormat.EMAIL)
        self.assertEqual(intent.email_purpose, EmailPurpose.APPROVAL_REQUEST)
        self.assertEqual(intent.recipient, "Plant Manager")

    def test_classify_confirmation_request_email(self) -> None:
        """User: 'Draft an email requesting confirmation of the hot work permit status to the Operations Team.'"""
        q = "Draft an email requesting confirmation of the hot work permit status to the Operations Team."
        intent = TaskIntentClassifier.classify(q)

        self.assertEqual(intent.output_format, OutputFormat.EMAIL)
        self.assertEqual(intent.email_purpose, EmailPurpose.CONFIRMATION_REQUEST)
        self.assertEqual(intent.recipient, "Operations Team")

    def test_classify_notification_email(self) -> None:
        """User: 'Draft an email notifying the team about the upcoming maintenance schedule.'"""
        q = "Draft an email notifying the team about the upcoming maintenance schedule."
        intent = TaskIntentClassifier.classify(q)

        self.assertEqual(intent.operation, TaskOperation.DRAFT_NOTIFICATION)
        self.assertEqual(intent.output_format, OutputFormat.EMAIL)
        self.assertEqual(intent.email_purpose, EmailPurpose.NOTIFICATION)
        self.assertEqual(intent.recipient, "Team")

    def test_classify_action_request_email(self) -> None:
        """User: 'Draft an email asking the team to take action on hot work isolation valves.'"""
        q = "Draft an email asking the team to take action on hot work isolation valves."
        intent = TaskIntentClassifier.classify(q)

        self.assertEqual(intent.operation, TaskOperation.DRAFT_ACTION_REQUEST)
        self.assertEqual(intent.output_format, OutputFormat.EMAIL)
        self.assertEqual(intent.email_purpose, EmailPurpose.ACTION_REQUEST)

    def test_classify_report_generation(self) -> None:
        """User: 'Generate a report on CDU-1 inspection findings.'"""
        q = "Generate a report on CDU-1 inspection findings."
        intent = TaskIntentClassifier.classify(q)

        self.assertEqual(intent.operation, TaskOperation.GENERATE_REPORT)
        self.assertEqual(intent.output_format, OutputFormat.REPORT)
        self.assertEqual(intent.archetype, PromptArchetype.REPORT)

    def test_classify_procedure_generation(self) -> None:
        """User: 'Generate a procedure for pump isolation and LOTO.'"""
        q = "Generate a procedure for pump isolation and LOTO."
        intent = TaskIntentClassifier.classify(q)

        self.assertEqual(intent.operation, TaskOperation.GENERATE_PROCEDURE)
        self.assertEqual(intent.output_format, OutputFormat.PROCEDURE)
        self.assertEqual(intent.archetype, PromptArchetype.SOP_RETRIEVAL)

    def test_classify_checklist_generation(self) -> None:
        """User: 'Generate a checklist for hot work safety precautions.'"""
        q = "Generate a checklist for hot work safety precautions."
        intent = TaskIntentClassifier.classify(q)

        self.assertEqual(intent.operation, TaskOperation.GENERATE_CHECKLIST)
        self.assertEqual(intent.output_format, OutputFormat.CHECKLIST)

    def test_entity_extraction_general_vs_specific(self) -> None:
        """Verify explicit equipment tags vs general queries."""
        general_q = "Summarize the safety requirements for hot work."
        intent_gen = TaskIntentClassifier.classify(general_q)
        self.assertTrue(intent_gen.is_general_query)
        self.assertEqual(len(intent_gen.explicit_entities), 0)

        specific_q = "Summarize the safety requirements for hot work on E-330."
        intent_spec = TaskIntentClassifier.classify(specific_q)
        self.assertFalse(intent_spec.is_general_query)
        self.assertIn("E-330", intent_spec.explicit_entities)

    def test_prompt_builder_injects_intent_directives(self) -> None:
        """Verify PromptBuilder dynamically includes TaskIntent instructions."""
        builder = PromptBuilder()
        q = "Draft an email summarizing the safety requirements for hot work for the site safety team. Use only the provided documentation."
        prompt = builder.build_prompt(
            query=q,
            candidates=[],
            citations=[],
        )

        text = prompt.prompt_text
        self.assertIn("TASK OBJECTIVE: DRAFT A PROFESSIONAL EMAIL", text)
        self.assertIn("Dear Site Safety Team,", text)
        self.assertIn("STRICT GROUNDING MODE ACTIVE", text)
        self.assertIn("DO NOT ask the recipient to confirm, approve", text)

    def test_answer_alignment_validator_sanitizes_hijacked_email_intent(self) -> None:
        """Test AnswerAlignmentValidator corrects hijacked email subject, recipient, and demands."""
        validator = AnswerAlignmentValidator()
        query = "Draft an email summarizing the safety requirements for hot work for the site safety team. Use only the provided documentation."

        bad_generated_email = (
            "Subject: Request for Confirmation of Hot Work Permit for E-330\n\n"
            "Dear Maintenance Manager,\n\n"
            "I am writing to request your immediate attention to the ongoing safety review for E-330 in Storage Area. "
            "As per our recent team review, documentation is needed.\n\n"
            "Specifically, we require confirmation of the following:\n"
            "• Applicable Personal Protective Equipment (PPE)\n"
            "• Isolation of valves\n\n"
            "Best regards,\n"
            "Engineering Team"
        )

        report = validator.evaluate(
            query=query,
            answer_text=bad_generated_email,
            source_context="Hot work requires PPE, isolation of valves, and gas testing.",
        )

        cleaned = report.cleaned_text
        self.assertIn("Dear Site Safety Team,", cleaned)
        self.assertNotIn("Dear Maintenance Manager,", cleaned)
        self.assertNotIn("Request for Confirmation", cleaned)
        self.assertIn("Subject:", cleaned)
        self.assertNotIn("for E-330", cleaned.splitlines()[0])  # Subject line doesn't have unrequested E-330
        self.assertNotIn("I am writing to request your immediate attention", cleaned)

    def test_deterministic_llm_email_generation(self) -> None:
        """Verify DeterministicTestLLM outputs properly formatted summary email without E-330 or confirmation requests."""
        model = DeterministicTestLLM()
        prompt = (
            "=== SYSTEM INSTRUCTIONS ===\nYou are MRPL Sovereign AI Assistant.\n"
            "=== USER QUERY ===\nDraft an email summarizing the safety requirements for hot work for the site safety team. Use only the provided documentation.\n"
            "=== VERIFIED CONTEXT ===\n"
            "--- SOURCE [1] Source: E294_Hot_Work.pdf ---\n"
            "Hot work activities require mandatory gas testing for combustible gases prior to ignition.\n"
            "Personnel must wear flame-retardant PPE and safety goggles during hot work operations.\n"
            "Positive isolation and Lockout/Tagout (LOTO) must be verified on all connected process lines.\n"
            "=== INSTRUCTIONS FOR EMAIL DRAFTING ===\nASSISTANT: "
        )

        gen_out = model.generate(prompt)
        text = gen_out.text

        self.assertIn("Subject:", text)
        self.assertIn("Summary", text)
        self.assertIn("Dear Site Safety Team,", text)
        self.assertIn("Best regards,", text)
        self.assertNotIn("[1]", text)  # No bracket citations in email
        self.assertNotIn("E-330", text)
        self.assertNotIn("Maintenance Manager", text)
        self.assertNotIn("Request for Confirmation", text)


if __name__ == "__main__":
    unittest.main()
