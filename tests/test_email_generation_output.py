"""Comprehensive Test Suite for Email Generation Output Integrity.

Verifies:
1. ResponseFormatter.sanitize_email_output strips all citations, document IDs,
   page numbers, chunk headers, preambles, postambles, and References/Sources blocks.
2. GenerationPipeline.generate produces clean email bodies with zero provenance leak.
3. RAG system internally maintains citations, grounding reports, and confidence scores.
4. RAGResponse.format_clean_cli_output outputs strictly the email without ### Sources.
5. RAGResponse.format_cli_output(detailed=True) cleanly isolates user email in Section 1
   while preserving engineering audit provenance in Section 2.
6. Works generically across diverse email types (Summary, Approval, Action, Notification)
   without hardcoding any equipment, document, or example.
"""

from __future__ import annotations

import unittest

from rag_engine.generation.generation_config import GenerationConfig
from rag_engine.generation.generation_pipeline import GenerationPipeline
from rag_engine.generation.guardrails.answer_alignment_validator import AnswerAlignmentValidator
from rag_engine.generation.models.deterministic_test_llm import DeterministicTestLLM
from rag_engine.generation.prompt.prompt_templates import PromptArchetype
from rag_engine.generation.prompt.task_intent import (
    EmailPurpose,
    OutputFormat,
    TaskIntentClassifier,
)
from rag_engine.generation.response_formatter import ResponseFormatter
from rag_engine.pipeline.rag_pipeline import RAGResponse
from rag_engine.retrieval.base_retriever import CitationBundle, RetrievalResult, ScoredRetrievalChunk
from rag_engine.schemas.chunk import Chunk, ChunkHierarchy, ChunkMetadata


class TestEmailGenerationOutput(unittest.TestCase):
    """Test suite for verifiable email generation output isolation."""

    def setUp(self) -> None:
        self.doc_name = "E294_Hot_Work_Standard.pdf"
        self.citations = [
            CitationBundle(
                chunk_id="chk-01",
                document_id="DOC-HW-001",
                document_name=self.doc_name,
                page_number=3,
                section_title="Gas Testing & PPE",
                equipment_tag="E-330",
                verbatim_quote="Gas testing for combustible gases must be performed prior to ignition.",
                citation_id="[1]",
            ),
            CitationBundle(
                chunk_id="chk-02",
                document_id="DOC-HW-002",
                document_name="Refinery_Safety_Manual.docx",
                page_number=7,
                section_title="Isolation Boundaries",
                equipment_tag="E-330",
                verbatim_quote="Positive isolation and Lockout/Tagout (LOTO) must be verified on all connected process lines.",
                citation_id="[2]",
            ),
        ]

    def test_sanitize_email_output_strips_all_prohibited_elements(self) -> None:
        """Verify sanitize_email_output strips preambles, citations, doc IDs, page numbers, chunk text, and sources."""
        raw_email = (
            "Here is the requested drafted email for the safety team:\n\n"
            "Subject: Safety Requirements for Hot Work — Summary\n\n"
            "Dear Site Safety Team,\n\n"
            "--- SOURCE [1] Source: E294_Hot_Work_Standard.pdf ---\n"
            "Please find below a summary of the documented safety requirements [1]:\n"
            "• Gas testing for combustible gases must be performed prior to ignition as per E294_Hot_Work_Standard.pdf (Page 3) [1].\n"
            "• Personnel must wear flame-retardant PPE and safety goggles during hot work operations [2].\n"
            "• Positive isolation and Lockout/Tagout (LOTO) must be verified on all connected process lines (documented in Refinery_Safety_Manual.docx on page 7).\n\n"
            "Best regards,\n"
            "[Engineering Team]\n\n"
            "Hope this helps! Let me know if you need anything else.\n\n"
            "### References & Provenance\n"
            "- **[1]** E294_Hot_Work_Standard.pdf | Page 3 | Section: Gas Testing & PPE\n"
            "  > \"Gas testing for combustible gases must be performed prior to ignition.\"\n"
            "- **[2]** Refinery_Safety_Manual.docx | Page 7 | Section: Isolation Boundaries\n"
            "  > \"Positive isolation and Lockout/Tagout (LOTO) must be verified...\""
        )

        sanitized = ResponseFormatter.sanitize_email_output(
            raw_email,
            document_names=[self.doc_name, "Refinery_Safety_Manual.docx", "DOC-HW-001"],
        )

        # 1. Contains only the email itself
        self.assertTrue(sanitized.startswith("Subject: Safety Requirements for Hot Work — Summary"))
        self.assertTrue(sanitized.endswith("[Engineering Team]"))

        # 2. No conversational preambles or postambles
        self.assertNotIn("Here is the requested drafted email", sanitized)
        self.assertNotIn("Hope this helps", sanitized)

        # 3. No References, Sources, or Provenance
        self.assertNotIn("### References", sanitized)
        self.assertNotIn("Provenance", sanitized)
        self.assertNotIn("References & Provenance", sanitized)

        # 4. No citations
        self.assertNotIn("[1]", sanitized)
        self.assertNotIn("[2]", sanitized)

        # 5. No document IDs or file extensions
        self.assertNotIn("E294_Hot_Work_Standard.pdf", sanitized)
        self.assertNotIn("Refinery_Safety_Manual.docx", sanitized)
        self.assertNotIn(".pdf", sanitized)
        self.assertNotIn(".docx", sanitized)

        # 6. No page numbers
        self.assertNotIn("Page 3", sanitized)
        self.assertNotIn("page 7", sanitized)

        # 7. No chunk headers
        self.assertNotIn("--- SOURCE", sanitized)

        # 8. Essential verified content remains intact
        self.assertIn("Dear Site Safety Team,", sanitized)
        self.assertIn("Gas testing for combustible gases must be performed prior to ignition", sanitized)
        self.assertIn("Personnel must wear flame-retardant PPE", sanitized)
        self.assertIn("Positive isolation and Lockout/Tagout", sanitized)
        self.assertIn("Best regards,", sanitized)

    def test_generation_pipeline_email_generation_contains_only_email(self) -> None:
        """Verify GenerationPipeline.generate with PromptArchetype.EMAIL outputs only the email itself."""
        config = GenerationConfig(default_model_name="deterministic_test", cache_enabled=False)
        pipeline = GenerationPipeline(config=config, model=DeterministicTestLLM())

        query = "Draft an email summarizing the safety requirements for hot work for the site safety team. Use only the provided documentation."
        retrieval_res = RetrievalResult(
            query=query,
            candidates=[
                ScoredRetrievalChunk(
                    chunk=Chunk(
                        chunk_id="chk-01",
                        content="Hot work activities require mandatory gas testing for combustible gases prior to ignition.",
                        metadata=ChunkMetadata(
                            document_id="DOC-HW-001",
                            document_name=self.doc_name,
                            page_number=3,
                            source_path=f"/docs/{self.doc_name}",
                            equipment_tag="E-330",
                        ),
                        hierarchy=ChunkHierarchy(document_id="DOC-HW-001"),
                    ),
                    score=0.92,
                ),
                ScoredRetrievalChunk(
                    chunk=Chunk(
                        chunk_id="chk-02",
                        content="Personnel must wear flame-retardant PPE and safety goggles during hot work operations.",
                        metadata=ChunkMetadata(
                            document_id="DOC-HW-001",
                            document_name=self.doc_name,
                            page_number=4,
                            source_path=f"/docs/{self.doc_name}",
                            equipment_tag="E-330",
                        ),
                        hierarchy=ChunkHierarchy(document_id="DOC-HW-001"),
                    ),
                    score=0.88,
                ),
            ],
            citations=self.citations,
            formatted_context=(
                "--- [1] Source: E294_Hot_Work_Standard.pdf ---\n"
                "Hot work activities require mandatory gas testing for combustible gases prior to ignition.\n\n"
                "--- [2] Source: Refinery_Safety_Manual.docx ---\n"
                "Personnel must wear flame-retardant PPE and safety goggles during hot work operations."
            ),
        )

        gen_resp = pipeline.generate(
            query=query,
            retrieval_result=retrieval_res,
            archetype=PromptArchetype.EMAIL,
        )

        # Check answer body
        ans = gen_resp.answer
        self.assertTrue(ans.startswith("Subject:"))
        self.assertIn("Dear Site Safety Team,", ans)
        self.assertIn("Best regards,", ans)
        self.assertTrue(ans.endswith("[Engineering Team]"))

        # Invariant: No citations, doc IDs, page numbers, or references in answer
        self.assertNotIn("### References", ans)
        self.assertNotIn("Provenance", ans)
        self.assertNotIn("Sources", ans)
        self.assertNotIn("[1]", ans)
        self.assertNotIn("[2]", ans)
        self.assertNotIn(".pdf", ans)
        self.assertNotIn("Page 3", ans)

        # Invariant: RAG system internally maintains citations, grounding, and confidence
        self.assertEqual(len(gen_resp.citations), 2)
        self.assertTrue(gen_resp.grounding_report.is_grounded)
        self.assertGreater(gen_resp.confidence.composite_score, 0.5)

    def test_rag_response_format_clean_cli_output_suppresses_sources_for_email(self) -> None:
        """Verify format_clean_cli_output does not append ### Sources for email queries."""
        email_text = (
            "Subject: Safety Requirements for Hot Work — Summary\n\n"
            "Dear Site Safety Team,\n\n"
            "Please find below a summary of the documented safety requirements:\n"
            "• Gas testing must be performed prior to ignition.\n"
            "• Personnel must wear flame-retardant PPE.\n\n"
            "Best regards,\n"
            "[Engineering Team]"
        )

        resp = RAGResponse(
            query="Draft an email summarizing the safety requirements for hot work for the site safety team. Use only the provided documentation.",
            answer=email_text,
            session_id="test_sess",
            citations=self.citations,
            retrieval_result=RetrievalResult(query="test", candidates=[], citations=self.citations),
            generation_response=None,  # type: ignore[arg-type]
            confidence_score=0.95,
            is_grounded=True,
            total_latency_ms=120.0,
            model_used="deterministic_test",
        )

        clean_cli = resp.format_clean_cli_output()
        self.assertEqual(clean_cli, email_text)
        self.assertNotIn("### Sources", clean_cli)
        self.assertNotIn("[1]", clean_cli)

    def test_rag_response_detailed_cli_output_keeps_section_isolation(self) -> None:
        """Verify detailed CLI output has clean email in Section 1 and internal audit in Section 2."""
        email_text = (
            "Subject: Safety Requirements for Hot Work — Summary\n\n"
            "Dear Site Safety Team,\n\n"
            "Please find below a summary of the documented safety requirements:\n"
            "• Gas testing must be performed prior to ignition.\n"
            "• Personnel must wear flame-retardant PPE.\n\n"
            "Best regards,\n"
            "[Engineering Team]"
        )

        resp = RAGResponse(
            query="Draft an email summarizing the safety requirements for hot work for the site safety team. Use only the provided documentation.",
            answer=email_text,
            session_id="test_sess",
            citations=self.citations,
            retrieval_result=RetrievalResult(query="test", candidates=[], citations=self.citations),
            generation_response=None,  # type: ignore[arg-type]
            confidence_score=0.95,
            is_grounded=True,
            total_latency_ms=120.0,
            model_used="deterministic_test",
        )

        detailed_cli = resp.format_cli_output(detailed=True)
        self.assertIn("GENERATED EMAIL", detailed_cli)
        self.assertIn("SOURCES / EVIDENCE", detailed_cli)

        # Section 1 contains the clean email
        sec1 = detailed_cli.split("SOURCES / EVIDENCE")[0]
        self.assertIn("Subject: Safety Requirements for Hot Work — Summary", sec1)
        self.assertNotIn("### References", sec1)
        self.assertNotIn("[1] E294_Hot_Work_Standard.pdf", sec1)

        # Section 2 contains the internal audit trail for engineers
        sec2 = detailed_cli.split("SOURCES / EVIDENCE")[1]
        self.assertIn("[1] E294_Hot_Work_Standard.pdf", sec2)
        self.assertIn("Page 3", sec2)

    def test_generic_email_purposes(self) -> None:
        """Verify that email output isolation applies generically across varied email queries."""
        config = GenerationConfig(default_model_name="deterministic_test", cache_enabled=False)
        pipeline = GenerationPipeline(config=config, model=DeterministicTestLLM())

        queries = [
            (
                "Draft an email requesting approval for boiler maintenance to the Plant Manager.",
                "Request for Approval",
                "Boiler maintenance requires formal shutdown authorization and safety review prior to execution.",
            ),
            (
                "Draft an email asking the operations team to take action on valve calibration.",
                "Action Required",
                "Action is required to calibrate relief valves according to statutory compliance intervals.",
            ),
            (
                "Draft an email notifying the team about the upcoming turnaround schedule.",
                "Notification",
                "The upcoming plant turnaround is scheduled to commence next month across all operating units.",
            ),
        ]

        for query, expected_subj_keyword, text_evidence in queries:
            candidates = [
                ScoredRetrievalChunk(
                    chunk=Chunk(
                        chunk_id=f"chk-{expected_subj_keyword[:4].lower()}",
                        content=text_evidence,
                        metadata=ChunkMetadata(
                            document_id="DOC-HW-001",
                            document_name=self.doc_name,
                            page_number=1,
                            source_path=f"/docs/{self.doc_name}",
                        ),
                        hierarchy=ChunkHierarchy(document_id="DOC-HW-001"),
                    ),
                    score=0.90,
                )
            ]
            retrieval_res = RetrievalResult(
                query=query,
                candidates=candidates,
                citations=self.citations,
                formatted_context=f"--- [1] Source: {self.doc_name} ---\n{text_evidence}",
            )

            resp = pipeline.generate(query=query, retrieval_result=retrieval_res, archetype=PromptArchetype.EMAIL)
            ans = resp.answer

            # Must start with Subject
            self.assertTrue(ans.startswith("Subject:"), f"Failed for {query}")
            self.assertIn(expected_subj_keyword, ans, f"Failed subject keyword for {query}")
            # Must end with sign-off
            self.assertTrue(ans.endswith("[Engineering Team]"), f"Failed signoff for {query}")
            # Zero leakage
            self.assertNotIn("### References", ans)
            self.assertNotIn("Provenance", ans)
            self.assertNotIn("[1]", ans)
            self.assertNotIn(".pdf", ans)
            self.assertNotIn(".docx", ans)
            self.assertNotIn("Page", ans)


if __name__ == "__main__":
    unittest.main()
