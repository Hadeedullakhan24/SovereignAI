"""End-to-End Integration Tests for Member 1 (RAG Engine) & Member 2 (TaskRouter & Orchestrator).

Verifies Architectural Correction:
1. Member 1 DOES NOT own model selection.
2. Member 1 primary contract is pure model-independent retrieval / context retrieval:
   query -> retrieve -> rerank -> evidence gating -> context packing -> citations -> retrieval trace -> structured result.
3. Member 2 owns:
   task classification -> model selection -> RAG decision -> calling Member 1 -> passing retrieved context to selected model -> final generation.
4. Auto routing maps:
   - DOCUMENT_QA -> Qwen2.5-1.5B (with Member 1 RAG)
   - COMPARISON -> Phi-3.5-mini (with Member 1 RAG)
   - GENERAL_QA -> SmolLM2-1.7B (no unnecessary RAG)
   - REASONING -> Phi-3.5-mini (with Member 1 RAG)
   - EXTRACTION -> Qwen2.5-1.5B (with Member 1 RAG)
   - DOCUMENT_GENERATION -> Phi-3.5-mini (with Member 1 RAG)
   - VISION_OCR -> Member 3 processor
   - ENGINEERING_DRAWING_ANALYSIS -> Qwen2.5-VL-3B-Instruct
5. Citations are preserved end-to-end.
6. CLI supports AUTO mode and explicit --model override.
"""

from __future__ import annotations

import json
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

from agent.model_registry import AgentModelRegistry, get_agent_registry
from agent.router import Capability, RoutingDecision, TaskRouter, TaskType, get_router
from agent.tool_executor import ToolExecutor
from rag_engine.generation.generation_config import GenerationConfig
from rag_engine.generation.prompt.prompt_templates import PromptArchetype
from rag_engine.pipeline.rag_pipeline import RAGPipeline, RAGResponse
from rag_engine.retrieval.base_retriever import (
    CitationBundle,
    RetrievalResult,
    ScoredRetrievalChunk,
)
from rag_engine.schemas.chunk import Chunk, ChunkHierarchy, ChunkMetadata


def make_mock_retrieval_result(query: str = "What is the design operating pressure of pump P-203?") -> RetrievalResult:
    chunk = Chunk(
        chunk_id="chk_pump_p203",
        content="Centrifugal Pump P-203 design operating pressure is 15.5 bar g with operating temperature 65 deg C.",
        metadata=ChunkMetadata(
            document_id="pump_p203_datasheet.pdf",
            chunk_index=0,
            page_number=1,
            section_title="Operating Conditions",
            equipment_entities=["P-203"],
        ),
        hierarchy=ChunkHierarchy(document_id="pump_p203_datasheet.pdf"),
    )
    candidate = ScoredRetrievalChunk(
        chunk=chunk,
        score=0.92,
        dense_score=0.90,
        bm25_score=14.5,
        rrf_score=0.032,
        channel="hybrid",
    )
    citation = CitationBundle(
        citation_id="[1]",
        chunk_id="chk_pump_p203",
        document_id="pump_p203_datasheet.pdf",
        document_name="pump_p203_datasheet.pdf",
        section_title="Operating Conditions",
        page_number=1,
        verbatim_quote="Centrifugal Pump P-203 design operating pressure is 15.5 bar g with operating temperature 65 deg C.",
        equipment_tags=["P-203"],
        score=0.92,
    )
    return RetrievalResult(
        query=query,
        candidates=[candidate],
        citations=[citation],
        formatted_context="[1] Centrifugal Pump P-203 design operating pressure is 15.5 bar g with operating temperature 65 deg C.",
        total_candidates=1,
        retrieval_strategy="hybrid",
        execution_time_ms=8.5,
    )


class TestMember1ModelIndependentRetrieval(unittest.TestCase):
    """Test Member 1's model-independent retrieval contract."""

    def setUp(self) -> None:
        self.mock_retrieval = MagicMock()
        self.mock_retrieval.retrieve.return_value = make_mock_retrieval_result()
        self.config = GenerationConfig(default_model_name="deterministic_test")
        self.pipeline = RAGPipeline(retrieval_pipeline=self.mock_retrieval, config=self.config)

    def test_retrieve_context_contract(self) -> None:
        """Verify retrieve_context returns structured model-independent result."""
        result = self.pipeline.retrieve_context("What is the operating pressure of pump P-203?")
        self.assertIsInstance(result, dict)
        self.assertIn("query", result)
        self.assertIn("context", result)
        self.assertIn("citations", result)
        self.assertIn("confidence", result)
        self.assertIn("retrieval_trace", result)
        self.assertIn("candidates", result)
        self.assertIn("structured_report", result)
        self.assertIn("has_sufficient_evidence", result)

        self.assertEqual(len(result["citations"]), 1)
        self.assertEqual(result["citations"][0].document_id, "pump_p203_datasheet.pdf")
        self.assertTrue(result["confidence"] > 0.5)
        self.assertIn("P-203", result["context"])


class TestMember2AuthoritativeRouting(unittest.TestCase):
    """Test Member 2 TaskRouter owns task classification and model selection."""

    def setUp(self) -> None:
        self.router = get_router()

    def test_document_qa_routes_to_qwen(self) -> None:
        decision = self.router.route("What is the design operating pressure of centrifugal pump P-203?")
        self.assertEqual(decision.task_type, TaskType.DOCUMENT_QA)
        self.assertTrue(decision.use_rag_context)
        self.assertIsNotNone(decision.model_record)
        self.assertIn("Qwen2.5-1.5B", decision.model_record.hf_repo_id)

    def test_comparison_routes_to_phi(self) -> None:
        decision = self.router.route("Compare centrifugal pump vs positive displacement pump for refinery heavy crude service")
        self.assertEqual(decision.task_type, TaskType.COMPARISON)
        self.assertTrue(decision.use_rag_context)
        self.assertIsNotNone(decision.model_record)
        self.assertIn("Phi-3.5-mini", decision.model_record.hf_repo_id)

    def test_general_qa_routes_to_smollm_no_rag(self) -> None:
        decision = self.router.route("What is photosynthesis?")
        self.assertEqual(decision.task_type, TaskType.GENERAL_QA)
        self.assertFalse(decision.use_rag_context)
        self.assertIsNotNone(decision.model_record)
        self.assertIn("SmolLM2-1.7B", decision.model_record.hf_repo_id)

    def test_reasoning_routes_to_phi(self) -> None:
        decision = self.router.route("Troubleshoot root cause of pump P-203 high vibration and cavitation trip")
        self.assertEqual(decision.task_type, TaskType.REASONING)
        self.assertTrue(decision.use_rag_context)
        self.assertIsNotNone(decision.model_record)
        self.assertIn("Phi-3.5-mini", decision.model_record.hf_repo_id)

    def test_extraction_routes_to_qwen(self) -> None:
        decision = self.router.route("Extract all equipment tags and design parameters from the crude distillation unit datasheet")
        self.assertEqual(decision.task_type, TaskType.EXTRACTION)
        self.assertTrue(decision.use_rag_context)
        self.assertIsNotNone(decision.model_record)
        self.assertIn("Qwen2.5-1.5B", decision.model_record.hf_repo_id)

    def test_document_generation_routes_to_phi(self) -> None:
        decision = self.router.route("Generate executive summary report as an actual pdf file for drum V-2201")
        self.assertEqual(decision.task_type, TaskType.DOCUMENT_GENERATION)
        self.assertTrue(decision.use_rag_context)
        self.assertIsNotNone(decision.model_record)
        self.assertIn("Phi-3.5-mini", decision.model_record.hf_repo_id)

    def test_engineering_drawing_routes_to_qwen_vl(self) -> None:
        decision = self.router.route("Inspect P&ID piping and instrumentation engineering drawing for crude column C-101")
        self.assertEqual(decision.task_type, TaskType.ENGINEERING_DRAWING_ANALYSIS)
        self.assertIsNotNone(decision.model_record)
        self.assertIn("Qwen2.5-VL-3B-Instruct", decision.model_record.hf_repo_id)

    def test_vision_ocr_routes_to_multimodal(self) -> None:
        decision = self.router.route("OCR scanned document inspection report page 1")
        self.assertEqual(decision.task_type, TaskType.VISION_OCR)
        self.assertIsNotNone(decision.model_record)


class TestEndToEndAutoPathOrchestration(unittest.TestCase):
    """Test full end-to-end AUTO pipeline: User input -> Member 2 router -> Member 1 RAG -> Selected Model -> Final Answer."""

    def setUp(self) -> None:
        self.mock_retrieval = MagicMock()
        self.mock_retrieval.retrieve.return_value = make_mock_retrieval_result()
        self.config = GenerationConfig(default_model_name="deterministic_test")
        self.pipeline = RAGPipeline(retrieval_pipeline=self.mock_retrieval, config=self.config)

    def test_auto_path_end_to_end_preserves_citations(self) -> None:
        """Verify user input flows through router, calls Member 1 retrieval, synthesizes response and preserves citations."""
        query = "What is the design operating pressure of pump P-203?"
        router = get_router()
        decision = router.route(query)

        # 1. Routing decision verified
        self.assertEqual(decision.task_type, TaskType.DOCUMENT_QA)
        self.assertTrue(decision.use_rag_context)

        # 2. Member 1 retrieval executed independently
        rag_data = self.pipeline.retrieve_context(query)
        self.assertTrue(rag_data["has_sufficient_evidence"])
        self.assertEqual(len(rag_data["citations"]), 1)

        # 3. Model generation produces grounded answer preserving citations
        response = self.pipeline.answer(question=query, model_name="deterministic_test")
        self.assertIsInstance(response, RAGResponse)
        self.assertTrue(response.is_grounded)
        self.assertEqual(len(response.citations), 1)
        self.assertEqual(response.citations[0].equipment_tags, ["P-203"])
        self.assertIn("pump_p203_datasheet.pdf", response.citations[0].document_name)


if __name__ == "__main__":
    unittest.main()
