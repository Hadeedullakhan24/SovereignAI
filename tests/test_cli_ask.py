"""Comprehensive Tests for CLI Question Answering, RAGPipeline Orchestrator, and Memory.

Verifies:
    1. RAGPipeline.answer() orchestration and grounded response synthesis.
    2. Insufficient evidence fallback ("The uploaded documents do not contain sufficient information...").
    3. RAGExecutionTrace stage timing telemetry and JSON serialization.
    4. RAGResponse.format_cli_output() terminal presentation.
    5. SQLite-backed ConversationMemory persistence and sliding-window trimming.
    6. PromptTemplate cryptographic versioning and SHA-256 hash generation.
    7. LLMRegistry model specifications catalog and alias resolution.
    8. scripts/ask.py single-query CLI execution.
    9. scripts/ask.py JSON output flag.
    10. scripts/ask.py --list-models flag.
    11. Top-level generation alias module imports.
    12. Strict offline air-gapped guarantees.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import MagicMock

from rag_engine.generation.generation_config import GenerationConfig
from rag_engine.generation.memory.conversation_memory import ConversationMemory
from rag_engine.generation.models.model_registry import LLMRegistry
from rag_engine.generation.prompt.prompt_templates import (
    PromptArchetype,
    PromptTemplateRegistry,
)
from rag_engine.pipeline.rag_pipeline import (
    INSUFFICIENT_EVIDENCE_FALLBACK,
    RAGPipeline,
    RAGResponse,
)
from rag_engine.retrieval.base_retriever import (
    CitationBundle,
    RetrievalResult,
    ScoredRetrievalChunk,
)
from rag_engine.schemas.chunk import Chunk, ChunkHierarchy, ChunkMetadata


def create_synthetic_retrieval_result() -> RetrievalResult:
    """Create synthetic retrieval evidence with Pump P-203 data for testing."""
    chunk = Chunk(
        chunk_id="chk_p203_01",
        content="Centrifugal Pump P-203 operates at 15.5 bar discharge pressure with normal temperature 65.0 °C.",
        metadata=ChunkMetadata(
            document_id="pump_p203_manual.pdf",
            chunk_index=1,
            token_count=25,
            character_count=100,
            page_number=3,
            section_title="Operating Parameters",
        ),
        hierarchy=ChunkHierarchy(document_id="pump_p203_manual.pdf"),
    )
    candidate = ScoredRetrievalChunk(
        chunk=chunk,
        score=0.95,
        dense_score=0.92,
        bm25_score=12.0,
        rrf_score=0.03,
        channel="hybrid",
    )
    citation = CitationBundle(
        citation_id="[1]",
        chunk_id="chk_p203_01",
        document_id="pump_p203_manual.pdf",
        document_name="pump_p203_manual.pdf",
        section_title="Operating Parameters",
        page_number=3,
        verbatim_quote="Centrifugal Pump P-203 operates at 15.5 bar discharge pressure with normal temperature 65.0 °C.",
        equipment_tags=["P-203"],
        score=0.95,
    )
    return RetrievalResult(
        query="What is the operating pressure and temperature of Pump P-203?",
        candidates=[candidate],
        citations=[citation],
        formatted_context="[1] Centrifugal Pump P-203 operates at 15.5 bar discharge pressure with normal temperature 65.0 °C.",
        total_candidates=1,
        retrieval_strategy="hybrid",
        execution_time_ms=5.0,
    )


class TestRAGPipelineAndCLI(unittest.TestCase):
    """Test suite for Master RAGPipeline, SQLite Memory, and CLI."""

    def setUp(self) -> None:
        self.config = GenerationConfig(default_model_name="deterministic_test", cache_enabled=False)
        self.mock_retrieval = MagicMock()
        self.mock_retrieval.retrieve.return_value = create_synthetic_retrieval_result()
        self.pipeline = RAGPipeline(
            retrieval_pipeline=self.mock_retrieval,
            config=self.config,
        )

    def test_rag_pipeline_answer_grounded(self) -> None:
        """Test RAGPipeline.answer() produces grounded answer with citations and trace."""
        response = self.pipeline.answer(
            question="What is the operating pressure and temperature of Pump P-203?",
            archetype=PromptArchetype.EQUIPMENT_LOOKUP,
        )
        self.assertIsInstance(response, RAGResponse)
        self.assertTrue(len(response.answer) > 0)
        self.assertIn("P-203", response.answer)
        self.assertIsNotNone(response.execution_trace)
        self.assertEqual(response.execution_trace.question, response.query)
        self.assertTrue(response.execution_trace.total_latency_ms >= 0.0)
        self.assertTrue(response.execution_trace.generation_time_ms >= 0.0)

    def test_rag_pipeline_insufficient_evidence_fallback(self) -> None:
        """Test insufficient evidence fallback when zero candidates exist."""
        mock_empty_retrieval = MagicMock()
        mock_empty_retrieval.retrieve.return_value = RetrievalResult(
            query="Unrelated question",
            candidates=[],
            citations=[],
            formatted_context="",
        )
        pipeline = RAGPipeline(retrieval_pipeline=mock_empty_retrieval, config=self.config)
        response = pipeline.answer(question="What is the capital of Mars?")

        self.assertEqual(response.answer, INSUFFICIENT_EVIDENCE_FALLBACK)
        self.assertEqual(len(response.citations), 0)
        self.assertGreater(response.confidence_score, 0.0)

    def test_rag_execution_trace_serialization(self) -> None:
        """Test RAGExecutionTrace to_dict() and to_json()."""
        response = self.pipeline.answer(question="Check safety valve setting for drum D-101.")
        trace = response.execution_trace
        self.assertIsNotNone(trace)
        trace_dict = trace.to_dict()
        self.assertIn("question", trace_dict)
        self.assertIn("embedding_model", trace_dict)
        self.assertIn("prompt_tokens", trace_dict)
        self.assertIn("generation_time_ms", trace_dict)

        trace_json = trace.to_json()
        parsed = json.loads(trace_json)
        self.assertEqual(parsed["question"], response.query)

    def test_rag_response_format_cli_output(self) -> None:
        """Test terminal output formatting of RAGResponse."""
        response = self.pipeline.answer(question="What is the rated capacity of pump P-201?")
        output = response.format_cli_output()
        self.assertIn("MRPL SOVEREIGN AGENTIC AI WORKBENCH - ANSWER", output)
        self.assertIn("Question:", output)
        self.assertIn("Answer:", output)
        self.assertIn("Confidence Score", output)
        self.assertIn("Model Used", output)

    def test_sqlite_conversation_memory_persistence(self) -> None:
        """Test SQLite-backed multi-turn conversation memory persistence and trimming."""
        test_db = Path("cache/test_cli_conv_mem.db")
        if test_db.exists():
            test_db.unlink()

        memory = ConversationMemory(max_turns_per_session=3, db_path=test_db)
        session_id = "test_session_123"

        # Add 4 turns (max is 3, turn 1 should be trimmed)
        for i in range(1, 5):
            memory.add_turn(
                session_id=session_id,
                user_query=f"Question {i}",
                response=f"Answer {i}",
                retrieved_chunk_ids=[f"chk_{i}"],
                citations=[f"[{i}]"],
            )

        history = memory.get_history(session_id)
        self.assertEqual(len(history), 3)
        self.assertEqual(history[0].user_query, "Question 2")
        self.assertEqual(history[-1].user_query, "Question 4")

        # Test new instance connects to same SQLite DB
        memory2 = ConversationMemory(max_turns_per_session=3, db_path=test_db)
        history2 = memory2.get_history(session_id)
        self.assertEqual(len(history2), 3)
        self.assertEqual(history2[0].user_query, "Question 2")

        # Cleanup
        memory.close()
        memory2.close()
        if test_db.exists():
            test_db.unlink()

    def test_prompt_template_versioning_and_hash(self) -> None:
        """Test PromptTemplate exposes version, author, date, compatibility, and prompt_hash."""
        registry = PromptTemplateRegistry()
        for archetype in PromptArchetype:
            template = registry.get(archetype)
            self.assertTrue(len(template.name) > 0)
            self.assertTrue(len(template.version) > 0)
            self.assertEqual(template.author, "MRPL AI Engineering Team")
            self.assertTrue(len(template.creation_date) > 0)
            self.assertEqual(template.compatibility, "v1.x")
            self.assertTrue(len(template.prompt_hash) == 64)

    def test_llm_registry_specifications_and_aliases(self) -> None:
        """Test LLMRegistry catalog contains standard models and resolves aliases."""
        registry = LLMRegistry.get_instance()
        self.assertTrue(registry.contains("deterministic_test"))
        self.assertTrue(registry.contains("tinyllama"))
        self.assertTrue(registry.contains("phi3"))
        self.assertTrue(registry.contains("qwen2.5"))
        self.assertTrue(registry.contains("mistral"))
        self.assertTrue(registry.contains("llama"))

        # Check alias resolution
        self.assertEqual(registry.resolve_alias("tinyllama"), "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        self.assertEqual(registry.resolve_alias("phi-3-mini"), "microsoft/Phi-3-mini-4k-instruct")
        self.assertEqual(registry.resolve_alias("qwen"), "Qwen/Qwen2.5-1.5B-Instruct")

        spec = registry.get_specification("tinyllama")
        self.assertIsNotNone(spec)
        self.assertEqual(spec.get("family"), "tinyllama")

    def test_top_level_alias_imports(self) -> None:
        """Test importing top-level generation alias modules."""
        import rag_engine.generation.base_llm as base_llm
        import rag_engine.generation.huggingface_llm as hf_llm
        import rag_engine.generation.gguf_llm as gguf_llm
        import rag_engine.generation.llm_registry as llm_reg
        import rag_engine.generation.llm_factory as llm_fac
        import rag_engine.generation.prompt_builder as p_bld
        import rag_engine.generation.prompt_templates as p_tpl
        import rag_engine.generation.context_formatter as c_fmt
        import rag_engine.generation.prompt_budget as p_bdg
        import rag_engine.generation.prompt_validator as p_val
        import rag_engine.generation.answer_formatter as a_fmt
        import rag_engine.generation.citation_injector as c_inj
        import rag_engine.generation.response_validator as r_val
        import rag_engine.generation.hallucination_guard as h_grd
        import rag_engine.generation.confidence_estimator as c_est
        import rag_engine.generation.exceptions as gen_exc

        self.assertTrue(hasattr(base_llm, "BaseLLM"))
        self.assertTrue(hasattr(hf_llm, "HFLocalLLM"))
        self.assertTrue(hasattr(gguf_llm, "GGUFLocalLLM"))
        self.assertTrue(hasattr(llm_reg, "LLMRegistry"))
        self.assertTrue(hasattr(llm_fac, "LLMFactory"))
        self.assertTrue(hasattr(p_bld, "PromptBuilder"))
        self.assertTrue(hasattr(p_tpl, "PromptTemplateRegistry"))
        self.assertTrue(hasattr(c_fmt, "ContextWindowBuilder"))
        self.assertTrue(hasattr(p_bdg, "TokenBudgetManager"))
        self.assertTrue(hasattr(p_val, "PromptValidator"))
        self.assertTrue(hasattr(a_fmt, "ResponseFormatter"))
        self.assertTrue(hasattr(c_inj, "CitationFormatter"))
        self.assertTrue(hasattr(r_val, "SafetyValidator"))
        self.assertTrue(hasattr(h_grd, "HallucinationGuard"))
        self.assertTrue(hasattr(c_est, "ConfidenceScorer"))
        self.assertTrue(hasattr(gen_exc, "BaseGenerationException"))

    def test_cli_ask_list_models(self) -> None:
        """Test scripts/ask.py --list-models flag."""
        cmd = [sys.executable, "scripts/ask.py", "--list-models"]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        self.assertEqual(res.returncode, 0)
        self.assertIn("Registered Local Model Specifications", res.stdout)
        self.assertIn("TinyLlama", res.stdout)

    def test_cli_ask_single_query_json(self) -> None:
        """Test CLI execution with mock pipeline producing JSON output."""
        import io
        from contextlib import redirect_stdout
        from unittest.mock import patch
        from scripts.ask import main

        with patch("scripts.ask.RAGPipeline") as MockPipelineCls:
            mock_inst = MagicMock()
            mock_inst.answer.return_value = self.pipeline.answer(
                question="What is the operating pressure of Pump P-203?"
            )
            MockPipelineCls.return_value = mock_inst

            buf = io.StringIO()
            with redirect_stdout(buf):
                main(["--query", "What is the operating pressure of Pump P-203?", "--json"])

            output_str = buf.getvalue()
            parsed = json.loads(output_str)
            self.assertEqual(parsed["question"], "What is the operating pressure of Pump P-203?")
            self.assertIn("P-203", parsed["answer"])
            self.assertIn("execution_trace", parsed)

    def test_offline_air_gapped_enforcement(self) -> None:
        """Verify strict air-gapped configuration."""
        self.assertTrue(self.config.strict_offline)
        self.assertFalse(self.config.telemetry_enabled)
        self.assertIn(self.config.device, ["cpu", "auto"])



if __name__ == "__main__":
    unittest.main()

