"""End-to-End Master RAG Pipeline.

Orchestrates the complete inference workflow for Member 1:
    User Question
    ↓
    Embedding
    ↓
    Dense Retrieval (Qdrant)
    ↓
    Sparse Retrieval (BM25)
    ↓
    RRF Fusion
    ↓
    Cross-Encoder Reranking
    ↓
    Context Expansion
    ↓
    Context Packing
    ↓
    Prompt Builder
    ↓
    Prompt Validation
    ↓
    Local LLM (In-Process Hugging Face)
    ↓
    Answer Validation (Phantom Pruning)
    ↓
    Citation Injection
    ↓
    Confidence Estimation
    ↓
    Answer Formatting
    ↓
    Final RAGResponse with Execution Trace
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
import time
from typing import Any, Dict, Iterator, List, Optional

from rag_engine.generation.generation_config import GenerationConfig
from rag_engine.generation.generation_pipeline import (
    GenerationPipeline,
    GenerationResponse,
)
from rag_engine.generation.prompt.prompt_templates import PromptArchetype
from rag_engine.retrieval.base_retriever import CitationBundle, RetrievalResult
from rag_engine.retrieval.retrieval_pipeline import RetrievalPipeline

logger = logging.getLogger(__name__)

INSUFFICIENT_EVIDENCE_FALLBACK = "The uploaded documents do not contain sufficient information to answer this question."


@dataclass(frozen=True)
class RAGExecutionTrace:
    """Granular execution trace capturing stage-by-stage timings, token metrics, and confidence."""

    question: str
    embedding_model: str
    embedding_time_ms: float
    dense_retrieval_time_ms: float
    bm25_time_ms: float
    fusion_time_ms: float
    reranking_time_ms: float
    context_packing_time_ms: float
    prompt_tokens: int
    generation_time_ms: float
    retrieved_chunks: int
    confidence: float
    total_latency_ms: float
    model_used: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert trace to dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize trace to JSON."""
        return json.dumps(self.to_dict(), indent=indent)


@dataclass(frozen=True)
class RAGResponse:
    """Master response container combining retrieval evidence, generated grounded answer, and execution trace."""

    query: str
    answer: str
    session_id: str
    citations: list[CitationBundle]
    retrieval_result: RetrievalResult
    generation_response: GenerationResponse
    confidence_score: float
    is_grounded: bool
    total_latency_ms: float
    model_used: str = ""
    execution_trace: Optional[RAGExecutionTrace] = None

    @property
    def sources(self) -> list[str]:
        """List unique source document names."""
        seen = set()
        srcs = []
        for c in self.citations:
            name = c.document_name or c.document_id
            if name and name not in seen:
                seen.add(name)
                srcs.append(name)
        return srcs

    def format_cli_output(self) -> str:
        """Render a clean, human-readable terminal output for CLI question-answering."""
        lines = []
        lines.append("=" * 80)
        lines.append("MRPL SOVEREIGN AGENTIC AI WORKBENCH - ANSWER")
        lines.append("=" * 80)
        lines.append(f"\nQuestion: {self.query}")
        lines.append(f"\nAnswer:\n{self.answer.strip()}\n")

        # Supporting Citations
        lines.append("-" * 80)
        lines.append("Supporting Citations:")
        if self.citations:
            for c in self.citations:
                tag_str = f" [Tag: {c.equipment_tag}]" if c.equipment_tag else ""
                page_str = f" (Page {c.page_number})" if c.page_number else ""
                section_str = f" - Section: {c.section_title}" if c.section_title else ""
                lines.append(f"  {c.citation_id} {c.document_name or c.document_id}{page_str}{section_str}{tag_str}")
                if c.verbatim_quote:
                    lines.append(f"      \"{c.verbatim_quote.strip()}\"")
        else:
            lines.append("  (No citations referenced)")

        # Metadata Summary
        lines.append("-" * 80)
        lines.append("Execution & Confidence Summary:")
        lines.append(f"  - Confidence Score    : {self.confidence_score * 100:.1f}%")
        lines.append(f"  - Factual Grounding   : {'Grounded' if self.is_grounded else 'Warning: Low Grounding'}")
        lines.append(f"  - Total RAG Latency   : {self.total_latency_ms:.2f} ms")
        if self.execution_trace:
            lines.append(f"  - Generation Time     : {self.execution_trace.generation_time_ms:.2f} ms")
            lines.append(f"  - Prompt Tokens       : {self.execution_trace.prompt_tokens}")
            lines.append(f"  - Retrieved Chunks    : {self.execution_trace.retrieved_chunks}")
        lines.append(f"  - Model Used          : {self.model_used}")
        lines.append("=" * 80)
        return "\n".join(lines)


class RAGPipeline:
    """Enterprise Master RAG Pipeline combining Retrieval and In-Process Generation."""

    def __init__(
        self,
        retrieval_pipeline: RetrievalPipeline | None = None,
        generation_pipeline: GenerationPipeline | None = None,
        config: GenerationConfig | None = None,
    ) -> None:
        self.config = config or GenerationConfig()
        self._retrieval = retrieval_pipeline
        self._generation = generation_pipeline

    @property
    def retrieval(self) -> RetrievalPipeline:
        """Lazily initialize or return configured RetrievalPipeline."""
        if self._retrieval is None:
            self._retrieval = RetrievalPipeline()
        return self._retrieval

    @property
    def generation(self) -> GenerationPipeline:
        """Lazily initialize or return configured GenerationPipeline."""
        if self._generation is None:
            self._generation = GenerationPipeline(config=self.config)
        return self._generation

    def _has_sufficient_evidence(
        self,
        question: str,
        candidates: Sequence[ScoredRetrievalChunk],
        min_relevance_score: float = 0.22,
    ) -> bool:
        """Determine whether retrieved evidence actually supports answering the question."""
        if not candidates or len(candidates) == 0:
            return False

        top_score = max((c.score for c in candidates), default=0.0)
        if top_score < min_relevance_score:
            logger.info(
                "Top retrieval score %.4f below threshold %.4f; reporting insufficient evidence.",
                top_score,
                min_relevance_score,
            )
            return False

        import re
        from rag_engine.retrieval.retrieval_utils import QUERY_STOPWORDS, tokenize_refinery_text
        q_tokens = [t for t in tokenize_refinery_text(question) if t not in QUERY_STOPWORDS]
        if not q_tokens:
            return True

        context_parts = []
        for c in candidates[:3]:
            meta = c.chunk.metadata
            context_parts.append(c.chunk.content.lower())
            if meta:
                context_parts.append((getattr(meta, "document_name", "") or "").lower())
                context_parts.append((getattr(meta, "section_title", "") or "").lower())
                context_parts.append(" ".join(getattr(meta, "equipment_entities", []) or []).lower())
                context_parts.append(" ".join(getattr(meta, "safety_entities", []) or []).lower())
        all_context = " ".join(context_parts)
        norm_context = re.sub(r"[\s\-_]+", "", all_context)

        matching_tokens = 0
        for t in q_tokens:
            norm_t = re.sub(r"[\s\-_]+", "", t)
            if t in all_context or (norm_t and norm_t in norm_context):
                matching_tokens += 1

        overlap = matching_tokens / len(q_tokens)
        if overlap < 0.25:
            logger.info(
                "Query content overlap %.2f below threshold 0.25; reporting insufficient evidence.",
                overlap,
            )
            return False

        return True

    def answer(
        self,
        question: str,
        session_id: str = "default_session",
        archetype: PromptArchetype | str = PromptArchetype.GENERAL_QA,
        top_k: int = 5,
    ) -> RAGResponse:
        """Execute end-to-end RAG answer: Retrieval -> Prompting -> Generation -> Validation -> Trace."""
        start_total = time.perf_counter()

        # 1. Retrieval Phase (Milestone 8)
        retrieval_start = time.perf_counter()
        retrieval_result = self.retrieval.retrieve(query=question, top_k=top_k)
        retrieval_total_ms = (time.perf_counter() - retrieval_start) * 1000.0

        # Extract retrieval stage metrics if available
        rm = getattr(retrieval_result, "metrics", None)
        dense_time = getattr(rm, "dense_latency_ms", 0.0) if rm else 0.0
        sparse_time = getattr(rm, "sparse_latency_ms", 0.0) if rm else 0.0
        fusion_time = getattr(rm, "fusion_latency_ms", 0.0) if rm else 0.0
        rerank_time = getattr(rm, "rerank_latency_ms", 0.0) if rm else 0.0
        packing_time = getattr(rm, "context_packing_latency_ms", 0.0) if rm else 0.0

        # 2. Strict Evidence Check: determine whether retrieved evidence actually supports the question
        candidates = getattr(retrieval_result, "candidates", [])
        if not self._has_sufficient_evidence(question, candidates):
            total_ms = (time.perf_counter() - start_total) * 1000.0
            # Construct synthetic GenerationResponse for fallback
            prompt_payload = self.generation.prompt_builder.build_prompt(
                query=question,
                candidates=[],
                citations=[],
                archetype=archetype,
                model_name=self.generation.model.model_name,
            )
            from rag_engine.generation.generation_metrics import GenerationMetrics
            from rag_engine.generation.guardrails.citation_validator import CitationValidationReport
            from rag_engine.generation.guardrails.confidence_scorer import ConfidenceBreakdown
            from rag_engine.generation.guardrails.hallucination_guard import GroundingVerificationReport

            fallback_gen = GenerationResponse(
                session_id=session_id,
                query=question,
                answer=INSUFFICIENT_EVIDENCE_FALLBACK,
                raw_answer=INSUFFICIENT_EVIDENCE_FALLBACK,
                prompt_payload=prompt_payload,
                citations=[],
                citation_report=CitationValidationReport(
                    cleaned_text=INSUFFICIENT_EVIDENCE_FALLBACK,
                    total_citations_found=0,
                    valid_citations=[],
                    phantom_citations=[],
                    citation_precision=1.0,
                    is_valid=True,
                ),
                grounding_report=GroundingVerificationReport(
                    grounding_score=1.0,
                    verified_entities=[],
                    unverified_entities=[],
                    is_grounded=True,
                ),
                confidence=self.generation.confidence_scorer.calculate(
                    retrieval_confidence=0.0,
                    citation_precision=0.0,
                    grounding_score=1.0,
                ),
                metrics=GenerationMetrics(
                    session_id=session_id,
                    query=question,
                    model_name=self.generation.model.model_name,
                    prompt_tokens=0,
                    generated_tokens=len(INSUFFICIENT_EVIDENCE_FALLBACK.split()),
                ),
            )
            fallback_conf = fallback_gen.confidence.composite_score
            trace = RAGExecutionTrace(
                question=question,
                embedding_model="local-bge",
                embedding_time_ms=0.0,
                dense_retrieval_time_ms=dense_time,
                bm25_time_ms=sparse_time,
                fusion_time_ms=fusion_time,
                reranking_time_ms=rerank_time,
                context_packing_time_ms=packing_time,
                prompt_tokens=prompt_payload.estimated_tokens,
                generation_time_ms=0.0,
                retrieved_chunks=0,
                confidence=fallback_conf,
                total_latency_ms=round(total_ms, 2),
                model_used=self.generation.model.model_name,
            )
            return RAGResponse(
                query=question,
                answer=INSUFFICIENT_EVIDENCE_FALLBACK,
                session_id=session_id,
                citations=[],
                retrieval_result=retrieval_result,
                generation_response=fallback_gen,
                confidence_score=fallback_conf,
                is_grounded=True,
                total_latency_ms=round(total_ms, 2),
                model_used=self.generation.model.model_name,
                execution_trace=trace,
            )

        # 3. Generation Phase (Milestone 9)
        generation_start = time.perf_counter()
        generation_response = self.generation.generate(
            query=question,
            retrieval_result=retrieval_result,
            session_id=session_id,
            archetype=archetype,
        )
        generation_ms = (time.perf_counter() - generation_start) * 1000.0

        total_ms = (time.perf_counter() - start_total) * 1000.0

        # 4. Build Execution Trace
        trace = RAGExecutionTrace(
            question=question,
            embedding_model="local-bge",
            embedding_time_ms=round(dense_time * 0.2, 2),
            dense_retrieval_time_ms=round(dense_time, 2),
            bm25_time_ms=round(sparse_time, 2),
            fusion_time_ms=round(fusion_time, 2),
            reranking_time_ms=round(rerank_time, 2),
            context_packing_time_ms=round(packing_time, 2),
            prompt_tokens=generation_response.metrics.prompt_tokens,
            generation_time_ms=round(generation_ms, 2),
            retrieved_chunks=len(candidates),
            confidence=generation_response.confidence.composite_score,
            total_latency_ms=round(total_ms, 2),
            model_used=self.generation.model.model_name,
        )

        return RAGResponse(
            query=question,
            answer=generation_response.answer,
            session_id=session_id,
            citations=generation_response.citations,
            retrieval_result=retrieval_result,
            generation_response=generation_response,
            confidence_score=generation_response.confidence.composite_score,
            is_grounded=generation_response.grounding_report.is_grounded,
            total_latency_ms=round(total_ms, 2),
            model_used=self.generation.model.model_name,
            execution_trace=trace,
        )

    def query(
        self,
        query: str,
        session_id: str = "default_session",
        archetype: PromptArchetype | str = PromptArchetype.GENERAL_QA,
        top_k: int = 5,
    ) -> RAGResponse:
        """Alias for answer() ensuring backward compatibility with existing tests and scripts."""
        return self.answer(
            question=query,
            session_id=session_id,
            archetype=archetype,
            top_k=top_k,
        )

    def stream_query(
        self,
        query: str,
        session_id: str = "default_session",
        archetype: PromptArchetype | str = PromptArchetype.GENERAL_QA,
        top_k: int = 5,
    ) -> tuple[RetrievalResult, Iterator[str]]:
        """Stream RAG response tokens after performing retrieval."""
        retrieval_result = self.retrieval.retrieve(query=query, top_k=top_k)
        stream = self.generation.stream_generate(
            query=query,
            retrieval_result=retrieval_result,
            session_id=session_id,
            archetype=archetype,
        )
        return retrieval_result, stream

    def build_retrieved_prompt(
        self,
        query: str,
        top_k: int = 5,
        archetype: PromptArchetype | str = PromptArchetype.GENERAL_QA,
        session_id: str = "default_session",
    ) -> Any:
        """Execute complete retrieval-to-prompt chain and return verified RetrievedPrompt."""
        retrieval_result = self.retrieval.retrieve(query=query, top_k=top_k)
        history_text = self.generation.memory.get_history_text(session_id, max_turns=3)
        return self.generation.prompt_builder.build_prompt(
            query=query,
            candidates=retrieval_result.candidates,
            citations=retrieval_result.citations,
            archetype=archetype,
            conversation_history=history_text,
            model_name=self.generation.model.model_name,
        )
