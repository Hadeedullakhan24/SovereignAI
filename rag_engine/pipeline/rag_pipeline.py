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
from typing import Any, Dict, Iterator, List, Optional, Sequence

from rag_engine.generation.generation_config import GenerationConfig
from rag_engine.generation.generation_pipeline import (
    GenerationPipeline,
    GenerationResponse,
)
from rag_engine.generation.prompt.prompt_templates import (
    PromptArchetype,
    detect_task_type,
)
from rag_engine.generation.response_formatter import ResponseFormatter
from rag_engine.retrieval.base_retriever import (
    CitationBundle,
    RetrievalResult,
    ScoredRetrievalChunk,
)
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

    def format_clean_cli_output(self) -> str:
        """Render clean, human-readable terminal output adhering to modern conversational interface."""
        clean_ans = ResponseFormatter.strip_provenance(self.answer).strip()
        concise_sources = ResponseFormatter.format_concise_sources(
            self.citations,
            query=self.query,
            answer=clean_ans,
        )
        if concise_sources:
            return f"{clean_ans}\n\n{concise_sources}"
        return clean_ans

    def format_cli_output(self, detailed: bool = True) -> str:
        """Render terminal output. If detailed=True, adheres to 3-section format; otherwise returns clean output."""
        if not detailed:
            return self.format_clean_cli_output()

        detected = detect_task_type(self.query)
        header_map = {
            PromptArchetype.EMAIL: "GENERATED EMAIL",
            PromptArchetype.APPROVAL_NOTE: "GENERATED APPROVAL NOTE",
            PromptArchetype.REPORT: "GENERATED REPORT",
            PromptArchetype.SUMMARY: "GENERATED SUMMARY",
            PromptArchetype.COMPARISON: "GENERATED COMPARISON",
            PromptArchetype.EXTRACTION: "GENERATED EXTRACTION",
            PromptArchetype.SAFETY_COMPLIANCE: "GENERATED SAFETY REQUIREMENTS",
            PromptArchetype.SOP_RETRIEVAL: "GENERATED PROCEDURE",
            PromptArchetype.PROCEDURE: "GENERATED PROCEDURE",
            PromptArchetype.SPECIFICATION: "GENERATED SPECIFICATION",
            PromptArchetype.EQUIPMENT_LOOKUP: "GENERATED SPECIFICATION",
            PromptArchetype.MAINTENANCE: "GENERATED MAINTENANCE SUMMARY" if "summary" in self.query.lower() else "GENERATED ANSWER",
            PromptArchetype.TROUBLESHOOTING: "GENERATED TROUBLESHOOTING",
            PromptArchetype.ANALYSIS: "GENERATED ANALYSIS",
            PromptArchetype.GENERAL_QA: "GENERATED ANSWER",
        }

        clean_ans = ResponseFormatter.strip_provenance(self.answer)
        if "insufficient information" in clean_ans.lower() or "insufficient evidence" in clean_ans.lower() or not self.is_grounded:
            section_title = "GENERATED RESULT"
        else:
            section_title = header_map.get(detected, "GENERATED ANSWER")

        lines: list[str] = []
        lines.append("=" * 80)
        lines.append(f"MRPL SOVEREIGN AGENTIC AI WORKBENCH - {section_title}")
        lines.append("=" * 80)
        lines.append(f"Question: {self.query}\n")

        # --------------------------------------------------
        # SECTION 1: USER RESULT FIRST
        # --------------------------------------------------
        lines.append("-" * 80)
        lines.append(f"{section_title}")
        lines.append("-" * 80)
        lines.append(clean_ans.strip())
        lines.append("")

        # --------------------------------------------------
        # SECTION 2: SOURCES / EVIDENCE SECOND
        # --------------------------------------------------
        lines.append("-" * 80)
        lines.append("SOURCES / EVIDENCE")
        lines.append("-" * 80)
        if self.citations:
            for idx, c in enumerate(self.citations, 1):
                anchor = getattr(c, "citation_id", f"[{idx}]")
                doc = getattr(c, "document_name", None) or getattr(c, "document_id", "Document")
                page = f"Page {c.page_number}" if getattr(c, "page_number", None) else None
                section = f"Section: {c.section_title}" if getattr(c, "section_title", None) else None
                tag = f"Tag: {c.equipment_tag}" if getattr(c, "equipment_tag", None) else None

                meta_parts = [p for p in [page, section, tag] if p]
                meta_str = f" ({' | '.join(meta_parts)})" if meta_parts else ""

                lines.append(f"  {anchor} {doc}{meta_str}")
                if getattr(c, "verbatim_quote", None):
                    quote = c.verbatim_quote.strip().replace("\n", " ")
                    if len(quote) > 160:
                        quote = quote[:157] + "..."
                    lines.append(f'      "{quote}"')
        else:
            lines.append("  (No citations referenced or required)")
        lines.append("")

        # --------------------------------------------------
        # SECTION 3: AI / RAG DETAILS LAST
        # --------------------------------------------------
        lines.append("-" * 80)
        lines.append("AI / RAG DETAILS")
        lines.append("-" * 80)
        lines.append(f"  - Confidence Score    : {self.confidence_score * 100:.1f}%")
        lines.append(f"  - Factual Grounding   : {'Grounded' if self.is_grounded else 'Warning: Low Grounding'}")
        if self.execution_trace:
            lines.append(f"  - Retrieved Chunks    : {self.execution_trace.retrieved_chunks}")
        elif getattr(self.retrieval_result, "candidates", None):
            lines.append(f"  - Retrieved Chunks    : {len(self.retrieval_result.candidates)}")
        lines.append(f"  - Model Used          : {self.model_used or 'qwen2.5-1.5b-instruct'}")
        if self.execution_trace and self.execution_trace.generation_time_ms is not None:
            lines.append(f"  - Generation Time     : {self.execution_trace.generation_time_ms:.2f} ms")
        lines.append(f"  - Total RAG Latency   : {self.total_latency_ms:.2f} ms")
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
        effective_archetype = (
            detect_task_type(question)
            if archetype in (PromptArchetype.GENERAL_QA, "general_qa")
            else archetype
        )
        generation_start = time.perf_counter()
        generation_response = self.generation.generate(
            query=question,
            retrieval_result=retrieval_result,
            session_id=session_id,
            archetype=effective_archetype,
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
        effective_archetype = (
            detect_task_type(query)
            if archetype in (PromptArchetype.GENERAL_QA, "general_qa")
            else archetype
        )
        retrieval_result = self.retrieval.retrieve(query=query, top_k=top_k)
        stream = self.generation.stream_generate(
            query=query,
            retrieval_result=retrieval_result,
            session_id=session_id,
            archetype=effective_archetype,
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
