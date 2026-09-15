"""Generation Pipeline — Master In-Process Generation Gateway.

Coordinates the 11-stage generation lifecycle:
1. Conversation Memory retrieval
2. Safety validation on user query
3. Token-budgeted Context & Prompt assembly
4. Deterministic Generation Cache check
5. In-process local LLM execution (via Hugging Face or DeterministicTestLLM)
6. Citation anchor validation & phantom pruning
7. Hallucination guard technical entity cross-check
8. Composite confidence scoring
9. Response formatting with bibliographic provenance
10. Cache storage and session memory update
11. Telemetry and event publishing
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import time
from typing import Any, Dict, Iterator, Optional, Sequence

from rag_engine.generation.generation_config import GenerationConfig
from rag_engine.generation.generation_events import (
    GenerationEvent,
    GenerationEventBus,
    GenerationEventType,
)
from rag_engine.generation.generation_exceptions import (
    PromptAssemblyError,
    SafetyViolationError,
)
from rag_engine.generation.generation_metrics import (
    GenerationMetrics,
    GenerationMetricsCollector,
    get_current_process_memory_mb,
)
from rag_engine.generation.guardrails.answer_alignment_validator import (
    AlignmentEvaluationReport,
    AnswerAlignmentValidator,
)
from rag_engine.generation.guardrails.citation_validator import (
    CitationValidationReport,
    CitationValidator,
)
from rag_engine.generation.guardrails.confidence_scorer import (
    ConfidenceBreakdown,
    ConfidenceScorer,
)
from rag_engine.generation.guardrails.hallucination_guard import (
    GroundingVerificationReport,
    HallucinationGuard,
)
from rag_engine.generation.guardrails.safety_validator import SafetyValidator
from rag_engine.generation.memory.conversation_memory import ConversationMemory
from rag_engine.generation.memory.generation_cache import GenerationCache
from rag_engine.generation.models.base_model import BaseLocalLLM
from rag_engine.generation.models.model_factory import LLMFactory
from rag_engine.generation.prompt.prompt_builder import PromptBuilder, PromptPayload
from rag_engine.generation.prompt.prompt_templates import PromptArchetype
from rag_engine.generation.response_formatter import ResponseFormatter
from rag_engine.generation.streaming_manager import StreamingManager
from rag_engine.retrieval.base_retriever import CitationBundle, RetrievalResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GenerationResponse:
    """Complete, validated generation result ready for end-user presentation."""

    session_id: str
    query: str
    answer: str
    raw_answer: str
    prompt_payload: Optional[PromptPayload]
    citations: list[CitationBundle]
    citation_report: CitationValidationReport
    grounding_report: GroundingVerificationReport
    confidence: ConfidenceBreakdown
    metrics: GenerationMetrics
    alignment_report: Optional[AlignmentEvaluationReport] = None
    cache_hit: bool = False


class GenerationPipeline:
    """Master generation gateway orchestrating in-process local generation."""

    def __init__(
        self,
        config: GenerationConfig | None = None,
        model: BaseLocalLLM | None = None,
        prompt_builder: PromptBuilder | None = None,
        memory: ConversationMemory | None = None,
        cache: GenerationCache | None = None,
    ) -> None:
        self.config = config or GenerationConfig()
        self.model = model or LLMFactory.get_instance().create(
            self.config.default_model_name,
            config=self.config,
        )
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.memory = memory or ConversationMemory()
        self.cache = cache or (GenerationCache(self.config.cache_db_path) if self.config.cache_enabled else None)

        self.citation_validator = CitationValidator(
            strip_phantom_citations=self.config.guardrails.strip_phantom_citations
        )
        self.hallucination_guard = HallucinationGuard(
            tolerance_threshold=self.config.guardrails.hallucination_tolerance_threshold
        )
        self.alignment_validator = AnswerAlignmentValidator()
        self.safety_validator = SafetyValidator()
        self.confidence_scorer = ConfidenceScorer()
        self.streaming_manager = StreamingManager()

        self.metrics_collector = GenerationMetricsCollector.get_instance()
        self.event_bus = GenerationEventBus.get_instance()

    def generate(
        self,
        query: str,
        retrieval_result: RetrievalResult,
        session_id: str = "default_session",
        archetype: PromptArchetype | str = PromptArchetype.GENERAL_QA,
    ) -> GenerationResponse:
        """Execute synchronous 11-stage generation pipeline."""
        start_total = time.perf_counter()

        # 1. Safety Check on Query Input
        if self.config.guardrails.enable_safety_validation:
            safety_res = self.safety_validator.validate_input(query)
            if not safety_res.is_safe:
                self.event_bus.publish(
                    GenerationEvent(
                        event_type=GenerationEventType.SAFETY_FLAGGED,
                        session_id=session_id,
                        query=query,
                        data={"flags": safety_res.flags},
                    )
                )
                raise SafetyViolationError(f"Query rejected by safety validator: {safety_res.flags}")

        # 2. Retrieve Conversation Memory
        history_text = self.memory.get_history_text(session_id, max_turns=3)

        # 3. Build Deterministic Prompt
        start_prompt = time.perf_counter()
        prompt_payload = self.prompt_builder.build_prompt(
            query=query,
            candidates=retrieval_result.candidates,
            citations=retrieval_result.citations,
            archetype=archetype,
            conversation_history=history_text,
            model_name=self.model.model_name,
        )
        prompt_ms = (time.perf_counter() - start_prompt) * 1000.0

        self.event_bus.publish(
            GenerationEvent(
                event_type=GenerationEventType.PROMPT_BUILT,
                session_id=session_id,
                query=query,
                data={"prompt_hash": prompt_payload.prompt_hash, "tokens": prompt_payload.token_count},
            )
        )

        # 4. Check Generation Cache
        chunk_ids = [c.chunk.chunk_id for c in retrieval_result.candidates]
        cache_key = GenerationCache.compute_cache_key(
            prompt_hash=prompt_payload.prompt_hash,
            chunk_ids=chunk_ids,
            model_name=self.model.model_name,
        )

        if self.cache is not None and self.config.cache_enabled:
            cached = self.cache.get(cache_key)
            if cached is not None:
                # A cache entry is not evidence.  Revalidate it against the
                # current selected context; indexes and source versions can
                # change between requests.
                cached_citation_report = self.citation_validator.validate(
                    generated_text=cached.response_text,
                    valid_anchors=prompt_payload.chunk_to_anchor_map,
                    valid_sources=[c.document_id for c in retrieval_result.citations],
                    citation_context=retrieval_result.citations,
                )
                cached_grounding = self.hallucination_guard.verify(
                    cached_citation_report.cleaned_text, retrieval_result.formatted_context
                )
                if not (cached_citation_report.is_valid and cached_grounding.is_grounded):
                    cached = None
            if cached is not None:
                self.event_bus.publish(
                    GenerationEvent(
                        event_type=GenerationEventType.CACHE_HIT,
                        session_id=session_id,
                        query=query,
                        data={"cache_key": cache_key},
                    )
                )
                total_ms = (time.perf_counter() - start_total) * 1000.0
                metrics = GenerationMetrics(
                    session_id=session_id,
                    query=query,
                    model_name=self.model.model_name,
                    prompt_tokens=prompt_payload.token_count,
                    generated_tokens=self.model.count_tokens(cached.response_text),
                    prompt_build_latency_ms=round(prompt_ms, 2),
                    total_pipeline_latency_ms=round(total_ms, 2),
                    cache_hit=True,
                    memory_rss_mb=get_current_process_memory_mb(),
                )
                self.metrics_collector.record(metrics)

                # Reconstruct dummy validation reports for cached output
                cit_report = CitationValidationReport(
                    cleaned_text=cached.response_text,
                    total_citations_found=len(cached.citations),
                    valid_citations=cached.citations,
                    phantom_citations=[],
                    citation_precision=1.0,
                    is_valid=True,
                )
                ground_report = cached_grounding
                conf = self.confidence_scorer.calculate(1.0, 1.0, ground_report.grounding_score)

                return GenerationResponse(
                    session_id=session_id,
                    query=query,
                    answer=cached.response_text,
                    raw_answer=cached.response_text,
                    prompt_payload=prompt_payload,
                    citations=list(retrieval_result.citations),
                    citation_report=cit_report,
                    grounding_report=ground_report,
                    confidence=conf,
                    metrics=metrics,
                    cache_hit=True,
                )

        # 5. Local Model Execution
        self.event_bus.publish(
            GenerationEvent(
                event_type=GenerationEventType.GENERATION_STARTED,
                session_id=session_id,
                query=query,
                data={"model": self.model.model_name},
            )
        )

        start_model = time.perf_counter()
        gen_output = self.model.generate(
            prompt=prompt_payload.prompt_text,
            config=self.config.inference,
        )
        model_ms = (time.perf_counter() - start_model) * 1000.0

        # 6. Citation Validation & Phantom Pruning
        start_guard = time.perf_counter()
        valid_sources = [
            c.document_id for c in retrieval_result.citations if getattr(c, "document_id", None)
        ] + [
            getattr(c, "document_name", "") for c in retrieval_result.citations if getattr(c, "document_name", None)
        ]
        cit_report = self.citation_validator.validate(
            generated_text=gen_output.text,
            valid_anchors=prompt_payload.chunk_to_anchor_map,
            valid_sources=valid_sources,
            citation_context=retrieval_result.citations,
        )

        # 7. Hallucination Guard Cross-Verification
        context_text = retrieval_result.formatted_context
        ground_report = self.hallucination_guard.verify(
            generated_text=cit_report.cleaned_text,
            source_context=context_text,
        )

        # 8. Answer Alignment & Quality Evaluation
        alignment_report = self.alignment_validator.evaluate(
            query=query,
            answer_text=ground_report.cleaned_text or cit_report.cleaned_text,
            source_context=context_text,
            is_grounded=ground_report.is_grounded,
            is_citation_valid=cit_report.is_valid,
        )

        # 9. Composite Confidence Calculation
        avg_retrieval_score = (
            sum(c.score for c in retrieval_result.candidates) / len(retrieval_result.candidates)
            if retrieval_result.candidates
            else 0.5
        )
        effective_grounding = ground_report.grounding_score if alignment_report.is_question_aligned else min(ground_report.grounding_score, 0.4)
        confidence = self.confidence_scorer.calculate(
            retrieval_confidence=avg_retrieval_score,
            citation_precision=cit_report.citation_precision,
            grounding_score=effective_grounding,
        )
        guard_ms = (time.perf_counter() - start_guard) * 1000.0

        # 10. Format Response with Provenance References
        clean_ans = alignment_report.cleaned_text
        # A response that fails either grounding or answer/evidence alignment
        # is never formatted as a trustworthy answer or turned into an artifact.
        if not ground_report.is_grounded or not alignment_report.is_question_aligned:
            clean_ans = "Not documented in the available evidence."
        final_answer = ResponseFormatter.format_with_provenance(
            answer_text=clean_ans,
            citations=retrieval_result.citations,
        )

        # Output Safety Validation (credential redaction)
        if self.config.guardrails.enable_safety_validation:
            output_safety = self.safety_validator.validate_output(final_answer)
            final_answer = output_safety.redacted_text

        # 10. Update Conversation Memory & Generation Cache
        self.memory.add_turn(
            session_id=session_id,
            user_query=query,
            response=final_answer,
            retrieved_chunk_ids=chunk_ids,
            citations=cit_report.valid_citations,
            metadata={"confidence": confidence.composite_score},
        )

        if self.cache is not None and self.config.cache_enabled:
            self.cache.put(
                cache_key=cache_key,
                prompt_hash=prompt_payload.prompt_hash,
                model_name=self.model.model_name,
                response_text=final_answer,
                citations=cit_report.valid_citations,
                metadata={"confidence": confidence.composite_score},
            )

        # 11. Record Telemetry & Publish Completed Event
        total_ms = (time.perf_counter() - start_total) * 1000.0
        tok_per_sec = (
            gen_output.completion_tokens / (model_ms / 1000.0)
            if model_ms > 0
            else 0.0
        )

        metrics = GenerationMetrics(
            session_id=session_id,
            query=query,
            model_name=self.model.model_name,
            prompt_tokens=gen_output.prompt_tokens,
            generated_tokens=gen_output.completion_tokens,
            prompt_build_latency_ms=round(prompt_ms, 2),
            model_generation_latency_ms=round(model_ms, 2),
            guardrail_latency_ms=round(guard_ms, 2),
            total_pipeline_latency_ms=round(total_ms, 2),
            tokens_per_second=round(tok_per_sec, 2),
            cache_hit=False,
            memory_rss_mb=get_current_process_memory_mb(),
            total_citations=cit_report.total_citations_found,
            verified_citations=len(cit_report.valid_citations),
            stripped_citations=len(cit_report.phantom_citations),
            grounding_score=ground_report.grounding_score,
        )
        self.metrics_collector.record(metrics)

        self.event_bus.publish(
            GenerationEvent(
                event_type=GenerationEventType.GENERATION_COMPLETED,
                session_id=session_id,
                query=query,
                data={
                    "total_ms": metrics.total_pipeline_latency_ms,
                    "confidence": confidence.composite_score,
                    "tokens": metrics.generated_tokens,
                },
            )
        )

        return GenerationResponse(
            session_id=session_id,
            query=query,
            answer=final_answer,
            raw_answer=gen_output.text,
            prompt_payload=prompt_payload,
            citations=list(retrieval_result.citations),
            citation_report=cit_report,
            grounding_report=ground_report,
            confidence=confidence,
            metrics=metrics,
            alignment_report=alignment_report,
            cache_hit=False,
        )

    def stream_generate(
        self,
        query: str,
        retrieval_result: RetrievalResult,
        session_id: str = "default_session",
        archetype: PromptArchetype | str = PromptArchetype.GENERAL_QA,
    ) -> Iterator[str]:
        """Stream generated response tokens incrementally."""
        history_text = self.memory.get_history_text(session_id, max_turns=3)
        prompt_payload = self.prompt_builder.build_prompt(
            query=query,
            candidates=retrieval_result.candidates,
            citations=retrieval_result.citations,
            archetype=archetype,
            conversation_history=history_text,
            model_name=self.model.model_name,
        )

        stream = self.model.stream_generate(
            prompt=prompt_payload.prompt_text,
            config=self.config.inference,
        )

        for token in stream:
            yield token
