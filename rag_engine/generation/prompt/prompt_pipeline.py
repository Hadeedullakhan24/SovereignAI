"""Dedicated Prompt Pipeline coordinating context assembly, compression, validation, and cryptographic lineage."""

from __future__ import annotations

import logging
import time
from typing import Any, Optional, Sequence

from rag_engine.generation.prompt.citation_formatter import CitationFormatter
from rag_engine.generation.prompt.context_compressor import ContextCompressor
from rag_engine.generation.prompt.context_window_builder import ContextWindowBuilder
from rag_engine.generation.prompt.conversation_formatter import ConversationFormatter
from rag_engine.generation.prompt.prompt_builder import PromptBuilder
from rag_engine.generation.prompt.prompt_config import PromptConfig
from rag_engine.generation.prompt.prompt_events import (
    PromptEvent,
    PromptEventBus,
    PromptEventType,
)
from rag_engine.generation.prompt.prompt_exceptions import (
    PromptBudgetExceededError,
    PromptValidationError,
)
from rag_engine.generation.prompt.prompt_metrics import (
    PromptMetrics,
    PromptMetricsCollector,
)
from rag_engine.generation.prompt.prompt_templates import (
    PromptArchetype,
    PromptTemplateRegistry,
)
from rag_engine.generation.prompt.prompt_validator import PromptValidator
from rag_engine.generation.prompt.system_prompt_manager import SystemPromptManager
from rag_engine.generation.prompt.token_budget_manager import TokenBudgetManager
from rag_engine.retrieval.base_retriever import CitationBundle, RetrievalResult, ScoredRetrievalChunk
from rag_engine.schemas.prompt import RetrievedPrompt

logger = logging.getLogger(__name__)


class PromptPipeline:
    """Master prompt execution pipeline delivering verified, auditable RetrievedPrompt payloads."""

    def __init__(
        self,
        config: Optional[PromptConfig] = None,
        builder: Optional[PromptBuilder] = None,
        system_prompt_manager: Optional[SystemPromptManager] = None,
        compressor: Optional[ContextCompressor] = None,
        conversation_formatter: Optional[ConversationFormatter] = None,
        citation_formatter: Optional[CitationFormatter] = None,
        validator: Optional[PromptValidator] = None,
    ) -> None:
        self.config = config or PromptConfig()
        self.builder = builder or PromptBuilder(
            prompt_version=self.config.prompt_version,
            retrieval_version=self.config.retrieval_version,
        )
        self.system_prompt_mgr = system_prompt_manager or SystemPromptManager(default_persona=self.config.persona)
        self.compressor = compressor or ContextCompressor()
        self.conversation_formatter = conversation_formatter or ConversationFormatter(default_style=self.config.conversation_style)
        self.citation_formatter = citation_formatter or CitationFormatter(default_style=self.config.citation_style)
        self.validator = validator or PromptValidator()

        self.event_bus = PromptEventBus.get_instance()
        self.metrics_collector = PromptMetricsCollector.get_instance()

    def process(
        self,
        query: str,
        retrieval_result: RetrievalResult,
        archetype: PromptArchetype | str = PromptArchetype.GENERAL_QA,
        conversation_turns: Optional[Sequence[Any]] = None,
        model_name: str = "local_llm",
        custom_system_instructions: Optional[str] = None,
    ) -> RetrievedPrompt:
        """Execute end-to-end prompt synthesis pipeline."""
        t0 = time.perf_counter()

        self.event_bus.publish(
            PromptEvent(
                event_type=PromptEventType.PROMPT_BUILD_START,
                query=query,
                data={"archetype": str(archetype)},
            )
        )

        # 1. Format conversation memory
        history_text = ""
        if conversation_turns:
            budget = getattr(self.builder.budget_manager.config, "memory_budget", 800)
            history_text = self.conversation_formatter.format_history(
                turns=conversation_turns,
                max_tokens=budget,
                format_style=self.config.conversation_style,
            )

        # 2. Build Core Prompt Payload
        retrieved_prompt = self.builder.build_prompt(
            query=query,
            candidates=retrieval_result.candidates,
            citations=retrieval_result.citations,
            archetype=archetype,
            conversation_history=history_text,
            model_name=model_name,
        )

        # 3. Validation Gate
        val_result = self.validator.validate_prompt(
            retrieved_prompt,
            max_context_length=getattr(self.builder.budget_manager.config, "max_context_window", 4096),
        )

        if not val_result.is_valid and self.config.strict_validation:
            self.event_bus.publish(
                PromptEvent(
                    event_type=PromptEventType.PROMPT_VALIDATION_FAILED,
                    query=query,
                    data={"errors": val_result.errors},
                )
            )
            raise PromptValidationError(f"Prompt failed validation: {'; '.join(val_result.errors)}")

        self.event_bus.publish(
            PromptEvent(
                event_type=PromptEventType.PROMPT_VALIDATED,
                query=query,
                data={"is_valid": val_result.is_valid, "tokens": val_result.total_tokens},
            )
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # 4. Record Telemetry
        self.metrics_collector.record(
            PromptMetrics(
                query=query,
                archetype=str(archetype),
                total_tokens=retrieved_prompt.token_count,
                system_tokens=retrieved_prompt.system_tokens,
                context_tokens=retrieved_prompt.context_tokens,
                history_tokens=retrieved_prompt.history_tokens,
                query_tokens=retrieved_prompt.query_tokens,
                context_chunks_count=len(retrieval_result.candidates),
                citations_count=len(retrieval_result.citations),
                assembly_latency_ms=round(elapsed_ms, 3),
            )
        )

        return retrieved_prompt
