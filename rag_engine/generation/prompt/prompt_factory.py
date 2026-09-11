"""Prompt Factory for dynamic creation and configuration of prompt builders."""

from __future__ import annotations

import threading
from typing import Optional

from rag_engine.generation.prompt.citation_formatter import CitationFormatter
from rag_engine.generation.prompt.context_compressor import ContextCompressor
from rag_engine.generation.prompt.context_window_builder import ContextWindowBuilder
from rag_engine.generation.prompt.conversation_formatter import ConversationFormatter
from rag_engine.generation.prompt.prompt_builder import PromptBuilder
from rag_engine.generation.prompt.prompt_config import PromptConfig
from rag_engine.generation.prompt.prompt_templates import PromptTemplateRegistry
from rag_engine.generation.prompt.prompt_validator import PromptValidator
from rag_engine.generation.prompt.system_prompt_manager import SystemPromptManager
from rag_engine.generation.prompt.token_budget_manager import TokenBudgetManager


class PromptFactory:
    """Thread-safe singleton factory for prompt components and pipeline instances."""

    _instance: PromptFactory | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._builders: dict[str, PromptBuilder] = {}
        self._f_lock = threading.RLock()

    @classmethod
    def get_instance(cls) -> PromptFactory:
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def create_prompt_builder(
        self,
        config: Optional[PromptConfig] = None,
        reuse_instance: bool = True,
    ) -> PromptBuilder:
        """Create or retrieve a fully wired PromptBuilder instance."""
        cfg = config or PromptConfig()
        cache_key = f"{cfg.prompt_version}:{cfg.retrieval_version}:{cfg.persona}"

        with self._f_lock:
            if reuse_instance and cache_key in self._builders:
                return self._builders[cache_key]

            templates = PromptTemplateRegistry()
            context_builder = ContextWindowBuilder()
            budget_manager = TokenBudgetManager()

            builder = PromptBuilder(
                template_registry=templates,
                context_builder=context_builder,
                budget_manager=budget_manager,
                prompt_version=cfg.prompt_version,
                retrieval_version=cfg.retrieval_version,
            )

            if reuse_instance:
                self._builders[cache_key] = builder

            return builder

    def create_system_prompt_manager(self) -> SystemPromptManager:
        return SystemPromptManager()

    def create_context_compressor(self) -> ContextCompressor:
        return ContextCompressor()

    def create_conversation_formatter(self) -> ConversationFormatter:
        return ConversationFormatter()

    def create_citation_formatter(self) -> CitationFormatter:
        return CitationFormatter()

    def create_prompt_validator(self) -> PromptValidator:
        return PromptValidator()

    def clear(self) -> None:
        with self._f_lock:
            self._builders.clear()


# Global factory helper
global_prompt_factory = PromptFactory.get_instance()
