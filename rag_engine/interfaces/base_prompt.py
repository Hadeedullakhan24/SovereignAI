"""Abstract base interfaces for Milestone 9 Prompt Engineering and Context Assembly.

Defines the formal contracts for:
- BasePromptBuilder
- BasePromptTemplate
- BasePromptContextBuilder
- BaseTokenBudgetManager
- BaseContextCompressor
- BaseConversationFormatter
- BaseCitationFormatter
- BasePromptValidator
- BaseSystemPromptManager
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence

from rag_engine.schemas.chunk import Chunk
from rag_engine.schemas.prompt import (
    ContextWindow,
    PromptValidationResult,
    RetrievedPrompt,
    SystemPrompt,
)


class BasePromptTemplate(ABC):
    """Abstract contract for prompt templates."""

    @abstractmethod
    def format(self, **kwargs: Any) -> str:
        """Render template with provided variable bindings."""
        raise NotImplementedError

    @property
    @abstractmethod
    def template_name(self) -> str:
        """Name of the template."""
        raise NotImplementedError


class BaseSystemPromptManager(ABC):
    """Abstract contract for system prompt and persona lifecycle management."""

    @abstractmethod
    def get_system_prompt(
        self,
        archetype: str = "general_qa",
        custom_instructions: Optional[str] = None,
        **kwargs: Any,
    ) -> SystemPrompt:
        """Retrieve or synthesize the system prompt for a domain archetype."""
        raise NotImplementedError


class BaseTokenBudgetManager(ABC):
    """Abstract contract for token budget partitioning across prompt sections."""

    @abstractmethod
    def allocate(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Compute budget allocation across system, context, memory, and query."""
        raise NotImplementedError

    @abstractmethod
    def estimate_tokens(self, text: str) -> int:
        """Count or estimate tokens in a string."""
        raise NotImplementedError


class BaseContextCompressor(ABC):
    """Abstract contract for context compression and selective information extraction."""

    @abstractmethod
    def compress(
        self,
        context_chunks: Sequence[Chunk],
        token_budget: int,
        query: Optional[str] = None,
    ) -> list[Chunk]:
        """Compress context chunks to strictly satisfy the token budget."""
        raise NotImplementedError


class BasePromptContextBuilder(ABC):
    """Abstract contract for assembling retrieval candidates into a formatted context window."""

    @abstractmethod
    def build_context_window(
        self,
        candidates: Sequence[Any],
        citations: Sequence[Any],
        token_budget: int,
    ) -> ContextWindow:
        """Pack candidates and citations into a context window."""
        raise NotImplementedError


class BaseConversationFormatter(ABC):
    """Abstract contract for multi-turn conversation memory formatting."""

    @abstractmethod
    def format_history(
        self,
        turns: Sequence[Any],
        max_tokens: int,
        format_style: str = "markdown",
    ) -> str:
        """Render conversation history within a token budget."""
        raise NotImplementedError


class BaseCitationFormatter(ABC):
    """Abstract contract for citation formatting and bibliographic referencing."""

    @abstractmethod
    def format_citations(
        self,
        citations: Sequence[Any],
        format_style: str = "inline_anchors",
    ) -> str:
        """Format citations into specified layout (inline, footnote, tabular)."""
        raise NotImplementedError


class BasePromptValidator(ABC):
    """Abstract contract for prompt safety, token limit, and placeholder validation."""

    @abstractmethod
    def validate_prompt(
        self,
        prompt: RetrievedPrompt | str,
        max_context_length: Optional[int] = None,
    ) -> PromptValidationResult:
        """Perform validation on synthesized prompt."""
        raise NotImplementedError


class BasePromptBuilder(ABC):
    """Master abstract contract for prompt synthesis and cryptographic lineage stamping."""

    @abstractmethod
    def build_prompt(
        self,
        query: str,
        candidates: Sequence[Any],
        citations: Sequence[Any],
        archetype: str = "general_qa",
        conversation_history: str = "",
        model_name: str = "local_llm",
        **kwargs: Any,
    ) -> RetrievedPrompt:
        """Synthesize auditable, deterministic prompt payload."""
        raise NotImplementedError
