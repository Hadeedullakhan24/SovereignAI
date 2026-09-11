"""Domain-specific exceptions for Prompt Engineering and Context Assembly."""

from __future__ import annotations


class PromptException(Exception):
    """Base exception for all prompt builder errors."""


class PromptValidationError(PromptException):
    """Raised when prompt syntax, token limit, or security checks fail."""


class PromptTemplateNotFoundError(PromptException):
    """Raised when an unrecognized prompt template archetype is requested."""


class PromptBudgetExceededError(PromptException):
    """Raised when the combined prompt exceeds the hard context window limit."""


class ContextCompressionError(PromptException):
    """Raised when context compression fails to satisfy target budget."""


class SystemPromptError(PromptException):
    """Raised when system prompt synthesis or persona configuration fails."""


class ConversationFormattingError(PromptException):
    """Raised when conversation history serialization fails."""


class CitationFormattingError(PromptException):
    """Raised when citation reference rendering fails."""
