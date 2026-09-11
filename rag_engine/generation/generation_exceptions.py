"""Domain exceptions for Milestone 9: Enterprise Generation Engine."""

from __future__ import annotations


class BaseGenerationException(Exception):
    """Root exception for all Generation Engine errors."""
    pass


class PromptAssemblyError(BaseGenerationException):
    """Raised when prompt construction or template formatting fails."""
    pass


class ContextAssemblyError(BaseGenerationException):
    """Raised when context window packing or table preserving formatting fails."""
    pass


class TokenLimitExceededError(BaseGenerationException):
    """Raised when total prompt tokens exceed the configured context window budget."""
    pass


class SafetyViolationError(BaseGenerationException):
    """Raised when prompt injection, system manipulation, or credential leaks are detected."""
    pass


class HallucinationDetectedError(BaseGenerationException):
    """Raised when generated response contains ungrounded technical facts or fake metrics."""
    pass


class CitationValidationError(BaseGenerationException):
    """Raised when citations in generated text fail provenance verification."""
    pass


class LLMModelError(BaseGenerationException):
    """Base exception for model management and inference operations."""
    pass


class ModelNotFoundError(LLMModelError):
    """Raised when requested model weights directory is not found on local disk."""
    pass


class ModelLoadingError(LLMModelError):
    """Raised when local weights fail to load into memory."""
    pass


class GenerationInferenceError(LLMModelError):
    """Raised when in-process model execution fails during forward pass."""
    pass


class GenerationCacheError(BaseGenerationException):
    """Raised when SQLite generation cache operations fail."""
    pass


class StreamingError(BaseGenerationException):
    """Raised when token streaming iteration fails."""
    pass


class ConversationMemoryError(BaseGenerationException):
    """Raised when conversational history persistence or retrieval fails."""
    pass
