"""Domain-specific exceptions for the Enterprise Chunking Engine."""

from __future__ import annotations

from typing import Optional


class ChunkingError(Exception):
    """Base exception for all chunking engine errors."""

    def __init__(self, message: str, strategy_name: Optional[str] = None) -> None:
        super().__init__(message)
        self.message = message
        self.strategy_name = strategy_name

    def __str__(self) -> str:
        if self.strategy_name:
            return f"[{self.strategy_name}] {self.message}"
        return self.message


class EmptyDocumentError(ChunkingError):
    """Raised when an input document has no readable content to chunk."""


class ValidationRejectionError(ChunkingError):
    """Raised when a chunk fails quality validation criteria (empty, tiny, oversized, duplicate)."""


class StrategyNotFoundError(ChunkingError):
    """Raised when an unregistered or unsupported chunking strategy is requested."""


class OversizedChunkError(ChunkingError):
    """Raised when an atomic unit cannot be reduced below the maximum token limit."""
