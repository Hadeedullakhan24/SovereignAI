"""Domain-specific exceptions for the Cleaning & Normalization Engine."""

from __future__ import annotations

from typing import Optional


class CleaningError(Exception):
    """Base exception for all cleaning and normalization errors."""

    def __init__(self, message: str, stage_name: Optional[str] = None) -> None:
        super().__init__(message)
        self.message = message
        self.stage_name = stage_name

    def __str__(self) -> str:
        if self.stage_name:
            return f"[{self.stage_name}] {self.message}"
        return self.message


class PipelineStageError(CleaningError):
    """Raised when a specific pipeline stage fails during document cleaning."""


class TokenProtectionError(CleaningError):
    """Raised when a protected engineering token is corrupted, modified, or missing."""


class PluginError(CleaningError):
    """Raised when a custom cleaning plugin fails to load, register, or execute."""


class CorruptedDocumentError(CleaningError):
    """Raised when a parsed document structure is unrecoverably malformed."""
