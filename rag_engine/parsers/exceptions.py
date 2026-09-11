"""Custom exception hierarchy for the document parsing engine."""

from typing import Optional


class ParserError(Exception):
    """Base exception for all document parsing errors."""

    def __init__(
        self,
        message: str,
        document_id: Optional[str] = None,
        source_path: Optional[str] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.document_id = document_id
        self.source_path = source_path
        self.cause = cause

    def __str__(self) -> str:
        parts = [self.message]
        if self.document_id:
            parts.append(f"document_id='{self.document_id}'")
        if self.source_path:
            parts.append(f"source_path='{self.source_path}'")
        if self.cause:
            parts.append(f"caused_by={type(self.cause).__name__}('{self.cause}')")
        return " | ".join(parts)


class ParserInitializationError(ParserError):
    """Raised when a parser fails to initialize or missing internal dependencies."""


class UnsupportedDocumentError(ParserError):
    """Raised when no parser can handle the provided document."""


class ParsingFailedError(ParserError):
    """Raised when parsing fails across all drivers and fallbacks."""


class CorruptedContentError(ParserError):
    """Raised when document content or binary streams are corrupt or truncated."""


class ParserTimeoutError(ParserError):
    """Raised when parsing exceeds configured timeout threshold."""


class ValidationError(ParserError):
    """Raised when document fails critical structural validation."""


class ProfileError(ParserError):
    """Raised when a refinery profile is invalid or cannot be applied."""
