"""
Loader Exceptions — Custom Exception Hierarchy for Document Loaders.

Sovereign On-Premise Agentic AI Workbench (SIH26117 / MRPL)
"""

from __future__ import annotations


class LoaderError(Exception):
    """Base exception for all document loading failures."""

    def __init__(self, message: str, file_path: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.file_path = file_path

    def __str__(self) -> str:
        if self.file_path:
            return f"[{self.file_path}] {self.message}"
        return self.message


class UnsupportedFormatError(LoaderError):
    """Raised when an unrecognized or unregistered file format is encountered."""
    pass


class CorruptedDocumentError(LoaderError):
    """Raised when a document fails integrity verification or is unreadable."""
    pass


class LoaderInitializationError(LoaderError):
    """Raised when a loader cannot be initialized or lacks required dependencies."""
    pass


class ValidationFailedError(LoaderError):
    """Raised when pre-load validation checks fail."""
    pass
