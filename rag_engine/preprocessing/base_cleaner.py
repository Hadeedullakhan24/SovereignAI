"""Abstract Base Class for cleaners and normalization stages."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from rag_engine.schemas.parsed_document import CleanParsedDocument, ParsedDocument


class BaseCleaner(ABC):
    """Abstract Base Class for all cleaning components and specialized cleaners."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique name of the cleaner."""
        ...

    @property
    def version(self) -> str:
        """Return the version string of the cleaner."""
        return "1.0.0"

    @abstractmethod
    def clean(self, document: ParsedDocument) -> CleanParsedDocument:
        """Execute cleaning pipeline on a ParsedDocument and return CleanParsedDocument."""
        ...

    def health(self) -> dict[str, Any]:
        """Return component health status."""
        return {
            "name": self.name,
            "version": self.version,
            "status": "HEALTHY",
        }
