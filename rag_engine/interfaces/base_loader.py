"""
Interface: BaseLoader

Abstract contract for document loaders. Each file format (PDF, DOCX, MD,
CSV, TXT, etc.) provides a concrete implementation of this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag_engine.schemas.document import Document


class BaseLoader(ABC):
    """Abstract base class for document loaders."""

    @abstractmethod
    def load(self, file_path: Path) -> Document:
        """
        Load a single document from the given file path.

        Args:
            file_path: Absolute or relative path to the source file.

        Returns:
            A Document schema instance with content and metadata.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file format is not supported by this loader.
        """
        ...

    @abstractmethod
    def supported_formats(self) -> list[str]:
        """
        Return the list of file extensions this loader supports.

        Returns:
            List of lowercase extensions including the dot, e.g. [".pdf"].
        """
        ...

    def can_load(self, file_path: Path) -> bool:
        """
        Check if this loader supports the given file.

        Args:
            file_path: Path to check.

        Returns:
            True if the file extension is in supported_formats().
        """
        return file_path.suffix.lower() in self.supported_formats()
