"""
Interface: BaseChunker

Abstract contract for text chunking strategies. Implementations include
fixed-size, semantic, and recursive chunking.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag_engine.schemas.chunk import Chunk
    from rag_engine.schemas.document import Document
    from rag_engine.schemas.parsed_document import ParsedDocument


class BaseChunker(ABC):
    """Abstract base class for document chunking strategies."""

    @abstractmethod
    def chunk(self, document: Document | ParsedDocument) -> list[Chunk]:
        """
        Split a document into chunks.

        Args:
            document: The Document or ParsedDocument to chunk.

        Returns:
            Ordered list of Chunk instances.

        Raises:
            ValueError: If the document content is empty.
        """
        ...

    @abstractmethod
    def get_strategy_name(self) -> str:
        """
        Return the name of the chunking strategy.

        Returns:
            Strategy identifier string, e.g. "fixed", "semantic", "recursive".
        """
        ...
