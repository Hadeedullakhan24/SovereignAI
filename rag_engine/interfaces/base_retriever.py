"""
Interface: BaseRetriever

Abstract contract for document retrieval strategies. Implementations
include dense retrieval, hybrid retrieval, and contextual retrieval.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag_engine.schemas.retrieved_document import RetrievedDocument


class BaseRetriever(ABC):
    """Abstract base class for retrieval strategies."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> RetrievedDocument:
        """
        Retrieve relevant chunks for the given query.

        Args:
            query: Natural language query string.
            top_k: Maximum number of chunks to retrieve.

        Returns:
            RetrievedDocument containing scored chunks and metadata.

        Raises:
            ValueError: If query is empty.
        """
        ...

    @abstractmethod
    def get_strategy_name(self) -> str:
        """
        Return the name of the retrieval strategy.

        Returns:
            Strategy identifier string, e.g. "dense", "hybrid", "contextual".
        """
        ...
