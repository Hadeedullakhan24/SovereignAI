"""
Interfaces package — Abstract Base Classes for the RAG Engine.

All contracts are defined here. Service modules implement these interfaces.
Consumers program against interfaces, never concrete implementations.
"""

from rag_engine.interfaces.base_chunker import BaseChunker
from rag_engine.interfaces.base_embedder import BaseEmbedder
from rag_engine.interfaces.base_loader import BaseLoader
from rag_engine.interfaces.base_prompt import (
    BaseCitationFormatter,
    BaseContextCompressor,
    BaseConversationFormatter,
    BasePromptBuilder,
    BasePromptContextBuilder,
    BasePromptTemplate,
    BasePromptValidator,
    BaseSystemPromptManager,
    BaseTokenBudgetManager,
)
from rag_engine.interfaces.base_retriever import BaseRetriever
from rag_engine.interfaces.base_vector_store import BaseVectorStore

__all__ = [
    # Milestones 1-5
    "BaseLoader",
    "BaseChunker",
    # Milestone 6
    "BaseEmbedder",
    # Milestone 7
    "BaseVectorStore",
    # Milestone 8
    "BaseRetriever",
    # Milestone 9
    "BasePromptBuilder",
    "BasePromptTemplate",
    "BasePromptContextBuilder",
    "BaseTokenBudgetManager",
    "BaseContextCompressor",
    "BaseConversationFormatter",
    "BaseCitationFormatter",
    "BasePromptValidator",
    "BaseSystemPromptManager",
]
