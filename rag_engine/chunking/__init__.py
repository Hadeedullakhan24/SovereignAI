"""Enterprise Chunking Engine for Sovereign On-Premise Agentic AI Workbench (SIH26117).

Provides multi-strategy, hierarchical, deterministic, retrieval-optimized chunking:
- Fixed-size chunking (token & character based)
- Recursive chunking (document -> section -> paragraph -> sentence -> token window)
- Section-aware chunking (manuals, P&IDs, standards, safety codes)
- Table-aware chunking (captions, headers, row integrity)
- List-aware chunking (SOP procedures, checklists, bullet points)
"""

from rag_engine.chunking.base_chunker import BaseChunker
from rag_engine.chunking.chunk_context import ChunkContext
from rag_engine.chunking.chunk_events import (
    ChunkCreated,
    ChunkEvent,
    ChunkEventBus,
    ChunkingFailed,
    ChunkingFinished,
    ChunkingStarted,
    ChunkValidationFailed,
    get_chunk_event_bus,
)
from rag_engine.chunking.chunk_factory import ChunkFactory, get_chunk_factory
from rag_engine.chunking.chunk_health import ChunkHealthReport, health
from rag_engine.chunking.chunk_metrics import (
    ChunkMetricsCollector,
    get_chunk_metrics_collector,
)
from rag_engine.chunking.chunk_registry import (
    ChunkRegistry,
    get_chunk_registry,
    register_chunker,
)
from rag_engine.chunking.chunk_utils import (
    compute_sha256,
    count_characters,
    count_words,
    estimate_tokens,
    generate_deterministic_chunk_id,
    split_into_list_items,
    split_into_paragraphs,
    split_into_sentences,
)
from rag_engine.chunking.chunk_validator import ChunkValidator
from rag_engine.chunking.exceptions import (
    ChunkingError,
    EmptyDocumentError,
    OversizedChunkError,
    StrategyNotFoundError,
    ValidationRejectionError,
)
from rag_engine.chunking.fixed_chunker import FixedChunker
from rag_engine.chunking.hierarchy_builder import HierarchyBuilder
from rag_engine.chunking.list_chunker import ListChunker
from rag_engine.chunking.metadata_inheritance import MetadataInheritor
from rag_engine.chunking.recursive_chunker import RecursiveChunker
from rag_engine.chunking.section_chunker import SectionChunker
from rag_engine.chunking.table_chunker import TableChunker

__all__ = [
    # Base and Context
    "BaseChunker",
    "ChunkContext",
    # Factory & Registry
    "ChunkFactory",
    "get_chunk_factory",
    "ChunkRegistry",
    "get_chunk_registry",
    "register_chunker",
    # Concrete Strategies
    "FixedChunker",
    "RecursiveChunker",
    "SectionChunker",
    "TableChunker",
    "ListChunker",
    # Pipeline & Lifecycle
    "ChunkValidator",
    "HierarchyBuilder",
    "MetadataInheritor",
    "ChunkMetricsCollector",
    "get_chunk_metrics_collector",
    "ChunkEventBus",
    "get_chunk_event_bus",
    "ChunkHealthReport",
    "health",
    # Events
    "ChunkEvent",
    "ChunkingStarted",
    "ChunkCreated",
    "ChunkValidationFailed",
    "ChunkingFinished",
    "ChunkingFailed",
    # Exceptions
    "ChunkingError",
    "EmptyDocumentError",
    "ValidationRejectionError",
    "StrategyNotFoundError",
    "OversizedChunkError",
    # Utilities
    "estimate_tokens",
    "count_words",
    "count_characters",
    "compute_sha256",
    "generate_deterministic_chunk_id",
    "split_into_sentences",
    "split_into_paragraphs",
    "split_into_list_items",
]
