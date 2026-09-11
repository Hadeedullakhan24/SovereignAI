"""ChunkFactory: Master factory for instantiating and auto-selecting chunker strategies."""

from __future__ import annotations

import threading
from typing import Any, Optional, Union

from rag_engine.chunking.base_chunker import BaseChunker
from rag_engine.chunking.chunk_context import ChunkContext
from rag_engine.chunking.chunk_events import ChunkEventBus, get_chunk_event_bus
from rag_engine.chunking.chunk_metrics import ChunkMetricsCollector, get_chunk_metrics_collector
from rag_engine.chunking.chunk_registry import ChunkRegistry, get_chunk_registry
from rag_engine.chunking.chunk_validator import ChunkValidator
from rag_engine.chunking.fixed_chunker import FixedChunker
from rag_engine.chunking.list_chunker import ListChunker
from rag_engine.chunking.recursive_chunker import RecursiveChunker
from rag_engine.chunking.section_chunker import SectionChunker
from rag_engine.chunking.table_chunker import TableChunker
from rag_engine.schemas.document import Document
from rag_engine.schemas.parsed_document import CleanParsedDocument, ParsedDocument


class ChunkFactory:
    """Enterprise factory for creating chunker strategy instances.
    
    Provides explicit strategy instantiation, document-adaptive strategy selection,
    and parameter configuration while ensuring thread safety.
    """

    def __init__(
        self,
        registry: Optional[ChunkRegistry] = None,
        event_bus: Optional[ChunkEventBus] = None,
        metrics: Optional[ChunkMetricsCollector] = None,
    ) -> None:
        self._registry = registry or get_chunk_registry()
        self._event_bus = event_bus or get_chunk_event_bus()
        self._metrics = metrics or get_chunk_metrics_collector()
        self._lock = threading.RLock()

    def create(
        self,
        strategy_name: str,
        context: Optional[ChunkContext] = None,
        **kwargs: Any,
    ) -> BaseChunker:
        """Instantiate a chunker by strategy identifier."""
        with self._lock:
            chunker_cls = self._registry.get(strategy_name)
            ctx = context or ChunkContext(strategy_name=strategy_name, **kwargs)
            validator = ChunkValidator(context=ctx)
            return chunker_cls(
                context=ctx,
                event_bus=self._event_bus,
                metrics=self._metrics,
                validator=validator,
            )

    def for_document(
        self,
        document: Union[Document, ParsedDocument, CleanParsedDocument],
        context: Optional[ChunkContext] = None,
    ) -> BaseChunker:
        """Auto-select the most appropriate chunking strategy based on document structure."""
        with self._lock:
            # Check document structure
            tables = getattr(document, "tables", [])
            sections = getattr(document, "sections", [])
            doc_cat = str(getattr(document, "category", "")).lower()

            # 1. Documents dominated by tables (CSVs, inspection spreadsheets)
            if tables and len(tables) > len(sections) * 2:
                strat = "table"
            # 2. Procedures, SOPs, checklists
            elif "procedure" in doc_cat or "sop" in doc_cat or "checklist" in doc_cat:
                strat = "list"
            # 3. Formal engineering manuals, standards with rich section hierarchies
            elif sections and len(sections) >= 2:
                strat = "section"
            # 4. Default enterprise robust strategy
            else:
                strat = "recursive"

            ctx = context or ChunkContext(strategy_name=strat)
            return self.create(strat, context=ctx)

    def create_fixed(
        self,
        target_tokens: int = 512,
        overlap: int = 64,
        unit: str = "token",
        **kwargs: Any,
    ) -> FixedChunker:
        """Create a configured FixedChunker instance."""
        ctx = ChunkContext(
            strategy_name="fixed",
            target_tokens=target_tokens,
            overlap_tokens=overlap,
            chunking_unit=unit,
            **kwargs,
        )
        return FixedChunker(
            context=ctx,
            mode=unit,
            event_bus=self._event_bus,
            metrics=self._metrics,
            validator=ChunkValidator(ctx),
        )

    def create_recursive(
        self,
        target_tokens: int = 512,
        overlap: int = 64,
        **kwargs: Any,
    ) -> RecursiveChunker:
        """Create a configured RecursiveChunker instance."""
        ctx = ChunkContext(
            strategy_name="recursive",
            target_tokens=target_tokens,
            overlap_tokens=overlap,
            **kwargs,
        )
        return RecursiveChunker(
            context=ctx,
            event_bus=self._event_bus,
            metrics=self._metrics,
            validator=ChunkValidator(ctx),
        )

    def create_section(
        self,
        target_tokens: int = 512,
        overlap: int = 64,
        preserve_headers: bool = True,
        **kwargs: Any,
    ) -> SectionChunker:
        """Create a configured SectionChunker instance."""
        ctx = ChunkContext(
            strategy_name="section",
            target_tokens=target_tokens,
            overlap_tokens=overlap,
            preserve_headers=preserve_headers,
            **kwargs,
        )
        return SectionChunker(
            context=ctx,
            event_bus=self._event_bus,
            metrics=self._metrics,
            validator=ChunkValidator(ctx),
        )

    def create_table(
        self,
        target_tokens: int = 512,
        repeat_headers_on_split: bool = True,
        **kwargs: Any,
    ) -> TableChunker:
        """Create a configured TableChunker instance."""
        ctx = ChunkContext(
            strategy_name="table",
            target_tokens=target_tokens,
            repeat_headers_on_split=repeat_headers_on_split,
            **kwargs,
        )
        return TableChunker(
            context=ctx,
            event_bus=self._event_bus,
            metrics=self._metrics,
            validator=ChunkValidator(ctx),
        )

    def create_list(
        self,
        target_tokens: int = 512,
        preserve_list_cohesion: bool = True,
        **kwargs: Any,
    ) -> ListChunker:
        """Create a configured ListChunker instance."""
        ctx = ChunkContext(
            strategy_name="list",
            target_tokens=target_tokens,
            preserve_list_cohesion=preserve_list_cohesion,
            **kwargs,
        )
        return ListChunker(
            context=ctx,
            event_bus=self._event_bus,
            metrics=self._metrics,
            validator=ChunkValidator(ctx),
        )


# Global factory instance
_GLOBAL_FACTORY = ChunkFactory()


def get_chunk_factory() -> ChunkFactory:
    """Get the global ChunkFactory singleton."""
    return _GLOBAL_FACTORY
