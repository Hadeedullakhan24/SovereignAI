"""Hierarchy builder establishing parent-child and sequential chunk linkage."""

from __future__ import annotations

from typing import Optional
from rag_engine.schemas.chunk import Chunk, ChunkHierarchy


class HierarchyBuilder:
    """Links sequential chunks and builds bidirectional parent-child document hierarchies."""

    @classmethod
    def link_chunks(cls, chunks: list[Chunk]) -> list[Chunk]:
        """Establish prev_chunk_id and next_chunk_id references across a sequence of chunks."""
        if not chunks:
            return []

        total = len(chunks)
        linked: list[Chunk] = []

        for i, chk in enumerate(chunks):
            prev_id: Optional[str] = chunks[i - 1].chunk_id if i > 0 else None
            next_id: Optional[str] = chunks[i + 1].chunk_id if i < total - 1 else None

            existing_hierarchy = chk.hierarchy
            depth = existing_hierarchy.hierarchy_depth if existing_hierarchy else 1
            if chk.metadata.heading_path:
                depth = max(1, len(chk.metadata.heading_path))

            hierarchy = ChunkHierarchy(
                document_id=chk.metadata.document_id,
                parent_section_id=chk.metadata.section_id,
                heading_path=chk.metadata.heading_path,
                prev_chunk_id=prev_id,
                next_chunk_id=next_id,
                hierarchy_depth=depth,
            )

            linked_chunk = chk.model_copy(update={"hierarchy": hierarchy})
            linked.append(linked_chunk)

        return linked
