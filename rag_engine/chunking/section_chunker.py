"""Section-aware chunker preserving engineering manual and report boundaries."""

from __future__ import annotations

from typing import Optional, Union

from rag_engine.chunking.base_chunker import BaseChunker
from rag_engine.chunking.chunk_context import ChunkContext
from rag_engine.chunking.chunk_utils import (
    estimate_tokens,
    split_into_paragraphs,
    split_into_sentences,
)
from rag_engine.chunking.metadata_inheritance import MetadataInheritor
from rag_engine.schemas.chunk import Chunk
from rag_engine.schemas.document import Document
from rag_engine.schemas.parsed_document import CleanParsedDocument, ParsedDocument, Section


class SectionChunker(BaseChunker):
    """Chunker that strictly preserves section boundaries for refinery manuals and reports."""

    def __init__(
        self,
        context: Optional[ChunkContext] = None,
        event_bus: Any = None,
        metrics: Any = None,
        validator: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(context=context, event_bus=event_bus, metrics=metrics, validator=validator)

    @property
    def strategy_name(self) -> str:
        return "section"

    def _chunk_document(
        self, document: Union[Document, ParsedDocument, CleanParsedDocument]
    ) -> list[Chunk]:
        """Chunk document respecting section boundaries (H1-H6)."""
        sections: list[Section] = getattr(document, "sections", [])
        if not sections:
            # Fallback if raw document has no sections: treat entire doc as one section
            full_text = getattr(document, "content", "")
            if hasattr(document, "get_full_text"):
                full_text = document.get_full_text()
            if not full_text.strip():
                return []
            sections = [
                Section(
                    section_id="sec_root",
                    title=getattr(document, "title", "Document"),
                    content=full_text,
                    level=1,
                )
            ]

        chunks: list[Chunk] = []
        curr_idx = 0

        for sec in sections:
            sec_chunks = self._process_section(
                sec=sec,
                document=document,
                start_index=curr_idx,
                heading_breadcrumbs=[sec.title] if sec.title else [],
            )
            chunks.extend(sec_chunks)
            curr_idx += len(sec_chunks)

        return chunks

    def _process_section(
        self,
        sec: Section,
        document: Union[Document, ParsedDocument, CleanParsedDocument],
        start_index: int,
        heading_breadcrumbs: list[str],
    ) -> list[Chunk]:
        """Process an individual section into one or more boundary-contained chunks."""
        chunks: list[Chunk] = []
        curr_idx = start_index

        content = sec.content.strip()
        if not content:
            # Recurse child subsections if any
            for sub in sec.subsections:
                sub_path = heading_breadcrumbs + ([sub.title] if sub.title else [])
                sub_chunks = self._process_section(sub, document, curr_idx, sub_path)
                chunks.extend(sub_chunks)
                curr_idx += len(sub_chunks)
            return chunks

        # Optional heading context prefix
        prefix = ""
        if self.context.include_heading_context and heading_breadcrumbs:
            prefix = f"[{' > '.join(heading_breadcrumbs)}]\n\n"

        tokens = estimate_tokens(content)

        # If section fits within max_chunk_tokens, keep it entirely cohesive!
        if tokens <= self.context.max_chunk_tokens:
            chunk_text = f"{prefix}{content}" if prefix else content
            meta = MetadataInheritor.inherit(
                document=document,
                chunk_content=chunk_text,
                chunk_index=curr_idx,
                section=sec,
                page_number=sec.page_number,
                heading_path=heading_breadcrumbs,
                context=self.context,
            )
            chk = Chunk.create(
                document_id=meta.document_id,
                content=chunk_text,
                chunk_index=curr_idx,
                document_name=meta.document_name,
                source_path=meta.source_path,
                page_number=meta.page_number,
                section_title=meta.section_title,
                section_id=meta.section_id,
                heading_path=meta.heading_path,
                category=meta.category,
                subcategory=meta.subcategory,
                plant_unit=meta.plant_unit,
                equipment_entities=meta.equipment_entities,
                safety_entities=meta.safety_entities,
                operating_parameters=meta.operating_parameters,
                chunk_strategy=self.strategy_name,
                token_count=estimate_tokens(chunk_text),
            )
            chunks.append(chk)
            curr_idx += 1
        else:
            # Section is too large: split across paragraphs and sentences within the section
            paras = split_into_paragraphs(content)
            units: list[str] = []
            for p in paras:
                if estimate_tokens(p) > self.context.target_chunk_tokens:
                    units.extend(split_into_sentences(p))
                else:
                    units.append(p)

            buffer: list[str] = []
            buf_tokens = 0

            for u in units:
                u_tokens = estimate_tokens(u)
                if buf_tokens + u_tokens > self.context.target_chunk_tokens and buffer:
                    body = "\n\n".join(buffer).strip()
                    chunk_text = f"{prefix}{body}" if prefix else body
                    meta = MetadataInheritor.inherit(
                        document=document,
                        chunk_content=chunk_text,
                        chunk_index=curr_idx,
                        section=sec,
                        page_number=sec.page_number,
                        heading_path=heading_breadcrumbs,
                        context=self.context,
                    )
                    chk = Chunk.create(
                        document_id=meta.document_id,
                        content=chunk_text,
                        chunk_index=curr_idx,
                        document_name=meta.document_name,
                        source_path=meta.source_path,
                        page_number=meta.page_number,
                        section_title=meta.section_title,
                        section_id=meta.section_id,
                        heading_path=meta.heading_path,
                        category=meta.category,
                        subcategory=meta.subcategory,
                        plant_unit=meta.plant_unit,
                        equipment_entities=meta.equipment_entities,
                        safety_entities=meta.safety_entities,
                        operating_parameters=meta.operating_parameters,
                        chunk_strategy=self.strategy_name,
                        token_count=estimate_tokens(chunk_text),
                    )
                    chunks.append(chk)
                    curr_idx += 1
                    buffer = [u]
                    buf_tokens = u_tokens
                else:
                    buffer.append(u)
                    buf_tokens += u_tokens

            if buffer:
                body = "\n\n".join(buffer).strip()
                chunk_text = f"{prefix}{body}" if prefix else body
                meta = MetadataInheritor.inherit(
                    document=document,
                    chunk_content=chunk_text,
                    chunk_index=curr_idx,
                    section=sec,
                    page_number=sec.page_number,
                    heading_path=heading_breadcrumbs,
                    context=self.context,
                )
                chk = Chunk.create(
                    document_id=meta.document_id,
                    content=chunk_text,
                    chunk_index=curr_idx,
                    document_name=meta.document_name,
                    source_path=meta.source_path,
                    page_number=meta.page_number,
                    section_title=meta.section_title,
                    section_id=meta.section_id,
                    heading_path=meta.heading_path,
                    category=meta.category,
                    subcategory=meta.subcategory,
                    plant_unit=meta.plant_unit,
                    equipment_entities=meta.equipment_entities,
                    safety_entities=meta.safety_entities,
                    operating_parameters=meta.operating_parameters,
                    chunk_strategy=self.strategy_name,
                    token_count=estimate_tokens(chunk_text),
                )
                chunks.append(chk)
                curr_idx += 1

        # Process nested child subsections
        for sub in sec.subsections:
            sub_path = heading_breadcrumbs + ([sub.title] if sub.title else [])
            sub_chunks = self._process_section(sub, document, curr_idx, sub_path)
            chunks.extend(sub_chunks)
            curr_idx += len(sub_chunks)

        return chunks
