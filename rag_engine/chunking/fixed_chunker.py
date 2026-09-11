"""Fixed-size chunker supporting token-based and character-based sliding windows with overlap."""

from __future__ import annotations

import re
from typing import Optional, Union

from rag_engine.chunking.base_chunker import BaseChunker
from rag_engine.chunking.chunk_context import ChunkContext
from rag_engine.chunking.chunk_utils import TOKEN_REGEX, estimate_tokens
from rag_engine.chunking.metadata_inheritance import MetadataInheritor
from rag_engine.schemas.chunk import Chunk
from rag_engine.schemas.document import Document
from rag_engine.schemas.parsed_document import CleanParsedDocument, ParsedDocument, Section


class FixedChunker(BaseChunker):
    """Fixed-size chunker supporting token-based and character-based sliding windows."""

    def __init__(
        self,
        context: Optional[ChunkContext] = None,
        event_bus: Any = None,
        metrics: Any = None,
        validator: Any = None,
        mode: str = "token",  # "token" or "char"
        **kwargs: Any,
    ) -> None:
        super().__init__(context=context, event_bus=event_bus, metrics=metrics, validator=validator)
        self.mode = mode

    @property
    def strategy_name(self) -> str:
        return "fixed_char" if self.mode == "char" else "fixed"

    def _chunk_document(
        self, document: Union[Document, ParsedDocument, CleanParsedDocument]
    ) -> list[Chunk]:
        """Split document text into fixed-size windows with configured overlap."""
        chunks: list[Chunk] = []
        doc_id = getattr(document, "document_id", getattr(document, "doc_id", "doc"))

        # If ParsedDocument with sections, process section by section to keep page/section metadata
        sections: list[Section] = getattr(document, "sections", [])
        if sections:
            chunk_idx = 0
            for sec in sections:
                if not sec.content.strip():
                    continue

                sec_chunks = self._chunk_text(
                    text=sec.content,
                    document=document,
                    start_chunk_index=chunk_idx,
                    section=sec,
                    page_number=sec.page_number,
                )
                chunks.extend(sec_chunks)
                chunk_idx += len(sec_chunks)
        else:
            # Flat document content
            full_text = getattr(document, "content", "")
            if hasattr(document, "get_full_text"):
                full_text = document.get_full_text()
            if full_text.strip():
                chunks = self._chunk_text(
                    text=full_text,
                    document=document,
                    start_chunk_index=0,
                )

        return chunks

    def _chunk_text(
        self,
        text: str,
        document: Union[Document, ParsedDocument, CleanParsedDocument],
        start_chunk_index: int = 0,
        section: Optional[Section] = None,
        page_number: Optional[int] = None,
    ) -> list[Chunk]:
        """Produce fixed chunks for a single text block."""
        if self.mode == "char":
            return self._chunk_by_characters(
                text=text,
                document=document,
                start_chunk_index=start_chunk_index,
                section=section,
                page_number=page_number,
            )
        return self._chunk_by_tokens(
            text=text,
            document=document,
            start_chunk_index=start_chunk_index,
            section=section,
            page_number=page_number,
        )

    def _chunk_by_tokens(
        self,
        text: str,
        document: Union[Document, ParsedDocument, CleanParsedDocument],
        start_chunk_index: int,
        section: Optional[Section] = None,
        page_number: Optional[int] = None,
    ) -> list[Chunk]:
        """Sliding window over whitespace/punctuation tokens."""
        words = text.split()
        if not words:
            return []

        target_tokens = self.context.target_chunk_tokens
        # In words: approx target_tokens * 0.75 words per token
        target_words = max(10, int(target_tokens * 0.75))
        overlap_words = max(0, int(self.context.chunk_overlap_tokens * 0.75))
        step_words = max(1, target_words - overlap_words)

        chunks: list[Chunk] = []
        curr_idx = start_chunk_index
        i = 0
        total_words = len(words)

        while i < total_words:
            window = words[i : i + target_words]
            chunk_content = " ".join(window).strip()
            if not chunk_content:
                i += step_words
                continue

            metadata = MetadataInheritor.inherit(
                document=document,
                chunk_content=chunk_content,
                chunk_index=curr_idx,
                section=section,
                page_number=page_number,
                context=self.context,
            )

            tok_count = estimate_tokens(chunk_content)
            chk = Chunk.create(
                document_id=metadata.document_id,
                content=chunk_content,
                chunk_index=curr_idx,
                document_name=metadata.document_name,
                source_path=metadata.source_path,
                page_number=metadata.page_number,
                section_title=metadata.section_title,
                section_id=metadata.section_id,
                heading_path=metadata.heading_path,
                category=metadata.category,
                subcategory=metadata.subcategory,
                plant_unit=metadata.plant_unit,
                equipment_entities=metadata.equipment_entities,
                safety_entities=metadata.safety_entities,
                operating_parameters=metadata.operating_parameters,
                chunk_strategy=self.strategy_name,
                token_count=tok_count,
            )
            chunks.append(chk)
            curr_idx += 1

            if i + target_words >= total_words:
                break
            i += step_words

        return chunks

    def _chunk_by_characters(
        self,
        text: str,
        document: Union[Document, ParsedDocument, CleanParsedDocument],
        start_chunk_index: int,
        section: Optional[Section] = None,
        page_number: Optional[int] = None,
    ) -> list[Chunk]:
        """Sliding window over raw character offsets with boundary snapping."""
        clean_text = text.strip()
        if not clean_text:
            return []

        # Target characters
        target_chars = self.context.target_chunk_tokens
        overlap_chars = self.context.chunk_overlap_tokens
        step_chars = max(1, target_chars - overlap_chars)

        chunks: list[Chunk] = []
        curr_idx = start_chunk_index
        start = 0
        total_len = len(clean_text)

        while start < total_len:
            end = min(total_len, start + target_chars)

            # Snap end to nearest space if not at end of text
            if end < total_len:
                next_space = clean_text.find(" ", end)
                prev_space = clean_text.rfind(" ", start, end)
                if prev_space > start:
                    end = prev_space
                elif next_space != -1 and next_space - end < 50:
                    end = next_space

            chunk_content = clean_text[start:end].strip()
            if chunk_content:
                metadata = MetadataInheritor.inherit(
                    document=document,
                    chunk_content=chunk_content,
                    chunk_index=curr_idx,
                    section=section,
                    page_number=page_number,
                    char_start=start,
                    char_end=end,
                    context=self.context,
                )

                tok_count = estimate_tokens(chunk_content)
                chk = Chunk.create(
                    document_id=metadata.document_id,
                    content=chunk_content,
                    chunk_index=curr_idx,
                    document_name=metadata.document_name,
                    source_path=metadata.source_path,
                    page_number=metadata.page_number,
                    section_title=metadata.section_title,
                    section_id=metadata.section_id,
                    heading_path=metadata.heading_path,
                    category=metadata.category,
                    subcategory=metadata.subcategory,
                    plant_unit=metadata.plant_unit,
                    equipment_entities=metadata.equipment_entities,
                    safety_entities=metadata.safety_entities,
                    operating_parameters=metadata.operating_parameters,
                    chunk_strategy=self.strategy_name,
                    char_start=start,
                    char_end=end,
                    token_count=tok_count,
                )
                chunks.append(chk)
                curr_idx += 1

            if end >= total_len:
                break
            start += step_chars

        return chunks
