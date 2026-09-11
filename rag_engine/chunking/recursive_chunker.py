"""Recursive hierarchical chunker: Document -> Section -> Paragraph -> Sentence -> Token window."""

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
from rag_engine.schemas.parsed_document import CleanParsedDocument, ParsedDocument, Section, Table


class RecursiveChunker(BaseChunker):
    """Recursively decomposes text using document hierarchy:

    Document -> Section -> Subsection -> Paragraph -> Sentence -> Token window.
    Never breaks a sentence unless unavoidable.
    """

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
        return "recursive"

    def _chunk_document(
        self, document: Union[Document, ParsedDocument, CleanParsedDocument]
    ) -> list[Chunk]:
        """Decompose document recursively into semantic chunks."""
        chunks: list[Chunk] = []
        chunk_idx = 0

        # 1. Process structured sections if available
        sections: list[Section] = getattr(document, "sections", [])
        if sections:
            for sec in sections:
                sec_chunks = self._chunk_section(
                    section=sec,
                    document=document,
                    start_chunk_index=chunk_idx,
                    ancestor_headings=[sec.title] if sec.title else [],
                )
                chunks.extend(sec_chunks)
                chunk_idx += len(sec_chunks)
        else:
            # Flat text fallback
            full_text = getattr(document, "content", "")
            if hasattr(document, "get_full_text"):
                full_text = document.get_full_text()

            if full_text.strip():
                flat_chunks = self._chunk_text_recursively(
                    text=full_text,
                    document=document,
                    start_chunk_index=0,
                )
                chunks.extend(flat_chunks)
                chunk_idx += len(flat_chunks)

        # 2. Process structured tables as cohesive chunks if preserve_tables is True
        if self.context.preserve_tables:
            tables: list[Table] = getattr(document, "tables", [])
            for tbl in tables:
                tbl_chunk = self._create_table_chunk(tbl, document, chunk_idx)
                if tbl_chunk:
                    chunks.append(tbl_chunk)
                    chunk_idx += 1

        return chunks

    def _chunk_section(
        self,
        section: Section,
        document: Union[Document, ParsedDocument, CleanParsedDocument],
        start_chunk_index: int,
        ancestor_headings: list[str],
    ) -> list[Chunk]:
        """Chunk a section, descending into subsections or breaking paragraphs."""
        chunks: list[Chunk] = []
        curr_idx = start_chunk_index

        content = section.content.strip()
        if not content:
            # Still process subsections if any
            for sub in section.subsections:
                sub_path = ancestor_headings + ([sub.title] if sub.title else [])
                sub_chunks = self._chunk_section(sub, document, curr_idx, sub_path)
                chunks.extend(sub_chunks)
                curr_idx += len(sub_chunks)
            return chunks

        tokens = estimate_tokens(content)

        # Base case: entire section fits in target chunk window!
        if tokens <= self.context.target_chunk_tokens:
            meta = MetadataInheritor.inherit(
                document=document,
                chunk_content=content,
                chunk_index=curr_idx,
                section=section,
                page_number=section.page_number,
                heading_path=ancestor_headings,
                context=self.context,
            )
            chk = Chunk.create(
                document_id=meta.document_id,
                content=content,
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
                token_count=tokens,
            )
            chunks.append(chk)
            curr_idx += 1
        else:
            # Section exceeds target size: split by paragraphs
            sec_chunks = self._chunk_text_recursively(
                text=content,
                document=document,
                start_chunk_index=curr_idx,
                section=section,
                page_number=section.page_number,
                heading_path=ancestor_headings,
            )
            chunks.extend(sec_chunks)
            curr_idx += len(sec_chunks)

        # Recursively process child subsections
        for sub in section.subsections:
            sub_path = ancestor_headings + ([sub.title] if sub.title else [])
            sub_chunks = self._chunk_section(sub, document, curr_idx, sub_path)
            chunks.extend(sub_chunks)
            curr_idx += len(sub_chunks)

        return chunks

    def _chunk_text_recursively(
        self,
        text: str,
        document: Union[Document, ParsedDocument, CleanParsedDocument],
        start_chunk_index: int,
        section: Optional[Section] = None,
        page_number: Optional[int] = None,
        heading_path: Optional[list[str]] = None,
    ) -> list[Chunk]:
        """Split text across paragraphs, sentences, and tokens without bisecting sentences."""
        paragraphs = split_into_paragraphs(text)
        if not paragraphs:
            return []

        chunks: list[Chunk] = []
        curr_idx = start_chunk_index

        current_para_buffer: list[str] = []
        current_token_count = 0

        for para in paragraphs:
            para_tokens = estimate_tokens(para)

            # If single paragraph exceeds target window, process accumulated buffer first
            if para_tokens > self.context.target_chunk_tokens:
                if current_para_buffer:
                    buffer_text = "\n\n".join(current_para_buffer).strip()
                    chk = self._create_chunk(
                        content=buffer_text,
                        document=document,
                        chunk_index=curr_idx,
                        section=section,
                        page_number=page_number,
                        heading_path=heading_path,
                    )
                    chunks.append(chk)
                    curr_idx += 1
                    current_para_buffer.clear()
                    current_token_count = 0

                # Break the large paragraph into sentences
                sentence_chunks = self._chunk_paragraph_by_sentences(
                    para=para,
                    document=document,
                    start_chunk_index=curr_idx,
                    section=section,
                    page_number=page_number,
                    heading_path=heading_path,
                )
                chunks.extend(sentence_chunks)
                curr_idx += len(sentence_chunks)
                continue

            # Check if adding para exceeds target tokens
            if current_token_count + para_tokens > self.context.target_chunk_tokens:
                buffer_text = "\n\n".join(current_para_buffer).strip()
                if buffer_text:
                    chk = self._create_chunk(
                        content=buffer_text,
                        document=document,
                        chunk_index=curr_idx,
                        section=section,
                        page_number=page_number,
                        heading_path=heading_path,
                    )
                    chunks.append(chk)
                    curr_idx += 1

                current_para_buffer = [para]
                current_token_count = para_tokens
            else:
                current_para_buffer.append(para)
                current_token_count += para_tokens

        # Flush any remaining buffer
        if current_para_buffer:
            buffer_text = "\n\n".join(current_para_buffer).strip()
            if buffer_text:
                chk = self._create_chunk(
                    content=buffer_text,
                    document=document,
                    chunk_index=curr_idx,
                    section=section,
                    page_number=page_number,
                    heading_path=heading_path,
                )
                chunks.append(chk)

        return chunks

    def _chunk_paragraph_by_sentences(
        self,
        para: str,
        document: Union[Document, ParsedDocument, CleanParsedDocument],
        start_chunk_index: int,
        section: Optional[Section] = None,
        page_number: Optional[int] = None,
        heading_path: Optional[list[str]] = None,
    ) -> list[Chunk]:
        """Split an oversized paragraph into sentences without splitting sentences."""
        sentences = split_into_sentences(para)
        if not sentences:
            return []

        chunks: list[Chunk] = []
        curr_idx = start_chunk_index
        sent_buffer: list[str] = []
        buffer_tokens = 0

        for sent in sentences:
            s_tokens = estimate_tokens(sent)

            # If a single sentence exceeds max_chunk_tokens, token-split as unavoidable fallback
            if s_tokens > self.context.max_chunk_tokens:
                if sent_buffer:
                    chk = self._create_chunk(
                        content=" ".join(sent_buffer),
                        document=document,
                        chunk_index=curr_idx,
                        section=section,
                        page_number=page_number,
                        heading_path=heading_path,
                    )
                    chunks.append(chk)
                    curr_idx += 1
                    sent_buffer.clear()
                    buffer_tokens = 0

                # Token window fallback for giant sentence
                words = sent.split()
                w_idx = 0
                step = self.context.target_chunk_tokens - self.context.chunk_overlap_tokens
                while w_idx < len(words):
                    window = words[w_idx : w_idx + self.context.target_chunk_tokens]
                    win_text = " ".join(window)
                    chk = self._create_chunk(
                        content=win_text,
                        document=document,
                        chunk_index=curr_idx,
                        section=section,
                        page_number=page_number,
                        heading_path=heading_path,
                    )
                    chunks.append(chk)
                    curr_idx += 1
                    w_idx += step
                continue

            if buffer_tokens + s_tokens > self.context.target_chunk_tokens:
                chk = self._create_chunk(
                    content=" ".join(sent_buffer),
                    document=document,
                    chunk_index=curr_idx,
                    section=section,
                    page_number=page_number,
                    heading_path=heading_path,
                )
                chunks.append(chk)
                curr_idx += 1
                sent_buffer = [sent]
                buffer_tokens = s_tokens
            else:
                sent_buffer.append(sent)
                buffer_tokens += s_tokens

        if sent_buffer:
            chk = self._create_chunk(
                content=" ".join(sent_buffer),
                document=document,
                chunk_index=curr_idx,
                section=section,
                page_number=page_number,
                heading_path=heading_path,
            )
            chunks.append(chk)

        return chunks

    def _create_table_chunk(
        self,
        table: Table,
        document: Union[Document, ParsedDocument, CleanParsedDocument],
        chunk_index: int,
    ) -> Optional[Chunk]:
        """Convert a structured table into a dedicated tabular chunk."""
        table_text = table.normalized_text.strip()
        if not table_text:
            return None

        # Include caption if present
        content = f"Table: {table.caption}\n{table_text}" if table.caption else table_text

        meta = MetadataInheritor.inherit(
            document=document,
            chunk_content=content,
            chunk_index=chunk_index,
            table=table,
            page_number=table.page_number,
            context=self.context,
        )

        return Chunk.create(
            document_id=meta.document_id,
            content=content,
            chunk_index=chunk_index,
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
            is_table_chunk=True,
            table_id=table.table_id,
            chunk_strategy=self.strategy_name,
            token_count=estimate_tokens(content),
        )

    def _create_chunk(
        self,
        content: str,
        document: Union[Document, ParsedDocument, CleanParsedDocument],
        chunk_index: int,
        section: Optional[Section] = None,
        page_number: Optional[int] = None,
        heading_path: Optional[list[str]] = None,
    ) -> Chunk:
        """Helper to create an individual chunk with inherited metadata."""
        meta = MetadataInheritor.inherit(
            document=document,
            chunk_content=content,
            chunk_index=chunk_index,
            section=section,
            page_number=page_number,
            heading_path=heading_path,
            context=self.context,
        )

        return Chunk.create(
            document_id=meta.document_id,
            content=content,
            chunk_index=chunk_index,
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
            token_count=estimate_tokens(content),
        )
