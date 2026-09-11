"""ListChunker: Strategy preserving bullet lists, numbered lists, and procedure cohesion."""

from __future__ import annotations

import re
from typing import Any, Optional, Union

from rag_engine.chunking.base_chunker import BaseChunker
from rag_engine.chunking.chunk_context import ChunkContext
from rag_engine.chunking.chunk_utils import (
    compute_sha256,
    count_characters,
    count_words,
    estimate_tokens,
    generate_deterministic_chunk_id,
    split_into_list_items,
)
from rag_engine.chunking.metadata_inheritance import MetadataInheritor
from rag_engine.schemas.chunk import Chunk, ChunkHierarchy
from rag_engine.schemas.document import Document
from rag_engine.schemas.parsed_document import CleanParsedDocument, ParsedDocument


# Pattern matching start of list items: numbered, bulleted, or step-based
LIST_ITEM_START_REGEX = re.compile(
    r"^(?:\d+[\.\)]|\([0-9a-zA-Z]+\)|[a-zA-Z][\.\)]|[\*\-\•\◦\–]|Step\s+\d+:?|Procedure\s+\d+:?|Phase\s+\d+:?)\s+",
    re.IGNORECASE,
)


class ListChunker(BaseChunker):
    """List-aware chunking strategy.
    
    Guarantees that bullet lists, numbered lists, checklist items, and operating
    procedures (e.g. refinery startup/shutdown sequences) stay together whenever possible.
    Never splits across an individual list item boundary.
    Long lists exceeding target token limits are partitioned at list item boundaries
    with procedure/list header breadcrumbs preserved.
    """

    @property
    def strategy_name(self) -> str:
        return "list"

    def _chunk_document(self, document: Union[Document, ParsedDocument, CleanParsedDocument]) -> list[Chunk]:
        raw_chunks: list[Chunk] = []
        sections = getattr(document, "sections", [])

        chunk_idx = 0

        if sections:
            for sec in sections:
                sec_text = (getattr(sec, "normalized_text", None) or getattr(sec, "content", "")).strip()
                if not sec_text:
                    continue

                sec_chunks = self._chunk_section_lists(document, sec_text, sec, chunk_idx)
                raw_chunks.extend(sec_chunks)
                chunk_idx += len(sec_chunks)

        else:
            full_text = getattr(document, "content", "")
            if hasattr(document, "get_full_text"):
                full_text = document.get_full_text()

            if full_text.strip():
                raw_chunks = self._chunk_section_lists(document, full_text, None, 0)

        return raw_chunks

    def _chunk_section_lists(
        self,
        document: Union[Document, ParsedDocument, CleanParsedDocument],
        text: str,
        section: Any,
        start_index: int,
    ) -> list[Chunk]:
        """Parse text into paragraphs and list blocks, keeping lists cohesive."""
        chunks: list[Chunk] = []
        doc_id = getattr(document, "document_id", getattr(document, "doc_id", "unknown_doc"))
        page_no = getattr(section, "page_number", None) or 1
        sec_title = getattr(section, "title", None) or ""
        h_path = [sec_title] if sec_title else ["Lists"]

        blocks = self._extract_content_blocks(text)
        chunk_idx = start_index

        for block_text, is_list in blocks:
            if not block_text.strip():
                continue

            if is_list:
                list_chunks = self._chunk_list_block(
                    document=document,
                    list_text=block_text,
                    section=section,
                    page_number=page_no,
                    start_index=chunk_idx,
                    section_title=sec_title,
                )
                chunks.extend(list_chunks)
                chunk_idx += len(list_chunks)
            else:
                # Prose block: partition by paragraphs
                paragraphs = [p.strip() for p in block_text.split("\n\n") if p.strip()]
                curr_para_batch: list[str] = []
                curr_tokens = 0

                for p in paragraphs:
                    p_toks = estimate_tokens(p)
                    if curr_tokens + p_toks > self.context.target_tokens and curr_para_batch:
                        prose_content = "\n\n".join(curr_para_batch)
                        chk = self._build_chunk(
                            document=document,
                            content=prose_content,
                            chunk_index=chunk_idx,
                            page_number=page_no,
                            section=section,
                            heading_path=h_path,
                            is_list=False,
                        )
                        chunks.append(chk)
                        chunk_idx += 1
                        curr_para_batch = [p]
                        curr_tokens = p_toks
                    else:
                        curr_para_batch.append(p)
                        curr_tokens += p_toks

                if curr_para_batch:
                    prose_content = "\n\n".join(curr_para_batch)
                    chk = self._build_chunk(
                        document=document,
                        content=prose_content,
                        chunk_index=chunk_idx,
                        page_number=page_no,
                        section=section,
                        heading_path=h_path,
                        is_list=False,
                    )
                    chunks.append(chk)
                    chunk_idx += 1

        return chunks

    def _extract_content_blocks(self, text: str) -> list[tuple[str, bool]]:
        """Separate text into sequential blocks of either (prose, False) or (list, True)."""
        lines = text.splitlines()
        blocks: list[tuple[str, bool]] = []
        current_lines: list[str] = []
        in_list = False

        for line in lines:
            stripped = line.strip()
            if not stripped:
                if current_lines:
                    current_lines.append("")
                continue

            is_list_line = bool(LIST_ITEM_START_REGEX.match(stripped))

            if is_list_line:
                if not in_list:
                    # Flush prose
                    if current_lines:
                        blocks.append(("\n".join(current_lines).strip(), False))
                        current_lines = []
                    in_list = True
                current_lines.append(line)
            else:
                # Continuation of list item (indented or wrapped line)
                if in_list and (line.startswith("  ") or line.startswith("\t") or len(stripped) < 80):
                    current_lines.append(line)
                else:
                    if in_list:
                        # Flush list
                        if current_lines:
                            blocks.append(("\n".join(current_lines).strip(), True))
                            current_lines = []
                        in_list = False
                    current_lines.append(line)

        if current_lines:
            blocks.append(("\n".join(current_lines).strip(), in_list))

        return [(b_text, is_l) for b_text, is_l in blocks if b_text.strip()]

    def _chunk_list_block(
        self,
        document: Union[Document, ParsedDocument, CleanParsedDocument],
        list_text: str,
        section: Any,
        page_number: int,
        start_index: int,
        section_title: str,
    ) -> list[Chunk]:
        """Bundle list items up to target token limit, partitioning across item boundaries if needed."""
        items = split_into_list_items(list_text)
        if not items:
            items = [list_text]

        chunks: list[Chunk] = []
        current_items: list[str] = []
        current_tokens = 0
        chunk_idx = start_index

        h_path = [section_title, "Procedure / List"] if section_title else ["Procedure / List"]

        for item in items:
            item_toks = estimate_tokens(item)
            if current_tokens + item_toks > self.context.target_tokens and current_items:
                content = "\n".join(current_items)
                chk = self._build_chunk(
                    document=document,
                    content=content,
                    chunk_index=chunk_idx,
                    page_number=page_number,
                    section=section,
                    heading_path=h_path,
                    is_list=True,
                )
                chunks.append(chk)
                chunk_idx += 1
                
                # Breadcrumb continuation for large procedures
                breadcrumb = f"[Continuation of {section_title or 'Procedure'}]\n" if section_title else ""
                current_items = [f"{breadcrumb}{item}"]
                current_tokens = estimate_tokens(current_items[0])
            else:
                current_items.append(item)
                current_tokens += item_toks

        if current_items:
            content = "\n".join(current_items)
            chk = self._build_chunk(
                document=document,
                content=content,
                chunk_index=chunk_idx,
                page_number=page_number,
                section=section,
                heading_path=h_path,
                is_list=True,
            )
            chunks.append(chk)

        return chunks

    def _build_chunk(
        self,
        document: Union[Document, ParsedDocument, CleanParsedDocument],
        content: str,
        chunk_index: int,
        page_number: int,
        section: Any,
        heading_path: list[str],
        is_list: bool,
    ) -> Chunk:
        doc_id = getattr(document, "document_id", getattr(document, "doc_id", "unknown_doc"))
        content_hash = compute_sha256(content)
        chunk_id = generate_deterministic_chunk_id(doc_id, page_number, chunk_index, content_hash)

        metadata = MetadataInheritor.inherit_metadata(
            document=document,
            chunk_content=content,
            chunk_index=chunk_index,
            page_number=page_number,
            section=section,
            table=None,
            context=self.context,
            heading_path=heading_path,
            is_list=is_list,
        )

        return Chunk(
            chunk_id=chunk_id,
            content=content,
            token_count=estimate_tokens(content),
            character_count=count_characters(content),
            word_count=count_words(content),
            metadata=metadata,
            hierarchy=ChunkHierarchy(
                parent_doc_id=doc_id,
                chunk_id=chunk_id,
                chunk_index=chunk_index,
            ),
        )
