"""TableChunker: Strategy preserving table structure, headers, captions, and row integrity."""

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
)
from rag_engine.chunking.metadata_inheritance import MetadataInheritor
from rag_engine.schemas.chunk import Chunk, ChunkHierarchy
from rag_engine.schemas.document import Document
from rag_engine.schemas.parsed_document import CleanParsedDocument, ParsedDocument, Table


class TableChunker(BaseChunker):
    """Table-aware chunking strategy.
    
    Guarantees that tables are never randomly split across cell boundaries.
    Preserves table captions, headers, rows, page numbers, and document references.
    If a table exceeds the target token limit, rows are batched into chunks while
    repeating column headers and caption breadcrumbs on every chunk.
    Non-table document sections are segmented cleanly by paragraphs.
    """

    @property
    def strategy_name(self) -> str:
        return "table"

    def _chunk_document(self, document: Union[Document, ParsedDocument, CleanParsedDocument]) -> list[Chunk]:
        raw_chunks: list[Chunk] = []
        doc_id = getattr(document, "document_id", getattr(document, "doc_id", "unknown_doc"))
        doc_tables: list[Table] = getattr(document, "tables", [])

        chunk_idx = 0

        # If document has structured Table objects
        if doc_tables:
            # 1. Process structured tables
            for tbl in doc_tables:
                tbl_chunks = self.chunk_table(tbl, document, start_index=chunk_idx)
                raw_chunks.extend(tbl_chunks)
                chunk_idx += len(tbl_chunks)

            # 2. Process any remaining text in sections that is not in tables
            sections = getattr(document, "sections", [])
            if sections:
                for sec in sections:
                    sec_text = (getattr(sec, "normalized_text", None) or getattr(sec, "content", "")).strip()
                    # Avoid re-chunking raw text that only consists of table representation
                    if len(sec_text) < 10:
                        continue
                    
                    # Split prose by paragraphs
                    paragraphs = [p.strip() for p in sec_text.split("\n\n") if p.strip()]
                    current_para_batch: list[str] = []
                    current_tokens = 0
                    
                    for p in paragraphs:
                        # Skip if paragraph is a duplicate of a table markdown
                        p_toks = estimate_tokens(p)
                        if current_tokens + p_toks > self.context.target_tokens and current_para_batch:
                            content = "\n\n".join(current_para_batch)
                            chk = self._create_prose_chunk(
                                document, content, chunk_idx, sec, getattr(sec, "page_number", None)
                            )
                            raw_chunks.append(chk)
                            chunk_idx += 1
                            current_para_batch = [p]
                            current_tokens = p_toks
                        else:
                            current_para_batch.append(p)
                            current_tokens += p_toks

                    if current_para_batch:
                        content = "\n\n".join(current_para_batch)
                        chk = self._create_prose_chunk(
                            document, content, chunk_idx, sec, getattr(sec, "page_number", None)
                        )
                        raw_chunks.append(chk)
                        chunk_idx += 1

            elif hasattr(document, "content") or hasattr(document, "get_full_text"):
                # If no sections, process document content
                full_text = getattr(document, "content", "")
                if hasattr(document, "get_full_text"):
                    full_text = document.get_full_text()
                # If full text has non-table content, chunk paragraphs
                paras = [p.strip() for p in full_text.split("\n\n") if p.strip() and not p.startswith("|")]
                for p in paras:
                    if len(p) > 20:
                        chk = self._create_prose_chunk(document, p, chunk_idx, None, None)
                        raw_chunks.append(chk)
                        chunk_idx += 1

        else:
            # Document has no structured Table objects: search for Markdown/ASCII tables in text
            raw_chunks = self._chunk_text_with_markdown_tables(document)

        return raw_chunks

    def chunk_table(
        self,
        table: Table,
        document: Union[Document, ParsedDocument, CleanParsedDocument],
        start_index: int = 0,
    ) -> list[Chunk]:
        """Convert a single Table object into one or more retrieval-optimized Chunks."""
        chunks: list[Chunk] = []
        doc_id = getattr(document, "document_id", getattr(document, "doc_id", "unknown_doc"))
        page_no = table.page_number or 1

        headers = table.headers or []
        rows = table.rows or []
        caption = table.caption.strip() if table.caption else ""

        if not rows and not headers:
            # If table is represented purely as raw_text or normalized_text
            raw = (table.normalized_text or table.raw_text or "").strip()
            if not raw:
                return []
            content = f"### Table: {caption}\n\n{raw}" if caption else raw
            return [self._build_table_chunk(document, content, start_index, table, page_no)]

        # Check size of full table markdown
        full_md = self._render_markdown_table(headers, rows, caption=caption)
        full_tokens = estimate_tokens(full_md)

        if full_tokens <= self.context.max_tokens or len(rows) <= 1:
            # Entire table fits within one chunk
            chk = self._build_table_chunk(document, full_md, start_index, table, page_no)
            chunks.append(chk)
            return chunks

        # Table exceeds token limit: batch rows while repeating headers on each chunk
        row_batches: list[list[list[str]]] = []
        current_batch: list[list[str]] = []
        header_toks = estimate_tokens(self._render_markdown_table(headers, []))

        for row in rows:
            row_toks = estimate_tokens(" | ".join(row))
            current_batch_toks = estimate_tokens(self._render_markdown_table(headers, current_batch)) if current_batch else header_toks
            
            if current_batch_toks + row_toks > self.context.target_tokens and current_batch:
                row_batches.append(current_batch)
                current_batch = [row]
            else:
                current_batch.append(row)

        if current_batch:
            row_batches.append(current_batch)

        total_parts = len(row_batches)
        for part_idx, batch in enumerate(row_batches, start=1):
            part_caption = f"{caption} (Part {part_idx} of {total_parts})" if caption else f"Table (Part {part_idx} of {total_parts})"
            part_md = self._render_markdown_table(headers, batch, caption=part_caption)
            chk = self._build_table_chunk(document, part_md, start_index + part_idx - 1, table, page_no)
            chunks.append(chk)

        return chunks

    def _render_markdown_table(self, headers: list[str], rows: list[list[str]], caption: str = "") -> str:
        """Render headers and rows into clean markdown table format."""
        lines: list[str] = []
        if caption:
            lines.append(f"**Table: {caption}**\n")
        
        if headers:
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        elif rows:
            col_count = max(len(r) for r in rows)
            lines.append("| " + " | ".join([f"Col {i+1}" for i in range(col_count)]) + " |")
            lines.append("| " + " | ".join(["---"] * col_count) + " |")

        for row in rows:
            lines.append("| " + " | ".join(str(c) for c in row) + " |")

        return "\n".join(lines)

    def _build_table_chunk(
        self,
        document: Union[Document, ParsedDocument, CleanParsedDocument],
        content: str,
        chunk_index: int,
        table: Table,
        page_number: int,
    ) -> Chunk:
        doc_id = getattr(document, "document_id", getattr(document, "doc_id", "unknown_doc"))
        content_hash = compute_sha256(content)
        chunk_id = generate_deterministic_chunk_id(doc_id, page_number, chunk_index, content_hash)
        
        metadata = MetadataInheritor.inherit_metadata(
            document=document,
            chunk_content=content,
            chunk_index=chunk_index,
            page_number=page_number,
            section=None,
            table=table,
            context=self.context,
            heading_path=[f"Table: {table.caption}"] if table.caption else ["Tables"],
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

    def _create_prose_chunk(
        self,
        document: Union[Document, ParsedDocument, CleanParsedDocument],
        content: str,
        chunk_index: int,
        section: Any,
        page_number: Optional[int],
    ) -> Chunk:
        doc_id = getattr(document, "document_id", getattr(document, "doc_id", "unknown_doc"))
        resolved_page = page_number or (getattr(section, "page_number", None) if section else 1) or 1
        content_hash = compute_sha256(content)
        chunk_id = generate_deterministic_chunk_id(doc_id, resolved_page, chunk_index, content_hash)

        sec_title = getattr(section, "title", None) if section else None
        h_path = [sec_title] if sec_title else ["General"]

        metadata = MetadataInheritor.inherit_metadata(
            document=document,
            chunk_content=content,
            chunk_index=chunk_index,
            page_number=resolved_page,
            section=section,
            table=None,
            context=self.context,
            heading_path=h_path,
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

    def _chunk_text_with_markdown_tables(
        self, document: Union[Document, ParsedDocument, CleanParsedDocument]
    ) -> list[Chunk]:
        """Detect markdown table blocks within plain text and isolate them as table chunks."""
        raw_chunks: list[Chunk] = []
        doc_id = getattr(document, "document_id", getattr(document, "doc_id", "unknown_doc"))

        full_text = getattr(document, "content", "")
        if hasattr(document, "get_full_text"):
            full_text = document.get_full_text()

        if not full_text.strip():
            return []

        # Regex detecting markdown table blocks: consecutive lines starting with '|'
        lines = full_text.splitlines()
        blocks: list[tuple[str, bool]] = []  # (text, is_table)
        current_lines: list[str] = []
        in_table = False

        for line in lines:
            stripped = line.strip()
            is_tbl_line = stripped.startswith("|") and stripped.endswith("|")
            if is_tbl_line:
                if not in_table:
                    if current_lines:
                        blocks.append(("\n".join(current_lines), False))
                        current_lines = []
                    in_table = True
                current_lines.append(line)
            else:
                if in_table:
                    if current_lines:
                        blocks.append(("\n".join(current_lines), True))
                        current_lines = []
                    in_table = False
                current_lines.append(line)

        if current_lines:
            blocks.append(("\n".join(current_lines), in_table))

        chunk_idx = 0
        for text_block, is_tbl in blocks:
            text_block = text_block.strip()
            if not text_block:
                continue

            if is_tbl:
                # Parse markdown lines into headers and rows
                tbl_lines = [l.strip() for l in text_block.splitlines() if l.strip()]
                headers: list[str] = []
                rows: list[list[str]] = []
                for idx, t_line in enumerate(tbl_lines):
                    cells = [c.strip() for c in t_line.strip("|").split("|")]
                    if idx == 0:
                        headers = cells
                    elif idx == 1 and all(set(c).issubset({"-", ":", " "}) for c in cells):
                        continue  # markdown separator row
                    else:
                        rows.append(cells)

                synthetic_table = Table(
                    table_id=f"tbl_md_{chunk_idx:04d}",
                    caption=f"Extracted Table {chunk_idx + 1}",
                    headers=headers,
                    rows=rows,
                    row_count=len(rows),
                    col_count=len(headers),
                    page_number=1,
                    raw_text=text_block,
                    normalized_text=text_block,
                )
                tbl_chunks = self.chunk_table(synthetic_table, document, start_index=chunk_idx)
                raw_chunks.extend(tbl_chunks)
                chunk_idx += len(tbl_chunks)
            else:
                # Split prose block by paragraphs
                paras = [p.strip() for p in text_block.split("\n\n") if p.strip()]
                for p in paras:
                    chk = self._create_prose_chunk(document, p, chunk_idx, None, 1)
                    raw_chunks.append(chk)
                    chunk_idx += 1

        return raw_chunks
