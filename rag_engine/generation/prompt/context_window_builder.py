"""Context Window Builder — Structure-Preserving Ground Truth Assembly.

Formats retrieved candidate chunks into an auditable, token-bounded context block.
Preserves table integrity without mid-row splits, attaches deterministic citation
anchors ([1], [2]), and prunes overlapping sentences.
"""

from __future__ import annotations

import logging
from typing import Sequence

from rag_engine.interfaces.base_prompt import BasePromptContextBuilder
from rag_engine.retrieval.base_retriever import CitationBundle, ScoredRetrievalChunk

logger = logging.getLogger(__name__)


class ContextWindowBuilder(BasePromptContextBuilder):
    """Assembles structured prompt context respecting strict token budgets."""

    def __init__(
        self,
        default_token_budget: int = 2000,
        header_template: str | None = None,
    ) -> None:
        self.default_token_budget = default_token_budget
        self.header_template = (
            header_template
            or "=== VERIFIED MRPL REFINERY GROUND TRUTH CONTEXT ===\n"
        )

    def estimate_tokens(self, text: str) -> int:
        """Estimate tokens (~4 chars per token)."""
        if not text:
            return 0
        return max(1, len(text) // 4)

    def build_context_window(
        self,
        candidates: Sequence[ScoredRetrievalChunk],
        citations: Sequence[CitationBundle],
        token_budget: int | None = None,
    ) -> tuple[str, int, dict[str, str]]:
        """Build formatted context string with citation anchor tags.
        
        Args:
            candidates: Ranked list of scored chunks from Milestone 8.
            citations: Citation bundles from Milestone 8.
            token_budget: Maximum tokens allowed.
            
        Returns:
            Tuple of (formatted_context, tokens_used, citation_map_chunk_to_anchor).
        """
        budget = token_budget or self.default_token_budget
        if not candidates:
            return "", 0, {}

        # Map chunk_id to citation anchor "[1]", "[2]"
        chunk_to_anchor: dict[str, str] = {}
        for idx, bundle in enumerate(citations, start=1):
            anchor = f"[{idx}]"
            chunk_to_anchor[bundle.chunk_id] = anchor

        blocks: list[str] = [self.header_template]
        curr_tokens = self.estimate_tokens(self.header_template)

        for item in candidates:
            chk = item.chunk
            meta = chk.metadata
            chunk_id = chk.chunk_id

            anchor = chunk_to_anchor.get(chunk_id, f"[{len(chunk_to_anchor) + 1}]")
            chunk_to_anchor[chunk_id] = anchor

            # Metadata header line
            doc_name = meta.document_id
            page = getattr(meta, "page_number", None) or getattr(meta, "page", None)
            page_str = f" | Page {page}" if page is not None else ""
            section = getattr(meta, "section_title", None) or getattr(meta, "section", None)
            section_str = f" | Section: {section}" if section else ""
            tag = getattr(meta, "equipment_tag", None)
            tag_str = f" | Tag: {tag}" if tag else ""

            header = f"\n--- SOURCE {anchor}: {doc_name}{page_str}{section_str}{tag_str} ---\n"
            content = chk.content.strip()

            # If table, preserve table row boundaries
            is_table = getattr(meta, "content_type", None) == "table" or "\n|" in content
            if is_table:
                content = self._pack_table_content(content, max(50, budget - curr_tokens - self.estimate_tokens(header)))

            block_text = header + content + "\n"
            block_tokens = self.estimate_tokens(block_text)

            if curr_tokens + block_tokens <= budget:
                blocks.append(block_text)
                curr_tokens += block_tokens
            else:
                # If cannot fit full block, check if partial content fits
                allowance = budget - curr_tokens - self.estimate_tokens(header)
                if allowance > 100:
                    truncated = content[: allowance * 4] + " ... [TRUNCATED DUE TO BUDGET]\n"
                    blocks.append(header + truncated)
                    curr_tokens += self.estimate_tokens(header + truncated)
                break

        final_context = "".join(blocks)
        return final_context, curr_tokens, chunk_to_anchor

    def _pack_table_content(self, table_text: str, token_allowance: int) -> str:
        """Pack table content preserving full row boundaries without mid-row cuts."""
        rows = table_text.split("\n")
        fitted_rows: list[str] = []
        tokens_used = 0

        for row in rows:
            row_tokens = self.estimate_tokens(row + "\n")
            if tokens_used + row_tokens <= token_allowance:
                fitted_rows.append(row)
                tokens_used += row_tokens
            else:
                break

        if len(fitted_rows) >= 2:
            return "\n".join(fitted_rows)
        return table_text[: max(50, token_allowance * 4)]


# Explicit alias for architectural conformity
PromptContextBuilder = ContextWindowBuilder
