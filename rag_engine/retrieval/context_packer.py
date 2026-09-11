"""Token-Aware Context Packer.

Assembles and formats retrieved candidate chunks and citations into a clean,
bounded context window for downstream LLMs without truncating tables mid-row.
"""

from __future__ import annotations

import logging
from typing import Sequence

from rag_engine.retrieval.base_retriever import CitationBundle, ScoredRetrievalChunk
from rag_engine.retrieval.retrieval_utils import estimate_token_count

logger = logging.getLogger(__name__)


class ContextPacker:
    """Assembles structured prompt context respecting strict token budgets."""

    def __init__(
        self,
        default_token_budget: int = 3000,
        header_template: Optional[str] = None,
    ) -> None:
        self.default_token_budget = default_token_budget
        self.header_template = (
            header_template
            or "=== VERIFIED GROUND TRUTH CONTEXT (SIH26117 / MRPL) ===\n"
        )

    def pack(
        self,
        candidates: Sequence[ScoredRetrievalChunk],
        citations: Sequence[CitationBundle],
        token_budget: Optional[int] = None,
    ) -> tuple[str, int]:
        """Pack retrieved chunks into a formatted context string bounded by token budget.
        
        Args:
            candidates: Ordered high-relevance chunks.
            citations: Corresponding citation anchors.
            token_budget: Maximum tokens allowed.
            
        Returns:
            Tuple of (formatted_context_string, total_tokens_packed).
        """
        budget = token_budget or self.default_token_budget
        if not candidates:
            return "", 0

        # Build map of chunk_id -> citation anchor string
        cit_map = {c.chunk_id: c.citation_id for c in citations}

        lines: list[str] = [self.header_template]
        curr_tokens = estimate_token_count(self.header_template)

        for item in candidates:
            chk = item.chunk
            meta = chk.metadata
            cit_anchor = cit_map.get(chk.chunk_id, "[*]")

            # Header info for this chunk
            header_parts = [f"Source: {meta.document_name or 'Document'}"]
            if meta.page_number is not None:
                header_parts.append(f"Page: {meta.page_number}")
            if meta.section_title:
                header_parts.append(f"Section: {meta.section_title}")
            if meta.equipment_entities:
                header_parts.append(f"Equipment: {', '.join(meta.equipment_entities)}")
            if meta.safety_entities:
                header_parts.append(f"Safety/Standards: {', '.join(meta.safety_entities)}")

            chunk_header = f"\n--- {cit_anchor} {' | '.join(header_parts)} ---\n"
            content = chk.content.strip()

            # Handle tables specially to avoid mid-row truncation
            if getattr(meta, "is_table_chunk", False) or "|---" in content or "\t" in content:
                packed_content = self._pack_table_content(
                    content, budget - curr_tokens - estimate_token_count(chunk_header)
                )
            else:
                packed_content = content

            if not packed_content:
                # Cannot fit this chunk into remaining budget
                continue

            chunk_block = chunk_header + packed_content + "\n"
            block_tokens = estimate_token_count(chunk_block)

            if curr_tokens + block_tokens <= budget:
                lines.append(chunk_block)
                curr_tokens += block_tokens
            else:
                # If cannot fit full chunk and not a table, check if we can fit partial paragraph
                remaining_budget = budget - curr_tokens - estimate_token_count(chunk_header)
                if remaining_budget > 100 and not getattr(meta, "is_table_chunk", False):
                    # Fit partial content by sentences
                    sentences = content.split(". ")
                    sub_content = []
                    sub_tok = 0
                    for sent in sentences:
                        s_tok = estimate_token_count(sent + ". ")
                        if sub_tok + s_tok <= remaining_budget:
                            sub_content.append(sent)
                            sub_tok += s_tok
                        else:
                            break
                    if sub_content:
                        fitted_text = ". ".join(sub_content) + "..."
                        partial_block = chunk_header + fitted_text + "\n"
                        lines.append(partial_block)
                        curr_tokens += estimate_token_count(partial_block)
                break

        final_context = "".join(lines)
        return final_context, curr_tokens

    def _pack_table_content(self, table_text: str, token_allowance: int) -> str:
        """Pack table content preserving full row boundaries without mid-row cuts."""
        rows = table_text.split("\n")
        fitted_rows: list[str] = []
        tokens_used = 0

        for row in rows:
            row_tokens = estimate_token_count(row + "\n")
            if tokens_used + row_tokens <= token_allowance:
                fitted_rows.append(row)
                tokens_used += row_tokens
            else:
                # Stop before breaking a row
                break

        # Require at least header + separator or 2 rows
        if len(fitted_rows) >= 2:
            return "\n".join(fitted_rows)
        return ""


# Enterprise aliases for context window formatting and token management
ContextWindowBuilder = ContextPacker
TokenBudgetManager = ContextPacker
