"""Context Compressor for intelligent context packing and token reduction."""

from __future__ import annotations

from enum import Enum
import re
from typing import Optional, Sequence

from rag_engine.interfaces.base_prompt import BaseContextCompressor
from rag_engine.schemas.chunk import Chunk, ChunkHierarchy, ChunkMetadata


class CompressionStrategy(str, Enum):
    """Available strategies for compressing context chunks."""

    TRUNCATE = "truncate"
    SELECTIVE_EXTRACTION = "selective_extraction"
    COMPACT = "compact"
    REDUNDANCY_PRUNING = "redundancy_pruning"


class ContextCompressor(BaseContextCompressor):
    """Compresses retrieved chunks to fit within designated context window budgets."""

    TECHNICAL_INDICATORS = [
        r"\b\d+(\.\d+)?\s*(bar|psi|kPa|MPa|°C|deg\s*C|K|m3/h|m/s|rpm|RPM|kW|MW)\b",
        r"\b[A-Z]{1,3}-\d{3,4}[A-Z]?\b",  # Tag numbers like P-101, C-201, XV-301
        r"\b(OISD|API|ASME|ISO|ASTM|IBR)-\d+\b",
        r"\|.*?\|.*?\|",  # Table markdown rows
    ]

    def __init__(self, strategy: CompressionStrategy = CompressionStrategy.SELECTIVE_EXTRACTION) -> None:
        self.strategy = strategy
        self._compiled_regexes = [re.compile(p, re.IGNORECASE) for p in self.TECHNICAL_INDICATORS]

    def compress(
        self,
        context_chunks: Sequence[Chunk],
        token_budget: int,
        query: Optional[str] = None,
    ) -> list[Chunk]:
        """Compress sequence of chunks to fit token budget."""
        if not context_chunks:
            return []

        # First pass: if already within budget, return unmodified
        current_tokens = sum(c.token_count or (len(c.content.split())) for c in context_chunks)
        if current_tokens <= token_budget:
            return list(context_chunks)

        compressed_chunks: list[Chunk] = []
        allocated_tokens = 0

        for chk in context_chunks:
            tokens_in_chk = chk.token_count or len(chk.content.split())
            if allocated_tokens + tokens_in_chk <= token_budget:
                compressed_chunks.append(chk)
                allocated_tokens += tokens_in_chk
            else:
                # Need compression on this or remaining chunks
                remaining_budget = token_budget - allocated_tokens
                if remaining_budget <= 20:
                    break

                if self.strategy == CompressionStrategy.SELECTIVE_EXTRACTION:
                    new_content = self._extract_technical_salience(chk.content, remaining_budget)
                elif self.strategy == CompressionStrategy.COMPACT:
                    new_content = self._compact_whitespace(chk.content, remaining_budget)
                else:
                    new_content = self._truncate_words(chk.content, remaining_budget)

                new_token_count = len(new_content.split())
                if new_token_count > 0:
                    compact_chk = Chunk(
                        chunk_id=chk.chunk_id,
                        chunk_hash=chk.chunk_hash,
                        content=new_content,
                        token_count=new_token_count,
                        character_count=len(new_content),
                        metadata=chk.metadata,
                        hierarchy=chk.hierarchy,
                    )
                    compressed_chunks.append(compact_chk)
                    allocated_tokens += new_token_count
                break

        return compressed_chunks

    def _extract_technical_salience(self, content: str, target_tokens: int) -> str:
        """Preserve tables and sentences containing critical refinery parameters."""
        lines = content.split("\n")
        selected_lines: list[str] = []
        token_sum = 0

        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue
            is_salient = any(rgx.search(line_str) for rgx in self._compiled_regexes)
            line_tokens = len(line_str.split())

            if is_salient or token_sum + line_tokens <= target_tokens:
                if token_sum + line_tokens <= target_tokens:
                    selected_lines.append(line_str)
                    token_sum += line_tokens
                else:
                    truncated = " ".join(line_str.split()[: target_tokens - token_sum])
                    if truncated:
                        selected_lines.append(truncated)
                    break

        return "\n".join(selected_lines) if selected_lines else self._truncate_words(content, target_tokens)

    def _compact_whitespace(self, content: str, target_tokens: int) -> str:
        """Minimizes extraneous spaces and blank lines."""
        cleaned = re.sub(r"\s+", " ", content).strip()
        return self._truncate_words(cleaned, target_tokens)

    def _truncate_words(self, content: str, target_tokens: int) -> str:
        words = content.split()
        if len(words) <= target_tokens:
            return content
        return " ".join(words[:target_tokens])
