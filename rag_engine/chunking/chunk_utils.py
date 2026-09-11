"""Utility functions for token estimation, sentence splitting, and deterministic hashing."""

from __future__ import annotations

import hashlib
import re
from typing import Optional

# Regex for approximate subword/punctuation tokenization (close to BPE/WordPiece)
TOKEN_REGEX = re.compile(r"\w+|[^\w\s]", re.UNICODE)

# Common engineering and English abbreviations that should not trigger sentence splits
ABBREVIATIONS = {
    "e.g.",
    "i.e.",
    "fig.",
    "figs.",
    "ref.",
    "no.",
    "rev.",
    "approx.",
    "min.",
    "max.",
    "temp.",
    "press.",
    "vol.",
    "dr.",
    "dept.",
    "equip.",
    "p&id.",
}

# Sentence boundary regex with negative lookahead for abbreviations and decimals
SENTENCE_SPLIT_REGEX = re.compile(
    r"(?<=[.?!])\s+(?=[A-Z0-9\"'(\[])",
)


def estimate_tokens(text: str) -> int:
    """Estimate token count deterministically without an LLM.

    Uses a fast regex token splitter counting alphanumeric words and individual punctuation,
    providing high correlation with BPE/WordPiece tokenizers.
    """
    if not text:
        return 0
    # Find all word tokens and punctuation
    tokens = TOKEN_REGEX.findall(text)
    return len(tokens)


def count_words(text: str) -> int:
    """Count whitespace-separated words."""
    if not text:
        return 0
    return len(text.split())


def count_characters(text: str) -> int:
    """Count total characters."""
    return len(text)


def compute_sha256(text: str) -> str:
    """Compute normalized SHA-256 hex digest of text."""
    clean = text.strip()
    return hashlib.sha256(clean.encode("utf-8")).hexdigest()


def generate_deterministic_chunk_id(
    document_id: str,
    page_number: Optional[int] = None,
    chunk_index: int = 0,
    content: str = "",
    section_id: Optional[str] = None,
) -> str:
    """Generate a deterministic, stable chunk identifier based on document and content hash.

    Format: chk_{doc_hash8}_{page_str}_{chunk_index:04d}_{content_hash8}
    Guarantees stable chunk IDs across identical document runs without UUID non-determinism.
    """
    doc_hash = hashlib.sha256(document_id.encode("utf-8")).hexdigest()[:8]
    page_str = f"p{page_number}" if page_number is not None else "p0"
    if len(content) == 64 and all(c in "0123456789abcdefABCDEF" for c in content):
        content_hash = content[:8].lower()
    elif content:
        content_hash = compute_sha256(content)[:8]
    else:
        content_hash = "00000000"

    sec_prefix = ""
    if section_id:
        clean_sec = re.sub(r"[^a-zA-Z0-9]", "", section_id)[:6]
        if clean_sec:
            sec_prefix = f"_{clean_sec}"

    return f"chk_{doc_hash}_{page_str}{sec_prefix}_{chunk_index:04d}_{content_hash}"


def split_into_paragraphs(text: str) -> list[str]:
    """Split text on double newlines while stripping empty blocks."""
    if not text:
        return []
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return paras


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences while protecting decimals, abbreviations, and units."""
    if not text:
        return []

    raw_candidates = SENTENCE_SPLIT_REGEX.split(text)
    sentences: list[str] = []
    buffer = ""

    for cand in raw_candidates:
        cand_strip = cand.strip()
        if not cand_strip:
            continue

        if buffer:
            cand_strip = f"{buffer} {cand_strip}"
            buffer = ""

        # Check if cand ends with an abbreviation (e.g. 'e.g.', 'Fig.')
        last_word = cand_strip.split()[-1].lower() if cand_strip.split() else ""
        if last_word in ABBREVIATIONS or re.search(r"\b\d+\.$", cand_strip):
            buffer = cand_strip
        else:
            sentences.append(cand_strip)

    if buffer:
        sentences.append(buffer)

    return sentences


def split_into_list_items(text: str) -> list[str]:
    """Split text into list items (bullet points or numbered items)."""
    if not text:
        return []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines
