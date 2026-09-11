"""Utility functions for string processing, similarity, and metric calculations."""

from __future__ import annotations

import difflib
import unicodedata
from typing import Sequence


def compute_string_similarity(str1: str, str2: str) -> float:
    """Compute normalized character sequence similarity ratio between 0.0 and 1.0."""
    s1, s2 = str1.strip().lower(), str2.strip().lower()
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return difflib.SequenceMatcher(None, s1, s2).ratio()


def count_characters(text: str) -> int:
    """Count non-null characters in text."""
    return len(text)


def compute_text_diff_stats(text_before: str, text_after: str) -> tuple[int, int]:
    """Compute (characters_removed, characters_normalized) between two versions of text.

    - characters_removed: net positive reduction in total length.
    - characters_normalized: character changes (substitutions or edits).
    """
    if text_before == text_after:
        return 0, 0

    len_before = len(text_before)
    len_after = len(text_after)
    chars_removed = max(0, len_before - len_after)

    # For high-throughput processing on large documents (>20,000 chars),
    # use fast O(N) length difference to avoid difflib quadratic slowdown
    if len_before > 20000 or len_after > 20000:
        chars_normalized = abs(len_before - len_after)
        return chars_removed, chars_normalized

    # Use SequenceMatcher matching blocks for smaller texts
    matcher = difflib.SequenceMatcher(None, text_before, text_after)
    matching_chars = sum(match.size for match in matcher.get_matching_blocks())
    chars_normalized = max(0, len_before - matching_chars)

    return chars_removed, chars_normalized


def is_mostly_ascii(text: str, threshold: float = 0.8) -> bool:
    """Check if text is predominantly ASCII characters."""
    if not text:
        return True
    ascii_count = sum(1 for c in text if ord(c) < 128)
    return (ascii_count / len(text)) >= threshold


def normalize_unicode_nfkc(text: str) -> str:
    """Apply standard NFKC normalization."""
    return unicodedata.normalize("NFKC", text)
