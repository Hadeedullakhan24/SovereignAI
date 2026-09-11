"""Retrieval Engine Utilities.

Common tokenizers, regular expressions, hashing utilities, and text helpers
engineered specifically for petroleum refinery domain data (SIH26117 / MRPL).
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Sequence

# Refinery Equipment Tag Patterns (e.g. P-203, P203, MOV-101, MOV101, HX-01, V-102, C-101)
EQUIPMENT_TAG_REGEX = re.compile(
    r"\b([A-Z]{1,4})[-_]?([0-9]{2,4}[A-Z]?)\b", re.IGNORECASE
)

# Common Refinery Standards (e.g. OISD-105, OISD-STD-105, OISD 105, API-610, API 650, ASME B31.3, PNGRB)
STANDARD_REGEX = re.compile(
    r"\b(OISD(?:[-_\s]?(?:STD|RP|GDN))?[-_\s]?[0-9]{2,3}|API[-_\s]?[0-9]{2,4}|ASME[-_\s]?[A-Z0-9\.]+|PNGRB[-_\s]?[A-Z0-9\.]+|ISO[-_\s]?[0-9]{4,5})\b",
    re.IGNORECASE,
)

# Standard question/prose stopwords to prevent noise in lexical search and reranking
QUERY_STOPWORDS = frozenset({
    "what", "is", "the", "a", "an", "and", "or", "of", "for", "in", "on", "at",
    "to", "by", "with", "from", "as", "are", "were", "was", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "can", "could", "should", "would",
    "shall", "will", "may", "might", "must", "about", "which", "who", "whom",
    "this", "that", "these", "those", "it", "its", "give", "tell", "show", "me",
    "please", "state", "describe", "explain",
})

# Plant Units (e.g. CDU-1, CDU, VDU, DCU, NHT, CCR, FCCU, HGU, MS-BLOCK)
PLANT_UNIT_REGEX = re.compile(
    r"\b(CDU[-_\s]?[12]?|VDU[-_\s]?[12]?|DCU|NHT|CCR|FCCU|HGU|DHDS|SRU|OM&S|ISOM|MS[-_\s]?BLOCK|CPP)\b",
    re.IGNORECASE,
)

# Pressure and Temperature values (e.g. 150 bar, 150bar, 10 kg/cm2, 250 deg C, 250°C)
PRESSURE_REGEX = re.compile(
    r"(\b[0-9]+(?:\.[0-9]+)?)\s*(bar|kg/cm2|kg/cm²|psi|mpa|kpa)\b", re.IGNORECASE
)
TEMPERATURE_REGEX = re.compile(
    r"(\b[0-9]+(?:\.[0-9]+)?)\s*(?:°|deg|degrees)?\s*([CcFfKk])\b|(\b[0-9]+(?:\.[0-9]+)?)\s*(?:°C|°F)\b",
    re.IGNORECASE,
)

# Line numbers (e.g. LINE-101-CS, 4"-LINE-101-CS)
LINE_NUMBER_REGEX = re.compile(
    r"\b(?:\d+\"?-)?(?:LINE|L)[-_][0-9]{2,4}[-_][A-Z0-9-]+\b", re.IGNORECASE
)

# Revision numbers (e.g. Rev 2, Rev. 0, Revision 3, R-01)
REVISION_REGEX = re.compile(
    r"\b(?:Rev(?:ision)?\.?|R)[-_\s]?([0-9]{1,3}|[A-Z])\b", re.IGNORECASE
)

# Page references (e.g. page 42, p. 15, pg 12)
PAGE_REF_REGEX = re.compile(
    r"\b(?:page|pg\.?|p\.)\s*([0-9]{1,4})\b", re.IGNORECASE
)

# Section references (e.g. section 3.1, sec. 2, clause 4.2)
SECTION_REF_REGEX = re.compile(
    r"\b(?:section|sec\.?|clause)\s*([0-9]+(?:\.[0-9]+)*)\b", re.IGNORECASE
)


def normalize_whitespace(text: str) -> str:
    """Collapse consecutive whitespace into single spaces and strip leading/trailing."""
    return re.sub(r"\s+", " ", text).strip()


def estimate_token_count(text: str) -> int:
    """Estimate token count for technical English text without external tokenizers.
    
    Approximately 4 characters per token for technical refinery English.
    """
    if not text:
        return 0
    clean = text.strip()
    if not clean:
        return 0
    # Average ~4 chars/token, bounded below by word count
    words = len(clean.split())
    by_chars = (len(clean) + 3) // 4
    return max(words, by_chars)


def tokenize_refinery_text(text: str) -> list[str]:
    """Tokenize technical refinery text preserving equipment tags, standards, and metrics.
    
    Splits on punctuation except hyphens inside alphanumeric identifiers (e.g. 'P-203', 'OISD-105').
    """
    if not text:
        return []
    # Replace punctuation other than hyphens and periods inside numbers with space
    # Matches words with internal hyphens or alphanumeric tokens
    tokens = re.findall(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*|[0-9]+(?:\.[0-9]+)?", text)
    return [t.lower() for t in tokens if t.strip()]


def compute_sha256(content: str) -> str:
    """Compute SHA-256 digest of input string."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def format_cache_key(
    query: str,
    strategy: str,
    top_k: int,
    filters: Any = None,
) -> str:
    """Generate deterministic cache key for query and configuration."""
    filter_repr = str(sorted(filters.items())) if isinstance(filters, dict) else str(filters)
    raw = f"{query.strip().lower()}::{strategy}::{top_k}::{filter_repr}"
    return compute_sha256(raw)
