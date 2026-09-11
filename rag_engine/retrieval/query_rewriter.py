"""Deterministic Query Rewriter.

Rewrites common refinery spelling variants, canonicalizes equipment tags and
standards, and cleans syntactic noise for deterministic retrieval consistency.
"""

from __future__ import annotations

import re

from rag_engine.retrieval.query_normalizer import QueryNormalizer
from rag_engine.retrieval.retrieval_utils import normalize_whitespace


class QueryRewriter:
    """Deterministic query rewriter with domain dictionaries."""

    # Common technical misspelling replacements
    _SPELLING_CORRECTIONS = {
        "lubricaton": "lubrication",
        "pressue": "pressure",
        "temprature": "temperature",
        "temperatue": "temperature",
        "centrifual": "centrifugal",
        "vibraton": "vibration",
        "overhual": "overhaul",
        "maintenence": "maintenance",
        "maintanance": "maintenance",
        "clearence": "clearance",
        "impellar": "impeller",
        "flange": "flange",
        "vavle": "valve",
    }

    # Standard canonicalization: e.g. "OISD 105" or "OISD_105" -> "OISD-105"
    _STD_CANONICAL = re.compile(
        r"\b(OISD|API|ASME|ISO)[-_\s]+([0-9]{2,4}[A-Z0-9\.]*)\b", re.IGNORECASE
    )

    @classmethod
    def rewrite(cls, query: str) -> str:
        """Rewrite and standardize query string."""
        if not query:
            return ""

        # First run through normalizer
        text = QueryNormalizer.normalize(query)

        # Standardize standards format
        text = cls._STD_CANONICAL.sub(lambda m: f"{m.group(1).upper()}-{m.group(2).upper()}", text)

        # Fix spelling variants
        tokens = text.split()
        corrected_tokens = []
        for tok in tokens:
            low = tok.lower().strip(".,?!:;")
            if low in cls._SPELLING_CORRECTIONS:
                # Retain punctuation
                repl = cls._SPELLING_CORRECTIONS[low]
                if tok.endswith((".", ",", "?", "!")):
                    repl += tok[-1]
                corrected_tokens.append(repl)
            else:
                corrected_tokens.append(tok)

        return normalize_whitespace(" ".join(corrected_tokens))
