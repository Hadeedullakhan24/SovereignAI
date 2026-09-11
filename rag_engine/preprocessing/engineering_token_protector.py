"""Engineering token protection engine for refinery and industrial documents."""

from __future__ import annotations

import re
from typing import NamedTuple

from rag_engine.preprocessing.exceptions import TokenProtectionError


class ProtectedMatch(NamedTuple):
    original: str
    placeholder: str
    start: int
    end: int
    token_type: str


class EngineeringTokenProtector:
    """Detects, masks, restores, and validates refinery engineering identifiers.

    Guarantees that critical tokens (e.g. Pump P-203, MOV-101, API-610, OISD-105,
    ASME, PNGRB, ISO, 10 bar, 250°C, kg/cm², MPa, psi, m³/hr, valve IDs, loop IDs)
    are never altered, truncated, or corrupted by cleaning stages.
    """

    # Deterministic regex patterns for critical refinery tokens
    PATTERNS: list[tuple[str, re.Pattern[str]]] = [
        # Equipment tags with prefix (e.g., "Pump P-203", "Compressor C-101")
        (
            "EQUIPMENT_WITH_PREFIX",
            re.compile(
                r"\b(?:Pump|Motor|Compressor|Turbine|Exchanger|Vessel|Tank|Furnace|Boiler|Reactor)\s+[A-Z]{1,4}[-_][0-9]{2,4}[A-Z]?\b",
                re.IGNORECASE,
            ),
        ),
        # Actuated & manual valves, instruments (e.g., MOV-101, PSV-201, XV-104, FCV-102, PT-101)
        (
            "VALVE_INSTRUMENT_TAG",
            re.compile(
                r"\b(?:MOV|PSV|PRV|XV|FCV|TCV|PCV|LCV|ESD|SOV|PT|TT|LT|FT|PIT|TIT|LIT|FIT|TIC|PIC|LIC|FIC)[-_][0-9]{2,4}[A-Z]?\b",
                re.IGNORECASE,
            ),
        ),
        # General equipment tag code (e.g., P-203, C-101, E-102, V-12)
        (
            "EQUIPMENT_TAG",
            re.compile(
                r"\b[A-Z]{1,3}[-_][0-9]{2,4}[A-Z]?\b",
            ),
        ),
        # Regulatory and engineering standards (e.g., API-610, API 610, OISD-105, ASME, PNGRB, ISO)
        (
            "STANDARDS",
            re.compile(
                r"\b(?:API[- ]?[0-9]{2,4}[A-Z]?|OISD[- ]?[0-9]{2,4}|ASME(?:\s+[A-Z0-9.-]+)?|PNGRB(?:\s+[A-Z0-9.-]+)?|ISO[- ]?[0-9]{3,5}(?:-[0-9]+)?|ASTM\s+[A-Z0-9.-]+|NFPA[- ]?[0-9]{2,4})\b",
                re.IGNORECASE,
            ),
        ),
        # Physical operating parameters: pressure, temperature, flow with values
        (
            "OPERATING_LIMITS",
            re.compile(
                r"\b-?\d+(?:\.\d+)?\s*(?:bar|psi|kg/cm²|kg/cm2|kPa|MPa|mbar|°C|deg\s*C|°F|deg\s*F|m³/hr|m3/hr|m³/h|m3/h|Nm³/hr|Nm3/hr|L/min|lpm|GPM|kg/hr|kg/h)\b",
                re.IGNORECASE,
            ),
        ),
        # Standalone physical units
        (
            "STANDALONE_UNITS",
            re.compile(
                r"\b(?:kg/cm²|kg/cm2|m³/hr|m3/hr|Nm³/hr|Nm3/hr|MPa|kPa|psi|bar|°C)\b",
            ),
        ),
        # Loop IDs & Line numbers
        (
            "LOOP_OR_LINE",
            re.compile(
                r"\b(?:Loop|Line|Pipe|P&ID|Drawing)\s*#?\s*[A-Z0-9\"'/_.-]+\b",
                re.IGNORECASE,
            ),
        ),
        # Document IDs & Revision tags
        (
            "DOC_REVISION",
            re.compile(
                r"\b(?:DOC[-_][A-Z0-9-]+|MRPL[-_][A-Z0-9-]+|Rev(?:\.|ision)?\s*[A-Z0-9.-]+)\b",
                re.IGNORECASE,
            ),
        ),
    ]

    def __init__(self) -> None:
        self._sentinel_prefix = "__ENG_TOKEN_"
        self._sentinel_suffix = "__"

    def extract_tokens(self, text: str) -> list[str]:
        """Extract a deduplicated list of all recognized engineering tokens from text."""
        tokens: set[str] = set()
        for _, pattern in self.PATTERNS:
            for match in pattern.finditer(text):
                token = match.group(0).strip()
                if len(token) > 1:
                    tokens.add(token)
        return sorted(tokens)

    def mask(self, text: str) -> tuple[str, dict[str, str]]:
        """Replace all engineering tokens with temporary sentinels.

        Returns (masked_text, placeholder_map) where placeholder_map maps
        `__ENG_TOKEN_idx__` back to original token string.
        """
        # Find all matches and their spans
        spans: list[tuple[int, int, str]] = []
        for _, pattern in self.PATTERNS:
            for match in pattern.finditer(text):
                spans.append((match.start(), match.end(), match.group(0)))

        if not spans:
            return text, {}

        # Resolve overlapping spans (keep the longest span)
        spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
        non_overlapping: list[tuple[int, int, str]] = []
        last_end = -1
        for start, end, orig in spans:
            if start >= last_end:
                non_overlapping.append((start, end, orig))
                last_end = end

        # Construct masked text and mapping
        parts: list[str] = []
        mapping: dict[str, str] = {}
        idx = 0
        curr_pos = 0

        for start, end, orig in non_overlapping:
            parts.append(text[curr_pos:start])
            placeholder = f"{self._sentinel_prefix}{idx}{self._sentinel_suffix}"
            mapping[placeholder] = orig
            parts.append(placeholder)
            curr_pos = end
            idx += 1

        parts.append(text[curr_pos:])
        return "".join(parts), mapping

    def unmask(self, text: str, mapping: dict[str, str]) -> str:
        """Restore all original engineering tokens from placeholders."""
        if not mapping:
            return text

        result = text
        for placeholder, original in mapping.items():
            result = result.replace(placeholder, original)
        return result

    def verify_tokens_preserved(
        self, original_text: str, cleaned_text: str, strict: bool = False
    ) -> list[str]:
        """Verify that all engineering tokens present in original_text exist in cleaned_text.

        Returns list of missing/corrupted tokens. If strict=True and any are missing,
        raises TokenProtectionError.
        """
        original_tokens = self.extract_tokens(original_text)
        missing_tokens: list[str] = []

        for token in original_tokens:
            # Check if token exists in cleaned_text verbatim
            if token not in cleaned_text:
                missing_tokens.append(token)

        if strict and missing_tokens:
            raise TokenProtectionError(
                f"Engineering token protection failed: {len(missing_tokens)} tokens missing "
                f"or corrupted after cleaning: {missing_tokens[:5]}"
            )
        return missing_tokens
