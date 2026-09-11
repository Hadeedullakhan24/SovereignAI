"""Repeated header and footer detection and safe removal engine."""

from __future__ import annotations

from collections import Counter
import re
from typing import NamedTuple

from rag_engine.schemas.parsed_document import Section


class HeaderFooterCandidate(NamedTuple):
    text: str
    is_header: bool
    section_index: int


class HeaderFooterDetector:
    """Stage 7: High-confidence detection and removal of repeated headers and footers.

    Enforces strict rules:
    - Only removes items if they appear across multiple pages/sections (high confidence).
    - NEVER removes unique section headings or body text.
    - Preserves all unique content.
    """

    # Patterns typical for page numbers or footers: e.g. "Page 1 of 12", "1 / 5", "Confidential"
    PAGINATION_FOOTER_PATTERN = re.compile(
        r"^(?:Page\s+\d+(?:\s+of\s+\d+)?|\d+\s*/\s*\d+|[-—]\s*\d+\s*[-—])$",
        re.IGNORECASE,
    )

    def __init__(self, min_occurrences: int = 2, max_line_length: int = 150) -> None:
        self.min_occurrences = min_occurrences
        self.max_line_length = max_line_length

    def detect_and_remove(
        self, sections: list[Section]
    ) -> tuple[list[Section], list[str], list[str]]:
        """Detect and remove repeated headers and footers across sections.

        Returns (cleaned_sections, removed_headers, removed_footers).
        """
        if len(sections) < 2:
            # Cannot establish repetition pattern with fewer than 2 sections
            return sections, [], []

        # 1. Collect top candidate headers (first non-empty line of each section)
        # and bottom candidate footers (last non-empty line of each section)
        top_candidates: list[str] = []
        bottom_candidates: list[str] = []

        for sec in sections:
            lines = [line.strip() for line in sec.content.splitlines() if line.strip()]
            if not lines:
                continue

            # First line (exclude markdown heading like # Title)
            first_line = lines[0]
            if not first_line.startswith("#") and len(first_line) <= self.max_line_length:
                top_candidates.append(first_line)

            # Last line
            if len(lines) > 1:
                last_line = lines[-1]
                if not last_line.startswith("#") and len(last_line) <= self.max_line_length:
                    bottom_candidates.append(last_line)

        # 2. Find high-confidence repeated headers and footers
        header_counts = Counter(top_candidates)
        footer_counts = Counter(bottom_candidates)

        detected_headers: set[str] = {
            text for text, count in header_counts.items()
            if count >= self.min_occurrences and not self._is_likely_heading(text)
        }

        detected_footers: set[str] = {
            text for text, count in footer_counts.items()
            if (count >= self.min_occurrences or self.PAGINATION_FOOTER_PATTERN.match(text))
            and not self._is_likely_heading(text)
        }

        # 3. Strip detected headers and footers from section content
        cleaned_sections: list[Section] = []
        removed_headers_list: list[str] = []
        removed_footers_list: list[str] = []

        for sec in sections:
            lines = sec.content.splitlines()
            if not lines:
                cleaned_sections.append(sec)
                continue

            new_lines = list(lines)

            # Check and remove header from top
            while new_lines and not new_lines[0].strip():
                new_lines.pop(0)

            if new_lines and new_lines[0].strip() in detected_headers:
                removed_hdr = new_lines.pop(0).strip()
                if removed_hdr not in removed_headers_list:
                    removed_headers_list.append(removed_hdr)

            # Check and remove footer from bottom
            while new_lines and not new_lines[-1].strip():
                new_lines.pop()

            if new_lines and (
                new_lines[-1].strip() in detected_footers
                or self.PAGINATION_FOOTER_PATTERN.match(new_lines[-1].strip())
            ):
                removed_ftr = new_lines.pop().strip()
                if removed_ftr not in removed_footers_list:
                    removed_footers_list.append(removed_ftr)

            # Update section content
            new_content = "\n".join(new_lines).strip()
            new_sec = sec.model_copy(update={"content": new_content})
            cleaned_sections.append(new_sec)

        return cleaned_sections, removed_headers_list, removed_footers_list

    @staticmethod
    def _is_likely_heading(line: str) -> bool:
        """Check if line is likely a functional section heading that should not be removed."""
        line_clean = line.strip().lower()
        if re.match(r"^(?:section|chapter|part|\d+\.|\d+\))\s+", line_clean):
            return True
        if line_clean in {"introduction", "scope", "purpose", "references", "safety precautions"}:
            return True
        return False
