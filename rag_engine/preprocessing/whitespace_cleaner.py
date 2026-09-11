"""Whitespace, line ending, paragraph wrap, and hyphen reconstruction cleaner."""

from __future__ import annotations

import re


class WhitespaceCleaner:
    """Handles Stages 3, 4, 5, and 6:

    - Stage 3: Whitespace normalization (collapse horizontal whitespace, strip line ends).
    - Stage 4: Line ending normalization (convert CRLF/CR to LF, limit consecutive newlines).
    - Stage 5: Broken paragraph reconstruction (rejoin soft-wrapped sentences).
    - Stage 6: Hyphenated word reconstruction (e.g. Oper-\\nating -> Operating).
    """

    # Stage 3: Horizontal whitespace (spaces, tabs)
    HORIZONTAL_WHITESPACE = re.compile(r"[^\S\n\r]+")

    # Stage 6: Hyphen split at line break or following space: e.g. "Oper-\nating" or "Oper- ating" -> "Operating"
    HYPHEN_SPLIT_PATTERN = re.compile(r"\b([A-Za-z]{2,})-\s*(?:\r?\n|[ \t]+)\s*([a-z]{2,})\b")

    # Prefixes that normally retain their hyphen in technical English
    RETAIN_HYPHEN_PREFIXES = {
        "self",
        "cross",
        "quasi",
        "semi",
        "ex",
        "all",
        "half",
    }

    @classmethod
    def clean_whitespace(cls, text: str) -> str:
        """Stage 3: Collapse multiple horizontal spaces and strip trailing/leading spaces per line."""
        if not text:
            return ""

        lines: list[str] = []
        for line in text.splitlines():
            # Collapse multiple spaces and tabs to a single space
            cleaned_line = cls.HORIZONTAL_WHITESPACE.sub(" ", line).strip()
            lines.append(cleaned_line)

        return "\n".join(lines)

    @classmethod
    def normalize_line_endings(cls, text: str) -> str:
        """Stage 4: Convert CRLF and CR to LF, and collapse 3+ consecutive newlines to 2."""
        if not text:
            return ""

        # Normalize \r\n and \r to \n
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")

        # Collapse 3 or more consecutive newlines down to 2 (\n\n) to preserve paragraph structure
        collapsed = re.sub(r"\n{3,}", "\n\n", normalized)
        return collapsed.strip()

    @classmethod
    def reconstruct_hyphenated_words(cls, text: str) -> str:
        """Stage 6: Reassemble words split across line breaks with hyphens (e.g. Oper-\\nating -> Operating)."""
        if not text:
            return ""

        def _replace_hyphen(match: re.Match[str]) -> str:
            part1 = match.group(1)
            part2 = match.group(2)
            # If part1 is a prefix that standardly retains hyphen, preserve it
            if part1.lower() in cls.RETAIN_HYPHEN_PREFIXES:
                return f"{part1}-{part2}"
            return f"{part1}{part2}"

        return cls.HYPHEN_SPLIT_PATTERN.sub(_replace_hyphen, text)

    @classmethod
    def reconstruct_broken_paragraphs(cls, text: str) -> str:
        """Stage 5: Rejoin broken lines that belong to the same sentence/paragraph.

        Preserves:
        - Markdown headings (# Heading)
        - Bullet points (- Item)
        - Numbered lists (1. Item)
        - Table rows (| Cell |)
        - True paragraph breaks (separated by blank lines)
        - Lines ending in terminal punctuation (. ? ! : ;)
        """
        if not text:
            return ""

        paragraphs = text.split("\n\n")
        reconstructed_paragraphs: list[str] = []

        for para in paragraphs:
            lines = [line.strip() for line in para.split("\n") if line.strip()]
            if not lines:
                continue

            merged_lines: list[str] = []
            for i, line in enumerate(lines):
                if not merged_lines:
                    merged_lines.append(line)
                    continue

                prev_line = merged_lines[-1]

                # Check if prev_line and current line should NOT be merged
                if (
                    cls._is_heading_or_block(prev_line)
                    or cls._is_heading_or_block(line)
                    or cls._is_list_item(line)
                    or cls._is_table_row(prev_line)
                    or cls._is_table_row(line)
                    or prev_line.endswith((".", "?", "!", ":", ";", "-"))
                    or (prev_line.endswith('"') and len(prev_line) > 1 and prev_line[-2] in ".?!")
                ):
                    merged_lines.append(line)
                else:
                    # Merge line with previous line separated by space
                    merged_lines[-1] = f"{prev_line} {line}"

            reconstructed_paragraphs.append("\n".join(merged_lines))

        return "\n\n".join(reconstructed_paragraphs)

    @staticmethod
    def _is_heading_or_block(line: str) -> bool:
        """Check if line is a heading or markdown block marker."""
        stripped = line.strip()
        return (
            stripped.startswith(("#", "===", "---", "```", ">"))
            or (stripped.isupper() and len(stripped) < 80)
        )

    @staticmethod
    def _is_list_item(line: str) -> bool:
        """Check if line starts with a list bullet or numbered index."""
        stripped = line.strip()
        return bool(
            re.match(r"^(?:[-*•▪▫►⁃◦○]|\d{1,4}[.)])\s+", stripped)
        )

    @staticmethod
    def _is_table_row(line: str) -> bool:
        """Check if line is a markdown or delimited table row."""
        stripped = line.strip()
        return stripped.startswith("|") and stripped.endswith("|")
