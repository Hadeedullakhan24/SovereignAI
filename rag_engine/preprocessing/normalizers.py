"""Normalizers for Unicode, encoding, bullets, and numbered lists."""

from __future__ import annotations

import re
import unicodedata


class UnicodeNormalizer:
    """Stage 1: Applies standard Unicode NFKC normalization."""

    @staticmethod
    def normalize(text: str) -> str:
        """Normalize Unicode characters using NFKC decomposition and recomposition."""
        if not text:
            return ""
        return unicodedata.normalize("NFKC", text)


class EncodingNormalizer:
    """Stage 2: Cleans corrupted encoding, control characters, zero-width chars, and smart quotes."""

    # Unprintable/invisible control characters (excluding standard whitespace \t, \n, \r)
    CONTROL_CHARS_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

    # Zero-width spaces, joiners, byte-order marks, soft hyphens
    ZERO_WIDTH_PATTERN = re.compile(r"[\u200b\u200c\u200d\ufeff\u00ad\u2060]")

    # Smart/curly quotes and typographical replacements
    QUOTE_REPLACEMENTS = {
        "\u2018": "'",  # left single quotation mark
        "\u2019": "'",  # right single quotation mark
        "\u201a": "'",  # single low-9 quotation mark
        "\u201b": "'",  # single high-reversed-9 quotation mark
        "\u201c": '"',  # left double quotation mark
        "\u201d": '"',  # right double quotation mark
        "\u201e": '"',  # double low-9 quotation mark
        "\u201f": '"',  # double high-reversed-9 quotation mark
        "\u2026": "...",  # horizontal ellipsis
    }

    @classmethod
    def normalize(cls, text: str) -> str:
        """Strip control characters, zero-width entities, and standardize typography."""
        if not text:
            return ""

        # Remove control characters
        result = cls.CONTROL_CHARS_PATTERN.sub("", text)

        # Remove zero-width spaces
        result = cls.ZERO_WIDTH_PATTERN.sub("", result)

        # Replace smart quotes and ellipsis
        for orig, replacement in cls.QUOTE_REPLACEMENTS.items():
            result = result.replace(orig, replacement)

        return result


class BulletNormalizer:
    """Stage 10: Standardizes heterogeneous bullet markers into Markdown standard '- '."""

    # Bullet markers at start of a line: •, *, ▪, ▫, ►, ▻, ⁃, ◦, ○, or en/em-dash bullet
    BULLET_LINE_PATTERN = re.compile(
        r"^([ \t]*)(?:[•▪▫►▻⁃◦○*]|\u2013|\u2014)(?:\s+|\t+)",
        re.MULTILINE,
    )

    @classmethod
    def normalize(cls, text: str) -> str:
        """Convert diverse bullet symbols at the start of lines to '- '."""
        if not text:
            return ""
        return cls.BULLET_LINE_PATTERN.sub(r"\1- ", text)


class ListNormalizer:
    """Stage 11: Normalizes diverse numbered list prefixes into standard Markdown '1. '."""

    # Numbered patterns at line start: e.g. "1)", "(1)", "1 -", "1."
    NUMBERED_PAREN_PATTERN = re.compile(
        r"^([ \t]*)(?:\((\d{1,4})\)|(\d{1,4})\)|\b(\d{1,4})\s*-\s+)(?=\s|\S)",
        re.MULTILINE,
    )

    # Clean redundant spacing after dot in standard numbered list: "1.   Item" -> "1. Item"
    EXCESS_NUMBERED_SPACE = re.compile(
        r"^([ \t]*\d{1,4}\.)[ \t]{2,}",
        re.MULTILINE,
    )

    @classmethod
    def normalize(cls, text: str) -> str:
        """Normalize varied numbered list prefixes into standard Markdown digit. format."""
        if not text:
            return ""

        def _replace_numbered(match: re.Match[str]) -> str:
            indent = match.group(1)
            num = match.group(2) or match.group(3) or match.group(4)
            return f"{indent}{num}. "

        result = cls.NUMBERED_PAREN_PATTERN.sub(_replace_numbered, text)
        result = cls.EXCESS_NUMBERED_SPACE.sub(r"\1 ", result)
        return result
