"""
Hash Generator — Streaming SHA-256 Digest Generation.

Sovereign On-Premise Agentic AI Workbench (SIH26117 / MRPL)
Provides memory-efficient, chunked SHA-256 hashing for all document types.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional


class HashGenerator:
    """Generates deterministic SHA-256 cryptographic hashes for files and text."""

    def __init__(self, chunk_size_bytes: int = 65536) -> None:
        """
        Initialize HashGenerator.

        Args:
            chunk_size_bytes: Read buffer size in bytes (default: 64 KB).
        """
        self.chunk_size_bytes = chunk_size_bytes

    def generate_file_hash(self, file_path: Path | str) -> str:
        """
        Compute the SHA-256 hex digest of a file using streaming chunks.

        Args:
            file_path: Path to the target file.

        Returns:
            64-character lowercase hexadecimal SHA-256 string.

        Raises:
            FileNotFoundError: If the file does not exist.
            PermissionError: If the file cannot be read.
            IOError: If an I/O error occurs during streaming.
        """
        path = Path(file_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"File not found or is not a regular file: {path}")

        hasher = hashlib.sha256()
        with open(path, "rb") as fh:
            while chunk := fh.read(self.chunk_size_bytes):
                hasher.update(chunk)

        return hasher.hexdigest()

    def generate_text_hash(self, text: str) -> str:
        """
        Compute SHA-256 digest of a text string.

        Args:
            text: Input string.

        Returns:
            64-character lowercase hexadecimal SHA-256 string.
        """
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def generate_bytes_hash(self, data: bytes) -> str:
        """
        Compute SHA-256 digest of raw bytes.

        Args:
            data: Raw bytes.

        Returns:
            64-character lowercase hexadecimal SHA-256 string.
        """
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def are_hashes_identical(hash1: str, hash2: str) -> bool:
        """
        Check if two hashes represent identical content.

        Args:
            hash1: First SHA-256 string.
            hash2: Second SHA-256 string.

        Returns:
            True if hashes match exactly.
        """
        if not hash1 or not hash2:
            return False
        return hash1.strip().lower() == hash2.strip().lower()
