"""
Loader Utilities — Helper functions for MIME detection, metadata, and token estimation.

Sovereign On-Premise Agentic AI Workbench (SIH26117 / MRPL)
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import mimetypes
from pathlib import Path
import re
from typing import Any, Optional


# Ensure common MIME types are registered
mimetypes.init()
mimetypes.add_type("application/pdf", ".pdf")
mimetypes.add_type("application/vnd.openxmlformats-officedocument.wordprocessingml.document", ".docx")
mimetypes.add_type("application/vnd.openxmlformats-officedocument.presentationml.presentation", ".pptx")
mimetypes.add_type("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", ".xlsx")
mimetypes.add_type("text/csv", ".csv")
mimetypes.add_type("text/markdown", ".md")
mimetypes.add_type("text/plain", ".txt")
mimetypes.add_type("image/jpeg", ".jpg")
mimetypes.add_type("image/jpeg", ".jpeg")
mimetypes.add_type("image/png", ".png")
mimetypes.add_type("image/bmp", ".bmp")
mimetypes.add_type("image/tiff", ".tiff")
mimetypes.add_type("image/tiff", ".tif")


def detect_mime_type(file_path: Path | str) -> str:
    """
    Detect the MIME type of a file using both magic bytes inspection and extension lookup.

    Args:
        file_path: Path to target file.

    Returns:
        MIME type string (e.g. 'application/pdf').
    """
    path = Path(file_path).resolve()
    if not path.exists():
        mime, _ = mimetypes.guess_type(path.name)
        return mime or "application/octet-stream"

    # Check magic bytes for common formats
    try:
        with open(path, "rb") as fh:
            magic = fh.read(16)

        if magic.startswith(b"%PDF-"):
            return "application/pdf"
        elif magic.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        elif magic.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        elif magic.startswith(b"BM"):
            return "image/bmp"
        elif magic.startswith(b"II*\x00") or magic.startswith(b"MM\x00*"):
            return "image/tiff"
        elif magic.startswith(b"PK\x03\x04") or magic.startswith(b"PK\x05\x06"):
            # Could be docx, pptx, xlsx - defer to extension lookup
            ext = path.suffix.lower()
            if ext == ".docx":
                return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            elif ext == ".pptx":
                return "application/vnd.openxmlformats-officedocument.presentationml.presentation"
            elif ext == ".xlsx":
                return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            return "application/zip"
    except Exception:
        pass

    # Fallback to extension guessing
    mime, _ = mimetypes.guess_type(path.name)
    return mime or "application/octet-stream"


def estimate_tokens(text: str) -> int:
    """
    Heuristic token count estimation (~4 characters per token for English/technical text).
    """
    if not text:
        return 0
    # Clean whitespace
    char_count = len(text)
    # Average across character ratio and word ratio (~0.75 words per token)
    words = len(text.split())
    return max(1, int(round((char_count / 4.0 + words * 1.33) / 2.0))) if words > 0 else 0


def count_words(text: str) -> int:
    """Return total word count."""
    return len(text.split()) if text else 0


def count_characters(text: str) -> int:
    """Return total character count."""
    return len(text) if text else 0


def compute_file_sha256(file_path: Path | str, chunk_size: int = 65536) -> str:
    """Compute SHA-256 checksum for a file."""
    p = Path(file_path).resolve()
    hasher = hashlib.sha256()
    with open(p, "rb") as fh:
        while chunk := fh.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_file_timestamps(file_path: Path | str) -> tuple[str, str]:
    """Return (created_at_iso, modified_at_iso) for a file."""
    p = Path(file_path).resolve()
    stat = p.stat()
    ctime_dt = datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc)
    mtime_dt = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
    return ctime_dt.isoformat(), mtime_dt.isoformat()
