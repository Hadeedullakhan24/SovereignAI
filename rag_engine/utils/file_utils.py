"""
File utilities — Path manipulation, format detection, and safe reading.

Sovereign On-Premise Agentic AI Workbench (SIH26117 / MRPL)
"""

from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Optional


def safe_read_file(path: Path | str, max_bytes: Optional[int] = None) -> bytes:
    """
    Safely read bytes from a file with an optional byte limit.

    Args:
        path: Path to the file.
        max_bytes: Maximum number of bytes to read (None reads all).

    Returns:
        Raw bytes.
    """
    p = Path(path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")
    with open(p, "rb") as fh:
        if max_bytes is not None:
            return fh.read(max_bytes)
        return fh.read()


def detect_format(path: Path | str) -> str:
    """Return the lowercase file extension including dot."""
    return Path(path).suffix.lower()


def get_file_size(path: Path | str) -> int:
    """Return the file size in bytes."""
    return Path(path).stat().st_size


def get_temp_path(prefix: str = "sov_rag_") -> Path:
    """Create and return a new temporary file path."""
    with tempfile.NamedTemporaryFile(prefix=prefix, delete=False) as tf:
        return Path(tf.name)


def list_files(
    directory: Path | str, extensions: Optional[set[str]] = None
) -> list[Path]:
    """
    Recursively list all files in directory, optionally filtering by extensions.

    Args:
        directory: Root search directory.
        extensions: Set of allowed extensions (lowercase with dot).

    Returns:
        List of matching Path instances.
    """
    d = Path(directory).resolve()
    if not d.is_dir():
        return []
    ext_set = {e.lower() for e in extensions} if extensions else None
    matches: list[Path] = []
    for item in d.rglob("*"):
        if item.is_file():
            if ext_set is None or item.suffix.lower() in ext_set:
                matches.append(item)
    return matches


def ensure_directory(path: Path | str) -> Path:
    """Create directory and parents if they do not exist."""
    p = Path(path).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p
