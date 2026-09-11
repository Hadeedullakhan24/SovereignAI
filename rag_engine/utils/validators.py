"""
Validators — Input validation for files, schemas, and configurations.

Sovereign On-Premise Agentic AI Workbench (SIH26117 / MRPL)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from dataset_engine.validator import DatasetValidator


def validate_file_exists(path: Path | str) -> bool:
    """Check if file exists and is a regular file."""
    p = Path(path).resolve()
    return p.exists() and p.is_file()


def validate_file_format(path: Path | str, allowed: set[str]) -> bool:
    """Check if file extension is within allowed set."""
    p = Path(path).resolve()
    return p.suffix.lower() in {ext.lower() for ext in allowed}


def validate_file_size(path: Path | str, max_bytes: int) -> bool:
    """Check if file size does not exceed max_bytes."""
    p = Path(path).resolve()
    try:
        return p.stat().st_size <= max_bytes
    except OSError:
        return False


def validate_chunk_size(size: int, min_size: int = 50, max_size: int = 4096) -> bool:
    """Validate that chunk size is within bounded range."""
    return min_size <= size <= max_size


def validate_embedding_dimension(vector: list[float], expected_dim: int) -> bool:
    """Validate that vector length equals expected dimension."""
    return len(vector) == expected_dim
