"""
Hash utilities — Content hashing for deduplication and cache keys.

Sovereign On-Premise Agentic AI Workbench (SIH26117 / MRPL)
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from dataset_engine.hash_generator import HashGenerator

_DEFAULT_GENERATOR = HashGenerator()


def hash_text(text: str, algorithm: str = "sha256") -> str:
    """Generate hex digest for text string."""
    if algorithm.lower() == "sha256":
        return _DEFAULT_GENERATOR.generate_text_hash(text)
    h = getattr(hashlib, algorithm.lower())()
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def hash_file(path: Path | str, algorithm: str = "sha256", chunk_size: int = 65536) -> str:
    """Generate hex digest for file contents using streaming chunks."""
    if algorithm.lower() == "sha256" and chunk_size == 65536:
        return _DEFAULT_GENERATOR.generate_file_hash(path)
    p = Path(path).resolve()
    h = getattr(hashlib, algorithm.lower())()
    with open(p, "rb") as fh:
        while chunk := fh.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def hash_embedding(vector: list[float]) -> str:
    """Generate a stable SHA-256 hash for an embedding vector float array."""
    data = ",".join(f"{x:.6f}" for x in vector).encode("utf-8")
    return hashlib.sha256(data).hexdigest()
