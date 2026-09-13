"""Shared path resolution helper for Member 3 evaluation harnesses.

Resolves project root and output directories without fragile hardcoded parent depth.
"""
from __future__ import annotations

from pathlib import Path


def get_project_root() -> Path:
    """Resolve repository root (C:\SovereignAI) robustly."""
    curr = Path(__file__).resolve()
    for p in [curr] + list(curr.parents):
        if (p / "datasets").exists() and (p / "member3_ocr").exists():
            return p
    return curr.parents[2]


def get_member3_root() -> Path:
    """Resolve member3_ocr package directory."""
    return get_project_root() / "member3_ocr"


def get_output_dir() -> Path:
    """Resolve member3_ocr/output directory."""
    return get_member3_root() / "output"


def get_models_dir() -> Path:
    """Resolve member3_ocr/models directory."""
    return get_member3_root() / "models"


def get_datasets_dir() -> Path:
    """Resolve C:\SovereignAI\datasets directory."""
    return get_project_root() / "datasets"
