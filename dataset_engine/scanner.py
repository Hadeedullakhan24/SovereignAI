"""
Dataset Scanner — Recursive Filesystem Discovery & Categorization.

Sovereign On-Premise Agentic AI Workbench (SIH26117 / MRPL)
Discovers all files in datasets/, classifies them by refinery domain category,
filters supported vs unsupported extensions, and produces structured records.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from rag_engine.config.constants import (
    SUPPORTED_ALL_FORMATS,
    SUPPORTED_DOCUMENT_FORMATS,
    SUPPORTED_IMAGE_FORMATS,
)


@dataclass(frozen=True)
class DiscoveredFile:
    """Represents a discovered file from filesystem scanning."""

    file_path: Path
    relative_path: str
    category: str
    subcategory: str
    extension: str
    size_bytes: int
    last_modified: str
    is_supported: bool


class DatasetScanner:
    """Recursively scans dataset directories to discover, categorize, and filter files."""

    def __init__(
        self,
        supported_extensions: Optional[set[str] | frozenset[str]] = None,
        ignored_extensions: Optional[set[str]] = None,
        ignored_directories: Optional[set[str]] = None,
    ) -> None:
        """
        Initialize DatasetScanner.

        Args:
            supported_extensions: Set of allowed extensions (lowercase with dot).
            ignored_extensions: Set of extensions to completely ignore.
            ignored_directories: Directory names to skip during recursive scan.
        """
        self.supported_extensions = (
            {ext.lower() for ext in supported_extensions}
            if supported_extensions
            else {ext.lower() for ext in SUPPORTED_ALL_FORMATS}
        )
        self.ignored_extensions = (
            {ext.lower() for ext in ignored_extensions}
            if ignored_extensions
            else {".pyc", ".gitkeep", ".gitignore", ".swp", ".ds_store"}
        )
        self.ignored_directories = (
            {d.lower() for d in ignored_directories}
            if ignored_directories
            else {".git", "__pycache__", ".venv", "node_modules", ".idea", ".vscode"}
        )

    def scan(self, dataset_dir: Path | str) -> list[DiscoveredFile]:
        """
        Recursively scan a directory and return all discovered supported and unsupported files.

        Args:
            dataset_dir: Path to the root dataset folder.

        Returns:
            List of DiscoveredFile instances.
        """
        root = Path(dataset_dir).resolve()
        if not root.exists():
            raise FileNotFoundError(f"Dataset root directory does not exist: {root}")

        discovered: list[DiscoveredFile] = []

        for item in root.rglob("*"):
            if not item.is_file():
                continue

            # Check if any parent directory is in ignored_directories
            parts_lower = [p.lower() for p in item.relative_to(root).parts]
            if any(ignored in parts_lower[:-1] for ignored in self.ignored_directories):
                continue

            ext = item.suffix.lower()
            if ext in self.ignored_extensions:
                continue

            # Determine category and subcategory from path relative to dataset root
            rel_parts = item.relative_to(root).parts
            category = rel_parts[0] if len(rel_parts) > 1 else "root"
            subcategory = rel_parts[1] if len(rel_parts) > 2 else ""

            # Normalized POSIX relative path
            rel_path_str = item.relative_to(root).as_posix()

            try:
                stat = item.stat()
                size_bytes = stat.st_size
                mtime_dt = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
                last_modified_str = mtime_dt.isoformat()
            except (OSError, PermissionError):
                size_bytes = 0
                last_modified_str = datetime.now(timezone.utc).isoformat()

            is_supported = ext in self.supported_extensions

            discovered.append(
                DiscoveredFile(
                    file_path=item,
                    relative_path=rel_path_str,
                    category=category,
                    subcategory=subcategory,
                    extension=ext,
                    size_bytes=size_bytes,
                    last_modified=last_modified_str,
                    is_supported=is_supported,
                )
            )

        return discovered

    def get_supported_files(self, dataset_dir: Path | str) -> list[DiscoveredFile]:
        """
        Scan directory and filter down to supported files only.

        Args:
            dataset_dir: Path to dataset directory.

        Returns:
            List of DiscoveredFile instances where is_supported is True.
        """
        all_files = self.scan(dataset_dir)
        return [f for f in all_files if f.is_supported]
