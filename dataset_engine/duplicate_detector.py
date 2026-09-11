"""
Duplicate Detector — Hash, Filename & Path Collision Analysis.

Sovereign On-Premise Agentic AI Workbench (SIH26117 / MRPL)
Identifies content duplicates, duplicate filenames across categories, and redundant
records. Strictly reports duplicates without automated deletion.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class DuplicateReport:
    """Consolidated duplicate analysis results."""

    hash_duplicates: dict[str, list[str]] = field(default_factory=dict)
    filename_duplicates: dict[str, list[str]] = field(default_factory=dict)
    path_duplicates: list[str] = field(default_factory=list)
    total_duplicate_instances: int = 0
    total_unique_hashes: int = 0


class DuplicateDetector:
    """Analyzes scanned and hashed files to locate exact duplicates and naming clashes."""

    def __init__(self) -> None:
        self._hash_to_paths: dict[str, list[str]] = defaultdict(list)
        self._filename_to_paths: dict[str, list[str]] = defaultdict(list)
        self._seen_canonical_paths: set[str] = set()
        self._path_duplicates: list[str] = []

    def register(self, file_path: Path | str, sha256_hash: str) -> None:
        """
        Register a file and its content hash into the detector registry.

        Args:
            file_path: File path on disk.
            sha256_hash: 64-character hex SHA-256 digest.
        """
        path = Path(file_path).resolve()
        canonical_str = path.as_posix()
        filename = path.name

        # Check path duplication
        if canonical_str in self._seen_canonical_paths:
            self._path_duplicates.append(canonical_str)
        else:
            self._seen_canonical_paths.add(canonical_str)

        # Track hash
        clean_hash = sha256_hash.strip().lower()
        if clean_hash:
            self._hash_to_paths[clean_hash].append(canonical_str)

        # Track filename
        self._filename_to_paths[filename].append(canonical_str)

    def analyze(self) -> DuplicateReport:
        """
        Generate a consolidated duplicate report.

        Returns:
            DuplicateReport instance.
        """
        # Exact content duplicates (groups with > 1 file)
        exact_dupes = {
            h: paths for h, paths in self._hash_to_paths.items() if len(paths) > 1
        }

        # Filename clashes (same name across different folders)
        name_dupes = {
            name: paths
            for name, paths in self._filename_to_paths.items()
            if len(paths) > 1
        }

        # Total redundant file instances (each group has 1 original and (N - 1) duplicates)
        total_dupe_instances = sum(len(paths) - 1 for paths in exact_dupes.values())

        return DuplicateReport(
            hash_duplicates=exact_dupes,
            filename_duplicates=name_dupes,
            path_duplicates=self._path_duplicates.copy(),
            total_duplicate_instances=total_dupe_instances,
            total_unique_hashes=len(self._hash_to_paths),
        )

    def is_duplicate_hash(self, sha256_hash: str) -> bool:
        """Return True if more than one registered file shares this hash."""
        clean = sha256_hash.strip().lower()
        return len(self._hash_to_paths.get(clean, [])) > 1

    def get_canonical_for_hash(self, sha256_hash: str) -> Optional[str]:
        """Return the primary (first-registered) path for a given hash."""
        clean = sha256_hash.strip().lower()
        paths = self._hash_to_paths.get(clean, [])
        return paths[0] if paths else None

    def reset(self) -> None:
        """Clear all registered files from memory."""
        self._hash_to_paths.clear()
        self._filename_to_paths.clear()
        self._seen_canonical_paths.clear()
        self._path_duplicates.clear()
