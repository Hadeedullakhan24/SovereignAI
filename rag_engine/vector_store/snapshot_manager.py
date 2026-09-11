"""Snapshot Manager.

Creates point-in-time tar.gz backup archives with cryptographic SHA-256 digests,
verifies archive integrity, and restores collections for disaster recovery.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import logging
from pathlib import Path
import shutil
import tarfile
from typing import Optional

from rag_engine.interfaces.base_vector_store import BaseVectorStore
from rag_engine.vector_store.exceptions import CollectionNotFoundError, SnapshotError

logger = logging.getLogger(__name__)


class SnapshotManager:
    """Manages creation, verification, and restoration of collection snapshots."""

    def __init__(
        self,
        store: BaseVectorStore,
        backup_dir: Path = Path("vector_db/backups"),
    ) -> None:
        self.store = store
        self.backup_dir = backup_dir
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def create_snapshot(
        self,
        collection_name: str,
        snapshot_name: Optional[str] = None,
    ) -> Path:
        """Create a point-in-time snapshot archive with SHA-256 integrity digest."""
        if not self.store.collection_exists(collection_name):
            raise CollectionNotFoundError(f"Collection '{collection_name}' does not exist.")

        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        name = snapshot_name or f"{collection_name}_{timestamp_str}"
        archive_path = self.backup_dir / f"{name}.tar.gz"

        try:
            self.store.create_snapshot(collection_name, archive_path)

            # Compute SHA-256 checksum over generated archive
            hasher = hashlib.sha256()
            with open(archive_path, "rb") as f:
                while chunk := f.read(65536):
                    hasher.update(chunk)
            digest = hasher.hexdigest()

            # Write checksum file
            checksum_path = archive_path.with_suffix(".tar.gz.sha256")
            with open(checksum_path, "w", encoding="utf-8") as f:
                f.write(f"{digest}  {archive_path.name}\n")

            logger.info("Created snapshot '%s' (SHA-256: %s)", archive_path.name, digest[:12])
            return archive_path
        except Exception as e:
            raise SnapshotError(f"Failed to create snapshot for '{collection_name}': {e}") from e

    def verify_snapshot(self, archive_path: Path) -> bool:
        """Verify the SHA-256 integrity of a snapshot archive against its digest file."""
        if not archive_path.exists():
            raise SnapshotError(f"Snapshot archive does not exist: {archive_path}")

        checksum_path = archive_path.with_suffix(".tar.gz.sha256")
        if not checksum_path.exists():
            raise SnapshotError(f"Missing checksum file for snapshot: {checksum_path}")

        with open(checksum_path, "r", encoding="utf-8") as f:
            line = f.readline().strip()
            expected_digest = line.split()[0]

        hasher = hashlib.sha256()
        with open(archive_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        actual_digest = hasher.hexdigest()

        if actual_digest != expected_digest:
            raise SnapshotError(
                f"Snapshot corruption detected! Expected {expected_digest}, computed {actual_digest}"
            )
        return True

    def restore_snapshot(
        self,
        collection_name: str,
        archive_path: Path,
        verify_first: bool = True,
    ) -> bool:
        """Restore collection from verified snapshot archive."""
        if verify_first:
            self.verify_snapshot(archive_path)

        try:
            return self.store.restore_snapshot(collection_name, archive_path)
        except Exception as e:
            raise SnapshotError(f"Failed to restore collection '{collection_name}': {e}") from e
