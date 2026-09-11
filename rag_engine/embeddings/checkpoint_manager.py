"""Checkpoint and Resume Manager for interrupted embedding workflows.

Persists ingestion progress so long-running embedding jobs can recover seamlessly
from interruptions without re-processing already embedded chunks.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import threading
from typing import Optional

from rag_engine.schemas.chunk import Chunk


class CheckpointManager:
    """Manages persistent checkpointing for indexing and embedding jobs."""

    def __init__(
        self,
        db_path: str | Path = "cache/embeddings/checkpoints.db",
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()

    def _init_db(self) -> None:
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=60.0) as conn:
                cursor = conn.cursor()
                cursor.execute("PRAGMA journal_mode=WAL;")
                cursor.execute("PRAGMA synchronous=NORMAL;")
                cursor.execute("PRAGMA busy_timeout=60000;")
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS job_checkpoints (
                        job_id TEXT NOT NULL,
                        chunk_id TEXT NOT NULL,
                        chunk_hash TEXT NOT NULL,
                        completed_at TEXT NOT NULL,
                        PRIMARY KEY (job_id, chunk_id)
                    );
                    """
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_checkpoint_job ON job_checkpoints(job_id);"
                )
                conn.commit()

    def record_completed(
        self,
        job_id: str,
        chunk_id: str,
        chunk_hash: str = "",
    ) -> None:
        """Mark a single chunk as successfully completed for a given job."""
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=60.0) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO job_checkpoints (job_id, chunk_id, chunk_hash, completed_at)
                    VALUES (?, ?, ?, ?);
                    """,
                    (job_id, chunk_id, chunk_hash, now_iso),
                )
                conn.commit()

    def record_completed_batch(
        self,
        job_id: str,
        chunk_records: list[tuple[str, str]],
    ) -> None:
        """Batch mark chunks as completed.

        Args:
            job_id: Identifier of the indexing run/job.
            chunk_records: List of tuples (chunk_id, chunk_hash).
        """
        if not chunk_records:
            return

        now_iso = datetime.now(timezone.utc).isoformat()
        rows = [(job_id, cid, chash, now_iso) for cid, chash in chunk_records]
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=60.0) as conn:
                cursor = conn.cursor()
                cursor.executemany(
                    """
                    INSERT OR REPLACE INTO job_checkpoints (job_id, chunk_id, chunk_hash, completed_at)
                    VALUES (?, ?, ?, ?);
                    """,
                    rows,
                )
                conn.commit()

    def get_completed_chunk_ids(self, job_id: str) -> set[str]:
        """Return the set of chunk IDs already completed for the job."""
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=60.0) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT chunk_id FROM job_checkpoints WHERE job_id = ?;",
                    (job_id,),
                )
                return {row[0] for row in cursor.fetchall()}

    def filter_unprocessed_chunks(
        self,
        job_id: str,
        chunks: list[Chunk],
    ) -> list[Chunk]:
        """Filter out chunks that have already been recorded as completed for this job."""
        completed_ids = self.get_completed_chunk_ids(job_id)
        if not completed_ids:
            return list(chunks)
        return [ch for ch in chunks if ch.chunk_id not in completed_ids]

    def clear_job(self, job_id: str) -> None:
        """Reset checkpoint records for a specific job."""
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=60.0) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM job_checkpoints WHERE job_id = ?;",
                    (job_id,),
                )
                conn.commit()

