"""Transaction Manager.

Provides Write-Ahead Logging (WAL) journaling and two-tier transaction management
to guarantee atomic ingestion and clean crash recovery.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import threading
import uuid

from typing import Any, Optional

from rag_engine.interfaces.base_vector_store import BaseVectorStore
from rag_engine.vector_store.exceptions import TransactionError

logger = logging.getLogger(__name__)


class TransactionRecord:
    """A single WAL transaction log entry."""

    def __init__(
        self,
        tx_id: str,
        action: str,  # START, COMMIT, ROLLBACK
        collection_name: str,
        chunk_ids: list[str],
        timestamp: Optional[str] = None,
    ) -> None:
        self.tx_id = tx_id
        self.action = action
        self.collection_name = collection_name
        self.chunk_ids = chunk_ids
        self.timestamp = timestamp or datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "tx_id": self.tx_id,
            "action": self.action,
            "collection_name": self.collection_name,
            "chunk_ids": self.chunk_ids,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TransactionRecord:
        return cls(
            tx_id=data["tx_id"],
            action=data["action"],
            collection_name=data["collection_name"],
            chunk_ids=data.get("chunk_ids", []),
            timestamp=data.get("timestamp"),
        )


class TransactionManager:
    """Manages transaction lifecycle and WAL journal persistence."""

    def __init__(
        self,
        store: BaseVectorStore,
        journal_dir: Path = Path("vector_db/journal"),
        enabled: bool = True,
    ) -> None:
        self.store = store
        self.journal_dir = journal_dir
        self.journal_file = journal_dir / "wal.jsonl"
        self.enabled = enabled
        self._lock = threading.RLock()
        self._active_txs: dict[str, TransactionRecord] = {}
        self._ensure_journal()

    def _ensure_journal(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            self.journal_dir.mkdir(parents=True, exist_ok=True)
            if not self.journal_file.exists():
                self.journal_file.touch()

    def _append_log(self, record: TransactionRecord) -> None:
        if not self.enabled:
            return
        with self._lock:
            with open(self.journal_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record.to_dict()) + "\n")

    def begin_transaction(self, collection_name: str, chunk_ids: Optional[list[str]] = None) -> str:
        """Begin a new write transaction."""
        tx_id = f"tx_{uuid.uuid4().hex[:12]}"
        record = TransactionRecord(
            tx_id=tx_id,
            action="START",
            collection_name=collection_name,
            chunk_ids=chunk_ids or [],
        )
        with self._lock:
            self._active_txs[tx_id] = record
            self._append_log(record)
        return tx_id

    def commit_transaction(self, tx_id: str) -> bool:
        """Commit an active transaction."""
        with self._lock:
            if tx_id not in self._active_txs:
                raise TransactionError(f"Transaction '{tx_id}' is not active or already finished.")
            start_rec = self._active_txs.pop(tx_id)
            commit_rec = TransactionRecord(
                tx_id=tx_id,
                action="COMMIT",
                collection_name=start_rec.collection_name,
                chunk_ids=start_rec.chunk_ids,
            )
            self._append_log(commit_rec)
        return True

    def rollback_transaction(self, tx_id: str) -> bool:
        """Roll back an uncommitted transaction and purge any partially written chunks."""
        with self._lock:
            if tx_id not in self._active_txs:
                return False
            start_rec = self._active_txs.pop(tx_id)

            # Purge written points if any
            if start_rec.chunk_ids and self.store.collection_exists(start_rec.collection_name):
                try:
                    self.store.delete_chunks(start_rec.collection_name, start_rec.chunk_ids)
                except Exception as e:
                    logger.error("Failed to purge chunks during rollback of '%s': %s", tx_id, e)

            rollback_rec = TransactionRecord(
                tx_id=tx_id,
                action="ROLLBACK",
                collection_name=start_rec.collection_name,
                chunk_ids=start_rec.chunk_ids,
            )
            self._append_log(rollback_rec)
        return True

    def recover_on_startup(self) -> int:
        """Inspect WAL journal on startup and clean up any uncommitted transactions."""
        with self._lock:
            if not self.journal_file.exists():
                return 0

            pending_starts: dict[str, TransactionRecord] = {}
            with open(self.journal_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = TransactionRecord.from_dict(json.loads(line))
                        if rec.action == "START":
                            pending_starts[rec.tx_id] = rec
                        elif rec.action in ("COMMIT", "ROLLBACK"):
                            pending_starts.pop(rec.tx_id, None)
                    except Exception:
                        continue

            uncommitted_count = len(pending_starts)
            for tx_id, rec in pending_starts.items():
                logger.warning("Rolling back uncommitted WAL transaction '%s' on startup.", tx_id)
                self._active_txs[tx_id] = rec
                self.rollback_transaction(tx_id)

            return uncommitted_count
