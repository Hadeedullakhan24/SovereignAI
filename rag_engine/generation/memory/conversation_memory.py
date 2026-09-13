"""Conversation Memory — Structured Multi-Turn Dialog Tracking with SQLite Persistence.

Maintains structured, auditable conversation turns preserving session lineage,
retrieved chunk identifiers, citations, and metadata with SQLite WAL persistence
and sliding-window context trimming.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sqlite3
import threading
from typing import Any, Dict, List, Optional
from rag_engine.config.runtime_paths import runtime_file

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TurnRecord:
    """Structured record of a single conversation turn."""

    session_id: str
    turn_index: int
    user_query: str
    response: str
    retrieved_chunk_ids: List[str]
    citations: List[str]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConversationMemory:
    """Thread-safe, SQLite-backed conversation memory manager for multi-turn refinery QA."""

    def __init__(
        self,
        max_turns_per_session: int = 10,
        db_path: Path | str | None = None,
    ) -> None:
        self.max_turns = max_turns_per_session
        self.db_path = Path(db_path) if db_path else runtime_file("cache", "conversation_memory.db")
        self._lock = threading.RLock()
        self._local = threading.local()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get or initialize thread-local SQLite connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            if str(self.db_path) == ":memory:":
                if not hasattr(self, "_shared_memory_conn"):
                    self._shared_memory_conn = sqlite3.connect(":memory:", check_same_thread=False)
                self._local.conn = self._shared_memory_conn
            else:
                self.db_path.parent.mkdir(parents=True, exist_ok=True)
                conn = sqlite3.connect(str(self.db_path), timeout=30.0)
                conn.execute("PRAGMA journal_mode = WAL;")
                conn.execute("PRAGMA synchronous = NORMAL;")
                self._local.conn = conn
        return self._local.conn

    def _init_db(self) -> None:
        """Initialize SQLite schema for conversation sessions."""
        with self._lock:
            conn = self._get_connection()
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS conversation_turns (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        turn_index INTEGER NOT NULL,
                        user_query TEXT NOT NULL,
                        response TEXT NOT NULL,
                        retrieved_chunk_ids TEXT NOT NULL,
                        citations TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        metadata TEXT NOT NULL
                    );
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_conv_session ON conversation_turns(session_id, turn_index);"
                )

    def add_turn(
        self,
        session_id: str,
        user_query: str,
        response: str,
        retrieved_chunk_ids: List[str] | None = None,
        citations: List[str] | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> TurnRecord:
        """Add a completed turn to session history with sliding-window eviction."""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COALESCE(MAX(turn_index), 0) + 1 FROM conversation_turns WHERE session_id = ?",
                (session_id,),
            )
            turn_index = cursor.fetchone()[0]

            record = TurnRecord(
                session_id=session_id,
                turn_index=turn_index,
                user_query=user_query.strip(),
                response=response.strip(),
                retrieved_chunk_ids=list(retrieved_chunk_ids or []),
                citations=list(citations or []),
                metadata=dict(metadata or {}),
            )

            with conn:
                conn.execute(
                    """
                    INSERT INTO conversation_turns (
                        session_id, turn_index, user_query, response,
                        retrieved_chunk_ids, citations, timestamp, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        record.session_id,
                        record.turn_index,
                        record.user_query,
                        record.response,
                        json.dumps(record.retrieved_chunk_ids),
                        json.dumps(record.citations),
                        record.timestamp,
                        json.dumps(record.metadata),
                    ),
                )

                # Evict oldest turns if exceeding limit
                conn.execute(
                    """
                    DELETE FROM conversation_turns
                    WHERE session_id = ? AND turn_index NOT IN (
                        SELECT turn_index FROM conversation_turns
                        WHERE session_id = ?
                        ORDER BY turn_index DESC
                        LIMIT ?
                    );
                    """,
                    (session_id, session_id, self.max_turns),
                )

            return record

    def get_history(self, session_id: str, max_turns: int | None = None) -> List[TurnRecord]:
        """Retrieve recent conversation turn records for a session in chronological order."""
        with self._lock:
            limit = max_turns or self.max_turns
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT session_id, turn_index, user_query, response,
                       retrieved_chunk_ids, citations, timestamp, metadata
                FROM (
                    SELECT * FROM conversation_turns
                    WHERE session_id = ?
                    ORDER BY turn_index DESC
                    LIMIT ?
                )
                ORDER BY turn_index ASC;
                """,
                (session_id, limit),
            )
            rows = cursor.fetchall()
            records: list[TurnRecord] = []
            for r in rows:
                records.append(
                    TurnRecord(
                        session_id=r[0],
                        turn_index=r[1],
                        user_query=r[2],
                        response=r[3],
                        retrieved_chunk_ids=json.loads(r[4]),
                        citations=json.loads(r[5]),
                        timestamp=r[6],
                        metadata=json.loads(r[7]),
                    )
                )
            return records

    def get_history_text(self, session_id: str, max_turns: int = 3) -> str:
        """Format recent turns into prompt-ready context string."""
        turns = self.get_history(session_id, max_turns)
        if not turns:
            return ""

        formatted: list[str] = []
        for t in turns:
            formatted.append(f"User: {t.user_query}")
            formatted.append(f"Assistant: {t.response}\n")
        return "\n".join(formatted)

    def clear_session(self, session_id: str) -> None:
        """Clear memory for a specific session."""
        with self._lock:
            conn = self._get_connection()
            with conn:
                conn.execute("DELETE FROM conversation_turns WHERE session_id = ?", (session_id,))

    def clear_all(self) -> None:
        """Clear all conversation memory across all sessions."""
        with self._lock:
            conn = self._get_connection()
            with conn:
                conn.execute("DELETE FROM conversation_turns;")

    def close(self) -> None:
        """Close SQLite connection and release file locks."""
        with self._lock:
            if hasattr(self, "_shared_memory_conn") and self._shared_memory_conn is not None:
                try:
                    self._shared_memory_conn.close()
                except Exception:
                    pass
                self._shared_memory_conn = None
            if hasattr(self._local, "conn") and self._local.conn is not None:
                try:
                    self._local.conn.close()
                except Exception:
                    pass
                self._local.conn = None

