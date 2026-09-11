"""Generation Cache — Deterministic SQLite WAL Response Caching.

Stores and reuses generated responses keyed by:
    SHA256(prompt_hash + retrieved_chunk_hashes + model_name + generation_params)
Eliminates redundant GPU/CPU inference on identical queries.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Dict, Generator, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CachedGeneration:
    """Cached generation record retrieved from persistent store."""

    cache_key: str
    prompt_hash: str
    response_text: str
    model_name: str
    citations: list[str]
    created_at: str
    metadata: Dict[str, Any]


class GenerationCache:
    """Persistent SQLite WAL cache for generated LLM responses."""

    def __init__(
        self,
        db_path: Path | str = "rag_engine/cache/generation_cache.db",
        default_ttl_seconds: int = 604800,  # 7 days
    ) -> None:
        self.db_path = Path(db_path)
        self.default_ttl = default_ttl_seconds
        self._lock = threading.Lock()
        self._init_db()

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager guaranteeing SQLite connection closure."""
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            check_same_thread=False,
        )
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Initialize SQLite database with WAL mode and schema."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL;")
            cursor.execute("PRAGMA synchronous=NORMAL;")
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS generation_cache (
                    cache_key TEXT PRIMARY KEY,
                    prompt_hash TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    response_text TEXT NOT NULL,
                    citations TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at REAL NOT NULL
                );
                """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_gen_cache_prompt ON generation_cache(prompt_hash);"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_gen_cache_expires ON generation_cache(expires_at);"
            )
            conn.commit()

    @staticmethod
    def compute_cache_key(
        prompt_hash: str,
        chunk_ids: Sequence[str],
        model_name: str,
        parameters: Dict[str, Any] | None = None,
    ) -> str:
        """Compute deterministic SHA-256 cache key."""
        hasher = hashlib.sha256()
        hasher.update(prompt_hash.encode("utf-8"))
        hasher.update(",".join(sorted(chunk_ids)).encode("utf-8"))
        hasher.update(model_name.lower().strip().encode("utf-8"))
        if parameters:
            param_str = json.dumps(parameters, sort_keys=True)
            hasher.update(param_str.encode("utf-8"))
        return hasher.hexdigest()

    def get(self, cache_key: str) -> Optional[CachedGeneration]:
        """Retrieve a valid, unexpired cached generation result."""
        now = time.time()
        with self._lock, self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT cache_key, prompt_hash, model_name, response_text, citations, metadata, created_at
                FROM generation_cache
                WHERE cache_key = ? AND expires_at > ?;
                """,
                (cache_key, now),
            )
            row = cursor.fetchone()
            if not row:
                return None

            return CachedGeneration(
                cache_key=row[0],
                prompt_hash=row[1],
                model_name=row[2],
                response_text=row[3],
                citations=json.loads(row[4]),
                metadata=json.loads(row[5]),
                created_at=str(row[6]),
            )

    def put(
        self,
        cache_key: str,
        prompt_hash: str,
        model_name: str,
        response_text: str,
        citations: Sequence[str],
        metadata: Dict[str, Any] | None = None,
        ttl_seconds: int | None = None,
    ) -> None:
        """Store generated response in cache."""
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        expires_at = time.time() + ttl
        cits_json = json.dumps(list(citations))
        meta_json = json.dumps(metadata or {})

        with self._lock, self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO generation_cache
                (cache_key, prompt_hash, model_name, response_text, citations, metadata, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                (cache_key, prompt_hash, model_name, response_text, cits_json, meta_json, expires_at),
            )
            conn.commit()

    def clear(self) -> None:
        """Wipe all records from generation cache."""
        with self._lock, self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM generation_cache;")
            conn.commit()
