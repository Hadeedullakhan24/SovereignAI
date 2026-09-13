"""Persistent SQLite Query Cache.

Caches complete RetrievalResult objects keyed by SHA256(query + retrieval_config)
with WAL journal mode, TTL expiration, and cache invalidation APIs.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import logging
from pathlib import Path
import sqlite3
import threading
import time
from typing import Generator, Optional

from rag_engine.retrieval.base_retriever import RetrievalResult
from rag_engine.retrieval.retrieval_exceptions import QueryCacheError
from rag_engine.config.runtime_paths import runtime_file

logger = logging.getLogger(__name__)


class QueryCache:
    """Thread-safe persistent SQLite query cache with WAL mode."""

    def __init__(
        self,
        db_path: str | Path | None = None,
        default_ttl_seconds: int = 86400,  # 24 hours
        enabled: bool = True,
    ) -> None:
        self.db_path = Path(db_path) if db_path is not None else runtime_file("cache", "retrieval_query_cache.db")
        self.default_ttl = default_ttl_seconds
        self.enabled = enabled
        self._lock = threading.RLock()

        if self.enabled:
            self._init_db()

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager guaranteeing connection closure on Windows."""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Initialize SQLite schema and WAL mode."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            try:
                with self._connect() as conn:
                    conn.execute("PRAGMA journal_mode=WAL;")
                    conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS query_cache (
                            cache_key TEXT PRIMARY KEY,
                            query TEXT NOT NULL,
                            strategy TEXT NOT NULL,
                            payload TEXT NOT NULL,
                            created_at REAL NOT NULL,
                            expires_at REAL NOT NULL,
                            hit_count INTEGER DEFAULT 0
                        );
                        """
                    )
                    conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_cache_expires ON query_cache (expires_at);"
                    )
                    conn.commit()
            except Exception as e:
                raise QueryCacheError(f"Failed to initialize SQLite query cache: {e}") from e

    def get(self, cache_key: str) -> Optional[RetrievalResult]:
        """Retrieve cached result if valid and not expired."""
        if not self.enabled:
            return None

        now = time.time()
        with self._lock:
            try:
                with self._connect() as conn:
                    cursor = conn.execute(
                        """
                        SELECT payload, expires_at FROM query_cache
                        WHERE cache_key = ?;
                        """,
                        (cache_key,),
                    )
                    row = cursor.fetchone()
                    if not row:
                        return None

                    payload_str, expires_at = row
                    if expires_at < now:
                        # Expired: purge entry
                        conn.execute("DELETE FROM query_cache WHERE cache_key = ?;", (cache_key,))
                        conn.commit()
                        return None

                    # Increment hit count
                    conn.execute(
                        "UPDATE query_cache SET hit_count = hit_count + 1 WHERE cache_key = ?;",
                        (cache_key,),
                    )
                    conn.commit()

                    data = json.loads(payload_str)
                    result = RetrievalResult.model_validate(data)
                    return result
            except Exception as e:
                logger.warning(f"Error reading query cache: {e}")
                return None

    def set(
        self,
        cache_key: str,
        result: RetrievalResult,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """Store RetrievalResult in SQLite cache."""
        if not self.enabled:
            return

        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        now = time.time()
        expires_at = now + ttl
        payload_str = result.model_dump_json()

        with self._lock:
            try:
                with self._connect() as conn:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO query_cache
                        (cache_key, query, strategy, payload, created_at, expires_at, hit_count)
                        VALUES (?, ?, ?, ?, ?, ?, 0);
                        """,
                        (cache_key, result.query, result.strategy_name, payload_str, now, expires_at),
                    )
                    conn.commit()
            except Exception as e:
                logger.warning(f"Failed to write query cache: {e}")

    def invalidate(self, cache_key: str) -> bool:
        """Invalidate a specific cache entry."""
        if not self.enabled:
            return False

        with self._lock:
            try:
                with self._connect() as conn:
                    res = conn.execute("DELETE FROM query_cache WHERE cache_key = ?;", (cache_key,))
                    conn.commit()
                    return res.rowcount > 0
            except Exception as e:
                logger.warning(f"Error invalidating cache key {cache_key}: {e}")
                return False

    def clear(self) -> int:
        """Purge all entries from the query cache."""
        if not self.enabled:
            return 0

        with self._lock:
            try:
                with self._connect() as conn:
                    res = conn.execute("DELETE FROM query_cache;")
                    conn.commit()
                    return res.rowcount
            except Exception as e:
                logger.warning(f"Error clearing cache: {e}")
                return 0

    def purge_expired(self) -> int:
        """Purge all expired cache entries."""
        if not self.enabled:
            return 0

        now = time.time()
        with self._lock:
            try:
                with self._connect() as conn:
                    res = conn.execute("DELETE FROM query_cache WHERE expires_at < ?;", (now,))
                    conn.commit()
                    return res.rowcount
            except Exception as e:
                logger.warning(f"Error purging expired cache: {e}")
                return 0

    def count(self) -> int:
        """Return total active entries in cache."""
        if not self.enabled:
            return 0

        with self._lock:
            try:
                with self._connect() as conn:
                    cursor = conn.execute("SELECT COUNT(*) FROM query_cache;")
                    row = cursor.fetchone()
                    return row[0] if row else 0
            except Exception as e:
                logger.warning(f"Error counting cache: {e}")
                return 0
