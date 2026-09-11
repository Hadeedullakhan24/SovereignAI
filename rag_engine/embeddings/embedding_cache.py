"""High-performance persistent SQLite vector cache with thread-safe access.

Avoids duplicate embedding computation by caching dense vectors under:
    SHA-256(model_name + ":" + chunk_hash)
Uses compact binary IEEE 754 float32 packing for fast retrieval and small storage footprint.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sqlite3
import struct
import threading
from typing import Any, Optional

from rag_engine.schemas.embedding import compute_vector_checksum


class EmbeddingCache:
    """Thread-safe, persistent SQLite cache for vector embeddings."""

    def __init__(
        self,
        db_path: str | Path = "cache/embeddings/embedding_cache.db",
        enable_wal: bool = True,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

        self._init_db(enable_wal)

    def _init_db(self, enable_wal: bool) -> None:
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=60.0) as conn:
                cursor = conn.cursor()
                if enable_wal:
                    cursor.execute("PRAGMA journal_mode=WAL;")
                cursor.execute("PRAGMA synchronous=NORMAL;")
                cursor.execute("PRAGMA busy_timeout=60000;")
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS embedding_cache (
                        cache_key TEXT PRIMARY KEY,
                        model_name TEXT NOT NULL,
                        chunk_hash TEXT NOT NULL,
                        vector_blob BLOB NOT NULL,
                        dimension INTEGER NOT NULL,
                        checksum TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    );
                    """
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_cache_model_chunk ON embedding_cache(model_name, chunk_hash);"
                )
                conn.commit()

    @staticmethod
    def compute_cache_key(model_name: str, chunk_hash: str) -> str:
        """Compute deterministic SHA-256 cache key from model name and chunk hash."""
        seed = f"{model_name}:{chunk_hash}"
        return hashlib.sha256(seed.encode("utf-8")).hexdigest()

    def get(self, model_name: str, chunk_hash: str) -> Optional[list[float]]:
        """Retrieve a cached vector if available, returning None on miss."""
        key = self.compute_cache_key(model_name, chunk_hash)
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=60.0) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT vector_blob, dimension FROM embedding_cache WHERE cache_key = ?;",
                    (key,),
                )
                row = cursor.fetchone()
                if row:
                    blob, dim = row
                    vector = list(struct.unpack(f"<{dim}f", blob))
                    self._hits += 1
                    return vector
                self._misses += 1
                return None

    def put(
        self,
        model_name: str,
        chunk_hash: str,
        embedding: list[float],
        dimension: Optional[int] = None,
    ) -> None:
        """Store an embedding vector into persistent cache."""
        dim = dimension or len(embedding)
        key = self.compute_cache_key(model_name, chunk_hash)
        blob = struct.pack(f"<{dim}f", *embedding)
        checksum = compute_vector_checksum(embedding)
        now_iso = datetime.now(timezone.utc).isoformat()

        with self._lock:
            with sqlite3.connect(self.db_path, timeout=60.0) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO embedding_cache
                    (cache_key, model_name, chunk_hash, vector_blob, dimension, checksum, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?);
                    """,
                    (key, model_name, chunk_hash, blob, dim, checksum, now_iso),
                )
                conn.commit()

    def get_batch(
        self, model_name: str, chunk_hashes: list[str]
    ) -> dict[str, list[float]]:
        """Batch retrieve cached vectors.

        Returns:
            Dict mapping chunk_hash -> vector for all cached hits.
        """
        if not chunk_hashes:
            return {}

        results: dict[str, list[float]] = {}
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=60.0) as conn:
                cursor = conn.cursor()
                # Query in batches of 500 for SQLite parameter limits
                chunk_size = 500
                for i in range(0, len(chunk_hashes), chunk_size):
                    batch = chunk_hashes[i : i + chunk_size]
                    placeholders = ",".join("?" for _ in batch)
                    cursor.execute(
                        f"""
                        SELECT chunk_hash, vector_blob, dimension
                        FROM embedding_cache
                        WHERE model_name = ? AND chunk_hash IN ({placeholders});
                        """,
                        [model_name] + batch,
                    )
                    rows = cursor.fetchall()
                    for ch_hash, blob, dim in rows:
                        results[ch_hash] = list(struct.unpack(f"<{dim}f", blob))

        hits = len(results)
        misses = len(chunk_hashes) - hits
        with self._lock:
            self._hits += hits
            self._misses += misses

        return results

    def put_batch(
        self,
        model_name: str,
        hash_to_vec: dict[str, list[float]],
        dimension: Optional[int] = None,
    ) -> None:
        """Batch store embeddings into persistent cache."""
        if not hash_to_vec:
            return

        now_iso = datetime.now(timezone.utc).isoformat()
        records = []
        for ch_hash, vec in hash_to_vec.items():
            dim = dimension or len(vec)
            key = self.compute_cache_key(model_name, ch_hash)
            blob = struct.pack(f"<{dim}f", *vec)
            checksum = compute_vector_checksum(vec)
            records.append((key, model_name, ch_hash, blob, dim, checksum, now_iso))

        with self._lock:
            with sqlite3.connect(self.db_path, timeout=60.0) as conn:
                cursor = conn.cursor()
                cursor.executemany(
                    """
                    INSERT OR REPLACE INTO embedding_cache
                    (cache_key, model_name, chunk_hash, vector_blob, dimension, checksum, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?);
                    """,
                    records,
                )
                conn.commit()

    def get_stats(self) -> dict[str, Any]:
        """Return cache telemetry and hit/miss ratios."""
        with self._lock:
            total_requests = self._hits + self._misses
            reuse_pct = (
                (self._hits / total_requests * 100.0) if total_requests > 0 else 0.0
            )

            total_entries = 0
            try:
                with sqlite3.connect(self.db_path, timeout=60.0) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM embedding_cache;")
                    total_entries = cursor.fetchone()[0]
            except Exception:
                pass

            return {
                "cache_hits": self._hits,
                "cache_misses": self._misses,
                "total_requests": total_requests,
                "cache_reuse_percentage": round(reuse_pct, 2),
                "total_entries": total_entries,
                "db_path": str(self.db_path),
            }

    def reset_stats(self) -> None:
        """Reset internal hit and miss counters."""
        with self._lock:
            self._hits = 0
            self._misses = 0

    def clear(self) -> None:
        """Purge all entries from cache database."""
        with self._lock:
            with sqlite3.connect(self.db_path, timeout=60.0) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM embedding_cache;")
                conn.commit()
            self._hits = 0
            self._misses = 0

