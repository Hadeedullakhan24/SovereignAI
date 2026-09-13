"""Deduplicate Qdrant collections by content-hash (chunk_hash payload field).

For each collection, scrolls all points and keeps only the first occurrence of
each unique chunk_hash, deleting all subsequent duplicates.

Usage:
    python scripts/deduplicate_qdrant.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("qdrant_dedup")

try:
    from qdrant_client import QdrantClient, models
except ImportError:
    logger.error("qdrant-client not installed. Run: pip install qdrant-client")
    sys.exit(1)


def deduplicate_collection(client: QdrantClient, collection_name: str) -> dict:
    """Remove duplicate vectors from a collection based on chunk_hash.

    A duplicate is defined as any point whose chunk_hash matches a point that
    was already encountered during a full collection scroll.  The FIRST
    occurrence (by scroll order) is kept; all later duplicates are deleted.

    Returns a summary dict with before/after counts and number deleted.
    """
    info = client.get_collection(collection_name)
    total_before = info.points_count or 0
    logger.info("[%s] Starting deduplication - %d points", collection_name, total_before)

    seen_hashes: dict[str, str] = {}   # chunk_hash -> first point id (str)
    duplicate_ids: list[str] = []

    offset = None
    total_scrolled = 0

    while True:
        records, next_offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=None,
            limit=500,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        if not records:
            break

        total_scrolled += len(records)

        for pt in records:
            payload = pt.payload or {}
            # Primary dedup key: chunk_hash (SHA-256 of content)
            chunk_hash = payload.get("chunk_hash", "")
            pt_id = str(pt.id)

            if not chunk_hash:
                # No hash - use chunk_id as fallback dedup key
                chunk_hash = payload.get("chunk_id", pt_id)

            if chunk_hash in seen_hashes:
                duplicate_ids.append(pt_id)
            else:
                seen_hashes[chunk_hash] = pt_id

        if next_offset is None:
            break
        offset = next_offset

    logger.info(
        "[%s] Scrolled %d points, found %d duplicates to remove.",
        collection_name, total_scrolled, len(duplicate_ids),
    )

    # Delete duplicates in batches of 500
    deleted = 0
    batch_size = 500
    for i in range(0, len(duplicate_ids), batch_size):
        batch = duplicate_ids[i : i + batch_size]
        client.delete(
            collection_name=collection_name,
            points_selector=models.PointIdsList(points=batch),
            wait=True,
        )
        deleted += len(batch)
        logger.info("[%s] Deleted batch %d/%d (%d points so far)", collection_name,
                    i // batch_size + 1, (len(duplicate_ids) + batch_size - 1) // batch_size,
                    deleted)

    info_after = client.get_collection(collection_name)
    total_after = info_after.points_count or 0
    logger.info(
        "[%s] Deduplication complete: %d -> %d points (%d removed).",
        collection_name, total_before, total_after, total_before - total_after,
    )
    return {
        "collection": collection_name,
        "points_before": total_before,
        "points_after": total_after,
        "duplicates_removed": total_before - total_after,
    }


def main() -> None:
    storage_path = Path("vector_db/qdrant")
    if not storage_path.exists():
        logger.error("Qdrant storage path does not exist: %s", storage_path)
        sys.exit(1)

    client = QdrantClient(path=str(storage_path))

    collections = [c.name for c in client.get_collections().collections]
    logger.info("Found collections: %s", collections)

    results = []
    for col in collections:
        if col.startswith("_"):
            logger.info("Skipping canary/internal collection: %s", col)
            continue
        result = deduplicate_collection(client, col)
        results.append(result)

    print("\n" + "=" * 70)
    print("  DEDUPLICATION REPORT")
    print("=" * 70)
    for r in results:
        print(f"  Collection : {r['collection']}")
        print(f"  Before     : {r['points_before']:,} points")
        print(f"  After      : {r['points_after']:,} points")
        print(f"  Removed    : {r['duplicates_removed']:,} duplicates")
        print()

    client.close()


if __name__ == "__main__":
    main()
