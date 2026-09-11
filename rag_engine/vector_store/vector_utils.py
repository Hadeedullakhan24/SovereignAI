"""Vector Utilities.

Helper functions for deterministic UUIDv5 generation, SHA-256 payload checksums,
and vector numerical transformations.
"""

from __future__ import annotations

import hashlib
import json
import struct
from typing import Any
import uuid


# Namespace for deterministic Qdrant point IDs
QDRANT_CHUNK_NAMESPACE = uuid.UUID("3c87e382-7e04-4f24-913a-a10c7bf7ea88")


def chunk_id_to_uuid(chunk_id: str) -> str:
    """Deterministically convert a string chunk_id into a valid RFC 4122 UUIDv5 string.

    Qdrant requires point IDs to be unsigned 64-bit integers or UUID strings.
    This guarantees 1:1 deterministic mapping without randomness.
    """
    return str(uuid.uuid5(QDRANT_CHUNK_NAMESPACE, chunk_id))


def compute_payload_checksum(payload: dict[str, Any]) -> str:
    """Compute a deterministic SHA-256 digest over a JSON-serializable payload dictionary."""
    serialized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def compute_vector_checksum(vector: list[float]) -> str:
    """Compute a deterministic SHA-256 checksum over float32 binary representation of vector."""
    packed = struct.pack(f"<{len(vector)}f", *vector)
    return hashlib.sha256(packed).hexdigest()
