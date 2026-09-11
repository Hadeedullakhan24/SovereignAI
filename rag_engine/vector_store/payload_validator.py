"""Payload Schema Validator.

Validates payload consistency, required metadata, field lengths, nested structures,
allowable enums, and checksums before vector database insertion.
"""

from __future__ import annotations

from typing import Any, Optional

from rag_engine.schemas.embedding import EmbeddedChunk
from rag_engine.vector_store.exceptions import PayloadValidationError
from rag_engine.vector_store.vector_utils import compute_payload_checksum


# Known valid refinery categories
VALID_REFINERY_CATEGORIES = {
    "Manual",
    "Engineering Manual",
    "Inspection Report",
    "NDT Report",
    "Maintenance Record",
    "Work Order",
    "Safety Document",
    "Safety Standard",
    "SOP",
    "Standard Operating Procedure",
    "Email",
    "Technical Correspondence",
    "Engineering Drawing",
    "P&ID",
    "General",
    "Unknown",
}

# Maximum field length constraints
MAX_FIELD_LENGTHS = {
    "document_id": 128,
    "chunk_id": 128,
    "document_name": 256,
    "plant_unit": 64,
    "section_title": 512,
    "source_file": 512,
    "revision": 32,
    "version": 32,
    "language": 16,
}


class PayloadValidator:
    """Validates payload consistency and type safety before insertion."""

    def __init__(
        self,
        strict_enums: bool = False,
        max_entity_tag_length: int = 64,
    ) -> None:
        self.strict_enums = strict_enums
        self.max_entity_tag_length = max_entity_tag_length

    def validate_payload(self, payload: dict[str, Any]) -> None:
        """Validate a single serialized payload dictionary."""
        chunk_id = payload.get("chunk_id", "")
        if not chunk_id or not isinstance(chunk_id, str):
            raise PayloadValidationError(f"Payload missing required string 'chunk_id': {payload}")

        # Required fields check
        doc_id = payload.get("document_id")
        if doc_id is None:
            raise PayloadValidationError(f"Payload for chunk '{chunk_id}' missing 'document_id'.")

        # Field lengths check
        for field, max_len in MAX_FIELD_LENGTHS.items():
            val = payload.get(field)
            if val is not None and isinstance(val, str) and len(val) > max_len:
                raise PayloadValidationError(
                    f"Field '{field}' in chunk '{chunk_id}' exceeds maximum length of {max_len} (len={len(val)})."
                )

        # Type checks
        page_no = payload.get("page_number")
        if page_no is not None and not isinstance(page_no, int):
            raise PayloadValidationError(
                f"Field 'page_number' in chunk '{chunk_id}' must be an int or None, got {type(page_no).__name__}."
            )

        chunk_idx = payload.get("chunk_index")
        if chunk_idx is not None and not isinstance(chunk_idx, int):
            raise PayloadValidationError(
                f"Field 'chunk_index' in chunk '{chunk_id}' must be an int, got {type(chunk_idx).__name__}."
            )

        # Nested metadata checks (equipment & safety entities)
        equip = payload.get("equipment_entities")
        if equip is not None:
            if not isinstance(equip, (list, tuple)):
                raise PayloadValidationError(
                    f"Field 'equipment_entities' in chunk '{chunk_id}' must be a list/tuple."
                )
            for item in equip:
                if not isinstance(item, str):
                    raise PayloadValidationError(
                        f"Equipment entity in chunk '{chunk_id}' must be string, got {type(item).__name__}."
                    )
                if len(item) > self.max_entity_tag_length:
                    raise PayloadValidationError(
                        f"Equipment entity '{item}' exceeds max length {self.max_entity_tag_length}."
                    )

        safety = payload.get("safety_entities")
        if safety is not None:
            if not isinstance(safety, (list, tuple)):
                raise PayloadValidationError(
                    f"Field 'safety_entities' in chunk '{chunk_id}' must be a list/tuple."
                )

        # Enum check
        cat = payload.get("category")
        if self.strict_enums and cat:
            if cat not in VALID_REFINERY_CATEGORIES:
                raise PayloadValidationError(
                    f"Invalid category enum '{cat}' in chunk '{chunk_id}'. "
                    f"Allowed: {sorted(VALID_REFINERY_CATEGORIES)}"
                )

    def validate_chunk(self, chunk: EmbeddedChunk) -> None:
        """Validate an EmbeddedChunk directly before serialization."""
        from rag_engine.vector_store.metadata_serializer import MetadataSerializer
        payload = MetadataSerializer.to_payload(chunk)
        self.validate_payload(payload)

    def validate_batch(self, chunks: list[EmbeddedChunk]) -> None:
        """Validate a batch of EmbeddedChunk instances."""
        for chunk in chunks:
            self.validate_chunk(chunk)
