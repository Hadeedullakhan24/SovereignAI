"""
Manifest Generator — Deterministic manifest.json Creation.

Sovereign On-Premise Agentic AI Workbench (SIH26117 / MRPL)
Produces structured, auditable manifest.json cataloging every discovered document
with UUIDs, hashes, refinery categories, and validation states.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Optional
import uuid

from rag_engine.schemas.manifest import ValidationStatus


class ManifestGenerator:
    """Builds and serializes manifest.json for validated datasets."""

    def __init__(self, category_to_doctype_map: Optional[dict[str, str]] = None) -> None:
        """
        Initialize ManifestGenerator.

        Args:
            category_to_doctype_map: Mapping from category names to standardized doc_types.
        """
        self.category_to_doctype = category_to_doctype_map or {}

    def build_document_entry(
        self,
        file_path: Path | str,
        relative_path: str,
        category: str,
        subcategory: str,
        extension: str,
        size_bytes: int,
        sha256_hash: str,
        status: ValidationStatus,
        last_modified: str,
        created_date: Optional[str] = None,
        is_duplicate: bool = False,
        duplicate_of: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Generate a standardized document record dictionary.
        """
        # Generate deterministic UUID based on normalized relative path
        doc_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, relative_path))
        doc_type = self.category_to_doctype.get(category, "general_document")

        created_str = (
            created_date
            if created_date
            else datetime.now(timezone.utc).isoformat()
        )

        processing_state = (
            "INDEXED"
            if status == ValidationStatus.VALID
            else "FAILED_VALIDATION"
        )

        return {
            "uuid": doc_uuid,
            "relative_path": relative_path,
            "absolute_path": Path(file_path).resolve().as_posix(),
            "category": category,
            "subcategory": subcategory,
            "document_type": doc_type,
            "extension": extension.lower(),
            "size_bytes": size_bytes,
            "hash": sha256_hash,
            "processing_state": processing_state,
            "created_date": created_str,
            "modified_date": last_modified,
            "status": status.value if hasattr(status, "value") else str(status),
            "is_duplicate": is_duplicate,
            "duplicate_of": duplicate_of,
            "error_message": error_message,
        }

    def generate_manifest(
        self,
        documents: list[dict[str, Any]],
        output_path: Path | str,
        dataset_root: str = "datasets",
    ) -> Path:
        """
        Assemble and atomically write manifest.json to disk.

        Args:
            documents: List of document records.
            output_path: Target file path for manifest.json.
            dataset_root: Root dataset identifier.

        Returns:
            Resolved Path to the created manifest.json.
        """
        out_file = Path(output_path).resolve()
        out_file.parent.mkdir(parents=True, exist_ok=True)

        manifest_payload = {
            "schema_version": "1.0.0",
            "system": "Sovereign On-Premise Agentic AI Workbench",
            "problem_statement": "SIH26117",
            "organization": "MRPL",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dataset_root": dataset_root,
            "total_documents": len(documents),
            "documents": documents,
        }

        # Write atomically using a temporary file
        temp_file = out_file.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as fh:
            json.dump(manifest_payload, fh, indent=2, ensure_ascii=False)

        temp_file.replace(out_file)
        return out_file
