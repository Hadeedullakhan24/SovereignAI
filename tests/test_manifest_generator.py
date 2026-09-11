"""Unit tests for dataset_engine.manifest_generator."""

import json
from pathlib import Path
import pytest

from dataset_engine.manifest_generator import ManifestGenerator
from rag_engine.schemas.manifest import ValidationStatus


def test_manifest_generation(tmp_path):
    generator = ManifestGenerator(category_to_doctype_map={"manuals": "technical_manual"})

    entry = generator.build_document_entry(
        file_path=tmp_path / "manuals" / "pump.pdf",
        relative_path="manuals/pump.pdf",
        category="manuals",
        subcategory="",
        extension=".pdf",
        size_bytes=1024,
        sha256_hash="abcdef1234567890" * 4,
        status=ValidationStatus.VALID,
        last_modified="2026-09-06T12:00:00Z",
    )

    assert entry["category"] == "manuals"
    assert entry["document_type"] == "technical_manual"
    assert entry["processing_state"] == "INDEXED"
    assert entry["status"] == "VALID"
    assert entry["uuid"] is not None

    manifest_file = tmp_path / "manifest.json"
    generator.generate_manifest([entry], manifest_file)

    assert manifest_file.exists()
    data = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert data["total_documents"] == 1
    assert data["documents"][0]["uuid"] == entry["uuid"]
