"""Unit tests for dataset_engine.validator."""

import zipfile
from pathlib import Path
import pytest

from dataset_engine.validator import DatasetValidator
from rag_engine.schemas.manifest import ValidationStatus


@pytest.fixture
def validator():
    return DatasetValidator(
        allowed_extensions={".pdf", ".docx", ".txt", ".json"},
        max_file_size_mb=10,
        min_file_size_bytes=1,
        enable_deep_integrity_check=True,
    )


def test_validator_valid_pdf(validator, tmp_path):
    f = tmp_path / "valid.pdf"
    f.write_bytes(b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\n%%EOF")
    res = validator.validate(f)
    assert res.is_valid is True
    assert res.status == ValidationStatus.VALID


def test_validator_invalid_pdf_header(validator, tmp_path):
    f = tmp_path / "corrupt.pdf"
    f.write_bytes(b"NOT_A_PDF_FILE")
    res = validator.validate(f)
    assert res.is_valid is False
    assert res.status == ValidationStatus.CORRUPTED
    assert "Invalid PDF header" in (res.error_message or "")


def test_validator_empty_file(validator, tmp_path):
    f = tmp_path / "empty.txt"
    f.write_bytes(b"")
    res = validator.validate(f)
    assert res.is_valid is False
    assert res.status == ValidationStatus.EMPTY


def test_validator_unsupported_format(validator, tmp_path):
    f = tmp_path / "script.py"
    f.write_text("print('hello')", encoding="utf-8")
    res = validator.validate(f)
    assert res.is_valid is False
    assert res.status == ValidationStatus.UNSUPPORTED


def test_validator_valid_json(validator, tmp_path):
    f = tmp_path / "data.json"
    f.write_text('{"plant": "MRPL", "unit": 203}', encoding="utf-8")
    res = validator.validate(f)
    assert res.is_valid is True
    assert res.status == ValidationStatus.VALID


def test_validator_invalid_json(validator, tmp_path):
    f = tmp_path / "broken.json"
    f.write_text('{"plant": "MRPL", unclosed', encoding="utf-8")
    res = validator.validate(f)
    assert res.is_valid is False
    assert res.status == ValidationStatus.CORRUPTED
