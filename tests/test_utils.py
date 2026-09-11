"""Unit tests for rag_engine.utils."""

from pathlib import Path
import pytest

from rag_engine.utils.file_utils import (
    safe_read_file,
    detect_format,
    get_file_size,
    list_files,
    ensure_directory,
    get_temp_path,
)
from rag_engine.utils.hash_utils import hash_text, hash_file, hash_embedding
from rag_engine.utils.validators import (
    validate_file_exists,
    validate_file_format,
    validate_file_size,
    validate_chunk_size,
    validate_embedding_dimension,
)


def test_file_utils(tmp_path):
    p = tmp_path / "sample.txt"
    p.write_text("Hello Sovereign AI", encoding="utf-8")

    assert detect_format(p) == ".txt"
    assert get_file_size(p) == 18
    assert safe_read_file(p, max_bytes=5) == b"Hello"

    temp = get_temp_path()
    assert temp.parent.exists()

    d = ensure_directory(tmp_path / "sub" / "dir")
    assert d.exists()

    files = list_files(tmp_path, extensions={".txt"})
    assert len(files) == 1


def test_hash_utils(tmp_path):
    p = tmp_path / "data.bin"
    p.write_bytes(b"vector bytes")

    h_text = hash_text("hello")
    assert len(h_text) == 64

    h_file = hash_file(p)
    assert len(h_file) == 64

    h_emb = hash_embedding([0.1, 0.2, 0.3])
    assert len(h_emb) == 64


def test_validators(tmp_path):
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"%PDF-1.4")

    assert validate_file_exists(p) is True
    assert validate_file_format(p, {".pdf"}) is True
    assert validate_file_format(p, {".docx"}) is False
    assert validate_file_size(p, 100) is True
    assert validate_file_size(p, 2) is False

    assert validate_chunk_size(512) is True
    assert validate_chunk_size(10) is False

    assert validate_embedding_dimension([0.1] * 768, 768) is True
    assert validate_embedding_dimension([0.1] * 512, 768) is False
