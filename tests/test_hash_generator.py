"""Unit tests for dataset_engine.hash_generator."""

import tempfile
from pathlib import Path
import pytest

from dataset_engine.hash_generator import HashGenerator


@pytest.fixture
def hash_gen():
    return HashGenerator(chunk_size_bytes=64)


def test_hash_text(hash_gen):
    text = "MRPL Refinery Safety Standard 2026"
    h1 = hash_gen.generate_text_hash(text)
    h2 = hash_gen.generate_text_hash(text)
    assert len(h1) == 64
    assert h1 == h2
    assert hash_gen.are_hashes_identical(h1, h2)


def test_hash_file(hash_gen, tmp_path):
    f = tmp_path / "sample.txt"
    content = b"P&ID Inspection Report for Hydrocracker Unit P-203\n" * 100
    f.write_bytes(content)

    digest = hash_gen.generate_file_hash(f)
    assert len(digest) == 64

    # Changing content changes hash
    f2 = tmp_path / "sample_modified.txt"
    f2.write_bytes(content + b"extra")
    digest2 = hash_gen.generate_file_hash(f2)
    assert digest != digest2
    assert not hash_gen.are_hashes_identical(digest, digest2)


def test_hash_missing_file(hash_gen):
    with pytest.raises(FileNotFoundError):
        hash_gen.generate_file_hash("non_existent_file_xyz_123.txt")
