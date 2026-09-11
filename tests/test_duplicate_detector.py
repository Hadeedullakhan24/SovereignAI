"""Unit tests for dataset_engine.duplicate_detector."""

from pathlib import Path
import pytest

from dataset_engine.duplicate_detector import DuplicateDetector


def test_duplicate_detection(tmp_path):
    detector = DuplicateDetector()

    p1 = tmp_path / "manuals" / "spec.pdf"
    p2 = tmp_path / "backup" / "spec_copy.pdf"
    p3 = tmp_path / "safety" / "spec.pdf"  # same name, different hash
    p4 = tmp_path / "unique" / "unique.docx"

    hash_a = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    hash_b = "ca978112ca1bbdcafac231b39a23dc4da7860814965f74754a4c79479c447602"
    hash_c = "5892f97c729793e92c163aaf569c21f44e1f75e2588026199d1a305e5b1207b1"

    detector.register(p1, hash_a)
    detector.register(p2, hash_a)  # hash duplicate
    detector.register(p3, hash_b)  # filename clash with p1
    detector.register(p4, hash_c)  # unique

    report = detector.analyze()

    assert report.total_duplicate_instances == 1
    assert hash_a in report.hash_duplicates
    assert len(report.hash_duplicates[hash_a]) == 2
    assert "spec.pdf" in report.filename_duplicates
    assert len(report.filename_duplicates["spec.pdf"]) == 2
    assert detector.is_duplicate_hash(hash_a) is True
    assert detector.is_duplicate_hash(hash_c) is False
    assert detector.get_canonical_for_hash(hash_a) == p1.resolve().as_posix()
