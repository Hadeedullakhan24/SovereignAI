"""Unit tests for dataset_engine.scanner."""

from pathlib import Path
import pytest

from dataset_engine.scanner import DatasetScanner


@pytest.fixture
def mock_dataset_dir(tmp_path):
    # Setup mock structure
    manuals = tmp_path / "manuals" / "pumps"
    manuals.mkdir(parents=True)
    (manuals / "pump_manual.pdf").write_bytes(b"%PDF-1.4 dummy")
    (manuals / "notes.txt").write_text("notes", encoding="utf-8")
    (manuals / "unsupported.exe").write_bytes(b"MZ dummy")

    safety = tmp_path / "safety_docs"
    safety.mkdir(parents=True)
    (safety / "safety_sop.docx").write_bytes(b"PK\x03\x04 dummy docx")

    # Ignored directory
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("git config", encoding="utf-8")

    return tmp_path


def test_scanner_discovers_and_categorizes(mock_dataset_dir):
    scanner = DatasetScanner(
        supported_extensions={".pdf", ".docx", ".txt"},
        ignored_extensions={".exe"},
    )
    discovered = scanner.scan(mock_dataset_dir)

    # Should ignore .git and .exe
    rel_paths = {d.relative_path for d in discovered}
    assert "manuals/pumps/pump_manual.pdf" in rel_paths
    assert "manuals/pumps/notes.txt" in rel_paths
    assert "safety_docs/safety_sop.docx" in rel_paths
    assert not any(".git" in p for p in rel_paths)
    assert not any(".exe" in p for p in rel_paths)

    # Check categories
    pdf_item = next(d for d in discovered if d.relative_path == "manuals/pumps/pump_manual.pdf")
    assert pdf_item.category == "manuals"
    assert pdf_item.subcategory == "pumps"
    assert pdf_item.is_supported is True


def test_scanner_get_supported_files(mock_dataset_dir):
    scanner = DatasetScanner(supported_extensions={".pdf"})
    supported = scanner.get_supported_files(mock_dataset_dir)
    assert len(supported) == 1
    assert supported[0].extension == ".pdf"
