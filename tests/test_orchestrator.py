"""Integration tests for dataset_engine.orchestrator."""

from pathlib import Path
import pytest

from dataset_engine.orchestrator import DatasetOrchestrator


@pytest.fixture
def mock_dataset_env(tmp_path):
    dataset_dir = tmp_path / "mock_datasets"
    manuals = dataset_dir / "manuals"
    manuals.mkdir(parents=True)

    # Valid PDF
    (manuals / "pump.pdf").write_bytes(b"%PDF-1.4 sample valid content %%EOF")
    # Duplicate of pump.pdf
    (manuals / "pump_copy.pdf").write_bytes(b"%PDF-1.4 sample valid content %%EOF")
    # Valid TXT
    (manuals / "guide.txt").write_text("MRPL operational guide", encoding="utf-8")
    # Empty file
    (manuals / "zero.txt").write_bytes(b"")

    output_dir = tmp_path / "mock_outputs"
    return dataset_dir, output_dir


def test_orchestrator_pipeline(mock_dataset_env):
    dataset_dir, output_dir = mock_dataset_env
    orchestrator = DatasetOrchestrator()

    result = orchestrator.run(dataset_dir=dataset_dir, output_dir=output_dir)

    assert result.total_files_scanned == 4
    assert result.valid_files_count == 3
    assert result.empty_files_count == 1
    assert result.duplicate_instances_count == 1
    assert result.manifest_path.exists()
    assert result.statistics_path.exists()
    assert result.health_report_path.exists()
