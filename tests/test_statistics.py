"""Unit tests for dataset_engine.statistics."""

import json
from pathlib import Path
import pytest

from dataset_engine.duplicate_detector import DuplicateReport
from dataset_engine.statistics import StatisticsGenerator


def test_statistics_and_health_report(tmp_path):
    generator = StatisticsGenerator()

    docs = [
        {"status": "VALID", "category": "manuals", "extension": ".pdf", "size_bytes": 1000},
        {"status": "VALID", "category": "manuals", "extension": ".docx", "size_bytes": 2000},
        {"status": "CORRUPTED", "category": "safety", "extension": ".pdf", "size_bytes": 500, "relative_path": "safety/bad.pdf", "error_message": "Invalid header"},
        {"status": "EMPTY", "category": "safety", "extension": ".txt", "size_bytes": 0},
    ]

    report = DuplicateReport(
        hash_duplicates={"hash1": ["a.pdf", "b.pdf"]},
        filename_duplicates={},
        total_duplicate_instances=1,
    )

    stats = generator.compute_statistics(docs, report, scan_duration_seconds=1.23)

    assert stats["total_files"] == 4
    assert stats["valid_files"] == 2
    assert stats["corrupted_files"] == 1
    assert stats["empty_files"] == 1
    assert stats["health_score"] == 50.0

    stats_file = tmp_path / "dataset_statistics.json"
    generator.generate_statistics_json(stats, stats_file)
    assert stats_file.exists()

    md_file = tmp_path / "dataset_health_report.md"
    generator.generate_health_report_md(stats, md_file)
    assert md_file.exists()
    content = md_file.read_text(encoding="utf-8")
    assert "Health Score" in content
    assert "50.0%" in content
