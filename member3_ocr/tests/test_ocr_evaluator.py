"""Unit tests for member3_ocr.ocr_evaluator.

Tests metric computation, normalization, safety boundaries, synthetic dataset loaders,
FUNSD loaders, evaluation aggregation, and report exporting.
All tests use temporary fixtures and deterministic mock backends;
NO dependency on the master dataset or network downloads.
"""
from __future__ import annotations


import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from member3_ocr.evaluation.ocr.ocr_evaluator import (
    EvaluationSample,
    EvaluationSummary,
    GroupMetrics,
    OCREvaluator,
    SampleEvaluationResult,
    aggregate_group_metrics,
    compute_cer,
    compute_confidence_calibration,
    compute_wer,
    levenshtein_distance,
    load_funsd_dataset_samples,
    load_synthetic_dataset_samples,
    match_block_to_gt_boxes,
    normalize_text_for_eval,
    validate_safe_output_path,
)
from member3_ocr.core.ocr_pipeline import (
    BackendCapabilities,
    BackendInfo,
    BackendRecognition,
    BoundingBox,
    OCRPipeline,
    TextBlock,
    TextLine,
    Word,
)


class MockEvalBackend:
    """Deterministic mock backend returning controlled text."""

    def __init__(self, text: str = "PUMP P-101 12.5 BAR APPROVED", confidence: float = 0.95) -> None:
        self.text = text
        self.confidence = confidence

    @property
    def backend_info(self) -> BackendInfo:
        return BackendInfo("mock_eval", "1.0", ("det", "rec"), "cpu", BackendCapabilities())

    def initialize(self) -> None:
        pass

    def recognize(self, image: np.ndarray) -> BackendRecognition:
        box = BoundingBox(10.0, 10.0, 200.0, 30.0)
        word = Word(self.text, self.confidence, box)
        line = TextLine(self.text, self.confidence, box, (word,))
        block = TextBlock("text-1", self.text, self.confidence, box, (line,))
        return BackendRecognition(blocks=(block,))


def _create_dummy_image(path: Path, width: int = 100, height: int = 50) -> None:
    arr = np.full((height, width, 3), 255, dtype=np.uint8)
    Image.fromarray(arr).save(path)


# 1. Normalization
def test_normalize_text_for_eval() -> None:
    raw = "  Equipment:   P-101 \n\t Pressure: 12.5 bar \r\n Status: APPROVED  "
    expected = "Equipment: P-101 Pressure: 12.5 bar Status: APPROVED"
    assert normalize_text_for_eval(raw) == expected
    assert normalize_text_for_eval("") == ""
    # Unicode NFKC normalization
    assert normalize_text_for_eval("10 \u2103") == "10 °C"


# 2. Levenshtein Distance
def test_levenshtein_distance() -> None:
    assert levenshtein_distance("kitten", "sitting") == 3
    assert levenshtein_distance("MRPL", "MRPL") == 0
    assert levenshtein_distance("", "test") == 4
    assert levenshtein_distance(["a", "b"], ["a", "c"]) == 1


# 3. CER and WER Computation
def test_compute_cer_and_wer() -> None:
    ref = "Equipment: P-101"
    hyp_exact = "Equipment: P-101"
    hyp_err = "Equipment: P-102"

    assert compute_cer(ref, hyp_exact) == 0.0
    assert compute_wer(ref, hyp_exact) == 0.0

    # One character different ('1' vs '2') out of 16 characters
    assert compute_cer(ref, hyp_err) == pytest.approx(1 / 16, abs=1e-3)
    # One word different ('P-101' vs 'P-102') out of 2 words
    assert compute_wer(ref, hyp_err) == pytest.approx(1 / 2, abs=1e-3)

    # Empty cases
    assert compute_cer("", "") == 0.0
    assert compute_wer("", "") == 0.0
    assert compute_cer("abc", "") == 1.0
    assert compute_wer("abc", "") == 1.0


# 4. Dataset Safety: Validate Output Paths Outside Datasets
def test_validate_safe_output_path(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "mock_datasets"
    dataset_dir.mkdir()
    safe_out = tmp_path / "mock_output"
    safe_out.mkdir()

    # Outside datasets -> valid
    validate_safe_output_path(safe_out, dataset_dir)

    # Inside datasets -> raises ValueError
    unsafe_out = dataset_dir / "eval_out"
    unsafe_out.mkdir()
    with pytest.raises(ValueError, match="Safety violation"):
        validate_safe_output_path(unsafe_out, dataset_dir)


# 5. Synthetic Dataset Loader
def test_load_synthetic_dataset_samples(tmp_path: Path) -> None:
    synth_dir = tmp_path / "synthetic_ocr_dataset"
    (synth_dir / "images").mkdir(parents=True)
    (synth_dir / "text").mkdir(parents=True)

    manifest = [
        {"doc_id": "form_001", "category": "synthetic_forms", "degraded": False, "meta": {"type": "form"}},
        {"doc_id": "table_001", "category": "synthetic_tables", "degraded": True, "meta": {"type": "table"}},
    ]
    (synth_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    _create_dummy_image(synth_dir / "images" / "form_001.png")
    (synth_dir / "text" / "form_001.txt").write_text("Form 1 Content", encoding="utf-8")

    _create_dummy_image(synth_dir / "images" / "table_001.jpg")
    (synth_dir / "text" / "table_001.txt").write_text("Table 1 Content", encoding="utf-8")

    samples = load_synthetic_dataset_samples(synth_dir)
    assert len(samples) == 2
    assert samples[0].sample_id == "form_001"
    assert samples[0].dataset == "synthetic"
    assert samples[0].category == "synthetic_forms"
    assert samples[0].degraded is False
    assert samples[0].ground_truth_text == "Form 1 Content"

    assert samples[1].sample_id == "table_001"
    assert samples[1].degraded is True


# 6. FUNSD Dataset Loader
def test_load_funsd_dataset_samples(tmp_path: Path) -> None:
    funsd_dir = tmp_path / "FUNSD"
    train_dir = funsd_dir / "training_data"
    (train_dir / "images").mkdir(parents=True)
    (train_dir / "annotations").mkdir(parents=True)

    _create_dummy_image(train_dir / "images" / "001.png")
    annotation_data = {
        "form": [
            {"box": [10, 50, 100, 70], "text": "Header Text", "label": "header", "id": 1},
            {"box": [10, 10, 100, 30], "text": "Top Title", "label": "header", "id": 0},
        ]
    }
    (train_dir / "annotations" / "001.json").write_text(json.dumps(annotation_data), encoding="utf-8")

    samples = load_funsd_dataset_samples(funsd_dir, split="training_data")
    assert len(samples) == 1
    sample = samples[0]
    assert sample.sample_id == "001"
    assert sample.dataset == "funsd_train"
    # Verify reading order sort: Top Title (top 10) should precede Header Text (top 50)
    lines = sample.ground_truth_text.splitlines()
    assert lines[0] == "Top Title"
    assert lines[1] == "Header Text"


# 7. Evaluator Single Sample & Metric Scoring
def test_evaluator_single_sample(tmp_path: Path) -> None:
    img_path = tmp_path / "test.png"
    _create_dummy_image(img_path)

    sample = EvaluationSample(
        sample_id="test_doc",
        dataset="synthetic",
        category="forms",
        degraded=False,
        image_path=img_path,
        ground_truth_path=img_path,
        ground_truth_text="PUMP P-101 12.5 BAR APPROVED",
    )

    backend = MockEvalBackend(text="PUMP P-101 12.5 BAR APPROVED", confidence=0.98)
    pipeline = OCRPipeline(backend)
    eval_out = tmp_path / "eval_out"
    evaluator = OCREvaluator(pipeline, output_dir=eval_out)

    res = evaluator.evaluate_sample(sample)
    assert res.sample_id == "test_doc"
    assert res.exact_match is True
    assert res.cer == 0.0
    assert res.wer == 0.0
    assert res.ocr_confidence == 0.98
    assert res.num_blocks == 1
    assert res.error_message is None


# 8. Evaluator Batch & Breakdown Aggregation
def test_evaluator_all_and_breakdowns(tmp_path: Path) -> None:
    img1 = tmp_path / "img1.png"
    img2 = tmp_path / "img2.png"
    _create_dummy_image(img1)
    _create_dummy_image(img2)

    sample1 = EvaluationSample(
        sample_id="s1",
        dataset="synthetic",
        category="forms",
        degraded=False,
        image_path=img1,
        ground_truth_path=img1,
        ground_truth_text="MATCHING TEXT",
    )
    sample2 = EvaluationSample(
        sample_id="s2",
        dataset="synthetic",
        category="tables",
        degraded=True,
        image_path=img2,
        ground_truth_path=img2,
        ground_truth_text="DIFFERENT TEXT",
    )

    # Backend outputs MATCHING TEXT
    backend = MockEvalBackend(text="MATCHING TEXT", confidence=0.90)
    pipeline = OCRPipeline(backend)
    eval_out = tmp_path / "eval_out"
    evaluator = OCREvaluator(pipeline, output_dir=eval_out)

    summary, results = evaluator.evaluate_all([sample1, sample2])

    assert summary.total_samples == 2
    assert summary.overall.sample_count == 2
    assert summary.overall.exact_match_count == 1
    assert summary.overall.exact_match_rate == 0.5

    # Breakdowns exist
    assert "forms" in summary.by_category
    assert "tables" in summary.by_category
    assert summary.by_category["forms"].exact_match_rate == 1.0
    assert summary.by_category["tables"].exact_match_rate == 0.0

    assert "clean" in summary.by_degradation
    assert "degraded" in summary.by_degradation
    assert summary.by_degradation["clean"].exact_match_rate == 1.0
    assert summary.by_degradation["degraded"].exact_match_rate == 0.0


# 9. Result Export: summary.json, per_sample.csv, failures.csv, README.md
def test_export_results(tmp_path: Path) -> None:
    img_path = tmp_path / "img.png"
    _create_dummy_image(img_path)

    sample = EvaluationSample(
        sample_id="doc_fail",
        dataset="synthetic",
        category="forms",
        degraded=False,
        image_path=img_path,
        ground_truth_path=img_path,
        ground_truth_text="TARGET GROUND TRUTH",
    )

    # Backend output differs -> CER > threshold
    backend = MockEvalBackend(text="COMPLETELY WRONG", confidence=0.5)
    pipeline = OCRPipeline(backend)
    eval_out = tmp_path / "eval_out"
    evaluator = OCREvaluator(pipeline, output_dir=eval_out, poor_cer_threshold=0.15)

    summary, results = evaluator.evaluate_all([sample])
    exported = evaluator.export_results(summary, results)

    assert exported["summary"].exists()
    assert exported["per_sample"].exists()
    assert exported["failures"].exists()
    assert exported["readme"].exists()

    # Check summary.json structure
    summary_data = json.loads(exported["summary"].read_text(encoding="utf-8"))
    assert summary_data["total_samples"] == 1
    assert "overall" in summary_data

    # Check failures.csv has the failed sample
    with open(exported["failures"], mode="r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 1
        assert reader[0]["sample_id"] == "doc_fail"
        assert "TARGET GROUND TRUTH" in reader[0]["ground_truth_snippet"]


# 10. Spatial Block-to-GT Matching and Calibration
def test_match_block_to_gt_boxes_and_calibration() -> None:
    gt_boxes = [
        {"box": [10.0, 10.0, 50.0, 30.0], "text": "PUMP"},
        {"box": [60.0, 10.0, 110.0, 30.0], "text": "P-101"},
        {"box": [10.0, 40.0, 90.0, 60.0], "text": "OPERATING"},
    ]
    # Block overlapping both PUMP and P-101
    matched = match_block_to_gt_boxes((5.0, 5.0, 120.0, 35.0), gt_boxes)
    assert matched == "PUMP P-101"

    # Non-overlapping block
    unmatched = match_block_to_gt_boxes((200.0, 200.0, 250.0, 250.0), gt_boxes)
    assert unmatched == ""

    # Test confidence calibration bucketing
    from member3_ocr.evaluation.ocr.ocr_evaluator import BlockConfidenceRecord
    sample_res = SampleEvaluationResult(
        sample_id="calib_test",
        dataset="synthetic",
        category="forms",
        degraded=False,
        image_path="",
        ground_truth_length=10,
        ocr_text_length=10,
        exact_match=True,
        cer=0.0,
        wer=0.0,
        ocr_confidence=0.96,
        num_blocks=2,
        processing_time_ms=10.0,
        ocr_text="TEST",
        ground_truth_text="TEST",
        block_records=(
            BlockConfidenceRecord("b1", 0.98, "PUMP P-101", "PUMP P-101", 0.0, 0.0),
            BlockConfidenceRecord("b2", 0.75, "OPRATING", "OPERATING", 0.11, 1.0),
        ),
    )
    buckets = compute_confidence_calibration([sample_res])
    b_map = {b.label: b for b in buckets}

    vh = b_map["very_high (>0.95)"]
    assert vh.block_count == 1
    assert vh.percent_of_blocks == 50.0
    assert vh.mean_cer == 0.0
    assert vh.mean_wer == 0.0

    med = b_map["medium (0.60-0.80)"]
    assert med.block_count == 1
    assert med.percent_of_blocks == 50.0
    assert med.mean_cer == 0.11
    assert med.mean_wer == 1.0

