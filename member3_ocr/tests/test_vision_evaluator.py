"""Unit tests for Member 3 Vision Pipeline Evaluation Harness.

Validates:
  - Manifest consistency and category coverage
  - Read-only sample file existence and rendering (JPG, PNG, PDF)
  - Drawing type matching accuracy
  - Hallucination detection rules (pressures, temperatures, safety claims, proximity)
  - P&ID inspection analysis
  - End-to-end evaluation execution with mock backend
  - Resume functionality
  - Exported artifacts (JSON, CSV, README.md) schema validation
  - Latency audit: model load time separation from per-inference time
  - Confidence method documentation (self_reported_heuristic)
  - Ground-truth keyword scoring (scene + equipment)
  - Repeat-run variance reporting
  - Device detection and resolution
  - Coverage gap detection
  - Full-manifest latency projection

All tests in this file run strictly OFFLINE with NO model downloads and NO real VLM inference.
"""
from __future__ import annotations


import csv
import json
from pathlib import Path
import shlex
import tempfile
import time

import numpy as np
import pytest

from member3_ocr.evaluation.vision.evaluate_vision import (
    CATEGORY_ALIASES,
    CATEGORY_DISPLAY_NAMES,
    CONFIDENCE_METHOD,
    MANIFEST_CATEGORIES,
    PRIMARY_TASK_MAP,
    REPRESENTATIVE_SAMPLES,
    DRAWING_SYNONYMS,
    EvaluationRecord,
    MockVisionEvaluatorBackend,
    SampleManifestItem,
    VisionEvaluationHarness,
    check_drawing_type_match,
    check_equipment_type_match,
    check_hallucinations,
    check_scene_keyword_match,
    detect_available_device,
    estimate_full_run_duration,
    estimate_run_duration,
    filter_samples_by_category,
    format_next_run_hint,
    generate_chunked_run_plan,
    get_next_suggested_category_chunk,
    inspect_pid_outputs,
    load_image_input,
    main,
    parse_args,
    resolve_device_for_harness,
)



def test_manifest_coverage_and_counts() -> None:
    """Validate that the sample manifest meets the representative target and category rules."""
    assert len(REPRESENTATIVE_SAMPLES) >= 12, "Manifest should have at least 12 representative samples"
    assert len(REPRESENTATIVE_SAMPLES) <= 20, "Manifest should not exceed 20 samples"

    categories = {s.category for s in REPRESENTATIVE_SAMPLES}
    required_categories = {
        "PID", "PFD", "Equipment_Drawings", "Instrumentation",
        "Pump_Diagrams", "Electrical", "corrosion_defects",
        "defect_detection", "infrared", "handwritten_notes"
    }
    assert required_categories.issubset(categories), f"Missing categories: {required_categories - categories}"

    # Verify task assignments
    for s in REPRESENTATIVE_SAMPLES:
        assert "understand" in s.tasks, f"{s.sample_id} missing 'understand' task"
        assert "caption" in s.tasks, f"{s.sample_id} missing 'caption' task"
        if s.dataset == "engineering_drawings":
            assert "drawing" in s.tasks, f"{s.sample_id} missing 'drawing' task"
        if s.category in ("corrosion_defects", "defect_detection", "infrared"):
            assert "observation" in s.tasks, f"{s.sample_id} missing 'observation' task"


def test_sample_files_exist_on_disk() -> None:
    """Verify that all samples defined in manifest actually exist in local datasets."""
    project_root = Path(__file__).resolve().parent.parent.parent
    for s in REPRESENTATIVE_SAMPLES:
        p = project_root / s.source_path
        assert p.exists(), f"Sample file not found: {p}"
        assert p.stat().st_size > 0, f"Sample file is empty: {p}"


def test_load_image_input_image_and_pdf(tmp_path: Path) -> None:
    """Test loading RGB arrays from regular image and PDF page 1."""
    project_root = Path(__file__).resolve().parent.parent.parent

    # Test JPG
    jpg_path = project_root / "datasets/engineering_drawings/PID/PID_002_Process_Example.jpg"
    arr_jpg = load_image_input(jpg_path)
    assert isinstance(arr_jpg, np.ndarray)
    assert arr_jpg.ndim == 3 and arr_jpg.shape[2] == 3
    assert arr_jpg.dtype == np.uint8

    # Test PDF
    pdf_path = project_root / "datasets/engineering_drawings/Equipment_Drawings/EQUIP_001_AlfaLaval_Plate_Heat_Exchanger_Drawing.pdf"
    arr_pdf = load_image_input(pdf_path)
    assert isinstance(arr_pdf, np.ndarray)
    assert arr_pdf.ndim == 3 and arr_pdf.shape[2] == 3
    assert arr_pdf.dtype == np.uint8

    # Test missing file
    with pytest.raises(FileNotFoundError):
        load_image_input(tmp_path / "non_existent.jpg")


def test_check_drawing_type_match() -> None:
    """Test matching predicted drawing types against expected categories."""
    assert check_drawing_type_match("Piping and Instrumentation Diagram", "PID") is True
    assert check_drawing_type_match("P&ID", "PID") is True
    assert check_drawing_type_match("Process Flow Diagram (PFD)", "PFD") is True
    assert check_drawing_type_match("Mechanical assembly drawing", "Equipment_Drawings") is True
    assert check_drawing_type_match("Centrifugal pump sectional cutaway", "Pump_Diagrams") is True
    assert check_drawing_type_match("Single-line power distribution diagram", "Electrical") is True

    # Mismatch
    assert check_drawing_type_match("Electrical wiring diagram", "PID") is False
    assert check_drawing_type_match("", "PID") is None
    assert check_drawing_type_match("P&ID", "corrosion_defects") is None


def test_check_hallucinations_detects_unsupported_assertions() -> None:
    """Verify that hallucination checks catch unsupported specs, pressures, safety claims, and proximity."""
    # 1. Unsupported pressure
    flags_press = check_hallucinations("The system operates at 12 bar pressure.", [], "PID")
    assert "unsupported_pressure_value" in flags_press

    # 2. Unsupported temperature
    flags_temp = check_hallucinations("Operating temperature is 180°C inside the drum.", [], "PID")
    assert "unsupported_temperature_value" in flags_temp

    # 3. Unsupported dimension in schematic
    flags_dim = check_hallucinations("Pipe diameter is 50 mm.", [], "PID")
    assert "unsupported_dimension" in flags_dim

    # 4. Unsupported safety conclusion
    flags_safe = check_hallucinations("Equipment is safe to operate without hazards.", [], "corrosion_defects")
    assert "unsupported_safety_conclusion" in flags_safe

    # 5. Unsupported proximity-based connectivity
    flags_conn = check_hallucinations("The pump is connected to the valve based on proximity.", [], "PID")
    assert "unsupported_connected_to_relationship" in flags_conn

    # 6. Invented equipment tag
    equip_items = [
        {"equipment_type": "Pump", "name_or_tag": "P-999", "evidence": "assumed tag from layout"},
    ]
    flags_tag = check_hallucinations("Pump visible", equip_items, "PID", known_tags=["P-101", "P-102"])
    assert any("invented_equipment_tag:P-999" in f for f in flags_tag)

    # 7. Clean, grounded text
    clean_text = "Drawing shows an industrial heat exchanger with inlet piping."
    flags_clean = check_hallucinations(clean_text, [], "PID")
    assert flags_clean == []


def test_pid_inspection_analysis() -> None:
    """Test P&ID specific inspection parsing."""
    task_res = {
        "equipment": [
            {"equipment_type": "Centrifugal Pump"},
            {"equipment_type": "Pneumatic Control Valve"},
            {"equipment_type": "Temperature Transmitter"},
        ],
        "visible_labels": [{"text": "P-101"}, {"text": "TT-201"}],
        "raw_output": "P&ID shows flow line connecting vessel to pump inlet with temperature transmitter loop.",
    }
    inspection = inspect_pid_outputs(task_res)
    assert inspection["labels_observed_count"] == 2
    assert inspection["equipment_identified_count"] == 3
    assert "Pneumatic Control Valve" in inspection["instruments_identified"]
    assert "Temperature Transmitter" in inspection["instruments_identified"]
    assert inspection["visual_structure_detected"] is True
    assert inspection["unsupported_connectivity_claims"] is False


def test_harness_end_to_end_mock(tmp_path: Path) -> None:
    """Test entire evaluation harness workflow with Mock backend generating JSON, CSV, and README."""
    harness = VisionEvaluationHarness(
        output_dir=tmp_path,
        is_mock=True,
    )
    assert harness.pipeline is not None
    assert harness.pipeline.backend.backend_info.name == "mock_vision_backend"

    # Evaluate 2 samples
    summary = harness.evaluate(limit=2)

    assert summary["evaluation_metadata"]["total_inferences"] > 0
    assert summary["evaluation_metadata"]["successful_inferences"] > 0
    assert summary["evaluation_metadata"]["failed_inferences"] == 0
    assert summary["evaluation_metadata"]["is_mock"] is True

    # Verify JSON output
    json_path = tmp_path / "vision_evaluation.json"
    assert json_path.exists()
    loaded_json = json.loads(json_path.read_text(encoding="utf-8"))
    assert "per_sample_inferences" in loaded_json
    assert len(loaded_json["per_sample_inferences"]) == summary["evaluation_metadata"]["total_inferences"]

    # Verify CSV output
    csv_path = tmp_path / "vision_evaluation.csv"
    assert csv_path.exists()
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert "sample_id" in header
        assert "task" in header
        assert "device_used" in header
        rows = list(reader)
        assert len(rows) == summary["evaluation_metadata"]["total_inferences"]

    # Verify README.md
    readme_path = tmp_path / "README.md"
    assert readme_path.exists()
    readme_text = readme_path.read_text(encoding="utf-8")
    assert "Member 3 Vision Pipeline Evaluation Report" in readme_text
    assert "Quantitative Measurements" in readme_text
    assert "Qualitative Observations" in readme_text
    assert "Hallucination" in readme_text and "Safety" in readme_text


def test_harness_resume_functionality(tmp_path: Path) -> None:
    """Test that resume=True correctly retains previous inferences and avoids re-execution."""
    harness = VisionEvaluationHarness(output_dir=tmp_path, is_mock=True)

    # 1. Run limit=1
    summary1 = harness.evaluate(limit=1, resume=False)
    inferences_sample1 = summary1["evaluation_metadata"]["total_inferences"]

    # 2. Run limit=2 with resume=True
    summary2 = harness.evaluate(limit=2, resume=True)
    total_inferences = summary2["evaluation_metadata"]["total_inferences"]

    assert total_inferences > inferences_sample1
    sample_ids = {r["sample_id"] for r in summary2["per_sample_inferences"]}
    assert len(sample_ids) == 2


def test_confidence_method_metadata(tmp_path: Path) -> None:
    """Test that confidence scoring method is strictly documented as self-reported heuristic."""
    assert CONFIDENCE_METHOD == "self_reported_heuristic"

    harness = VisionEvaluationHarness(output_dir=tmp_path, is_mock=True)
    summary = harness.evaluate(limit=1, resume=False)

    conf_meta = summary.get("confidence_metadata", {})
    assert conf_meta.get("confidence_method") == "self_reported_heuristic"
    assert "logprobs" in conf_meta.get("confidence_derivation", "").lower()

    # Verify per-sample inferences carry confidence_method
    for inf in summary["per_sample_inferences"]:
        assert inf.get("confidence_method") == "self_reported_heuristic"


def test_latency_audit_separation(tmp_path: Path) -> None:
    """Test that model loading time is isolated from per-inference execution latency."""
    harness = VisionEvaluationHarness(output_dir=tmp_path, is_mock=True)
    summary = harness.evaluate(limit=1, resume=False)

    eval_meta = summary["evaluation_metadata"]
    assert "model_load_time_ms" in eval_meta
    assert isinstance(eval_meta["model_load_time_ms"], (int, float))

    latency_sum = summary["latency_summary"]
    assert "processing_time_ms" in latency_sum["note"]
    assert "model_load_time_ms" in latency_sum["note"]
    assert "average_latency_ms" in latency_sum
    assert "per_task_latency" in latency_sum

    # Each inference has its individual processing_time_ms
    for inf in summary["per_sample_inferences"]:
        assert "processing_time_ms" in inf
        assert inf["processing_time_ms"] >= 0


def test_scene_keyword_matching_logic() -> None:
    """Test ground-truth keyword matching logic for scene understanding."""
    # Positive match with mixed case
    res = check_scene_keyword_match(
        predicted_text="This is an industrial distillation column with reflux drum.",
        expected_keywords=("distillation column", "reflux drum", "boiler"),
    )
    assert res["scored"] is True
    assert res["total_expected"] == 3
    assert set(res["matched_keywords"]) == {"distillation column", "reflux drum"}
    assert res["keyword_hit_rate"] == round(2 / 3, 4)

    # Empty expected keywords (unscored)
    res_empty = check_scene_keyword_match(
        predicted_text="Some equipment",
        expected_keywords=(),
    )
    assert res_empty["scored"] is False
    assert res_empty["total_expected"] == 0
    assert res_empty["keyword_hit_rate"] is None


def test_equipment_type_matching_logic() -> None:
    """Test ground-truth equipment type matching logic."""
    detected = ["centrifugal pump", "gate valve", "storage tank"]
    expected = ("pump", "valve", "heat exchanger")

    res = check_equipment_type_match(detected, expected)
    assert res["scored"] is True
    assert res["total_expected"] == 3
    assert set(res["matched_equipment"]) == {"pump", "valve"}
    assert res["equipment_hit_rate"] == round(2 / 3, 4)

    # Empty expected
    res_empty = check_equipment_type_match(detected, ())
    assert res_empty["scored"] is False
    assert res_empty["equipment_hit_rate"] is None


def test_repeat_variance_reporting(tmp_path: Path) -> None:
    """Test that repeat execution produces valid variance metrics."""
    harness = VisionEvaluationHarness(output_dir=tmp_path, is_mock=True)
    summary = harness.evaluate(limit=2, repeat=2, resume=False)

    var_report = summary.get("repeat_variance_report", {})
    assert len(var_report) > 0

    first_key = next(iter(var_report))
    item = var_report[first_key]
    assert item["repeat_count"] == 2
    assert "confidence_mean" in item
    assert "confidence_stdev" in item
    assert "hallucination_flag_counts" in item
    assert len(item["hallucination_flag_counts"]) == 2


def test_device_detection_and_resolution() -> None:
    """Test device auto-detection and explicit resolution."""
    dev, reason = detect_available_device()
    assert dev in ("cuda", "cpu")
    assert isinstance(reason, str) and len(reason) > 0

    # Explicit cpu
    res_dev, note = resolve_device_for_harness("cpu")
    assert res_dev == "cpu"
    assert "explicitly requested" in note

    # Auto resolution
    res_dev_auto, note_auto = resolve_device_for_harness("auto")
    assert res_dev_auto in ("cuda", "cpu")
    assert "auto-selected" in note_auto


def test_full_run_projection_calculation() -> None:
    """Test wall-clock projection calculations for the manifest."""
    proj = estimate_full_run_duration(REPRESENTATIVE_SAMPLES, avg_latency_ms_per_task=1000.0)
    assert proj["total_manifest_samples"] == len(REPRESENTATIVE_SAMPLES)
    assert proj["total_manifest_tasks"] > 0
    assert proj["estimated_total_ms"] == proj["total_manifest_tasks"] * 1000.0
    assert proj["estimated_total_minutes"] == round(proj["estimated_total_ms"] / 60000, 1)


def test_category_maps_are_consistent() -> None:
    """Verify that CATEGORY_ALIASES and PRIMARY_TASK_MAP keys strictly equal MANIFEST_CATEGORIES (Fix 1)."""
    assert set(CATEGORY_ALIASES.keys()) == set(MANIFEST_CATEGORIES)
    assert set(PRIMARY_TASK_MAP.keys()) == set(MANIFEST_CATEGORIES)
    for s in REPRESENTATIVE_SAMPLES:
        primary_task = PRIMARY_TASK_MAP[s.category]
        assert primary_task in s.tasks, f"{primary_task} not in {s.sample_id} tasks {s.tasks}"


def test_eta_fallback_on_cold_start(tmp_path: Path) -> None:
    """Verify that cold-start fallback estimate doesn't crash and clearly flags itself as an estimate (Fix 2)."""
    non_existent_json = tmp_path / "vision_evaluation.json"
    est = estimate_run_duration(
        REPRESENTATIVE_SAMPLES[:2],
        tasks_per_sample=1,
        prior_json_path=non_existent_json,
        is_quick=True,
    )
    assert est["is_fallback_estimate"] is True
    assert "No prior data for these settings — showing conservative estimate:" in est["note"]
    assert "Actual may differ." in est["note"]
    assert est["estimated_total_minutes"] > 0
    assert est["total_tasks_estimated"] == 2


def test_category_filter_by_alias() -> None:
    """Verify that filter_samples_by_category handles canonical names and aliases case-insensitively."""
    # Canonical
    pid_samples = filter_samples_by_category(REPRESENTATIVE_SAMPLES, ["PID"])
    assert len(pid_samples) > 0
    assert all(s.category == "PID" for s in pid_samples)

    # Alias (lower-case)
    pid_alias_samples = filter_samples_by_category(REPRESENTATIVE_SAMPLES, ["p&id"])
    assert pid_alias_samples == pid_samples

    # Multiple categories
    combined = filter_samples_by_category(REPRESENTATIVE_SAMPLES, ["pid", "process flow"])
    assert {s.category for s in combined} == {"PID", "PFD"}

    # Invalid category/alias
    with pytest.raises(ValueError, match="Unknown category"):
        filter_samples_by_category(REPRESENTATIVE_SAMPLES, ["invalid_unknown_cat"])


def test_list_categories_output(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify that --list-categories prints all manifest categories and exits cleanly."""
    rc = main(["--list-categories"])
    assert rc == 0
    captured = capsys.readouterr().out
    for cat in MANIFEST_CATEGORIES:
        assert cat in captured


def test_quick_mode_selects_primary_task(tmp_path: Path) -> None:
    """Verify that tasks_per_sample=1 selects PRIMARY_TASK_MAP task for each category."""
    harness = VisionEvaluationHarness(output_dir=tmp_path, is_mock=True, run_mode="quick")
    samples = REPRESENTATIVE_SAMPLES[:4]
    summary = harness.evaluate(samples=samples, tasks_per_sample=1)

    records = summary["per_sample_inferences"]
    assert len(records) == len(samples)
    for r in records:
        expected_task = PRIMARY_TASK_MAP[r["category"]]
        assert r["task"] == expected_task


def test_tasks_per_sample_limiter(tmp_path: Path) -> None:
    """Verify that tasks_per_sample limits the number of tasks evaluated per sample."""
    harness = VisionEvaluationHarness(output_dir=tmp_path, is_mock=True)
    summary = harness.evaluate(samples=REPRESENTATIVE_SAMPLES[:2], tasks_per_sample=2)
    records = summary["per_sample_inferences"]
    # 2 samples x 2 tasks = 4 inferences
    assert len(records) == 4
    sample_task_counts: dict[str, int] = {}
    for r in records:
        sample_task_counts[r["sample_id"]] = sample_task_counts.get(r["sample_id"], 0) + 1
    assert all(c == 2 for c in sample_task_counts.values())


def test_quick_mode_metadata_fields(tmp_path: Path) -> None:
    """Verify that run_mode and quick_mode_settings are preserved in evaluation_metadata."""
    harness = VisionEvaluationHarness(
        output_dir=tmp_path,
        is_mock=True,
        run_mode="quick",
        quick_mode_settings={"max_image_size": 192, "max_new_tokens": 24, "tasks_per_sample": 1},
    )
    summary = harness.evaluate(limit=1)
    meta = summary["evaluation_metadata"]
    assert meta["run_mode"] == "quick"
    assert meta["quick_mode_settings"] == {
        "max_image_size": 192,
        "max_new_tokens": 24,
        "tasks_per_sample": 1,
    }
    assert "stopped_early" in meta


def test_max_run_minutes_early_stop(tmp_path: Path) -> None:
    """Verify that max_run_minutes stops evaluation gracefully and records stopped_early."""
    harness = VisionEvaluationHarness(output_dir=tmp_path, is_mock=True)
    # Tiny budget (0.000001 min = 0.06 ms) forces early stop after the first task
    summary = harness.evaluate(limit=5, max_run_minutes=0.000001)
    assert summary["evaluation_metadata"]["stopped_early"] is True
    assert summary["evaluation_metadata"]["total_inferences"] < 5 * 2
    assert (tmp_path / "vision_evaluation.json").exists()


def test_interrupt_safety_and_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that KeyboardInterrupt triggers incremental flush and resume continues."""
    harness = VisionEvaluationHarness(output_dir=tmp_path, is_mock=True)

    call_count = 0
    orig_infer = harness.run_inference_on_sample_task

    def mock_infer(sample, task, image_array, repeat_index=0):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise KeyboardInterrupt()
        return orig_infer(sample, task, image_array, repeat_index)

    monkeypatch.setattr(harness, "run_inference_on_sample_task", mock_infer)

    with pytest.raises(KeyboardInterrupt):
        harness.evaluate(limit=2, resume=False)

    # Verify task 1 was saved via incremental flush
    json_path = tmp_path / "vision_evaluation.json"
    assert json_path.exists()
    flushed_data = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(flushed_data["per_sample_inferences"]) == 1

    # Resume without monkeypatch
    monkeypatch.undo()
    summary = harness.evaluate(limit=2, resume=True)
    assert summary["evaluation_metadata"]["total_inferences"] > 1


def test_threads_flag_parses() -> None:
    """Verify that --threads flag is parsed correctly by parse_args."""
    args_with = parse_args(["--threads", "4"])
    assert args_with.threads == 4

    args_without = parse_args([])
    assert args_without.threads is None


def test_category_resume_isolation(tmp_path: Path) -> None:
    """Verify multi-session --category + --resume isolation (Fix 3)."""
    harness = VisionEvaluationHarness(output_dir=tmp_path, is_mock=True)

    # Session 1: Run category PID with resume=True
    pid_samples = filter_samples_by_category(REPRESENTATIVE_SAMPLES, ["PID"])
    summary1 = harness.evaluate(samples=pid_samples, resume=True)
    pid_records = summary1["per_sample_inferences"]
    pid_inferences_count = len(pid_records)
    assert pid_inferences_count > 0
    assert all(r["category"] == "PID" for r in pid_records)

    # Session 2: Run category PFD with resume=True
    pfd_samples = filter_samples_by_category(REPRESENTATIVE_SAMPLES, ["PFD"])
    summary2 = harness.evaluate(samples=pfd_samples, resume=True)
    combined_records = summary2["per_sample_inferences"]

    # Assert X's results are present and unmodified
    pid_records_after = [r for r in combined_records if r["category"] == "PID"]
    assert pid_records_after == pid_records

    # Assert Y's results are newly added
    pfd_records_after = [r for r in combined_records if r["category"] == "PFD"]
    assert len(pfd_records_after) > 0

    # Assert no task from X was re-run or duplicated
    pid_sample_tasks = [(r["sample_id"], r["task"], r["repeat_index"]) for r in pid_records_after]
    assert len(pid_sample_tasks) == len(set(pid_sample_tasks))
    assert len(combined_records) == pid_inferences_count + len(pfd_records_after)

    # Assert combined CSV has both
    csv_path = tmp_path / "vision_evaluation.csv"
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        csv_cats = {row["category"] for row in reader}
        assert "PID" in csv_cats
        assert "PFD" in csv_cats


def test_coverage_gap_partial_state(tmp_path: Path) -> None:
    """Verify coverage-gap warning correctly lists all uncovered categories under partial chunking (Fix 5 / Bug 2)."""
    harness = VisionEvaluationHarness(output_dir=tmp_path, is_mock=True)
    # Evaluate 3 categories
    subset_cats = ["PID", "PFD", "Electrical"]
    samples = filter_samples_by_category(REPRESENTATIVE_SAMPLES, subset_cats)
    summary = harness.evaluate(samples=samples)

    cov = summary["coverage_summary"]
    uncovered = cov["categories_with_zero_real_model_coverage"]
    expected_uncovered = sorted(set(CATEGORY_DISPLAY_NAMES.values()) - {CATEGORY_DISPLAY_NAMES.get(c, c) for c in subset_cats})

    # Exactly 7 uncovered categories
    assert len(uncovered) == 7
    assert uncovered == expected_uncovered

    # Coverage gap warning explicitly lists all 7 uncovered categories
    warning = cov["coverage_gap_warning"]
    assert "WARNING: 7 categories have zero coverage:" in warning
    for cat in expected_uncovered:
        assert cat in warning


def test_latency_stats_scoped_by_settings_signature(tmp_path: Path) -> None:
    """Verify avg latency and full-run estimates only reflect matching-signature records (Bug 1 Fix)."""
    harness = VisionEvaluationHarness(
        output_dir=tmp_path,
        is_mock=True,
        run_mode="quick",
        max_image_size=192,
        max_new_tokens=24,
    )

    # Seed mixed-signature records:
    # 2 old full-mode records (avg 550,000 ms)
    full_records = [
        EvaluationRecord(
            sample_id="PID_01",
            source_path="fake/path/pid1.jpg",
            dataset="engineering_drawings",
            category="PID",
            task="drawing",
            success=True,
            processing_time_ms=500_000.0,
            run_mode="full",
            max_image_size=256,
            max_new_tokens=48,
        ),
        EvaluationRecord(
            sample_id="PID_01",
            source_path="fake/path/pid1.jpg",
            dataset="engineering_drawings",
            category="PID",
            task="understand",
            success=True,
            processing_time_ms=600_000.0,
            run_mode="full",
            max_image_size=256,
            max_new_tokens=48,
        ),
    ]

    # 2 quick-mode records (avg 250,000 ms)
    quick_records = [
        EvaluationRecord(
            sample_id="PID_02",
            source_path="fake/path/pid2.jpg",
            dataset="engineering_drawings",
            category="PID",
            task="drawing",
            success=True,
            processing_time_ms=200_000.0,
            run_mode="quick",
            max_image_size=192,
            max_new_tokens=24,
        ),
        EvaluationRecord(
            sample_id="PFD_01",
            source_path="fake/path/pfd1.png",
            dataset="engineering_drawings",
            category="PFD",
            task="drawing",
            success=True,
            processing_time_ms=300_000.0,
            run_mode="quick",
            max_image_size=192,
            max_new_tokens=24,
        ),
    ]

    all_records = full_records + quick_records
    summary = harness._generate_and_save_artifacts(
        records=all_records,
        total_time_s=100.0,
        target_samples=list(REPRESENTATIVE_SAMPLES[:2]),
    )

    lat = summary["latency_summary"]
    # Should reflect ONLY the 2 quick-mode records: (200,000 + 300,000) / 2 = 250,000.0 ms
    assert lat["average_latency_ms"] == 250_000.0
    assert lat["minimum_latency_ms"] == 200_000.0
    assert lat["maximum_latency_ms"] == 300_000.0
    assert lat["matching_signature_inferences_count"] == 2
    assert lat["confidence_caveat"] is None  # count >= 2, confident

    # Full run projection must use 250,000 ms
    proj = lat["full_run_projection"]
    assert proj["avg_latency_ms_per_task_basis"] == 250_000.0

    # Test single-record case triggers low confidence caveat
    summary_low_conf = harness._generate_and_save_artifacts(
        records=full_records + [quick_records[0]],
        total_time_s=50.0,
        target_samples=list(REPRESENTATIVE_SAMPLES[:2]),
    )
    lat_low = summary_low_conf["latency_summary"]
    assert lat_low["average_latency_ms"] == 200_000.0
    assert lat_low["matching_signature_inferences_count"] == 1
    assert "low confidence — based on 1 sample(s) at these settings" in lat_low["confidence_caveat"]
    assert "low confidence" in lat_low["full_run_projection"]["confidence_caveat"]


def test_category_maps_are_consistent() -> None:
    """Verify CATEGORY_ALIASES, PRIMARY_TASK_MAP, CATEGORY_DISPLAY_NAMES match MANIFEST_CATEGORIES (Bug 2 Fix)."""
    assert frozenset(CATEGORY_ALIASES.keys()) == MANIFEST_CATEGORIES
    assert frozenset(PRIMARY_TASK_MAP.keys()) == MANIFEST_CATEGORIES
    assert frozenset(CATEGORY_DISPLAY_NAMES.keys()) == MANIFEST_CATEGORIES

    # Every canonical display name is human-typeable and round-trips through filter_samples_by_category
    for canonical, display_name in CATEGORY_DISPLAY_NAMES.items():
        matched = filter_samples_by_category(REPRESENTATIVE_SAMPLES, [display_name])
        assert len(matched) > 0, f"Display name {display_name!r} failed to resolve to samples"
        assert all(s.category == canonical for s in matched)


def test_coverage_gap_names_match_cli_filter(tmp_path: Path) -> None:
    """Assert for each name in a simulated coverage-gap list, --category <that exact name> works (Bug 2 Fix)."""
    harness = VisionEvaluationHarness(output_dir=tmp_path, is_mock=True)
    # Simulate a run with only PID samples completed
    pid_samples = [s for s in REPRESENTATIVE_SAMPLES if s.category == "PID"]
    summary = harness.evaluate(samples=pid_samples, limit=2)

    gaps = summary["coverage_summary"]["categories_with_zero_real_model_coverage"]
    assert len(gaps) == 9  # 9 remaining gap categories

    # Assert every gap name is a canonical display name and successfully filters samples
    for gap_name in gaps:
        filtered = filter_samples_by_category(REPRESENTATIVE_SAMPLES, [gap_name])
        assert len(filtered) > 0, f"Simulated gap category {gap_name!r} failed to filter samples!"


def test_next_command_hint_is_valid(tmp_path: Path) -> None:
    """Assert generated hint string parses through arg parser producing a valid --category without placeholders (Bug 3 Fix)."""
    # Test partial coverage with 4 remaining categories
    uncovered = ["Equipment_Drawings", "Instrumentation", "Pump_Diagrams", "Electrical"]
    hint_str = format_next_run_hint(uncovered, run_mode="quick")

    assert "<" not in hint_str and ">" not in hint_str, f"Found placeholder in hint: {hint_str}"
    assert hint_str.startswith("python evaluate_vision.py")

    # Split using shlex
    tokens = shlex.split(hint_str)
    # Strip 'python' and 'evaluate_vision.py'
    cli_tokens = tokens[2:]
    args = parse_args(cli_tokens)

    assert args.category is not None
    assert len(args.category.strip()) > 0
    assert args.quick is True
    assert args.resume is True

    # Validate the suggested categories resolve to samples
    cat_names = [c.strip() for c in args.category.split(",") if c.strip()]
    matched = filter_samples_by_category(REPRESENTATIVE_SAMPLES, cat_names)
    assert len(matched) > 0
    # Next chunk for these uncovered categories is Equipment_Drawings,Instrumentation
    assert set(cat_names) == {"Equipment_Drawings", "Instrumentation"}



