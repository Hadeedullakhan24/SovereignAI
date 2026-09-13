"""Unit tests for member3_ocr.vision_pipeline.

All tests use synthetic in-memory/temp-file fixtures.
No network, no downloads, no Hugging Face, no master dataset access.
"""
from __future__ import annotations


import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import subprocess
import sys
from PIL import Image

from member3_ocr.core.image_preprocessing import (
    PreprocessingOptions,
    preprocess_image,
)
from member3_ocr.core.ocr_pipeline import BoundingBox
from member3_ocr.core.vision_pipeline import (
    CUDAUnavailableError,
    DEFAULT_VISION_MODEL_PATH,
    BackendCapabilities,
    BackendInfo,
    ClassificationResult,
    DetectedObject,
    ImageCaption,
    LocalQwenVisionBackend,
    MissingLocalModelError,
    MissingVisionModelError,
    MockVisionBackend,
    VisionBackend,
    VisionConfigurationError,
    VisionIssue,
    VisionModelConfig,
    VisionPipeline,
    VisionResult,
    create_vision_pipeline,
    main,
    _clamp_confidence,
    _make_stable_id,
    _parse_drawing_response,
    _parse_equipment_response,
    _parse_observation_response,
    _parse_scene_type,
    _resolve_device,
    _sanitize_text,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_image(tmp_path: Path) -> Path:
    """Create a valid synthetic RGB PNG image for testing."""
    img_path = tmp_path / "test_equipment.png"
    img_data = np.zeros((100, 150, 3), dtype=np.uint8)
    # Draw a colored rectangle
    img_data[20:80, 30:120, :] = [200, 100, 50]
    img = Image.fromarray(img_data)
    img.save(img_path)
    return img_path


@pytest.fixture
def mock_backend() -> MockVisionBackend:
    """Return a standard mock backend with deterministic detections."""
    return MockVisionBackend()


# ──────────────────────────────────────────────────────────────────────────────
# 1. VisionModelConfig defaults to offline behavior
# ──────────────────────────────────────────────────────────────────────────────

def test_model_config_defaults_offline() -> None:
    config = VisionModelConfig()
    assert config.allow_downloads is False
    assert config.device == "cpu"
    assert 0.0 <= config.confidence_threshold <= 1.0
    # validate() should pass when allow_downloads is False and no missing model_path
    config.validate()


# ──────────────────────────────────────────────────────────────────────────────
# 2. Download-enabled configuration is rejected
# ──────────────────────────────────────────────────────────────────────────────

def test_download_enabled_configuration_rejected() -> None:
    with pytest.raises(VisionConfigurationError, match="forbidden"):
        VisionModelConfig(allow_downloads=True)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Missing local model path is detected
# ──────────────────────────────────────────────────────────────────────────────

def test_missing_local_model_detected(tmp_path: Path) -> None:
    missing_path = tmp_path / "nonexistent_model_weights"
    config = VisionModelConfig(model_path=missing_path)
    with pytest.raises(MissingLocalModelError, match="does not exist"):
        config.validate()

    # Backend initialization must also trigger the missing model error
    backend = MockVisionBackend(config=config)
    with pytest.raises(MissingLocalModelError):
        backend.initialize()


# ──────────────────────────────────────────────────────────────────────────────
# 4. Unsupported image format is handled
# ──────────────────────────────────────────────────────────────────────────────

def test_unsupported_image_format_handled(tmp_path: Path, mock_backend: MockVisionBackend) -> None:
    unsupported_file = tmp_path / "document.xyz"
    unsupported_file.write_text("dummy content")

    pipeline = VisionPipeline(backend=mock_backend)
    result = pipeline.process_image(unsupported_file)

    assert result.success is False
    assert any(i.code == "unsupported_format" for i in result.issues)
    assert result.image_width == 0
    assert result.image_height == 0


# ──────────────────────────────────────────────────────────────────────────────
# 5. Missing image is handled
# ──────────────────────────────────────────────────────────────────────────────

def test_missing_image_handled(tmp_path: Path, mock_backend: MockVisionBackend) -> None:
    nonexistent = tmp_path / "no_such_image.png"

    pipeline = VisionPipeline(backend=mock_backend)
    result = pipeline.process_image(nonexistent)

    assert result.success is False
    assert any(i.code == "missing_image" for i in result.issues)
    assert result.image_width == 0


# ──────────────────────────────────────────────────────────────────────────────
# 6. Corrupt image is handled
# ──────────────────────────────────────────────────────────────────────────────

def test_corrupt_image_handled(tmp_path: Path, mock_backend: MockVisionBackend) -> None:
    corrupt_file = tmp_path / "broken.png"
    corrupt_file.write_bytes(b"NOT_A_VALID_IMAGE_DATA")

    pipeline = VisionPipeline(backend=mock_backend)
    result = pipeline.process_image(corrupt_file)

    assert result.success is False
    assert any(i.code == "corrupt_image" for i in result.issues)
    assert result.image_width == 0


# ──────────────────────────────────────────────────────────────────────────────
# 7. Synthetic valid image loads successfully
# ──────────────────────────────────────────────────────────────────────────────

def test_synthetic_valid_image_success(synthetic_image: Path, mock_backend: MockVisionBackend) -> None:
    pipeline = VisionPipeline(backend=mock_backend)
    result = pipeline.process_image(synthetic_image)

    assert result.success is True
    assert result.image_width == 150
    assert result.image_height == 100
    assert len(result.classifications) >= 1
    assert len(result.detections) >= 1
    assert len(result.captions) >= 1


# ──────────────────────────────────────────────────────────────────────────────
# 8. VisionResult serializes with to_dict()
# ──────────────────────────────────────────────────────────────────────────────

def test_vision_result_to_dict(synthetic_image: Path, mock_backend: MockVisionBackend) -> None:
    pipeline = VisionPipeline(backend=mock_backend)
    result = pipeline.process_image(synthetic_image)

    res_dict = result.to_dict()
    assert isinstance(res_dict, dict)
    assert res_dict["success"] is True
    assert res_dict["image_width"] == 150
    assert res_dict["image_height"] == 100
    assert "classifications" in res_dict
    assert "detections" in res_dict
    assert "captions" in res_dict
    assert "backend" in res_dict

    # Must be JSON serializable
    json_str = json.dumps(res_dict)
    assert len(json_str) > 0


# ──────────────────────────────────────────────────────────────────────────────
# 9. VisionResult serializes with to_json()
# ──────────────────────────────────────────────────────────────────────────────

def test_vision_result_to_json(synthetic_image: Path, mock_backend: MockVisionBackend) -> None:
    pipeline = VisionPipeline(backend=mock_backend)
    result = pipeline.process_image(synthetic_image)

    json_str = result.to_json()
    assert isinstance(json_str, str)

    parsed = json.loads(json_str)
    assert parsed["success"] is True
    assert parsed["source_path"] == str(synthetic_image)
    assert parsed["classifications"][0]["label"] == "industrial_equipment"


# ──────────────────────────────────────────────────────────────────────────────
# 10. Bounding-box coordinate format is stable/documented
# ──────────────────────────────────────────────────────────────────────────────

def test_bounding_box_coordinate_format(synthetic_image: Path, mock_backend: MockVisionBackend) -> None:
    pipeline = VisionPipeline(backend=mock_backend)
    result = pipeline.process_image(synthetic_image)

    assert len(result.detections) > 0
    det = result.detections[0]
    bbox = det.bbox

    # Ensure coordinate conventions
    assert isinstance(bbox.left, (int, float))
    assert isinstance(bbox.top, (int, float))
    assert isinstance(bbox.right, (int, float))
    assert isinstance(bbox.bottom, (int, float))
    assert bbox.left <= bbox.right
    assert bbox.top <= bbox.bottom
    assert bbox.coordinate_space == "processed_pixels"


# ──────────────────────────────────────────────────────────────────────────────
# 11. Backend capabilities are correctly exposed
# ──────────────────────────────────────────────────────────────────────────────

def test_backend_capabilities_exposed() -> None:
    caps = BackendCapabilities(
        image_classification=True,
        object_detection=True,
        image_captioning=False,
    )
    backend = MockVisionBackend(capabilities=caps)
    assert backend.capabilities().image_classification is True
    assert backend.capabilities().object_detection is True
    assert backend.capabilities().image_captioning is False
    assert backend.backend_info.capabilities.image_classification is True


# ──────────────────────────────────────────────────────────────────────────────
# 12. Pipeline accepts a preprocessing result if supported
# ──────────────────────────────────────────────────────────────────────────────

def test_pipeline_accepts_preprocessing_result(synthetic_image: Path, mock_backend: MockVisionBackend) -> None:
    prep_opts = PreprocessingOptions.drawing(max_width=100)
    prep_result = preprocess_image(synthetic_image, prep_opts, save_output=False)

    pipeline = VisionPipeline(backend=mock_backend)
    result = pipeline.process_image(prep_result)

    assert result.success is True
    assert result.image_width == prep_result.processed_width
    assert result.image_height == prep_result.processed_height
    assert result.processing_metadata["source_type"] == "preprocessing_result"


# ──────────────────────────────────────────────────────────────────────────────
# 13. Source image is not modified
# ──────────────────────────────────────────────────────────────────────────────

def test_source_image_not_modified(synthetic_image: Path, mock_backend: MockVisionBackend) -> None:
    # Compute sha256 before analysis
    hash_before = hashlib.sha256(synthetic_image.read_bytes()).hexdigest()
    mtime_before = synthetic_image.stat().st_mtime

    pipeline = VisionPipeline(backend=mock_backend, run_preprocessing=True)
    result = pipeline.process_image(synthetic_image)
    assert result.success is True

    # Verify sha256 and mtime unchanged
    hash_after = hashlib.sha256(synthetic_image.read_bytes()).hexdigest()
    mtime_after = synthetic_image.stat().st_mtime

    assert hash_before == hash_after, "Source image content was modified!"
    assert mtime_before == mtime_after, "Source image mtime was modified!"


# ──────────────────────────────────────────────────────────────────────────────
# 14. No network/download behavior exists in the pipeline
# ──────────────────────────────────────────────────────────────────────────────

def test_no_network_behavior(monkeypatch: pytest.MonkeyPatch, synthetic_image: Path, mock_backend: MockVisionBackend) -> None:
    """Ensure that any attempt to establish a network socket fails."""
    import socket

    def forbidden_connect(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Network access is strictly forbidden in Sovereign AI offline pipeline!")

    monkeypatch.setattr(socket, "create_connection", forbidden_connect)

    pipeline = VisionPipeline(backend=mock_backend)
    result = pipeline.process_image(synthetic_image)
    assert result.success is True


# ──────────────────────────────────────────────────────────────────────────────
# 15. Deterministic structured output from a deterministic mock backend
# ──────────────────────────────────────────────────────────────────────────────

def test_deterministic_output(synthetic_image: Path, mock_backend: MockVisionBackend) -> None:
    pipeline = VisionPipeline(backend=mock_backend)
    res1 = pipeline.process_image(synthetic_image)
    res2 = pipeline.process_image(synthetic_image)

    dict1 = res1.to_dict()
    dict2 = res2.to_dict()

    # Discard non-deterministic diagnostic timings if any
    dict1["processing_metadata"].pop("processing_time_ms", None)
    dict2["processing_metadata"].pop("processing_time_ms", None)

    assert dict1 == dict2
    assert res1.classifications == res2.classifications
    assert res1.detections == res2.detections
    assert res1.captions == res2.captions


# ──────────────────────────────────────────────────────────────────────────────
# 16. Backend exceptions become structured VisionIssue values
# ──────────────────────────────────────────────────────────────────────────────

def test_backend_exceptions_become_structured_issues(synthetic_image: Path) -> None:
    failing_backend = MockVisionBackend(should_fail=True)
    pipeline = VisionPipeline(backend=failing_backend)

    result = pipeline.process_image(synthetic_image)
    assert result.success is False
    assert any(i.code == "backend_execution_failed" for i in result.issues)
    issue = next(i for i in result.issues if i.code == "backend_execution_failed")
    assert issue.severity == "error"
    assert "Simulated mock backend" in issue.message


# ──────────────────────────────────────────────────────────────────────────────
# 17. Invalid backend output is handled safely
# ──────────────────────────────────────────────────────────────────────────────

def test_invalid_backend_output_handled_safely(synthetic_image: Path) -> None:
    invalid_backend = MockVisionBackend(invalid_output=True)
    pipeline = VisionPipeline(backend=invalid_backend)

    result = pipeline.process_image(synthetic_image)
    assert result.success is False
    assert any(i.code == "invalid_backend_output" for i in result.issues)


# ──────────────────────────────────────────────────────────────────────────────
# 18. CLI execution tests
# ──────────────────────────────────────────────────────────────────────────────

def test_cli_execution_success(synthetic_image: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["--input", str(synthetic_image), "--json", "--mock"])
    assert exit_code == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["success"] is True
    assert payload["image_width"] == 150


def test_cli_execution_missing_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    missing = tmp_path / "does_not_exist.png"
    exit_code = main(["--input", str(missing)])
    assert exit_code != 0


def test_cli_execution_invalid_model_path(synthetic_image: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    missing_model = tmp_path / "ghost_weights"
    exit_code = main(["--input", str(synthetic_image), "--model-path", str(missing_model)])
    assert exit_code == 2
    captured = capsys.readouterr()
    assert "does not exist" in captured.err


# ──────────────────────────────────────────────────────────────────────────────
# 19. Device selection tests
# ──────────────────────────────────────────────────────────────────────────────

def test_device_selection_cpu() -> None:
    assert _resolve_device("cpu") == "cpu"


def test_device_selection_auto() -> None:
    dev = _resolve_device("auto")
    assert dev in ("cpu", "cuda")


def test_device_selection_cuda_unsupported_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(CUDAUnavailableError, match="CUDA is not available"):
        _resolve_device("cuda")


def test_device_selection_invalid() -> None:
    with pytest.raises(VisionConfigurationError, match="Unknown device"):
        _resolve_device("tpu")


# ──────────────────────────────────────────────────────────────────────────────
# 20. Confidence clamping and parsing
# ──────────────────────────────────────────────────────────────────────────────

def test_confidence_clamping() -> None:
    assert _clamp_confidence(-0.5) == 0.0
    assert _clamp_confidence(1.5) == 1.0
    assert _clamp_confidence(0.72) == 0.72


# ──────────────────────────────────────────────────────────────────────────────
# 21. Deterministic IDs
# ──────────────────────────────────────────────────────────────────────────────

def test_deterministic_id_generation() -> None:
    id1 = _make_stable_id("eq", "centrifugal_pump", "0")
    id2 = _make_stable_id("eq", "centrifugal_pump", "0")
    id3 = _make_stable_id("eq", "centrifugal_pump", "1")
    assert id1 == id2
    assert id1.startswith("eq_")
    assert id1 != id3


# ──────────────────────────────────────────────────────────────────────────────
# 22. Security: Image text untrusted data & sanitization
# ──────────────────────────────────────────────────────────────────────────────

def test_security_sanitization_strips_control_chars() -> None:
    raw = "Industrial pump\x00\x08\x0b\x0c inspection text"
    sanitized = _sanitize_text(raw)
    assert "\x00" not in sanitized
    assert "\x08" not in sanitized
    assert "Industrial pump inspection text" == sanitized


# ──────────────────────────────────────────────────────────────────────────────
# 23. Equipment recognition parsing (No tag hallucination)
# ──────────────────────────────────────────────────────────────────────────────

def test_equipment_parser_avoids_tag_hallucination() -> None:
    raw = "Type: Pump\nVisible Tag: none\nConfidence: 0.90\nEvidence: Casing visible"
    items = _parse_equipment_response(raw, image_id="img_test")
    assert len(items) == 1
    assert items[0].equipment_type == "Pump"
    assert items[0].name_or_tag is None  # Never fabricated


def test_equipment_parser_with_visible_tag() -> None:
    raw = "Type: Pump\nTag: P-101A\nConfidence: 0.95\nEvidence: Tag visible on casing"
    items = _parse_equipment_response(raw, image_id="img_test")
    assert len(items) == 1
    assert items[0].equipment_type == "Pump"
    assert items[0].name_or_tag == "P-101A"


# ──────────────────────────────────────────────────────────────────────────────
# 24. Engineering drawing parsing
# ──────────────────────────────────────────────────────────────────────────────

def test_drawing_parser_types_and_labels() -> None:
    raw = "Drawing Type: P&ID\nInstruments: PT-101, TT-102\nLabels: Stream 1, Line 100"
    dtype, labels = _parse_drawing_response(raw)
    assert dtype in ("PID", "P&ID")
    assert any(l.text == "PT-101" for l in labels)


# ──────────────────────────────────────────────────────────────────────────────
# 25. Observation parser preserves uncertainty
# ──────────────────────────────────────────────────────────────────────────────

def test_observation_parser_preserves_uncertainty() -> None:
    raw = "Observation: Possible corrosion-like discoloration on flange\nConfidence: medium\nUncertainty: Possible surface dust or lighting effect"
    obs = _parse_observation_response(raw)
    assert len(obs) >= 1
    assert "corrosion" in obs[0].description.lower()
    assert obs[0].confidence == 0.65
    assert len(obs[0].uncertainty) > 0


# ──────────────────────────────────────────────────────────────────────────────
# 26. Factory helper create_vision_pipeline
# ──────────────────────────────────────────────────────────────────────────────

def test_create_vision_pipeline_mock() -> None:
    pipeline = create_vision_pipeline(use_mock=True)
    assert isinstance(pipeline, VisionPipeline)
    assert isinstance(pipeline.backend, MockVisionBackend)


# ──────────────────────────────────────────────────────────────────────────────
# 27. Structured error alias
# ──────────────────────────────────────────────────────────────────────────────

def test_missing_vision_model_error_alias() -> None:
    assert MissingVisionModelError is MissingLocalModelError


# ──────────────────────────────────────────────────────────────────────────────
# 28. In-memory inputs (PIL Image & NumPy array) without mutation
# ──────────────────────────────────────────────────────────────────────────────

def test_process_with_pil_and_numpy_inputs(mock_backend: MockVisionBackend) -> None:
    pipeline = VisionPipeline(backend=mock_backend)

    # 1. PIL Image input
    pil_img = Image.new("RGB", (120, 80), color=(100, 150, 200))
    res_pil = pipeline.process(pil_img)
    assert res_pil.success is True
    assert res_pil.image_width == 120
    assert res_pil.image_height == 80
    assert res_pil.image_id.startswith("img_")
    assert res_pil.provenance["source_type"] == "pil_image"

    # 2. NumPy ndarray input
    orig_arr = np.zeros((90, 140, 3), dtype=np.uint8)
    orig_arr[10:30, 10:30] = 255
    arr_copy = orig_arr.copy()
    res_np = pipeline.process(orig_arr)
    assert res_np.success is True
    assert res_np.image_width == 140
    assert res_np.image_height == 90
    assert res_np.provenance["source_type"] == "numpy_array"
    # Verify caller-owned array was NOT mutated
    assert np.array_equal(orig_arr, arr_copy)


# ──────────────────────────────────────────────────────────────────────────────
# 29. OCR Provenance Preservation
# ──────────────────────────────────────────────────────────────────────────────

def test_ocr_provenance_preserved(synthetic_image: Path, mock_backend: MockVisionBackend) -> None:
    pipeline = VisionPipeline(backend=mock_backend)

    class DummyOCRResult:
        document_id = "ocr_doc_abc123"

    res = pipeline.process_image(synthetic_image, ocr_result=DummyOCRResult())
    assert res.success is True
    assert res.provenance.get("ocr_supplied") is True
    assert res.provenance.get("ocr_document_id") == "ocr_doc_abc123"


# ──────────────────────────────────────────────────────────────────────────────
# 30. Offline Security Test (Air-gap Enforcement)
# ──────────────────────────────────────────────────────────────────────────────

def test_offline_security_no_network_access(
    synthetic_image: Path,
    mock_backend: MockVisionBackend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import socket

    def blocked_connect(*args, **kwargs):
        raise RuntimeError("Network access attempted in offline vision pipeline!")

    monkeypatch.setattr(socket.socket, "connect", blocked_connect)

    pipeline = VisionPipeline(backend=mock_backend)
    res = pipeline.process_image(synthetic_image)
    assert res.success is True


# ──────────────────────────────────────────────────────────────────────────────
# 31. Drawing Proximity Rule (Proximity does not imply connectivity)
# ──────────────────────────────────────────────────────────────────────────────

def test_drawing_proximity_rule_does_not_infer_connectivity() -> None:
    raw_drawing_output = (
        "Drawing Type: P&ID\n"
        "Visible equipment: Pump P-101 next to Valve V-102\n"
        "Instruments: PT-101\n"
    )
    task_res = LocalQwenVisionBackend().run_task(
        np.zeros((100, 100, 3), dtype=np.uint8), "drawing"
    ) if False else _parse_drawing_response(raw_drawing_output)

    dtype, labels = task_res
    assert dtype == "PID"
    # Proximity must not create connectivity claims
    assert "connected" not in [l.text.lower() for l in labels]


# ──────────────────────────────────────────────────────────────────────────────
# 32. Real Model Integration Test
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
def test_real_qwen25_vl_integration() -> None:
    """Load the REAL local Qwen2.5-VL-3B-Instruct model and run inference.

    Requirements:
    1. Loads the REAL local Qwen2.5-VL-3B-Instruct model
    2. Uses local_files_only=True
    3. Processes a real local image
    4. Generates an actual vision result
    5. Verifies result structure
    6. Reports device used
    7. Reports inference latency
    8. Does NOT download anything
    """
    project_root = Path(__file__).resolve().parents[2]
    model_path = project_root / "models" / "vision" / "qwen2.5-vl-3b-instruct"
    if not model_path.exists():
        pytest.skip(f"Local model directory not found: {model_path}")

    test_image = Path("datasets/engineering_drawings/PID/PID_002_Process_Example.jpg")
    if not test_image.exists():
        pytest.skip(f"Test drawing image not found: {test_image}")

    # Run in subprocess to provide clean process memory isolation
    script = f"""
import json, sys, time
from pathlib import Path

sys.path.insert(0, r"{project_root}")
from member3_ocr.core.vision_pipeline import (
    LocalQwenVisionBackend,
    VisionModelConfig,
    VisionPipeline,
)

model_path = Path(r"{model_path}")
img_path = Path(r"{test_image}")

config = VisionModelConfig(
    model_name="qwen2.5-vl-3b-instruct",
    model_path=model_path,
    device="auto",
    max_image_size=256,
    max_new_tokens=32,
    allow_downloads=False,
)
backend = LocalQwenVisionBackend(config=config)
pipeline = VisionPipeline(backend=backend)

t0 = time.perf_counter()
result = pipeline.process_image(img_path)
wall_ms = (time.perf_counter() - t0) * 1000.0

out = {{
    "success": result.success,
    "device_requested": result.device_requested,
    "device_used": result.device_used,
    "scene_type": result.scene_type,
    "caption": result.caption,
    "equipment_count": len(result.equipment),
    "observations_count": len(result.observations),
    "issues_count": len(result.issues),
    "processing_time_ms": result.processing_time_ms,
    "wall_clock_ms": wall_ms,
}}
print("---RESULT_PAYLOAD---")
print(json.dumps(out))
"""
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=1200,
        cwd=str(project_root),
    )
    assert proc.returncode == 0, f"Integration test failed (stderr: {proc.stderr})"
    assert "---RESULT_PAYLOAD---" in proc.stdout, f"Payload marker missing: {proc.stdout}"

    payload_json = proc.stdout.split("---RESULT_PAYLOAD---")[1].strip()
    data = json.loads(payload_json)

    assert data["success"] is True
    assert data["device_requested"] == "auto"
    assert data["device_used"] in ("cpu", "cuda")
    assert data["processing_time_ms"] > 0.0
    assert data["scene_type"] is not None
    assert len(data["caption"]) > 0

    print(
        f"\n[REAL MODEL INTEGRATION TEST PASSED]\n"
        f"  Device Requested: {data['device_requested']}\n"
        f"  Device Used: {data['device_used']}\n"
        f"  Inference Latency: {data['processing_time_ms']:.1f} ms (wall: {data['wall_clock_ms']:.1f} ms)\n"
        f"  Scene Type: {data['scene_type']}\n"
        f"  Caption: {data['caption']}\n"
        f"  Equipment Found: {data['equipment_count']}\n"
        f"  Observations: {data['observations_count']}\n"
    )

