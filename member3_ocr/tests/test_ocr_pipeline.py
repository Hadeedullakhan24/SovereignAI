"""Unit tests for the backend-neutral OCR pipeline; no model downloads required.

Covers all 23 verification requirements for member3_ocr.ocr_pipeline:
1. OCRModelConfig validates correctly.
2. Local model path is required.
3. Missing local model is detected.
4. allow_downloads=True is rejected.
5. Invalid image is handled.
6. Valid preprocessing result can be accepted.
7. OCRDocumentResult structure is valid.
8. OCRPageResult structure is valid.
9. TextBlock normalization works.
10. TextLine normalization works.
11. Word normalization works.
12. Bounding boxes use: [left, top, right, bottom].
13. Coordinate space is: processed_pixels.
14. Confidence values are preserved.
15. Multiple OCR blocks are normalized correctly.
16. Legacy/raw backend output does not leak outside adapter.
17. Backend failure becomes structured Issue.
18. JSON serialization works.
19. to_dict() works.
20. Deterministic mock backend gives deterministic output.
21. No UUID/random identifiers.
22. No model download.
23. No network access.
"""
from __future__ import annotations


import json
import re
import socket
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from member3_ocr.core.image_preprocessing import PreprocessingOptions, preprocess_image
from member3_ocr.core.ocr_pipeline import (
    BackendCapabilities,
    BackendInfo,
    BackendRecognition,
    BoundingBox,
    Issue,
    MissingLocalModelError,
    OCRBackendExecutionError,
    OCRConfigurationError,
    OCRDocumentResult,
    OCRPageResult,
    OCRPipeline,
    PaddleOCRBackend,
    PaddleOCRModelConfig,
    SCHEMA_VERSION,
    Table,
    TextBlock,
    TextLine,
    Word,
    _bbox_from_polygon,
    _normalize_confidence,
    _normalize_polygon,
)


class DeterministicMockBackend:
    """Mock OCR backend returning deterministic, predictable output."""

    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.initialized = False

    @property
    def backend_info(self) -> BackendInfo:
        return BackendInfo(
            name="mock_paddle",
            version="1.0",
            model_ids=("local/det", "local/rec"),
            device="cpu",
            capabilities=BackendCapabilities(),
        )

    def initialize(self) -> None:
        self.initialized = True

    def recognize(self, image: np.ndarray) -> BackendRecognition:
        if self.fail:
            raise OCRBackendExecutionError("synthetic backend failure")

        # Three deterministic blocks with distinct geometry
        b1 = BoundingBox(10.0, 10.0, 200.0, 30.0)
        w1 = Word("Equipment", 0.95, b1)
        l1 = TextLine("Equipment", 0.95, b1, (w1,))
        tb1 = TextBlock("text-1", "Equipment", 0.95, b1, (l1,))

        b2 = BoundingBox(10.0, 35.0, 150.0, 55.0)
        w2 = Word("Tag:", 0.92, b2)
        l2 = TextLine("Tag:", 0.92, b2, (w2,))
        tb2 = TextBlock("text-2", "Tag:", 0.92, b2, (l2,))

        b3 = BoundingBox(160.0, 35.0, 250.0, 55.0)
        w3 = Word("P-203", 0.98, b3)
        l3 = TextLine("P-203", 0.98, b3, (w3,))
        tb3 = TextBlock("text-3", "P-203", 0.98, b3, (l3,))

        return BackendRecognition(
            blocks=(tb1, tb2, tb3),
            warnings=(Issue("mock_info", "Deterministic mock recognition executed", severity="info"),),
        )


def _write_test_image(path: Path, width: int = 300, height: int = 150) -> None:
    arr = np.full((height, width, 3), 255, dtype=np.uint8)
    arr[20:40, 20:100] = 0
    Image.fromarray(arr).save(path)


# 1. OCRModelConfig validates correctly
def test_ocr_model_config_validates_correctly(tmp_path: Path) -> None:
    det_dir = tmp_path / "det_model"
    rec_dir = tmp_path / "rec_model"
    det_dir.mkdir()
    rec_dir.mkdir()

    config = PaddleOCRModelConfig(
        detection_model_dir=det_dir,
        recognition_model_dir=rec_dir,
        language="en",
        device="cpu",
    )
    # Should validate without raising an exception
    config.validate()
    assert config.detection_model_dir == det_dir
    assert config.recognition_model_dir == rec_dir
    assert not config.allow_model_download


# 2. Local model path is required
def test_local_model_path_is_required(tmp_path: Path) -> None:
    config = PaddleOCRModelConfig(
        detection_model_dir=tmp_path / "non_existent_det",
        recognition_model_dir=tmp_path / "non_existent_rec",
    )
    with pytest.raises(MissingLocalModelError):
        config.validate()


# 3. Missing local model is detected
def test_missing_local_model_is_detected(tmp_path: Path) -> None:
    det_dir = tmp_path / "existing_det"
    det_dir.mkdir()
    rec_dir = tmp_path / "missing_rec"

    config = PaddleOCRModelConfig(
        detection_model_dir=det_dir,
        recognition_model_dir=rec_dir,
    )
    with pytest.raises(MissingLocalModelError) as exc_info:
        config.validate()
    assert "Missing required local PaddleOCR model directory" in str(exc_info.value)
    assert "Automatic model download is disabled" in str(exc_info.value)


# 4. allow_downloads=True is rejected
def test_allow_downloads_true_is_rejected(tmp_path: Path) -> None:
    det_dir = tmp_path / "det"
    rec_dir = tmp_path / "rec"
    det_dir.mkdir()
    rec_dir.mkdir()

    config = PaddleOCRModelConfig(
        detection_model_dir=det_dir,
        recognition_model_dir=rec_dir,
        allow_model_download=True,
    )
    with pytest.raises(OCRConfigurationError, match="Model download is forbidden"):
        config.validate()


# 5. Invalid image is handled
def test_invalid_image_is_handled(tmp_path: Path) -> None:
    missing_image = tmp_path / "does_not_exist.png"
    pipeline = OCRPipeline(DeterministicMockBackend())
    result = pipeline.process_image(missing_image)

    assert len(result.pages) == 0
    assert len(result.errors) > 0
    assert result.errors[0].code == "invalid_image"
    assert result.errors[0].severity == "error"
    assert result.errors[0].recoverable is True


# 6. Valid preprocessing result can be accepted
def test_valid_preprocessing_result_accepted(tmp_path: Path) -> None:
    img_path = tmp_path / "valid.png"
    _write_test_image(img_path)

    prep_result = preprocess_image(img_path, PreprocessingOptions(grayscale=True))
    pipeline = OCRPipeline(DeterministicMockBackend())
    result = pipeline.process_image(prep_result, document_id="doc_from_prep", page_number=2)

    assert result.document_id == "doc_from_prep"
    assert len(result.pages) == 1
    assert result.pages[0].page_number == 2
    assert result.provenance["source_type"] == "preprocessing_result"
    assert "grayscale" in result.pages[0].preprocessing_operations


# 7. OCRDocumentResult structure is valid
def test_ocr_document_result_structure_valid() -> None:
    backend_info = BackendInfo("test", "1.0", ("m1",), "cpu", BackendCapabilities())
    page = OCRPageResult(
        page_number=1,
        original_width=100,
        original_height=100,
        processed_width=100,
        processed_height=100,
        text="Hello",
    )
    doc = OCRDocumentResult(
        document_id="doc-123",
        pages=(page,),
        backend=backend_info,
        provenance={"coordinate_space": "processed_pixels"},
    )
    assert doc.document_id == "doc-123"
    assert doc.schema_version == SCHEMA_VERSION
    assert len(doc.pages) == 1
    assert doc.backend.name == "test"
    assert isinstance(doc.total_processing_time_ms, float)
    assert doc.warnings == ()
    assert doc.errors == ()


# 8. OCRPageResult structure is valid
def test_ocr_page_result_structure_valid() -> None:
    box = BoundingBox(10, 10, 50, 20)
    word = Word("Test", 0.99, box)
    line = TextLine("Test", 0.99, box, (word,))
    block = TextBlock("text-1", "Test", 0.99, box, (line,))

    page = OCRPageResult(
        page_number=1,
        original_width=640,
        original_height=480,
        processed_width=640,
        processed_height=480,
        text="Test",
        blocks=(block,),
        tables=(),
        key_value_fields=(),
        preprocessing_operations=("grayscale",),
        preprocessing_time_ms=5.2,
        processing_time_ms=12.4,
    )
    assert page.page_number == 1
    assert page.original_width == 640
    assert page.original_height == 480
    assert page.processed_width == 640
    assert page.processed_height == 480
    assert page.text == "Test"
    assert len(page.blocks) == 1
    assert page.preprocessing_operations == ("grayscale",)
    assert page.preprocessing_time_ms == 5.2
    assert page.processing_time_ms == 12.4


# 9. TextBlock normalization works
def test_text_block_normalization_works() -> None:
    raw_v3 = [{
        "rec_polys": [[[10, 10], [50, 10], [50, 30], [10, 30]]],
        "rec_texts": ["Sample Text"],
        "rec_scores": [0.94],
    }]
    blocks = PaddleOCRBackend.normalize_raw_output(raw_v3)
    assert len(blocks) == 1
    block = blocks[0]
    assert block.id == "text-1"
    assert block.text == "Sample Text"
    assert block.confidence == pytest.approx(0.94)
    assert block.block_type == "text"
    assert block.bbox.left == 10.0
    assert block.bbox.top == 10.0
    assert block.bbox.right == 50.0
    assert block.bbox.bottom == 30.0


# 10. TextLine normalization works
def test_text_line_normalization_works() -> None:
    raw_v2 = [[
        [[[10, 20], [60, 20], [60, 40], [10, 40]], ("Line text", 0.88)],
    ]]
    blocks = PaddleOCRBackend.normalize_raw_output(raw_v2)
    assert len(blocks) == 1
    lines = blocks[0].lines
    assert len(lines) == 1
    assert lines[0].text == "Line text"
    assert lines[0].confidence == pytest.approx(0.88)
    assert lines[0].bbox.top == 20.0
    assert len(lines[0].words) == 1


# 11. Word normalization works
def test_word_normalization_works() -> None:
    raw_v2 = [[
        [[[15, 25], [75, 25], [75, 45], [15, 45]], ("WordText", 0.92)],
    ]]
    blocks = PaddleOCRBackend.normalize_raw_output(raw_v2)
    word = blocks[0].lines[0].words[0]
    assert word.text == "WordText"
    assert word.confidence == pytest.approx(0.92)
    assert word.bbox.left == 15.0
    assert word.bbox.top == 25.0
    assert word.bbox.right == 75.0
    assert word.bbox.bottom == 45.0
    assert word.polygon is not None
    assert len(word.polygon) == 4


# 12. Bounding boxes use: [left, top, right, bottom]
def test_bounding_boxes_use_left_top_right_bottom() -> None:
    poly = ((10.0, 20.0), (80.0, 20.0), (80.0, 50.0), (10.0, 50.0))
    bbox = _bbox_from_polygon(poly)
    assert bbox.left == 10.0
    assert bbox.top == 20.0
    assert bbox.right == 80.0
    assert bbox.bottom == 50.0
    assert bbox.left <= bbox.right
    assert bbox.top <= bbox.bottom


# 13. Coordinate space is: processed_pixels
def test_coordinate_space_is_processed_pixels() -> None:
    poly = ((5.0, 5.0), (25.0, 5.0), (25.0, 25.0), (5.0, 25.0))
    bbox = _bbox_from_polygon(poly)
    assert bbox.coordinate_space == "processed_pixels"

    raw = [[
        [[[10, 10], [50, 10], [50, 30], [10, 30]], ("Check", 0.9)],
    ]]
    blocks = PaddleOCRBackend.normalize_raw_output(raw)
    assert blocks[0].bbox.coordinate_space == "processed_pixels"
    assert blocks[0].lines[0].bbox.coordinate_space == "processed_pixels"
    assert blocks[0].lines[0].words[0].bbox.coordinate_space == "processed_pixels"


# 14. Confidence values are preserved
def test_confidence_values_are_preserved() -> None:
    # 0.0 - 1.0 range preserved
    assert _normalize_confidence(0.85) == pytest.approx(0.85)
    # Percentage (>1) scaled to 0-1
    assert _normalize_confidence(95.0) == pytest.approx(0.95)
    # None / non-finite handling
    assert _normalize_confidence(None) is None
    assert _normalize_confidence(float("nan")) is None
    assert _normalize_confidence(float("inf")) is None
    # Clamping negative
    assert _normalize_confidence(-0.5) == 0.0


# 15. Multiple OCR blocks are normalized correctly
def test_multiple_ocr_blocks_normalized_correctly() -> None:
    # 3 blocks in unordered positions; verify reading order sort (top-to-bottom, left-to-right)
    raw = [[
        [[[10, 100], [80, 100], [80, 120], [10, 120]], ("Bottom Line", 0.9)],
        [[[100, 10], [180, 10], [180, 30], [100, 30]], ("Top Right", 0.9)],
        [[[10, 10], [80, 10], [80, 30], [10, 30]], ("Top Left", 0.9)],
    ]]
    blocks = PaddleOCRBackend.normalize_raw_output(raw)
    texts = [b.text for b in blocks]
    assert texts == ["Top Left", "Top Right", "Bottom Line"]


# 16. Legacy/raw backend output does not leak outside adapter
def test_legacy_raw_backend_output_does_not_leak_outside_adapter(tmp_path: Path) -> None:
    img_path = tmp_path / "test.png"
    _write_test_image(img_path)

    pipeline = OCRPipeline(DeterministicMockBackend())
    result = pipeline.process_image(img_path)
    d = result.to_dict()

    # Verify no raw objects or paddle references leaked
    d_str = json.dumps(d)
    assert "PaddleOCR" not in d_str
    assert "ndarray" not in d_str
    assert isinstance(d["pages"], list)
    assert isinstance(d["pages"][0]["blocks"], list)


# 17. Backend failure becomes structured Issue
def test_backend_failure_becomes_structured_issue(tmp_path: Path) -> None:
    img_path = tmp_path / "test.png"
    _write_test_image(img_path)

    pipeline = OCRPipeline(DeterministicMockBackend(fail=True))
    result = pipeline.process_image(img_path)

    assert len(result.errors) > 0
    assert result.errors[0].code == "backend_execution_failed"
    assert result.errors[0].severity == "error"
    assert "synthetic backend failure" in result.errors[0].message

    assert len(result.pages) == 1
    assert result.pages[0].errors[0].code == "backend_execution_failed"
    assert result.pages[0].blocks == ()
    assert result.pages[0].tables == ()
    assert result.pages[0].key_value_fields == ()


# 18. JSON serialization works
def test_json_serialization_works(tmp_path: Path) -> None:
    img_path = tmp_path / "test.png"
    _write_test_image(img_path)

    pipeline = OCRPipeline(DeterministicMockBackend())
    result = pipeline.process_image(img_path)
    json_text = result.to_json(indent=2)

    assert isinstance(json_text, str)
    parsed = json.loads(json_text)
    assert parsed["document_id"] == "test"
    assert parsed["schema_version"] == "1.0"
    assert len(parsed["pages"]) == 1
    assert len(parsed["pages"][0]["blocks"]) == 3


# 19. to_dict() works
def test_to_dict_works(tmp_path: Path) -> None:
    img_path = tmp_path / "test.png"
    _write_test_image(img_path)

    pipeline = OCRPipeline(DeterministicMockBackend())
    result = pipeline.process_image(img_path)
    d = result.to_dict()

    assert isinstance(d, dict)
    assert d["document_id"] == "test"
    assert d["pages"][0]["blocks"][0]["id"] == "text-1"
    assert d["pages"][0]["blocks"][0]["lines"][0]["words"][0]["text"] == "Equipment"


# 20. Deterministic mock backend gives deterministic output
def test_deterministic_mock_backend_gives_deterministic_output(tmp_path: Path) -> None:
    img_path = tmp_path / "test.png"
    _write_test_image(img_path)

    pipeline = OCRPipeline(DeterministicMockBackend())
    res1 = pipeline.process_image(img_path, document_id="fixed-id")
    res2 = pipeline.process_image(img_path, document_id="fixed-id")

    dict1 = res1.to_dict()
    dict2 = res2.to_dict()

    # Total processing time can vary slightly by milliseconds, exclude it for deterministic comparison
    dict1.pop("total_processing_time_ms", None)
    dict2.pop("total_processing_time_ms", None)
    dict1["pages"][0].pop("processing_time_ms", None)
    dict2["pages"][0].pop("processing_time_ms", None)
    dict1["pages"][0].pop("preprocessing_time_ms", None)
    dict2["pages"][0].pop("preprocessing_time_ms", None)

    assert dict1 == dict2


# 21. No UUID/random identifiers
def test_no_uuid_or_random_identifiers(tmp_path: Path) -> None:
    img_path = tmp_path / "test.png"
    _write_test_image(img_path)

    pipeline = OCRPipeline(DeterministicMockBackend())
    result = pipeline.process_image(img_path, document_id="my_doc")

    uuid_pattern = re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
        re.IGNORECASE,
    )

    assert not uuid_pattern.match(result.document_id)
    assert result.document_id == "my_doc"

    for page in result.pages:
        for block in page.blocks:
            assert not uuid_pattern.match(block.id)
            assert block.id.startswith("text-")


# 22. No model download
def test_no_model_download_enforced(tmp_path: Path) -> None:
    det_dir = tmp_path / "det"
    rec_dir = tmp_path / "rec"
    det_dir.mkdir()
    rec_dir.mkdir()

    class MockEngine:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def predict(self, image: np.ndarray) -> Any:
            return iter([{"rec_polys": [], "rec_texts": [], "rec_scores": []}])

    config = PaddleOCRModelConfig(det_dir, rec_dir, allow_model_download=False)
    backend = PaddleOCRBackend(config, engine_factory=MockEngine)
    backend.initialize()

    assert backend.config.allow_model_download is False
    assert backend._engine.kwargs["text_detection_model_dir"] == str(det_dir)
    assert backend._engine.kwargs["text_recognition_model_dir"] == str(rec_dir)


# 23. No network access
def test_no_network_access(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _blocked_socket(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Network socket call attempted in offline test!")

    monkeypatch.setattr(socket, "socket", _blocked_socket)

    img_path = tmp_path / "offline_test.png"
    _write_test_image(img_path)

    pipeline = OCRPipeline(DeterministicMockBackend())
    result = pipeline.process_image(img_path)
    assert len(result.pages) == 1
    assert "Equipment" in result.pages[0].text


# 24. Regression: 2-D grayscale uint8 image normalized to 3-channel for PaddleOCR backend
def test_2d_grayscale_image_normalized_to_3channel_for_backend(tmp_path: Path) -> None:
    det_dir = tmp_path / "det"
    rec_dir = tmp_path / "rec"
    det_dir.mkdir()
    rec_dir.mkdir()

    class ShapeRecordingEngine:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.received_shape: tuple[int, ...] | None = None
            self.received_dtype: np.dtype | None = None

        def predict(self, image: np.ndarray) -> Any:
            self.received_shape = image.shape
            self.received_dtype = image.dtype
            return iter([{"rec_polys": [], "rec_texts": [], "rec_scores": []}])

    config = PaddleOCRModelConfig(det_dir, rec_dir, allow_model_download=False)
    backend = PaddleOCRBackend(config, engine_factory=ShapeRecordingEngine)
    backend.initialize()

    # 2-D uint8 grayscale image
    img_2d = np.full((120, 180), 200, dtype=np.uint8)
    res = backend.recognize(img_2d)

    assert res.blocks == ()
    engine = backend._engine
    assert engine.received_shape == (120, 180, 3)
    assert engine.received_dtype == np.uint8
    # Original image must not be mutated
    assert img_2d.shape == (120, 180)
    assert img_2d.ndim == 2


# 25. Regression: 3-channel image remains 3-channel
def test_3channel_image_remains_3channel(tmp_path: Path) -> None:
    det_dir = tmp_path / "det"
    rec_dir = tmp_path / "rec"
    det_dir.mkdir()
    rec_dir.mkdir()

    class ShapeRecordingEngine:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.received_shape: tuple[int, ...] | None = None

        def predict(self, image: np.ndarray) -> Any:
            self.received_shape = image.shape
            return iter([{"rec_polys": [], "rec_texts": [], "rec_scores": []}])

    config = PaddleOCRModelConfig(det_dir, rec_dir, allow_model_download=False)
    backend = PaddleOCRBackend(config, engine_factory=ShapeRecordingEngine)
    backend.initialize()

    img_3d = np.full((100, 150, 3), 128, dtype=np.uint8)
    res = backend.recognize(img_3d)

    assert res.blocks == ()
    assert backend._engine.received_shape == (100, 150, 3)


# 26. Regression: Unsupported dimensions and non-uint8 types are rejected clearly
def test_unsupported_dimensions_and_types_rejected(tmp_path: Path) -> None:
    det_dir = tmp_path / "det"
    rec_dir = tmp_path / "rec"
    det_dir.mkdir()
    rec_dir.mkdir()

    class DummyEngine:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def predict(self, image: np.ndarray) -> Any:
            return iter([])

    config = PaddleOCRModelConfig(det_dir, rec_dir, allow_model_download=False)
    backend = PaddleOCRBackend(config, engine_factory=DummyEngine)
    backend.initialize()

    # 1-D array
    with pytest.raises(OCRBackendExecutionError, match="Unsupported image dimensions"):
        backend.recognize(np.zeros(100, dtype=np.uint8))

    # 4-D array
    with pytest.raises(OCRBackendExecutionError, match="Unsupported image dimensions"):
        backend.recognize(np.zeros((1, 10, 10, 3), dtype=np.uint8))

    # 3-D array with 4 channels
    with pytest.raises(OCRBackendExecutionError, match="must have exactly 3 channels"):
        backend.recognize(np.zeros((10, 10, 4), dtype=np.uint8))

    # 2-D float array
    with pytest.raises(OCRBackendExecutionError, match="expected uint8"):
        backend.recognize(np.zeros((10, 10), dtype=np.float32))

    # Non-ndarray input
    with pytest.raises(OCRBackendExecutionError, match="must be a numpy ndarray"):
        backend.recognize("not_an_image")  # type: ignore[arg-type]


# 27. Real local PaddleOCR integration test (runs when local models exist, skipped otherwise)
REAL_DET_DIR = Path(r"C:\SovereignAI\member3_ocr\models\paddleocr\PP-OCRv5_mobile_det_infer")
REAL_REC_DIR = Path(r"C:\SovereignAI\member3_ocr\models\paddleocr\PP-OCRv5_mobile_rec_infer")
REAL_TEST_IMG = Path(r"C:\SovereignAI\member3_ocr\input\ocr_real_test.png")


@pytest.mark.skipif(
    not (REAL_DET_DIR.is_dir() and REAL_REC_DIR.is_dir() and REAL_TEST_IMG.is_file()),
    reason="Real PaddleOCR local model directories or test image not present",
)
def test_real_paddleocr_with_preprocessed_2d_image() -> None:
    """Integration test: real PaddleOCR with 2-D preprocessed image using document_ocr preset."""
    config = PaddleOCRModelConfig(
        detection_model_dir=REAL_DET_DIR,
        recognition_model_dir=REAL_REC_DIR,
        detection_model_name="PP-OCRv5_mobile_det",
        recognition_model_name="PP-OCRv5_mobile_rec",
        device="cpu",
        allow_model_download=False,
    )
    backend = PaddleOCRBackend(config)
    pipeline = OCRPipeline(backend)
    result = pipeline.process_image(REAL_TEST_IMG, document_id="real-regression-test")

    assert result.errors == ()
    assert len(result.pages) == 1
    page = result.pages[0]
    assert page.errors == ()
    assert len(page.blocks) >= 3
    assert "P-101" in page.text or "APPROVED" in page.text


# ──────────────────────────────────────────────────────────────────────────────
# 28-29. Adversarial input handling tests (mock backend — no real models needed)
#   These tests close the genuine gap identified in Step 0: the pipeline must
#   handle pathological inputs gracefully, producing structured errors rather
#   than unhandled exceptions.
# ──────────────────────────────────────────────────────────────────────────────

def test_ocr_handles_all_white_image(tmp_path: Path) -> None:
    """OCR pipeline must handle a blank all-white image without crashing.

    When the detection model finds no text regions, the pipeline must return
    a valid OCRDocumentResult with an empty blocks list — not raise an
    unhandled exception.  This uses a mock backend that always returns
    empty output, directly testing the pipeline's tolerance of no-text inputs.
    """
    from PIL import Image as PILImage

    # Create a valid 640x480 all-white PNG
    img_path = tmp_path / "all_white.png"
    arr = np.full((480, 640, 3), 255, dtype=np.uint8)
    PILImage.fromarray(arr).save(img_path)

    class EmptyBackend:
        """Mock backend that returns no text blocks (simulates blank-image detection)."""

        @property
        def backend_info(self) -> BackendInfo:
            return BackendInfo("empty_mock", "1.0", (), "cpu", BackendCapabilities())

        def initialize(self) -> None:
            pass

        def recognize(self, image: np.ndarray) -> BackendRecognition:
            return BackendRecognition(blocks=())  # No text detected

    pipeline = OCRPipeline(EmptyBackend())
    result = pipeline.process_image(source=img_path, document_id="white-image-test")

    # Must return a valid result — no unhandled exception
    assert result is not None, "Pipeline must not return None on blank input"
    assert isinstance(result, OCRDocumentResult)
    # Must have exactly 1 page processed
    assert len(result.pages) == 1
    page = result.pages[0]
    # No text should be detected
    assert page.blocks == (), f"Expected empty blocks on blank image, got {len(page.blocks)}"
    # No structured errors (blank image is valid input)
    assert page.errors == (), f"Expected no errors on valid blank image, got {page.errors}"


def test_ocr_handles_sub_pixel_image(tmp_path: Path) -> None:
    """OCR pipeline must handle a tiny 3x3 pixel image without crashing.

    Extremely small images may trigger edge cases in OCR model preprocessing
    (e.g. invalid resize targets). The pipeline must handle this gracefully —
    either as an empty result or as a structured Image error, never as a
    raw Python exception that propagates to the caller.
    """
    from PIL import Image as PILImage

    # Create a valid 3x3 black PNG
    img_path = tmp_path / "tiny.png"
    arr = np.zeros((3, 3, 3), dtype=np.uint8)
    PILImage.fromarray(arr).save(img_path)

    # Use the deterministic mock backend — any result is acceptable as long
    # as it doesn't crash with an unhandled exception
    backend = DeterministicMockBackend()
    pipeline = OCRPipeline(backend)

    try:
        result = pipeline.process_image(source=img_path, document_id="tiny-image-test")
        # Any non-exception result (empty, errors, or blocks) is acceptable
        assert result is not None, "Pipeline must return a result, even for tiny images"
        assert isinstance(result, OCRDocumentResult)
    except (ValueError, RuntimeError) as exc:
        pytest.fail(
            f"Pipeline raised an unhandled {type(exc).__name__} on a 3x3 image: {exc}"
        )
