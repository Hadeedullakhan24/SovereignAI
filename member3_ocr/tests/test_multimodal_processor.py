"""Unit tests for member3_ocr.multimodal_processor.

All tests use synthetic in-memory images and deterministic mock components.
No network, no model downloads, no datasets/ modifications.
"""
from __future__ import annotations


import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from member3_ocr.core.document_parser import DocumentParser
from member3_ocr.core.drawing_analyzer import (
    DrawingAnalysisResult,
    DrawingAnalyzer,
    DrawingLabel,
    DrawingMetadata,
    DrawingRegion,
    DrawingType,
    MockDrawingBackend,
)
from member3_ocr.core.image_preprocessing import (
    PreprocessingOptions,
    preprocess_image,
)
from member3_ocr.core.multimodal_processor import (
    MockOCRBackend,
    ModalityStatus,
    MultimodalIssue,
    MultimodalProcessingResult,
    MultimodalProcessor,
    MultimodalProcessorConfig,
    MultimodalResult,
    ProcessingError,
    RoutingDecision,
    classify_routing,
    export_for_agent,
    export_for_rag,
    main,
    process_document,
)
from member3_ocr.core.ocr_pipeline import (
    BackendCapabilities as OCRBackendCapabilities,
    BackendInfo as OCRBackendInfo,
    BoundingBox,
    OCRDocumentResult,
    OCRPageResult,
    OCRPipeline,
    TextBlock,
    TextLine,
    Word,
)
from member3_ocr.core.vision_pipeline import (
    ClassificationResult,
    DetectedObject,
    ImageCaption,
    MockVisionBackend,
    VisionPipeline,
    VisionResult,
)
from rag_engine.schemas.parsed_document import ParsedDocument


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_image(tmp_path: Path) -> Path:
    """Create a valid synthetic RGB PNG image for testing."""
    img_path = tmp_path / "synthetic_plant_asset.png"
    img_arr = np.full((120, 160, 3), 200, dtype=np.uint8)
    img_arr[30:90, 40:120] = [30, 80, 150]
    img = Image.fromarray(img_arr)
    img.save(img_path)
    return img_path


@pytest.fixture
def precomputed_ocr() -> OCRDocumentResult:
    """Return a precomputed OCR result."""
    bbox = BoundingBox(10.0, 10.0, 150.0, 40.0)
    word = Word("P-203", 0.99, bbox)
    line = TextLine("P-203", 0.99, bbox, (word,))
    block = TextBlock("tb_1", "P-203", 0.99, bbox, (line,))
    page = OCRPageResult(
        page_number=1,
        original_width=160,
        original_height=120,
        processed_width=160,
        processed_height=120,
        text="P-203",
        blocks=(block,),
    )
    backend_info = OCRBackendInfo("mock_ocr", "1.0", (), "cpu", OCRBackendCapabilities())
    return OCRDocumentResult("ocr_doc_precomputed", (page,), backend_info, {"source": "precomputed"})


@pytest.fixture
def precomputed_vision() -> VisionResult:
    """Return a precomputed Vision result."""
    return VisionResult(
        success=True,
        source_path="fake_source.png",
        image_width=160,
        image_height=120,
        classifications=(ClassificationResult("industrial_pump", 0.96),),
        detections=(
            DetectedObject(
                label="centrifugal_pump",
                confidence=0.92,
                bbox=BoundingBox(20.0, 20.0, 140.0, 100.0),
                attributes={"status": "active"},
            ),
        ),
        captions=(ImageCaption("Centrifugal pump in refinery unit.", 0.91),),
    )


@pytest.fixture
def precomputed_drawing() -> DrawingAnalysisResult:
    """Return a precomputed Drawing analysis result."""
    return DrawingAnalysisResult(
        success=True,
        source_path="fake_source.png",
        drawing_type=DrawingType.PID,
        metadata=DrawingMetadata(
            drawing_type=DrawingType.PID,
            title="P&ID Unit 1",
            drawing_number="DWG-001",
            source="precomputed",
        ),
        regions=(
            DrawingRegion(
                region_id="region_0",
                region_type="equipment",
                bbox=BoundingBox(20.0, 20.0, 100.0, 100.0),
                label="P-203",
                confidence=0.95,
                attributes={"tag": "P-203"},
            ),
        ),
        labels=(
            DrawingLabel(
                label_id="label_0",
                text="P-203",
                bbox=BoundingBox(20.0, 10.0, 80.0, 30.0),
                confidence=0.98,
                label_type="equipment_tag",
                source="precomputed",
            ),
        ),
        description="P&ID diagram depicting pump P-203.",
    )


# ──────────────────────────────────────────────────────────────────────────────
# 1. Config defaults
# ──────────────────────────────────────────────────────────────────────────────

def test_config_defaults() -> None:
    cfg = MultimodalProcessorConfig()
    assert cfg.enable_preprocessing is True
    assert cfg.enable_ocr is True
    assert cfg.enable_vision is True
    assert cfg.enable_drawing_analysis is True
    assert cfg.enable_document_parsing is False
    assert cfg.fail_fast is False
    cfg.validate()


# ──────────────────────────────────────────────────────────────────────────────
# 2. Missing input
# ──────────────────────────────────────────────────────────────────────────────

def test_missing_input(tmp_path: Path) -> None:
    processor = MultimodalProcessor()
    res = processor.process(tmp_path / "ghost_file.png")
    assert res.success is False
    assert any(i.code == "missing_image" for i in res.issues)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Unsupported input
# ──────────────────────────────────────────────────────────────────────────────

def test_unsupported_input(tmp_path: Path) -> None:
    bad_path = tmp_path / "data.xyz"
    bad_path.write_text("invalid")
    processor = MultimodalProcessor()
    res = processor.process(bad_path)
    assert res.success is False
    assert any(i.code == "unsupported_format" for i in res.issues)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Corrupt image
# ──────────────────────────────────────────────────────────────────────────────

def test_corrupt_image(tmp_path: Path) -> None:
    corrupt_path = tmp_path / "corrupt.png"
    corrupt_path.write_bytes(b"NOT_A_VALID_IMAGE_HEADER")
    processor = MultimodalProcessor()
    res = processor.process(corrupt_path)
    assert res.success is False
    assert any(i.code == "corrupt_image" for i in res.issues)


# ──────────────────────────────────────────────────────────────────────────────
# 5. Preprocessing-only flow
# ──────────────────────────────────────────────────────────────────────────────

def test_preprocessing_only_flow(synthetic_image: Path) -> None:
    cfg = MultimodalProcessorConfig(
        enable_preprocessing=True,
        enable_ocr=False,
        enable_vision=False,
        enable_drawing_analysis=False,
    )
    processor = MultimodalProcessor(config=cfg)
    res = processor.process(synthetic_image)
    assert res.success is True
    assert res.modality_statuses["preprocessing"] == ModalityStatus.SUCCESS
    assert res.modality_statuses["ocr"] == ModalityStatus.NOT_REQUESTED
    assert res.modality_statuses["vision"] == ModalityStatus.NOT_REQUESTED
    assert res.modality_statuses["drawing"] == ModalityStatus.NOT_REQUESTED


# ──────────────────────────────────────────────────────────────────────────────
# 6. OCR-only flow
# ──────────────────────────────────────────────────────────────────────────────

def test_ocr_only_flow(synthetic_image: Path) -> None:
    cfg = MultimodalProcessorConfig(
        enable_preprocessing=False,
        enable_ocr=True,
        enable_vision=False,
        enable_drawing_analysis=False,
    )
    processor = MultimodalProcessor(config=cfg)
    res = processor.process(synthetic_image)
    assert res.success is True
    assert res.modality_statuses["ocr"] == ModalityStatus.SUCCESS
    assert res.ocr_result is not None
    assert len(res.combined_text) >= 1
    assert res.combined_text[0]["source"] == "ocr"


# ──────────────────────────────────────────────────────────────────────────────
# 7. Vision-only flow
# ──────────────────────────────────────────────────────────────────────────────

def test_vision_only_flow(synthetic_image: Path) -> None:
    cfg = MultimodalProcessorConfig(
        enable_preprocessing=False,
        enable_ocr=False,
        enable_vision=True,
        enable_drawing_analysis=False,
    )
    processor = MultimodalProcessor(config=cfg)
    res = processor.process(synthetic_image)
    assert res.success is True
    assert res.modality_statuses["vision"] == ModalityStatus.SUCCESS
    assert res.vision_result is not None
    assert len(res.combined_structured["classifications"]) >= 1


# ──────────────────────────────────────────────────────────────────────────────
# 8. Drawing-only flow
# ──────────────────────────────────────────────────────────────────────────────

def test_drawing_only_flow(synthetic_image: Path) -> None:
    cfg = MultimodalProcessorConfig(
        enable_preprocessing=False,
        enable_ocr=False,
        enable_vision=False,
        enable_drawing_analysis=True,
    )
    processor = MultimodalProcessor(config=cfg)
    res = processor.process(synthetic_image)
    assert res.success is True
    assert res.modality_statuses["drawing"] == ModalityStatus.SUCCESS
    assert res.drawing_result is not None
    assert res.combined_structured["drawing_type"] is not None


# ──────────────────────────────────────────────────────────────────────────────
# 9. OCR + Vision aggregation
# ──────────────────────────────────────────────────────────────────────────────

def test_ocr_and_vision_aggregation(synthetic_image: Path) -> None:
    cfg = MultimodalProcessorConfig(
        enable_preprocessing=False,
        enable_ocr=True,
        enable_vision=True,
        enable_drawing_analysis=False,
    )
    processor = MultimodalProcessor(config=cfg)
    res = processor.process(synthetic_image)
    assert res.success is True
    sources = {item["source"] for item in res.combined_text}
    assert "ocr" in sources
    assert "vision" in sources


# ──────────────────────────────────────────────────────────────────────────────
# 10. OCR + Drawing aggregation
# ──────────────────────────────────────────────────────────────────────────────

def test_ocr_and_drawing_aggregation(synthetic_image: Path) -> None:
    cfg = MultimodalProcessorConfig(
        enable_preprocessing=False,
        enable_ocr=True,
        enable_vision=False,
        enable_drawing_analysis=True,
    )
    processor = MultimodalProcessor(config=cfg)
    res = processor.process(synthetic_image)
    assert res.success is True
    assert res.ocr_result is not None
    assert res.drawing_result is not None
    assert len(res.combined_structured["drawing_labels"]) >= 1


# ──────────────────────────────────────────────────────────────────────────────
# 11. Vision + Drawing aggregation
# ──────────────────────────────────────────────────────────────────────────────

def test_vision_and_drawing_aggregation(synthetic_image: Path) -> None:
    cfg = MultimodalProcessorConfig(
        enable_preprocessing=False,
        enable_ocr=False,
        enable_vision=True,
        enable_drawing_analysis=True,
    )
    processor = MultimodalProcessor(config=cfg)
    res = processor.process(synthetic_image)
    assert res.success is True
    assert res.vision_result is not None
    assert res.drawing_result is not None
    assert len(res.combined_structured["detections"]) >= 1
    assert len(res.combined_structured["regions"]) >= 1


# ──────────────────────────────────────────────────────────────────────────────
# 12. Full multimodal flow
# ──────────────────────────────────────────────────────────────────────────────

def test_full_multimodal_flow(synthetic_image: Path) -> None:
    cfg = MultimodalProcessorConfig(
        enable_preprocessing=True,
        enable_ocr=True,
        enable_vision=True,
        enable_drawing_analysis=True,
    )
    processor = MultimodalProcessor(config=cfg)
    res = processor.process(synthetic_image)
    assert res.success is True
    assert res.modality_statuses["preprocessing"] == ModalityStatus.SUCCESS
    assert res.modality_statuses["ocr"] == ModalityStatus.SUCCESS
    assert res.modality_statuses["vision"] == ModalityStatus.SUCCESS
    assert res.modality_statuses["drawing"] == ModalityStatus.SUCCESS
    assert res.ocr_result is not None
    assert res.vision_result is not None
    assert res.drawing_result is not None


# ──────────────────────────────────────────────────────────────────────────────
# 13. Existing PreprocessingResult support
# ──────────────────────────────────────────────────────────────────────────────

def test_existing_preprocessing_result_support(synthetic_image: Path) -> None:
    prep_res = preprocess_image(synthetic_image, PreprocessingOptions.general_vision(), save_output=False)
    processor = MultimodalProcessor()
    res = processor.process(prep_res)
    assert res.success is True
    assert res.image_dimensions == (prep_res.processed_width, prep_res.processed_height)
    assert res.modality_statuses["preprocessing"] == ModalityStatus.SUCCESS


# ──────────────────────────────────────────────────────────────────────────────
# 14-16. Precomputed results support
# ──────────────────────────────────────────────────────────────────────────────

def test_precomputed_results_support(
    synthetic_image: Path,
    precomputed_ocr: OCRDocumentResult,
    precomputed_vision: VisionResult,
    precomputed_drawing: DrawingAnalysisResult,
) -> None:
    processor = MultimodalProcessor()
    res = processor.process(
        synthetic_image,
        ocr_result=precomputed_ocr,
        vision_result=precomputed_vision,
        drawing_result=precomputed_drawing,
    )
    assert res.success is True
    assert res.ocr_result is precomputed_ocr
    assert res.vision_result is precomputed_vision
    assert res.drawing_result is precomputed_drawing
    assert res.provenance["ocr"]["source"] == "precomputed"
    assert res.provenance["vision"]["source"] == "precomputed"
    assert res.provenance["drawing"]["source"] == "precomputed"


# ──────────────────────────────────────────────────────────────────────────────
# 17. DocumentParser integration
# ──────────────────────────────────────────────────────────────────────────────

def test_document_parser_integration(synthetic_image: Path) -> None:
    cfg = MultimodalProcessorConfig(
        enable_ocr=True,
        enable_document_parsing=True,
    )
    processor = MultimodalProcessor(config=cfg)
    res = processor.process(synthetic_image)
    assert res.success is True
    assert res.parsed_document is not None
    assert isinstance(res.parsed_document, ParsedDocument)
    assert res.modality_statuses["document_parser"] == ModalityStatus.SUCCESS


# ──────────────────────────────────────────────────────────────────────────────
# 18. Document parser without OCR produces structured issue
# ──────────────────────────────────────────────────────────────────────────────

def test_document_parser_without_ocr_produces_issue(synthetic_image: Path) -> None:
    cfg = MultimodalProcessorConfig(
        enable_ocr=False,
        enable_document_parsing=True,
    )
    processor = MultimodalProcessor(config=cfg)
    res = processor.process(synthetic_image)
    assert res.parsed_document is None
    assert res.modality_statuses["document_parser"] == ModalityStatus.SKIPPED
    assert any(i.code == "DOCUMENT_PARSE_REQUIRES_OCR" for i in res.issues)


# ──────────────────────────────────────────────────────────────────────────────
# 19. fail_fast=True behavior
# ──────────────────────────────────────────────────────────────────────────────

def test_fail_fast_true_stops_on_failure(synthetic_image: Path) -> None:
    cfg = MultimodalProcessorConfig(
        enable_ocr=True,
        enable_vision=True,
        fail_fast=True,
    )
    failing_ocr = OCRPipeline(backend=MockOCRBackend(fail=True))
    processor = MultimodalProcessor(config=cfg, ocr_pipeline=failing_ocr)
    res = processor.process(synthetic_image)
    assert res.success is False
    assert res.modality_statuses["ocr"] == ModalityStatus.FAILED
    assert res.modality_statuses["vision"] == ModalityStatus.NOT_REQUESTED


# ──────────────────────────────────────────────────────────────────────────────
# 20. fail_fast=False behavior
# ──────────────────────────────────────────────────────────────────────────────

def test_fail_fast_false_continues_independent_modalities(synthetic_image: Path) -> None:
    cfg = MultimodalProcessorConfig(
        enable_ocr=True,
        enable_vision=True,
        fail_fast=False,
    )
    failing_ocr = OCRPipeline(backend=MockOCRBackend(fail=True))
    processor = MultimodalProcessor(config=cfg, ocr_pipeline=failing_ocr)
    res = processor.process(synthetic_image)
    # OCR failed, but Vision succeeded; fail_fast=False allows usable output
    assert res.success is True
    assert res.modality_statuses["ocr"] == ModalityStatus.FAILED
    assert res.modality_statuses["vision"] == ModalityStatus.SUCCESS
    assert res.vision_result is not None


# ──────────────────────────────────────────────────────────────────────────────
# 21-23. Modality isolation
# ──────────────────────────────────────────────────────────────────────────────

def test_ocr_failure_does_not_corrupt_vision(synthetic_image: Path) -> None:
    failing_ocr = OCRPipeline(backend=MockOCRBackend(fail=True))
    processor = MultimodalProcessor(ocr_pipeline=failing_ocr)
    res = processor.process(synthetic_image)
    assert res.modality_statuses["ocr"] == ModalityStatus.FAILED
    assert res.modality_statuses["vision"] == ModalityStatus.SUCCESS
    assert len(res.combined_structured["classifications"]) > 0


def test_vision_failure_does_not_corrupt_ocr(synthetic_image: Path) -> None:
    failing_vis = VisionPipeline(backend=MockVisionBackend(should_fail=True))
    processor = MultimodalProcessor(vision_pipeline=failing_vis)
    res = processor.process(synthetic_image)
    assert res.modality_statuses["vision"] == ModalityStatus.FAILED
    assert res.modality_statuses["ocr"] == ModalityStatus.SUCCESS
    assert res.ocr_result is not None


def test_drawing_failure_does_not_corrupt_ocr_or_vision(synthetic_image: Path) -> None:
    failing_drw = DrawingAnalyzer(backend=MockDrawingBackend(should_fail=True))
    processor = MultimodalProcessor(drawing_analyzer=failing_drw)
    res = processor.process(synthetic_image)
    assert res.modality_statuses["drawing"] == ModalityStatus.FAILED
    assert res.modality_statuses["ocr"] == ModalityStatus.SUCCESS
    assert res.modality_statuses["vision"] == ModalityStatus.SUCCESS


# ──────────────────────────────────────────────────────────────────────────────
# 24. Multimodal conflict detection
# ──────────────────────────────────────────────────────────────────────────────

def test_multimodal_conflict_detection(synthetic_image: Path) -> None:
    # OCR detects P-203
    ocr_backend = MockOCRBackend(text="P-203 Centrifugal Pump")
    ocr_pipe = OCRPipeline(backend=ocr_backend)

    # Drawing detects conflicting tag P-999
    conflicting_drawing_backend = MockDrawingBackend(
        labels=(
            DrawingLabel("l1", "P-999", BoundingBox(10.0, 10.0, 50.0, 50.0), label_type="equipment_tag"),
        ),
        regions=(
            DrawingRegion("r1", "equipment", BoundingBox(10.0, 10.0, 50.0, 50.0), label="P-999", attributes={"tag": "P-999"}),
        ),
    )
    drawing_pipe = DrawingAnalyzer(backend=conflicting_drawing_backend)

    processor = MultimodalProcessor(ocr_pipeline=ocr_pipe, drawing_analyzer=drawing_pipe)
    res = processor.process(synthetic_image)

    conflict_issues = [i for i in res.issues if i.code == "MULTIMODAL_CONFLICT"]
    assert len(conflict_issues) >= 1
    iss = conflict_issues[0]
    assert iss.details is not None
    assert "P-203" in str(iss.details["value_a"])
    assert "P-999" in str(iss.details["value_b"])


# ──────────────────────────────────────────────────────────────────────────────
# 25. Deterministic output
# ──────────────────────────────────────────────────────────────────────────────

def test_deterministic_output(synthetic_image: Path) -> None:
    processor = MultimodalProcessor()
    res1 = processor.process(synthetic_image)
    res2 = processor.process(synthetic_image)

    d1 = res1.to_dict()
    d2 = res2.to_dict()

    def _strip_timings(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {
                k: _strip_timings(v)
                for k, v in obj.items()
                if not (k.endswith(("_time_ms", "_time_seconds")) or "processing_time" in k)
            }
        if isinstance(obj, list):
            return [_strip_timings(v) for v in obj]
        return obj

    assert _strip_timings(d1) == _strip_timings(d2)


# ──────────────────────────────────────────────────────────────────────────────
# 26. Deterministic IDs
# ──────────────────────────────────────────────────────────────────────────────

def test_deterministic_ids(synthetic_image: Path) -> None:
    processor = MultimodalProcessor()
    res1 = processor.process(synthetic_image)
    res2 = processor.process(synthetic_image)

    # Identifiers must be stable across multiple runs on identical input
    labels1 = [item["label_id"] for item in res1.combined_structured["drawing_labels"]]
    labels2 = [item["label_id"] for item in res2.combined_structured["drawing_labels"]]
    assert labels1 == labels2

    regions1 = [item["region_id"] for item in res1.combined_structured["regions"]]
    regions2 = [item["region_id"] for item in res2.combined_structured["regions"]]
    assert regions1 == regions2


# ──────────────────────────────────────────────────────────────────────────────
# 27. Combined text aggregation
# ──────────────────────────────────────────────────────────────────────────────

def test_combined_text_aggregation(synthetic_image: Path) -> None:
    ocr_backend = MockOCRBackend(text="OCR Pump Description")
    ocr_pipe = OCRPipeline(backend=ocr_backend)

    drw_backend = MockDrawingBackend(
        labels=(DrawingLabel("l1", "Tag P-203", BoundingBox(10.0, 10.0, 50.0, 50.0)),),
        description="Drawing Pump Outline",
    )
    drw_pipe = DrawingAnalyzer(backend=drw_backend)

    processor = MultimodalProcessor(ocr_pipeline=ocr_pipe, drawing_analyzer=drw_pipe)
    res = processor.process(synthetic_image)

    texts = [item["text"] for item in res.combined_text]
    assert "OCR Pump Description" in texts
    assert "Drawing Pump Outline" in texts
    assert "Tag P-203" in texts


# ──────────────────────────────────────────────────────────────────────────────
# 28. Duplicate text handling
# ──────────────────────────────────────────────────────────────────────────────

def test_duplicate_text_handling(synthetic_image: Path) -> None:
    ocr_backend = MockOCRBackend(text="Duplicate Text Line\nUnique OCR Line")
    ocr_pipe = OCRPipeline(backend=ocr_backend)

    drw_backend = MockDrawingBackend(
        labels=(DrawingLabel("l1", "Duplicate Text Line", BoundingBox(10.0, 10.0, 50.0, 50.0)),),
        description="Unique Drawing Description",
    )
    drw_pipe = DrawingAnalyzer(backend=drw_backend)

    processor = MultimodalProcessor(ocr_pipeline=ocr_pipe, drawing_analyzer=drw_pipe)
    res = processor.process(synthetic_image)

    texts = [item["text"] for item in res.combined_text]
    # "Duplicate Text Line" should appear only once (first from OCR)
    assert texts.count("Duplicate Text Line") == 1
    assert "Unique OCR Line" in texts
    assert "Unique Drawing Description" in texts


# ──────────────────────────────────────────────────────────────────────────────
# 29. Provenance preservation
# ──────────────────────────────────────────────────────────────────────────────

def test_provenance_preservation(synthetic_image: Path) -> None:
    processor = MultimodalProcessor()
    res = processor.process(synthetic_image)
    assert "ocr" in res.provenance
    assert "vision" in res.provenance
    assert "drawing" in res.provenance


# ──────────────────────────────────────────────────────────────────────────────
# 30. JSON serialization
# ──────────────────────────────────────────────────────────────────────────────

def test_json_serialization(synthetic_image: Path) -> None:
    processor = MultimodalProcessor()
    res = processor.process(synthetic_image)

    json_str = res.to_json()
    assert isinstance(json_str, str)
    parsed = json.loads(json_str)
    assert parsed["success"] is True
    assert "ocr_result" in parsed
    assert "vision_result" in parsed
    assert "drawing_result" in parsed


# ──────────────────────────────────────────────────────────────────────────────
# 31. to_dict serialization
# ──────────────────────────────────────────────────────────────────────────────

def test_to_dict_serialization(synthetic_image: Path) -> None:
    processor = MultimodalProcessor()
    res = processor.process(synthetic_image)

    d = res.to_dict()
    assert isinstance(d, dict)
    assert d["success"] is True
    assert isinstance(d["combined_text"], list)
    assert isinstance(d["combined_structured"], dict)
    assert isinstance(d["modality_statuses"], dict)


# ──────────────────────────────────────────────────────────────────────────────
# 32. RAG payload generation
# ──────────────────────────────────────────────────────────────────────────────

def test_rag_payload_generation(synthetic_image: Path) -> None:
    processor = MultimodalProcessor()
    res = processor.process(synthetic_image)
    rag_payload = res.to_rag_payload()

    assert isinstance(rag_payload, dict)
    assert "source_identifier" in rag_payload
    assert "text_chunks" in rag_payload
    assert "entities" in rag_payload
    assert "drawing_relations" in rag_payload
    assert "provenance" in rag_payload


# ──────────────────────────────────────────────────────────────────────────────
# 33. RAG payload preserves modality
# ──────────────────────────────────────────────────────────────────────────────

def test_rag_payload_preserves_modality(synthetic_image: Path) -> None:
    processor = MultimodalProcessor()
    res = processor.process(synthetic_image)
    rag_payload = res.to_rag_payload()

    for chunk in rag_payload["text_chunks"]:
        assert "modality" in chunk
        assert chunk["modality"] in ("ocr", "vision", "drawing", "drawing_label")
    for entity in rag_payload["entities"]:
        assert "modality" in entity
        assert entity["modality"] in ("vision", "drawing")


# ──────────────────────────────────────────────────────────────────────────────
# 34. RAG payload preserves provenance
# ──────────────────────────────────────────────────────────────────────────────

def test_rag_payload_preserves_provenance(synthetic_image: Path) -> None:
    processor = MultimodalProcessor()
    res = processor.process(synthetic_image)
    rag_payload = res.to_rag_payload()

    assert "ocr" in rag_payload["provenance"]
    assert "vision" in rag_payload["provenance"]
    assert "drawing" in rag_payload["provenance"]


# ──────────────────────────────────────────────────────────────────────────────
# 35. Source image remains unchanged
# ──────────────────────────────────────────────────────────────────────────────

def test_source_image_unchanged(synthetic_image: Path) -> None:
    hash_before = hashlib.sha256(synthetic_image.read_bytes()).hexdigest()
    mtime_before = synthetic_image.stat().st_mtime

    processor = MultimodalProcessor()
    res = processor.process(synthetic_image)
    assert res.success is True

    hash_after = hashlib.sha256(synthetic_image.read_bytes()).hexdigest()
    mtime_after = synthetic_image.stat().st_mtime

    assert hash_before == hash_after
    assert mtime_before == mtime_after


# ──────────────────────────────────────────────────────────────────────────────
# 36. No writes to datasets/
# ──────────────────────────────────────────────────────────────────────────────

def test_no_writes_to_datasets() -> None:
    datasets_dir = Path(__file__).resolve().parents[2] / "datasets"
    before_mtime = datasets_dir.stat().st_mtime if datasets_dir.exists() else None

    # Processor must forbid output_directory inside datasets/
    with pytest.raises(Exception):
        MultimodalProcessorConfig(output_directory=datasets_dir).validate()

    if before_mtime is not None:
        assert datasets_dir.stat().st_mtime == before_mtime


# ──────────────────────────────────────────────────────────────────────────────
# 37. No network calls/downloads
# ──────────────────────────────────────────────────────────────────────────────

def test_no_network_behavior(monkeypatch: pytest.MonkeyPatch, synthetic_image: Path) -> None:
    import socket

    def forbidden_connect(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Network calls strictly forbidden in Sovereign AI offline pipeline!")

    monkeypatch.setattr(socket, "create_connection", forbidden_connect)

    processor = MultimodalProcessor()
    res = processor.process(synthetic_image)
    assert res.success is True


# ──────────────────────────────────────────────────────────────────────────────
# 38. Mock provenance is explicit
# ──────────────────────────────────────────────────────────────────────────────

def test_mock_provenance_explicit(synthetic_image: Path) -> None:
    processor = MultimodalProcessor()
    res = processor.process(synthetic_image)
    assert "mock" in res.provenance["ocr"]["backend_name"]
    assert "mock" in res.provenance["vision"]["backend_name"]
    assert "mock" in res.provenance["drawing"]["backend_name"]


# ──────────────────────────────────────────────────────────────────────────────
# 39. No raw model objects escape in to_dict()
# ──────────────────────────────────────────────────────────────────────────────

def test_no_raw_model_objects_escape(synthetic_image: Path) -> None:
    processor = MultimodalProcessor()
    res = processor.process(synthetic_image)
    d = res.to_dict()

    # Must be 100% pure JSON without OpenCV/PIL/tensors
    json_str = json.dumps(d)
    assert len(json_str) > 0


# ──────────────────────────────────────────────────────────────────────────────
# 40. CLI execution tests
# ──────────────────────────────────────────────────────────────────────────────

def test_cli_execution(synthetic_image: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["--input", str(synthetic_image), "--json"])
    assert exit_code == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["success"] is True
    assert "ocr" in payload["modality_statuses"]


# ──────────────────────────────────────────────────────────────────────────────
# 41. Helper for Synthetic OCR Results
# ──────────────────────────────────────────────────────────────────────────────

def _make_ocr(lines_text: list[str], *, width: int = 800, height: int = 1100) -> OCRDocumentResult:
    bbox = BoundingBox(10.0, 10.0, float(width - 20), 40.0)
    lines = []
    for line in lines_text:
        word = Word(text=line, confidence=0.95, bbox=bbox)
        lines.append(TextLine(text=line, confidence=0.95, bbox=bbox, words=(word,)))
    full_text = "\n".join(lines_text)
    block = TextBlock(id="tb_synth", text=full_text, confidence=0.95, bbox=bbox, lines=tuple(lines))
    page = OCRPageResult(
        page_number=1,
        original_width=width,
        original_height=height,
        processed_width=width,
        processed_height=height,
        text=full_text,
        blocks=(block,),
    )
    b_info = OCRBackendInfo(name="mock_ocr", version="1.0.0", model_ids=("mock_ocr",), device="cpu", capabilities=OCRBackendCapabilities())
    return OCRDocumentResult("doc_synth", (page,), b_info, {"source": "synthetic"})


# ──────────────────────────────────────────────────────────────────────────────
# 42. Task 1 Routing Classifier Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_routing_classifier_plain_document() -> None:
    ocr_res = _make_ocr([
        "Invoice Number: INV-99042",
        "Billing Address: 123 Industrial Parkway.",
        "Total Amount Due: $1,450.00.",
        "Please remit payment within 30 days of receipt.",
        "Terms and Conditions apply to all delivered equipment.",
    ], width=800, height=1100)

    decision, signals = classify_routing("invoice_doc.pdf", image_width=800, image_height=1100, ocr_result=ocr_res)
    assert decision == RoutingDecision.PLAIN_DOCUMENT
    assert signals["document_score"] >= signals["drawing_score"]
    assert "reasoning" in signals


def test_routing_classifier_engineering_drawing() -> None:
    ocr_res = _make_ocr([
        "PIPING & INSTRUMENTATION DIAGRAM",
        "UNIT 200 - CRUDE DISTILLATION",
        "P-101A CENTRIFUGAL PUMP",
        "FV-201 CONTROL VALVE",
        "TK-301 STORAGE TANK",
    ], width=2400, height=1200)

    decision, signals = classify_routing("unit_200_pid.png", image_width=2400, image_height=1200, ocr_result=ocr_res)
    assert decision == RoutingDecision.ENGINEERING_DRAWING
    assert signals["drawing_score"] >= 3.0
    assert signals["is_panoramic"] is True


def test_routing_classifier_mixed_content() -> None:
    ocr_res = _make_ocr([
        "Technical Specification and System Architecture Report.",
        "The crude distillation unit separates incoming petroleum into fractions.",
        "Operating temperature in the column exceeds 350 degrees Celsius.",
        "Refer to drawing schematic P-201 and piping layout for line 2-HC-102.",
        "Instrumentation includes PT-104 and FV-302 connected to the bypass line.",
    ], width=1800, height=1000)

    decision, signals = classify_routing("complex_tech_spec.png", image_width=1800, image_height=1000, ocr_result=ocr_res)
    assert decision in (RoutingDecision.MIXED, RoutingDecision.ENGINEERING_DRAWING)
    assert signals["drawing_score"] > 0
    assert signals["document_score"] > 0


# ──────────────────────────────────────────────────────────────────────────────
# 43. Orchestrator Force-Route Override
# ──────────────────────────────────────────────────────────────────────────────

def test_orchestrator_force_route_overrides(synthetic_image: Path) -> None:
    processor = MultimodalProcessor()

    # Force plain_document on an image
    res_doc = processor.orchestrate(synthetic_image, force_route="plain_document")
    assert res_doc.routing_decision == "plain_document"
    assert res_doc.routing_signals.get("forced") is True

    # Force engineering_drawing
    res_drw = processor.orchestrate(synthetic_image, force_route="engineering_drawing")
    assert res_drw.routing_decision == "engineering_drawing"
    assert res_drw.routing_signals.get("forced") is True

    # Force both / mixed
    res_both = processor.orchestrate(synthetic_image, force_route="both")
    assert res_both.routing_decision == "mixed"
    assert res_both.routing_signals.get("forced") is True


# ──────────────────────────────────────────────────────────────────────────────
# 44. Partial-Failure & Non-Fatal Robustness
# ──────────────────────────────────────────────────────────────────────────────

def test_orchestrator_partial_failure_non_fatal(synthetic_image: Path) -> None:
    # Set up an OCR pipeline with a failing backend
    failing_ocr = OCRPipeline(backend=MockOCRBackend(fail=True))
    processor = MultimodalProcessor(ocr_pipeline=failing_ocr)

    # Process document; should NOT raise exception, must return MultimodalProcessingResult with error
    res = processor.orchestrate(synthetic_image, force_route="plain_document")
    assert isinstance(res, MultimodalProcessingResult)
    assert res.has_errors is True
    assert any(e.stage == "ocr" for e in res.processing_errors)
    assert res.pipeline_versions["processor"] is not None


def test_orchestrator_missing_file_handled() -> None:
    processor = MultimodalProcessor()
    res = processor.orchestrate("non_existent_file_path_12345.png")
    assert isinstance(res, MultimodalProcessingResult)
    assert res.has_errors is True
    assert res.processing_errors[0].code == "FILE_NOT_FOUND"


# ──────────────────────────────────────────────────────────────────────────────
# 45. Unified RAG & Agent Export Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_unified_rag_and_agent_export(synthetic_image: Path) -> None:
    config = MultimodalProcessorConfig(enable_document_parsing=True)
    processor = MultimodalProcessor(config=config)

    # 1. Plain document path export
    res_doc = processor.orchestrate(synthetic_image, force_route="plain_document")
    rag_chunks = export_for_rag(res_doc)
    assert isinstance(rag_chunks, list)
    for c in rag_chunks:
        assert hasattr(c, "chunk_id")
        assert hasattr(c, "metadata")

    agent_doc = export_for_agent(res_doc)
    assert agent_doc.has_document is True
    assert agent_doc.document_id == res_doc.document_id
    summary = agent_doc.get_summary()
    assert summary["document_id"] == res_doc.document_id
    assert "section_count" in summary

    # 2. Drawing path export
    res_drw = processor.orchestrate(synthetic_image, force_route="engineering_drawing")
    drw_chunks = export_for_rag(res_drw)
    assert isinstance(drw_chunks, list)

    agent_drw = export_for_agent(res_drw)
    assert agent_drw.has_drawing is True
    eq_list = agent_drw.get_equipment_list()
    assert isinstance(eq_list, list)


def test_process_document_convenience_entry(synthetic_image: Path) -> None:
    res = process_document(synthetic_image, force_route="plain_document")
    assert isinstance(res, MultimodalProcessingResult)
    assert res.routing_decision == "plain_document"


def test_routing_classifier_visual_inspection() -> None:
    """Test that defect inspection images route to VISUAL_INSPECTION."""
    ocr_empty = _make_ocr([], width=800, height=800)
    
    # NEU surface defect image
    decision_neu, signals_neu = classify_routing(
        "datasets/Vision/neu_surface_defect/crazing_1.jpg",
        image_width=800,
        image_height=800,
        ocr_result=ocr_empty,
    )
    assert decision_neu == RoutingDecision.VISUAL_INSPECTION
    assert "visual" in signals_neu["reasoning"].lower()

    # PCB defect image
    decision_pcb, signals_pcb = classify_routing(
        "datasets/Vision/pcb_defect/missing_hole_01.jpg",
        image_width=600,
        image_height=600,
        ocr_result=ocr_empty,
    )
    assert decision_pcb == RoutingDecision.VISUAL_INSPECTION

    # Infrared image
    decision_ir, signals_ir = classify_routing(
        "datasets/Vision/infrared/panel_hotspot.jpg",
        image_width=640,
        image_height=480,
        ocr_result=ocr_empty,
    )
    assert decision_ir == RoutingDecision.VISUAL_INSPECTION


def test_orchestrator_visual_inspection_contract7(synthetic_image: Path) -> None:
    """Test Contract 7 visual inspection export for RAG and Agent."""
    config = MultimodalProcessorConfig(
        enable_preprocessing=False,
        enable_ocr=False,
        enable_drawing_analysis=False,
        enable_vision=True,
    )
    processor = MultimodalProcessor(config=config)
    # Use mock vision pipeline
    processor.vision_pipeline = VisionPipeline(backend=MockVisionBackend())

    res = processor.orchestrate(
        synthetic_image,
        force_route="visual_inspection",
    )

    assert isinstance(res, MultimodalProcessingResult)
    assert res.routing_decision == "visual_inspection"
    assert res.has_vision_analysis is True
    assert res.vision_analysis is not None
    assert res.vision_analysis.image_type == "visual_inspection"
    assert len(res.vision_analysis.observations) > 0

    # RAG Export
    rag_chunks = export_for_rag(res)
    assert len(rag_chunks) >= 2
    section_titles = [c.metadata.section_title for c in rag_chunks]
    assert "Inspection Summary" in section_titles
    assert "Visual Observations" in section_titles
    for c in rag_chunks:
        assert c.metadata.category == "visual_inspection"
        assert c.token_count > 0

    # Agent Export
    agent_doc = export_for_agent(res)
    assert agent_doc.has_vision_analysis is True
    assert agent_doc.visual_inspection is not None
    summary = agent_doc.get_summary()
    assert summary["has_vision_analysis"] is True
    assert summary["observation_count"] > 0
    obs = agent_doc.get_visual_observations()
    assert len(obs) > 0
    assert "description" in obs[0]
    assert agent_doc.has_defects() is True


