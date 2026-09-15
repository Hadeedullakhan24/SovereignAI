"""Regression tests for the Phase 1 structured production vision integration."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from agent.tool_executor import DEFAULT_PROJECT_ROOT, DEFAULT_SANDBOX_DIR, VisionInspectorTool
from member3_ocr.core.multimodal_processor import MultimodalProcessor, MultimodalProcessorConfig
from member3_ocr.core.vision_pipeline import (
    BackendCapabilities,
    BackendInfo,
    EquipmentItem,
    ImageCaption,
    LocalQwenVisionBackend,
    VisionModelConfig,
    VisionAnalysisPayload,
    VisionPipeline,
)


class CountingProductionVisionBackend:
    """Small production-shaped backend used to verify the integration contract."""

    def __init__(self) -> None:
        self.calls = 0

    @property
    def backend_info(self) -> BackendInfo:
        return BackendInfo(
            name="LocalQwenVisionBackend",
            version="Qwen2.5-VL-3B-Instruct",
            model_path="models/vision/qwen2.5-vl-3b-instruct",
            device="cpu",
            device_requested="auto",
            device_used="cpu",
            capabilities=BackendCapabilities(
                image_classification=True,
                image_captioning=True,
                visual_question_answering=True,
            ),
        )

    def capabilities(self) -> BackendCapabilities:
        return self.backend_info.capabilities

    def initialize(self) -> None:
        pass

    def analyze(self, image) -> VisionAnalysisPayload:
        self.calls += 1
        return VisionAnalysisPayload(
            captions=(ImageCaption("Visible P&ID process equipment.", 0.9),),
            metadata={
                "scene_type": "P&ID",
                "caption": "Visible P&ID process equipment.",
                "equipment": [{
                    "equipment_type": "Pump",
                    "name_or_tag": "P-101",
                    "confidence": 0.9,
                    "evidence": "Visible tag in the supplied image.",
                }],
                "observations": [{
                    "description": "A process drawing is visibly present.",
                    "confidence": 0.8,
                }],
                "visible_text": [{"text": "P-101", "confidence": 0.9}],
                "model_name": "qwen2.5-vl-3b-instruct",
                "model_path": "models/vision/qwen2.5-vl-3b-instruct",
                "device_used": "cpu",
                "device_requested": "auto",
                "inference_count": 1,
            },
        )


def _image(tmp_path: Path) -> Path:
    path = tmp_path / "PID_002_Process_Example.jpg"
    Image.new("RGB", (80, 60), "white").save(path)
    return path


def _processor(backend: CountingProductionVisionBackend, *, enable_vision: bool) -> MultimodalProcessor:
    return MultimodalProcessor(
        config=MultimodalProcessorConfig(
            enable_preprocessing=False,
            enable_ocr=False,
            enable_vision=enable_vision,
            enable_drawing_analysis=True,
        ),
        vision_pipeline=VisionPipeline(backend=backend),
    )


def test_ocr_only_does_not_execute_or_invent_vlm_results(tmp_path: Path) -> None:
    backend = CountingProductionVisionBackend()
    result = _processor(backend, enable_vision=False).orchestrate(
        _image(tmp_path), force_route="engineering_drawing"
    )

    assert backend.calls == 0
    assert result.has_vision_analysis is False
    assert result.processing_metadata["vision"]["executed"] is False
    assert result.drawing_analysis is not None
    assert all(item.source == "ocr" for item in result.drawing_analysis.equipment)
    assert all(item.tag != "P-101" for item in result.drawing_analysis.equipment)


def test_vlm_visual_route_creates_structured_vision_record_once(tmp_path: Path) -> None:
    backend = CountingProductionVisionBackend()
    result = _processor(backend, enable_vision=True).orchestrate(
        _image(tmp_path), force_route="visual_inspection"
    )

    assert backend.calls == 1
    assert result.has_vision_analysis is True
    assert result.vision_analysis is not None
    assert result.vision_analysis.equipment[0].name_or_tag == "P-101"
    vision = result.processing_metadata["vision"]
    assert vision["backend"] == "LocalQwenVisionBackend"
    assert vision["inference_count"] == 1
    assert vision["image_dimensions"] == (80, 60)
    assert vision["resolved_image_path"] == str(_image_path_resolved(result))


def test_drawing_vlm_equipment_is_vlm_provenanced_and_not_visual_inspection(tmp_path: Path) -> None:
    backend = CountingProductionVisionBackend()
    result = _processor(backend, enable_vision=True).orchestrate(
        _image(tmp_path), force_route="engineering_drawing"
    )

    assert backend.calls == 1
    assert result.has_vision_analysis is False
    assert result.drawing_analysis is not None
    equipment = {item.tag: item for item in result.drawing_analysis.equipment}
    assert equipment["P-101"].source == "vlm"
    assert result.processing_metadata["vision"]["analysis_kind"] == "drawing_fusion"


def test_vision_inspector_builds_real_qwen_pipeline_not_mock_backend() -> None:
    tool = VisionInspectorTool(DEFAULT_SANDBOX_DIR, DEFAULT_PROJECT_ROOT)
    processor = tool._get_processor(enable_vision=True)

    assert isinstance(processor.vision_pipeline.backend, LocalQwenVisionBackend)
    assert processor.vision_pipeline.backend.backend_info.name == "LocalQwenVisionBackend"


def test_qwen_backend_runs_one_generation_per_image(monkeypatch) -> None:
    backend = LocalQwenVisionBackend(VisionModelConfig(model_path="models/vision/qwen2.5-vl-3b-instruct"))
    backend._initialized = True
    backend._model = object()
    calls: list[str] = []

    def fake_vqa(_image, prompt: str) -> str:
        calls.append(prompt)
        return "P&ID with Pump P-101 visible."

    monkeypatch.setattr(backend, "_run_vqa", fake_vqa)
    payload = backend.analyze(np.zeros((20, 30, 3), dtype=np.uint8))

    assert len(calls) == 1
    assert payload.metadata["inference_count"] == 1


def _image_path_resolved(result) -> Path:
    """Avoid duplicating an image path assumption in the provenance assertion."""
    return Path(result.source_path).resolve()
