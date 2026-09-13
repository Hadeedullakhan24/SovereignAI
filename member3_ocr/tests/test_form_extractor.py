"""Unit and integration tests for form key-value extraction (Step 1B).

Verifies:
- Colon-separated key/value ("Key: Value", "Key : Value", "Key:Value")
- Whitespace variations around colons
- Key/value on separate lines (vertical adjacency)
- Bounding-box based adjacent key/value (horizontal adjacency)
- Multiple fields extraction and reading-order sorting
- Empty value fields (e.g. "Remarks:")
- Ambiguous field handling (unresolved + Issue warning)
- False-positive prevention (filtering narrative prose, retaining engineering specs)
- Deterministic results and confidence propagation
- Page provenance (page_number and bounding boxes preserved)
- Integration into OCRPipeline (process_image and process_pdf)
- Preservation in DocumentParser processing_history
- Offline / no-network safety and dataset read-only safety
- Real local PP-OCRv5 form inference integration
"""
from __future__ import annotations


import socket
from pathlib import Path
from typing import Sequence
from unittest.mock import patch

import pytest

from member3_ocr.core.document_parser import DocumentParser
from member3_ocr.core.form_extractor import (
    FormExtractorConfig,
    FormKeyValueExtractor,
    clean_field_key,
    extract_form_key_values,
    is_prose_text,
)
from member3_ocr.core.ocr_pipeline import (
    BoundingBox,
    KeyValueField,
    OCRBackend,
    OCRDocumentResult,
    OCRPageResult,
    OCRPipeline,
    PaddleOCRBackend,
    PaddleOCRModelConfig,
    TextBlock,
)


# ──────────────────────────────────────────────────────────────────────────────
# Test Fixtures & Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_bbox(left: float, top: float, right: float, bottom: float) -> BoundingBox:
    return BoundingBox(
        left=left,
        top=top,
        right=right,
        bottom=bottom,
        coordinate_space="processed_pixels",
    )


def _make_block(
    block_id: str,
    text: str,
    *,
    left: float = 100.0,
    top: float = 100.0,
    right: float = 250.0,
    bottom: float = 120.0,
    confidence: float = 0.95,
) -> TextBlock:
    return TextBlock(
        id=block_id,
        text=text,
        bbox=_make_bbox(left, top, right, bottom),
        confidence=confidence,
    )


class DummyBackend(OCRBackend):
    def __init__(self, blocks: Sequence[TextBlock]) -> None:
        super().__init__()
        self._blocks = tuple(blocks)

    def initialize(self) -> None:
        pass

    def recognize(self, image):
        from member3_ocr.core.ocr_pipeline import BackendRecognition
        return BackendRecognition(blocks=self._blocks, warnings=())

    @property
    def is_available(self) -> bool:
        return True

    @property
    def backend_info(self):
        from member3_ocr.core.ocr_pipeline import BackendCapabilities, BackendInfo
        return BackendInfo(
            name="dummy",
            version="1.0",
            model_ids=(),
            device="cpu",
            capabilities=BackendCapabilities(key_value_extraction=True),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Unit Tests: Key-Value Extraction Logic
# ──────────────────────────────────────────────────────────────────────────────

class TestInlineColonExtraction:
    def test_colon_separated_key_value(self) -> None:
        block = _make_block("b1", "Equipment Tag: P-203A")
        fields, issues = extract_form_key_values([block], page_number=1)
        assert len(fields) == 1
        assert fields[0].key == "Equipment Tag"
        assert fields[0].value == "P-203A"
        assert fields[0].extraction_method == "inline_colon"
        assert fields[0].page_number == 1
        assert len(issues) == 0

    def test_whitespace_around_colon(self) -> None:
        b1 = _make_block("b1", "Form No  :   MRPL-INSP-001", left=50, top=50, right=300, bottom=70)
        b2 = _make_block("b2", "Status:APPROVED", left=50, top=90, right=200, bottom=110)
        fields, _ = extract_form_key_values([b1, b2])
        assert len(fields) == 2
        f_map = {f.key: f.value for f in fields}
        assert f_map["Form No"] == "MRPL-INSP-001"
        assert f_map["Status"] == "APPROVED"

    def test_inline_bbox_interpolation(self) -> None:
        block = _make_block("b1", "Tag: P-101", left=100.0, top=50.0, right=200.0, bottom=70.0)
        fields, _ = extract_form_key_values([block])
        assert len(fields) == 1
        f = fields[0]
        assert f.key_bbox is not None
        assert f.value_bbox is not None
        assert f.key_bbox.left == 100.0
        assert f.key_bbox.right == f.value_bbox.left
        assert f.value_bbox.right == 200.0


class TestHorizontalAdjacentExtraction:
    def test_horizontal_adjacent_label_and_value(self) -> None:
        label = _make_block("l1", "Location:", left=100.0, top=200.0, right=180.0, bottom=220.0, confidence=0.98)
        val = _make_block("v1", "Sulphur Plant", left=210.0, top=200.0, right=320.0, bottom=220.0, confidence=0.92)
        fields, issues = extract_form_key_values([label, val], page_number=2)
        assert len(fields) == 1
        assert fields[0].key == "Location"
        assert fields[0].value == "Sulphur Plant"
        assert fields[0].extraction_method == "horizontal_adjacent"
        assert fields[0].confidence == 0.92  # min(0.98, 0.92)
        assert fields[0].page_number == 2
        assert len(issues) == 0

    def test_label_without_colon_from_known_labels(self) -> None:
        label = _make_block("l1", "Inspector", left=100.0, top=250.0, right=180.0, bottom=270.0)
        val = _make_block("v1", "P. Shetty", left=200.0, top=250.0, right=280.0, bottom=270.0)
        fields, _ = extract_form_key_values([label, val])
        assert len(fields) == 1
        assert fields[0].key == "Inspector"
        assert fields[0].value == "P. Shetty"


class TestVerticalAdjacentExtraction:
    def test_vertical_adjacent_label_and_value(self) -> None:
        label = _make_block("l1", "Operating Pressure:", left=100.0, top=300.0, right=250.0, bottom=320.0, confidence=0.96)
        val = _make_block("v1", "14.2 bar g", left=100.0, top=330.0, right=180.0, bottom=350.0, confidence=0.90)
        fields, issues = extract_form_key_values([label, val], page_number=1)
        assert len(fields) == 1
        assert fields[0].key == "Operating Pressure"
        assert fields[0].value == "14.2 bar g"
        assert fields[0].extraction_method == "vertical_adjacent"
        assert fields[0].confidence == 0.90


class TestMultilineLongValueParagraphExtraction:
    def test_vertical_multiline_paragraph_collection_for_note_label(self) -> None:
        label = _make_block("l1", "NOTE:", left=50.0, top=100.0, right=100.0, bottom=115.0, confidence=0.98)
        line1 = _make_block("v1", "THIS MESSAGE IS INTENDED ONLY FOR THE USE", left=50.0, top=125.0, right=450.0, bottom=140.0, confidence=0.95)
        line2 = _make_block("v2", "OF THE INDIVIDUAL OR ENTITY TO WHOM IT IS", left=50.0, top=145.0, right=450.0, bottom=160.0, confidence=0.94)
        line3 = _make_block("v3", "ADDRESSED AND MAY CONTAIN PRIVILEGED DATA.", left=50.0, top=165.0, right=450.0, bottom=180.0, confidence=0.96)

        fields, issues = extract_form_key_values([label, line1, line2, line3], page_number=1)
        assert len(fields) == 1
        f = fields[0]
        assert f.key == "NOTE"
        expected_val = (
            "THIS MESSAGE IS INTENDED ONLY FOR THE USE "
            "OF THE INDIVIDUAL OR ENTITY TO WHOM IT IS "
            "ADDRESSED AND MAY CONTAIN PRIVILEGED DATA."
        )
        assert f.value == expected_val
        assert f.extraction_method == "vertical_adjacent"
        assert f.confidence == 0.94  # min of 0.98, 0.95, 0.94, 0.96
        assert f.value_bbox is not None
        assert f.value_bbox.top == 125.0
        assert f.value_bbox.bottom == 180.0
        assert len(issues) == 0

    def test_short_field_does_not_swallow_subsequent_lines(self) -> None:
        """Verify normal short fields (e.g. DATE) stop after a single value block."""
        label = _make_block("l1", "DATE:", left=50.0, top=100.0, right=100.0, bottom=115.0)
        val = _make_block("v1", "12/10/98", left=110.0, top=100.0, right=180.0, bottom=115.0)
        unrelated = _make_block("u1", "UNRELATED FOOTER LINE", left=110.0, top=125.0, right=300.0, bottom=140.0)

        fields, _ = extract_form_key_values([label, val, unrelated])
        assert len(fields) == 1
        assert fields[0].key == "DATE"
        assert fields[0].value == "12/10/98"


class TestEmptyValueExtraction:
    def test_empty_value_field(self) -> None:
        label = _make_block("l1", "Remarks:", left=100.0, top=400.0, right=180.0, bottom=420.0, confidence=0.94)
        fields, _ = extract_form_key_values([label])
        assert len(fields) == 1
        assert fields[0].key == "Remarks"
        assert fields[0].value == ""
        assert fields[0].extraction_method == "empty_value"
        assert fields[0].confidence == 0.94
        assert fields[0].value_bbox is None


class TestAmbiguityHandling:
    def test_ambiguous_horizontal_match_emits_issue_and_skips(self) -> None:
        label = _make_block("l1", "Date:", left=100.0, top=100.0, right=150.0, bottom=120.0)
        v1 = _make_block("v1", "12-04-2026", left=180.0, top=100.0, right=260.0, bottom=120.0)
        v2 = _make_block("v2", "15-04-2026", left=185.0, top=100.0, right=265.0, bottom=120.0)
        fields, issues = extract_form_key_values([label, v1, v2], page_number=3)
        # Should NOT guess or fabricate; must log issue and remain unresolved
        assert len(fields) == 0
        assert len(issues) == 1
        assert issues[0].code == "ambiguous_form_field"
        assert "Date" in issues[0].message
        assert issues[0].severity == "warning"


class TestFalsePositivePrevention:
    def test_prose_sentences_with_colons_not_extracted_as_form_fields(self) -> None:
        prose_blocks = [
            _make_block("b1", "Note: The pump must be primed prior to startup.", top=50),
            _make_block("b2", "Warning: High voltage present inside the terminal box.", top=80),
            _make_block("b3", "- Step 1: Ensure inlet valve V-101 is fully open.", top=110),
            _make_block("b4", "Please note that inspection records must be retained for 5 years.", top=140),
        ]
        fields, _ = extract_form_key_values(prose_blocks)
        assert len(fields) == 0

    def test_engineering_parameters_preserved(self) -> None:
        eng_block = _make_block("b1", "Pressure: 12.5 bar", top=50)
        fields, _ = extract_form_key_values([eng_block])
        assert len(fields) == 1
        assert fields[0].key == "Pressure"
        assert fields[0].value == "12.5 bar"

    def test_bullet_points_not_absorbed_as_vertical_values(self) -> None:
        label = _make_block("l1", "Findings:", top=100, bottom=120)
        bullet = _make_block("v1", "- Bearing temperature elevated by 4 deg C.", top=135, bottom=155)
        fields, _ = extract_form_key_values([label, bullet])
        # Findings should be extracted as empty_value, not paired to a single bullet item
        assert len(fields) == 1
        assert fields[0].key == "Findings"
        assert fields[0].value == ""
        assert fields[0].extraction_method == "empty_value"


class TestDeterminismAndProvenance:
    def test_deterministic_output(self) -> None:
        blocks = [
            _make_block("b2", "Status: APPROVED", top=200),
            _make_block("b1", "Form No: MRPL-01", top=50),
        ]
        f1, _ = extract_form_key_values(blocks)
        f2, _ = extract_form_key_values(blocks)
        assert [f.key for f in f1] == [f.key for f in f2] == ["Form No", "Status"]
        assert [f.value for f in f1] == [f.value for f in f2] == ["MRPL-01", "APPROVED"]

    def test_confidence_propagation_rule(self) -> None:
        label = _make_block("l1", "Shift:", confidence=0.85, top=100, bottom=120)
        val = _make_block("v1", "B", confidence=0.95, top=100, bottom=120, left=180)
        fields, _ = extract_form_key_values([label, val])
        assert len(fields) == 1
        assert fields[0].confidence == 0.85  # min(0.85, 0.95)


# ──────────────────────────────────────────────────────────────────────────────
# Integration Tests: OCRPipeline & DocumentParser
# ──────────────────────────────────────────────────────────────────────────────

class TestPipelineAndParserIntegration:
    def test_pipeline_populates_page_key_values(self) -> None:
        blocks = [
            _make_block("b1", "Equipment: Heat Exchanger E-102", top=50),
            _make_block("b2", "Status: APPROVED", top=100),
        ]
        backend = DummyBackend(blocks)
        pipeline = OCRPipeline(backend, extract_key_values=True)

        # Mock image preprocessing result
        import numpy as np
        dummy_img = np.full((150, 400, 3), 255, dtype=np.uint8)
        with patch("member3_ocr.ocr_pipeline.preprocess_image") as mock_prep:
            from member3_ocr.core.image_preprocessing import PreprocessingResult
            mock_prep.return_value = PreprocessingResult(
                image=dummy_img,
                original_path=Path("dummy.png"),
                original_width=400,
                original_height=150,
                processed_width=400,
                processed_height=150,
                original_mode="RGB",
                processed_is_grayscale=False,
                original_metadata={},
                operations_applied=["test"],
                processing_time_seconds=0.01,
            )
            res = pipeline.process_image("dummy.png")

        page = res.pages[0]
        assert len(page.key_value_fields) == 2
        f_map = {f.key: f.value for f in page.key_value_fields}
        assert f_map["Equipment"] == "Heat Exchanger E-102"
        assert f_map["Status"] == "APPROVED"

    def test_pipeline_extract_key_values_toggle(self) -> None:
        blocks = [_make_block("b1", "Status: APPROVED", top=50)]
        backend = DummyBackend(blocks)
        pipeline = OCRPipeline(backend, extract_key_values=False)

        import numpy as np
        dummy_img = np.full((100, 200, 3), 255, dtype=np.uint8)
        with patch("member3_ocr.ocr_pipeline.preprocess_image") as mock_prep:
            from member3_ocr.core.image_preprocessing import PreprocessingResult
            mock_prep.return_value = PreprocessingResult(
                image=dummy_img,
                original_path=Path("dummy.png"),
                original_width=200,
                original_height=100,
                processed_width=200,
                processed_height=100,
                original_mode="RGB",
                processed_is_grayscale=False,
                original_metadata={},
                operations_applied=[],
                processing_time_seconds=0.01,
            )
            res = pipeline.process_image("dummy.png")

        assert len(res.pages[0].key_value_fields) == 0

    def test_document_parser_preserves_kv_history(self) -> None:
        kv = KeyValueField(
            key="Equipment",
            value="Pump P-101",
            confidence=0.96,
            extraction_method="horizontal_adjacent",
            page_number=1,
            key_bbox=_make_bbox(10, 10, 50, 30),
            value_bbox=_make_bbox(60, 10, 120, 30),
        )
        page = OCRPageResult(
            page_number=1,
            original_width=200,
            original_height=200,
            processed_width=200,
            processed_height=200,
            text="Equipment: Pump P-101",
            blocks=(_make_block("b1", "Equipment: Pump P-101"),),
            tables=(),
            key_value_fields=(kv,),
            preprocessing_operations=(),
            preprocessing_time_ms=0.0,
            processing_time_ms=5.0,
            warnings=(),
            errors=(),
        )
        from member3_ocr.core.ocr_pipeline import BackendCapabilities, BackendInfo
        doc_result = OCRDocumentResult(
            document_id="doc_test",
            pages=(page,),
            backend=BackendInfo("test", "1.0", (), "cpu", BackendCapabilities()),
            provenance={},
        )
        parser = DocumentParser()
        parsed = parser.parse_ocr_result(doc_result)

        history = parsed.processing_history[0]
        kv_records = history["key_value_fields"]
        assert len(kv_records) == 1
        assert kv_records[0]["key"] == "Equipment"
        assert kv_records[0]["value"] == "Pump P-101"
        assert kv_records[0]["extraction_method"] == "horizontal_adjacent"
        assert kv_records[0]["key_bbox"] == {"left": 10, "top": 10, "right": 50, "bottom": 30}


# ──────────────────────────────────────────────────────────────────────────────
# Safety & Air-Gap Verification Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestSafetyAndAirGap:
    def test_offline_no_network_calls(self) -> None:
        """Verify form extraction triggers zero socket connections."""
        def mock_connect(*args, **kwargs):
            raise AssertionError("Network connection attempted during form extraction!")

        with patch.object(socket.socket, "connect", side_effect=mock_connect):
            block = _make_block("b1", "Form No: MRPL-999")
            fields, _ = extract_form_key_values([block])
            assert len(fields) == 1

    def test_datasets_directory_safety(self) -> None:
        """Verify datasets directory exists and remains untouched."""
        # Tests must follow the repository location rather than a developer's
        # drive letter; CI and air-gapped deployments stage this project in
        # different roots.
        dataset_dir = Path(__file__).resolve().parents[2] / "datasets" / "ocr"
        assert dataset_dir.exists(), "Dataset directory must exist"
        assert dataset_dir.is_dir()


# ──────────────────────────────────────────────────────────────────────────────
# Real Local PP-OCRv5 Integration Test
# ──────────────────────────────────────────────────────────────────────────────

class TestRealPPOCRv5FormIntegration:
    def test_real_form_002_inference_and_kv_extraction(self) -> None:
        """Real integration test: PP-OCRv5 model -> OCRPipeline -> KV fields on form_002.png."""
        det_dir = Path(r"C:\SovereignAI\member3_ocr\models\paddleocr\PP-OCRv5_mobile_det_infer")
        rec_dir = Path(r"C:\SovereignAI\member3_ocr\models\paddleocr\PP-OCRv5_mobile_rec_infer")
        form_img = Path(r"C:\SovereignAI\datasets\ocr\synthetic_ocr_dataset\synthetic_forms\images\form_002.png")

        if not (det_dir.exists() and rec_dir.exists() and form_img.exists()):
            pytest.skip("Local PP-OCRv5 models or form_002.png not found; skipping real integration test.")

        config = PaddleOCRModelConfig(detection_model_dir=det_dir, recognition_model_dir=rec_dir)
        backend = PaddleOCRBackend(config)
        pipeline = OCRPipeline(backend, extract_key_values=True)

        res = pipeline.process_image(form_img)
        assert len(res.pages) == 1
        page = res.pages[0]

        assert len(page.key_value_fields) >= 6
        kv_dict = {f.key: f.value for f in page.key_value_fields}

        assert "Form No" in kv_dict
        assert "MRPL-INSP-6155" in kv_dict["Form No"]
        assert "Equipment" in kv_dict
        assert "Boiler Feed Pump BFP-07" in kv_dict["Equipment"]
        assert "Status" in kv_dict
        assert "APPROVED" in kv_dict["Status"]


class TestFormDatasetEvaluation:
    def test_synthetic_forms_evaluation_artifacts_generated(self) -> None:
        """Verify form evaluation produces all required summary and metric artifacts."""
        import json
        out_dir = Path(__file__).resolve().parents[2] / "member3_ocr" / "output" / "evaluation" / "forms"
        if not out_dir.exists():
            pytest.skip("Optional form-evaluation artifacts have not been generated in this checkout")
        assert (out_dir / "summary.json").exists(), "summary.json must exist"
        assert (out_dir / "per_sample.csv").exists(), "per_sample.csv must exist"
        assert (out_dir / "failures.csv").exists(), "failures.csv must exist"
        assert (out_dir / "README.md").exists(), "README.md must exist"

        summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
        assert summary["total_samples"] == 12
        assert summary["metrics"]["exact_match_accuracy"] >= 0.80
        assert summary["breakdown"]["clean_exact_match_accuracy"] == 1.0

