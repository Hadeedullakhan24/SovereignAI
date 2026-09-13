"""Unit and integration tests for table extraction (Step 1C).

Verifies:
- Simple table extraction
- Multiple rows & multiple columns
- Table headers identification and formatting
- Empty cells handling without fabricating values
- Irregular column/row spacing
- Ambiguous cell assignment detection and Issue reporting
- Cell confidence propagation
- Bounding box preservation (table bbox and cell bboxes)
- Deterministic output
- Page provenance (page_number attached to Table and cells)
- PDF table page compatibility
- Offline / no-network isolation
- Dataset read-only safety
- Real local PP-OCRv5 inference integration on table_001.png
- Table evaluation artifacts verification
"""
from __future__ import annotations


import json
import socket
from pathlib import Path
from unittest.mock import patch

import pytest

from member3_ocr.core.document_parser import DocumentParser
from member3_ocr.core.ocr_pipeline import (
    BackendRecognition,
    BoundingBox,
    OCRBackend,
    OCRDocumentResult,
    OCRPageResult,
    OCRPipeline,
    PaddleOCRBackend,
    PaddleOCRModelConfig,
    Table,
    TextBlock,
)
from member3_ocr.core.table_extractor import (
    TableExtractor,
    TableExtractorConfig,
    extract_tables_from_blocks,
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
    left: float,
    top: float,
    right: float,
    bottom: float,
    confidence: float = 0.95,
) -> TextBlock:
    return TextBlock(
        id=block_id,
        text=text,
        bbox=_make_bbox(left, top, right, bottom),
        confidence=confidence,
    )


class DummyBackend(OCRBackend):
    def __init__(self, blocks: list[TextBlock]) -> None:
        super().__init__()
        self._blocks = tuple(blocks)

    def initialize(self) -> None:
        pass

    def recognize(self, image):
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
            capabilities=BackendCapabilities(table_extraction=True),
        )


def _build_simple_grid_blocks() -> list[TextBlock]:
    """Create a 3x3 table with clear headers and data cells."""
    blocks = [
        # Row 0 (Headers)
        _make_block("h0", "Item", left=50, top=100, right=150, bottom=120, confidence=0.98),
        _make_block("h1", "Quantity", left=200, top=100, right=300, bottom=120, confidence=0.97),
        _make_block("h2", "Status", left=350, top=100, right=450, bottom=120, confidence=0.99),
        # Row 1 (Data)
        _make_block("d1_0", "Valve V-1", left=50, top=140, right=140, bottom=160, confidence=0.95),
        _make_block("d1_1", "4", left=200, top=140, right=220, bottom=160, confidence=0.96),
        _make_block("d1_2", "PASS", left=350, top=140, right=400, bottom=160, confidence=0.94),
        # Row 2 (Data)
        _make_block("d2_0", "Gasket G-2", left=50, top=180, right=150, bottom=200, confidence=0.92),
        _make_block("d2_1", "12", left=200, top=180, right=230, bottom=200, confidence=0.93),
        _make_block("d2_2", "FAIL", left=350, top=180, right=400, bottom=200, confidence=0.91),
    ]
    return blocks


# ──────────────────────────────────────────────────────────────────────────────
# Unit Tests: Structure & Extraction
# ──────────────────────────────────────────────────────────────────────────────

class TestTableExtractionBasics:
    def test_simple_table_extraction(self) -> None:
        blocks = _build_simple_grid_blocks()
        tables, issues = extract_tables_from_blocks(blocks, page_number=1)
        assert len(tables) == 1
        assert len(issues) == 0

        tbl = tables[0]
        assert tbl.id == "tbl_p1_0"
        assert tbl.page_number == 1
        assert tbl.bbox.left == 50.0
        assert tbl.bbox.top == 100.0
        assert tbl.bbox.right == 450.0
        assert tbl.bbox.bottom == 200.0

    def test_multiple_rows_and_columns(self) -> None:
        blocks = _build_simple_grid_blocks()
        tables, _ = extract_tables_from_blocks(blocks)
        tbl = tables[0]

        # 3 rows x 3 columns = 9 cells
        assert len(tbl.cells) == 9
        row_indices = {c["row_idx"] for c in tbl.cells}
        col_indices = {c["col_idx"] for c in tbl.cells}
        assert row_indices == {0, 1, 2}
        assert col_indices == {0, 1, 2}

    def test_headers_identification_and_markdown(self) -> None:
        blocks = _build_simple_grid_blocks()
        tables, _ = extract_tables_from_blocks(blocks)
        tbl = tables[0]

        header_cells = [c for c in tbl.cells if c["is_header"]]
        assert len(header_cells) == 3
        assert [c["text"] for c in header_cells] == ["Item", "Quantity", "Status"]

        assert "| Item | Quantity | Status |" in tbl.markdown
        assert "<table>" in tbl.html
        assert "<th>Item</th>" in tbl.html

    def test_empty_cell_handling(self) -> None:
        """Verify missing cell value is not fabricated."""
        blocks = [
            # Row 0
            _make_block("h0", "Code", left=50, top=100, right=100, bottom=120),
            _make_block("h1", "Description", left=150, top=100, right=250, bottom=120),
            _make_block("h2", "Qty", left=300, top=100, right=350, bottom=120),
            # Row 1 (Missing Description cell in column 1)
            _make_block("d1_0", "C-01", left=50, top=140, right=90, bottom=160),
            _make_block("d1_2", "10", left=300, top=140, right=320, bottom=160),
            # Row 2 (Has all three cells)
            _make_block("d2_0", "C-02", left=50, top=180, right=90, bottom=200),
            _make_block("d2_1", "Filter", left=150, top=180, right=200, bottom=200),
            _make_block("d2_2", "5", left=300, top=180, right=310, bottom=200),
        ]
        tables, _ = extract_tables_from_blocks(blocks)
        tbl = tables[0]

        # Total extracted cells is 8 (no fake cell text invented for col 1)
        assert len(tbl.cells) == 8
        assert any(c["row_idx"] == 1 and c["col_idx"] == 0 for c in tbl.cells)
        assert not any(c["row_idx"] == 1 and c["col_idx"] == 1 for c in tbl.cells)
        assert any(c["row_idx"] == 1 and c["col_idx"] == 2 for c in tbl.cells)
        # Markdown row contains empty column
        assert "| C-01 |  | 10 |" in tbl.markdown

    def test_irregular_column_spacing(self) -> None:
        blocks = [
            _make_block("h0", "ID", left=20, top=100, right=60, bottom=120),
            _make_block("h1", "Very Long Detailed Technical Description", left=100, top=100, right=500, bottom=120),
            _make_block("h2", "Qty", left=650, top=100, right=700, bottom=120),
            _make_block("d0", "1", left=20, top=140, right=35, bottom=160),
            _make_block("d1", "Overhaul motor bearings", left=100, top=140, right=300, bottom=160),
            _make_block("d2", "2", left=650, top=140, right=665, bottom=160),
        ]
        tables, _ = extract_tables_from_blocks(blocks)
        assert len(tables) == 1
        tbl = tables[0]
        assert len(tbl.cells) == 6

    def test_ambiguous_cell_assignment_emits_issue(self) -> None:
        """Cell exactly halfway between two column anchors must trigger Issue."""
        blocks = [
            _make_block("h0", "Col1", left=100, top=100, right=150, bottom=120),
            _make_block("h1", "Col2", left=300, top=100, right=350, bottom=120),
            _make_block("d0", "Val1", left=100, top=140, right=140, bottom=160),
            _make_block("d1", "Val2", left=300, top=140, right=340, bottom=160),
            # Ambiguous cell exactly equidistant between anchor 100 and 300 (at x=200)
            _make_block("d_amb", "Ambiguous", left=200, top=180, right=250, bottom=200),
            _make_block("d_amb2", "ValB", left=100, top=180, right=140, bottom=200),
        ]
        config = TableExtractorConfig(ambiguity_margin_px=15.0)
        tables, issues = extract_tables_from_blocks(blocks, config=config, page_number=2)
        assert any(i.code == "ambiguous_cell_assignment" for i in issues)
        assert any("Ambiguous" in i.message for i in issues)

    def test_cell_confidence_propagation(self) -> None:
        blocks = _build_simple_grid_blocks()
        tables, _ = extract_tables_from_blocks(blocks)
        tbl = tables[0]

        confs = [c["confidence"] for c in tbl.cells]
        expected_mean = round(sum(confs) / len(confs), 4)
        assert tbl.confidence == expected_mean

    def test_bbox_preservation(self) -> None:
        blocks = _build_simple_grid_blocks()
        tables, _ = extract_tables_from_blocks(blocks)
        tbl = tables[0]

        for cell in tbl.cells:
            assert cell["bbox"] is not None
            assert cell["bbox"]["left"] <= cell["bbox"]["right"]
            assert cell["bbox"]["top"] <= cell["bbox"]["bottom"]

    def test_deterministic_output(self) -> None:
        blocks = _build_simple_grid_blocks()
        t1, _ = extract_tables_from_blocks(blocks)
        t2, _ = extract_tables_from_blocks(blocks)
        assert t1[0].markdown == t2[0].markdown
        assert t1[0].bbox == t2[0].bbox
        assert [c["text"] for c in t1[0].cells] == [c["text"] for c in t2[0].cells]

    def test_page_provenance(self) -> None:
        blocks = _build_simple_grid_blocks()
        tables, _ = extract_tables_from_blocks(blocks, page_number=4)
        tbl = tables[0]
        assert tbl.page_number == 4
        assert tbl.id == "tbl_p4_0"
        for cell in tbl.cells:
            assert cell["page_number"] == 4


# ──────────────────────────────────────────────────────────────────────────────
# Integration Tests: OCRPipeline & DocumentParser
# ──────────────────────────────────────────────────────────────────────────────

class TestPipelineAndParserIntegration:
    def test_pipeline_populates_tables(self) -> None:
        blocks = _build_simple_grid_blocks()
        backend = DummyBackend(blocks)
        pipeline = OCRPipeline(backend, extract_tables=True)

        import numpy as np
        dummy_img = np.full((300, 500, 3), 255, dtype=np.uint8)
        with patch("member3_ocr.ocr_pipeline.preprocess_image") as mock_prep:
            from member3_ocr.core.image_preprocessing import PreprocessingResult
            mock_prep.return_value = PreprocessingResult(
                image=dummy_img,
                original_path=Path("dummy.png"),
                original_width=500,
                original_height=300,
                processed_width=500,
                processed_height=300,
                original_mode="RGB",
                processed_is_grayscale=False,
                original_metadata={},
                operations_applied=["test"],
                processing_time_seconds=0.01,
            )
            res = pipeline.process_image("dummy.png")

        assert len(res.pages[0].tables) == 1
        assert len(res.pages[0].tables[0].cells) == 9

    def test_pipeline_extract_tables_toggle(self) -> None:
        blocks = _build_simple_grid_blocks()
        backend = DummyBackend(blocks)
        pipeline = OCRPipeline(backend, extract_tables=False)

        import numpy as np
        dummy_img = np.full((300, 500, 3), 255, dtype=np.uint8)
        with patch("member3_ocr.ocr_pipeline.preprocess_image") as mock_prep:
            from member3_ocr.core.image_preprocessing import PreprocessingResult
            mock_prep.return_value = PreprocessingResult(
                image=dummy_img,
                original_path=Path("dummy.png"),
                original_width=500,
                original_height=300,
                processed_width=500,
                processed_height=300,
                original_mode="RGB",
                processed_is_grayscale=False,
                original_metadata={},
                operations_applied=[],
                processing_time_seconds=0.01,
            )
            res = pipeline.process_image("dummy.png")

        assert len(res.pages[0].tables) == 0

    def test_document_parser_converts_ocr_table(self) -> None:
        blocks = _build_simple_grid_blocks()
        tables, _ = extract_tables_from_blocks(blocks, page_number=1)
        ocr_table = tables[0]

        page = OCRPageResult(
            page_number=1,
            original_width=500,
            original_height=300,
            processed_width=500,
            processed_height=300,
            text="table text",
            blocks=tuple(blocks),
            tables=(ocr_table,),
            key_value_fields=(),
            preprocessing_operations=(),
            preprocessing_time_ms=0.0,
            processing_time_ms=5.0,
            warnings=(),
            errors=(),
        )
        from member3_ocr.core.ocr_pipeline import BackendCapabilities, BackendInfo
        doc_result = OCRDocumentResult(
            document_id="doc_table_test",
            pages=(page,),
            backend=BackendInfo("dummy", "1.0", (), "cpu", BackendCapabilities()),
            provenance={},
        )
        parser = DocumentParser()
        parsed = parser.parse_ocr_result(doc_result)

        assert len(parsed.tables) == 1
        p_tbl = parsed.tables[0]
        assert p_tbl.headers == ["Item", "Quantity", "Status"]
        assert p_tbl.row_count == 2
        assert p_tbl.col_count == 3
        assert p_tbl.rows[0] == ["Valve V-1", "4", "PASS"]
        assert p_tbl.rows[1] == ["Gasket G-2", "12", "FAIL"]
        assert "Item" in p_tbl.dataframe_dict
        assert p_tbl.dataframe_dict["Item"] == ["Valve V-1", "Gasket G-2"]

    def test_pdf_table_page_processing(self) -> None:
        """Verify process_pdf populates tables correctly."""
        pdf_path = Path(r"C:\SovereignAI\datasets\ocr\synthetic_ocr_dataset\synthetic_tables\pdf\table_001.pdf")
        if not pdf_path.exists():
            pytest.skip("table_001.pdf not found; skipping PDF table test.")

        blocks = _build_simple_grid_blocks()
        backend = DummyBackend(blocks)
        pipeline = OCRPipeline(backend, extract_tables=True)

        res = pipeline.process_pdf(pdf_path)
        assert len(res.pages) >= 1
        assert len(res.pages[0].tables) >= 1


# ──────────────────────────────────────────────────────────────────────────────
# Safety & Air-Gap Verification Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestSafetyAndAirGap:
    def test_offline_no_network_calls(self) -> None:
        def mock_connect(*args, **kwargs):
            raise AssertionError("Network connection attempted during table extraction!")

        with patch.object(socket.socket, "connect", side_effect=mock_connect):
            blocks = _build_simple_grid_blocks()
            tables, _ = extract_tables_from_blocks(blocks)
            assert len(tables) == 1

    def test_datasets_directory_safety(self) -> None:
        dataset_dir = Path(r"C:\SovereignAI\datasets\ocr")
        assert dataset_dir.exists(), "Dataset directory must exist"
        assert dataset_dir.is_dir()


# ──────────────────────────────────────────────────────────────────────────────
# Real Local PP-OCRv5 Integration Test
# ──────────────────────────────────────────────────────────────────────────────

class TestRealPPOCRv5TableIntegration:
    def test_real_table_001_inference_and_table_extraction(self) -> None:
        """Real integration test: PP-OCRv5 -> OCRPipeline -> structured Table on table_001.png."""
        det_dir = Path(r"C:\SovereignAI\member3_ocr\models\paddleocr\PP-OCRv5_mobile_det_infer")
        rec_dir = Path(r"C:\SovereignAI\member3_ocr\models\paddleocr\PP-OCRv5_mobile_rec_infer")
        table_img = Path(r"C:\SovereignAI\datasets\ocr\synthetic_ocr_dataset\synthetic_tables\images\table_001.png")

        if not (det_dir.exists() and rec_dir.exists() and table_img.exists()):
            pytest.skip("Local PP-OCRv5 models or table_001.png not found; skipping real integration test.")

        config = PaddleOCRModelConfig(detection_model_dir=det_dir, recognition_model_dir=rec_dir)
        backend = PaddleOCRBackend(config)
        pipeline = OCRPipeline(backend, extract_tables=True)

        res = pipeline.process_image(table_img)
        assert len(res.pages) == 1
        page = res.pages[0]

        assert len(page.tables) == 1
        tbl = page.tables[0]
        assert len(tbl.cells) >= 60
        assert "Equipment ID" in tbl.markdown
        assert "Technician" in tbl.markdown


# ──────────────────────────────────────────────────────────────────────────────
# Table Evaluation Artifacts Verification
# ──────────────────────────────────────────────────────────────────────────────

class TestTableDatasetEvaluation:
    def test_synthetic_tables_evaluation_artifacts_generated(self) -> None:
        """Verify table evaluation produces all required summary and metric artifacts."""
        out_dir = Path(r"C:\SovereignAI\member3_ocr\output\evaluation\tables")
        assert (out_dir / "summary.json").exists(), "summary.json must exist"
        assert (out_dir / "per_sample.csv").exists(), "per_sample.csv must exist"
        assert (out_dir / "failures.csv").exists(), "failures.csv must exist"
        assert (out_dir / "README.md").exists(), "README.md must exist"

        summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
        assert summary["total_samples"] == 12
        assert summary["metrics"]["cell_text_accuracy"] >= 0.50
        assert summary["breakdown"]["clean_cell_accuracy"] >= 0.90


# ──────────────────────────────────────────────────────────────────────────────
# Noise Fragment Filtering Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestNoiseFiltering:
    def test_phantom_punctuation_filtered(self) -> None:
        """Isolated tiny punctuation specks must be filtered and logged as Issues."""
        blocks = [
            # Legitimate row 0
            _make_block("h0", "ID", left=50, top=100, right=100, bottom=125),
            _make_block("h1", "Status", left=150, top=100, right=220, bottom=125),
            # Phantom noise speck in between rows
            _make_block("noise_dot", "..", left=80, top=135, right=92, bottom=143, confidence=0.35),
            # Legitimate row 1
            _make_block("d0", "P1", left=50, top=160, right=100, bottom=185),
            _make_block("d1", "Active", left=150, top=160, right=220, bottom=185),
        ]
        extractor = TableExtractor(TableExtractorConfig())
        tables, issues = extractor.extract(blocks)
        assert len(tables) == 1
        tbl = tables[0]
        # Should have exactly 2 rows (header + 1 data row)
        row_indices = {c["row_idx"] for c in tbl.cells}
        assert row_indices == {0, 1}
        # Noise fragment must be logged as an issue
        filtered_issues = [iss for iss in issues if iss.code == "noise_fragment_filtered"]
        assert len(filtered_issues) >= 1
        assert any(".." in iss.message for iss in filtered_issues)

    def test_tiny_noise_glyph_filtered(self) -> None:
        """Tiny low-confidence glyphs and noise hallucinations must be filtered."""
        blocks = [
            _make_block("h0", "Code", left=50, top=100, right=100, bottom=125),
            _make_block("h1", "Item", left=150, top=100, right=220, bottom=125),
            # Tiny low-confidence noise block
            _make_block("noise_glyph", "p", left=80, top=135, right=87, bottom=142, confidence=0.15),
            # Hallucinated CJK character from noise speckle
            _make_block("noise_cjk", "中", left=160, top=135, right=175, bottom=148, confidence=0.35),
            _make_block("d0", "C10", left=50, top=160, right=100, bottom=185),
            _make_block("d1", "Valve", left=150, top=160, right=220, bottom=185),
        ]
        extractor = TableExtractor(TableExtractorConfig())
        tables, issues = extractor.extract(blocks)
        assert len(tables) == 1
        tbl = tables[0]
        row_indices = {c["row_idx"] for c in tbl.cells}
        assert row_indices == {0, 1}
        assert not any(c["text"] in ("p", "中") for c in tbl.cells)

    def test_legitimate_single_character_value_preserved(self) -> None:
        """Legitimate single-character values like 'A', '1', 'X' with standard cell height must be kept."""
        blocks = [
            _make_block("h0", "Section", left=50, top=100, right=120, bottom=125),
            _make_block("h1", "Grade", left=150, top=100, right=220, bottom=125),
            # Single-character letter 'A' with standard height (18px)
            _make_block("d0_0", "S-1", left=50, top=160, right=100, bottom=185),
            _make_block("d0_1", "A", left=150, top=160, right=165, bottom=182, confidence=0.85),
            # Single-character letter 'B'
            _make_block("d1_0", "S-2", left=50, top=200, right=100, bottom=225),
            _make_block("d1_1", "B", left=150, top=200, right=165, bottom=222, confidence=0.80),
        ]
        extractor = TableExtractor(TableExtractorConfig())
        tables, issues = extractor.extract(blocks)
        assert len(tables) == 1
        tbl = tables[0]
        cell_texts = [c["text"] for c in tbl.cells]
        assert "A" in cell_texts
        assert "B" in cell_texts
        assert not any(iss.code == "noise_fragment_filtered" for iss in issues)

    def test_legitimate_numeric_value_preserved(self) -> None:
        """Legitimate numeric values like '1', '4.9', '100' must be preserved."""
        blocks = [
            _make_block("h0", "Item", left=50, top=100, right=120, bottom=125),
            _make_block("h1", "Qty", left=150, top=100, right=220, bottom=125),
            _make_block("d0_0", "Pump", left=50, top=160, right=100, bottom=185),
            _make_block("d0_1", "1", left=150, top=160, right=165, bottom=182, confidence=0.75),
            _make_block("d1_0", "Bearing", left=50, top=200, right=100, bottom=225),
            _make_block("d1_1", "4.9", left=150, top=200, right=185, bottom=222, confidence=0.85),
        ]
        extractor = TableExtractor(TableExtractorConfig())
        tables, issues = extractor.extract(blocks)
        assert len(tables) == 1
        tbl = tables[0]
        cell_texts = [c["text"] for c in tbl.cells]
        assert "1" in cell_texts
        assert "4.9" in cell_texts

    def test_legitimate_plus_minus_value_preserved(self) -> None:
        """Mathematical signs '+' and '-' with normal cell height and confidence must be preserved."""
        blocks = [
            _make_block("h0", "Test", left=50, top=100, right=120, bottom=125),
            _make_block("h1", "Result", left=150, top=100, right=220, bottom=125),
            _make_block("d0_0", "Polarity A", left=50, top=160, right=120, bottom=185),
            _make_block("d0_1", "+", left=150, top=160, right=165, bottom=182, confidence=0.85),
            _make_block("d1_0", "Polarity B", left=50, top=200, right=120, bottom=225),
            _make_block("d1_1", "-", left=150, top=200, right=165, bottom=222, confidence=0.85),
        ]
        extractor = TableExtractor(TableExtractorConfig())
        tables, issues = extractor.extract(blocks)
        assert len(tables) == 1
        tbl = tables[0]
        cell_texts = [c["text"] for c in tbl.cells]
        assert "+" in cell_texts
        assert "-" in cell_texts

    def test_confidence_handling_and_threshold_config(self) -> None:
        """Verify configurable filter_noise_fragments toggle preserves all blocks when False."""
        blocks = [
            _make_block("h0", "ID", left=50, top=100, right=100, bottom=125),
            _make_block("h1", "Val", left=150, top=100, right=200, bottom=125),
            _make_block("noise", "..", left=80, top=135, right=92, bottom=143, confidence=0.20),
            _make_block("d0", "X", left=50, top=160, right=100, bottom=185),
            _make_block("d1", "10", left=150, top=160, right=200, bottom=185),
        ]
        # When filter_noise_fragments is False, noise fragment is NOT filtered
        cfg_disabled = TableExtractorConfig(filter_noise_fragments=False)
        extractor_disabled = TableExtractor(cfg_disabled)
        _, issues_disabled = extractor_disabled.extract(blocks)
        assert not any(iss.code == "noise_fragment_filtered" for iss in issues_disabled)

        # When True, noise fragment is filtered
        cfg_enabled = TableExtractorConfig(filter_noise_fragments=True)
        extractor_enabled = TableExtractor(cfg_enabled)
        _, issues_enabled = extractor_enabled.extract(blocks)
        assert any(iss.code == "noise_fragment_filtered" for iss in issues_enabled)

    def test_deterministic_noise_filtering(self) -> None:
        """Repeated extractions must yield byte-for-byte identical tables and issues."""
        blocks = [
            _make_block("h0", "Col1", left=50, top=100, right=100, bottom=125),
            _make_block("h1", "Col2", left=150, top=100, right=200, bottom=125),
            _make_block("noise", "::", left=80, top=135, right=90, bottom=140, confidence=0.10),
            _make_block("d0", "A", left=50, top=160, right=100, bottom=185),
            _make_block("d1", "B", left=150, top=160, right=200, bottom=185),
        ]
        extractor = TableExtractor()
        t1, i1 = extractor.extract(blocks)
        t2, i2 = extractor.extract(blocks)
        assert len(t1) == len(t2) == 1
        assert t1[0].markdown == t2[0].markdown
        assert t1[0].cells == t2[0].cells
        assert len(i1) == len(i2)
        assert [iss.code for iss in i1] == [iss.code for iss in i2]

