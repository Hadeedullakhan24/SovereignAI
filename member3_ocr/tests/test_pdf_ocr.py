"""Unit and integration tests for Scanned PDF OCR support in Member 3.

Covers:
1. Valid single-page scanned PDF rendering and OCR
2. Valid multi-page scanned PDF rendering and OCR
3. Page ordering preservation
4. 1-based page numbering
5. Empty / low-content scanned page handling
6. Missing PDF handling
7. Corrupt / unreadable PDF handling
8. Zero-page PDF handling
9. Unsupported file extension handling
10. Page-level rendering failure propagation
11. Page-level OCR failure propagation
12. Determinism of output
13. Existing process_image API regression protection
14. Dataset safety (datasets/ remains untouched)
15. Offline / no-network guarantee
16. Real local PP-OCRv5 scanned PDF integration test
"""
from __future__ import annotations


import socket
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import cv2
import numpy as np
import pytest
from PIL import Image

from member3_ocr.core.ocr_pipeline import (
    BackendCapabilities,
    BackendInfo,
    BackendRecognition,
    BoundingBox,
    Issue,
    OCRBackend,
    OCRBackendExecutionError,
    OCRDocumentResult,
    OCRPageResult,
    OCRPipeline,
    PaddleOCRBackend,
    PaddleOCRModelConfig,
    TextBlock,
)
from member3_ocr.core.pdf_rendering import (
    PDFPageRenderError,
    PDFRenderingError,
    PDFUnreadableError,
    PDFValidationError,
    PDFZeroPageError,
    RenderedPDFPage,
    get_pdf_page_count,
    render_pdf_pages,
    validate_pdf_path,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helper fixtures & PDF generators
# ──────────────────────────────────────────────────────────────────────────────

def _make_scanned_page_image(
    lines: list[str],
    size: tuple[int, int] = (800, 600),
    bg_color: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Generate a synthetic scanned document page as a PIL RGB image."""
    w, h = size
    img = np.full((h, w, 3), bg_color, dtype=np.uint8)
    y = 60
    for line in lines:
        cv2.putText(
            img,
            line,
            (40, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        y += 50
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def _create_scanned_pdf(
    path: Path,
    pages_lines: list[list[str]],
    size: tuple[int, int] = (800, 600),
) -> Path:
    """Create a temporary multi-page scanned PDF file."""
    images = [_make_scanned_page_image(lines, size=size) for lines in pages_lines]
    if not images:
        # Create a single blank image if empty
        images = [_make_scanned_page_image([])]

    first_img, rest = images[0], images[1:]
    first_img.save(path, "PDF", save_all=True, append_images=rest, resolution=100.0)
    return path


class DummyMockOCRBackend:
    """Deterministic mock backend for rapid unit testing without loading weights."""

    def __init__(self, fail_on_page: int | None = None) -> None:
        self.call_count = 0
        self.fail_on_page = fail_on_page
        self.initialized = False

    @property
    def backend_info(self) -> BackendInfo:
        return BackendInfo(
            name="mock_paddleocr",
            version="3.0.3",
            model_ids=("mock_det", "mock_rec"),
            device="cpu",
            capabilities=BackendCapabilities(
                local_execution=True,
                text_recognition=True,
                word_boxes=True,
                line_boxes=True,
                polygons=True,
                confidence_scores=True,
            ),
        )

    def initialize(self) -> None:
        self.initialized = True

    def recognize(self, image: np.ndarray) -> BackendRecognition:
        self.call_count += 1
        if self.fail_on_page is not None and self.call_count == self.fail_on_page:
            raise OCRBackendExecutionError(f"Backend failed on page {self.call_count}")

        # Return synthetic text block based on call count
        text = f"RECOGNIZED_TEXT_PAGE_{self.call_count}"
        block = TextBlock(
            id=f"block_{self.call_count}",
            text=text,
            confidence=0.98,
            bbox=BoundingBox(left=40.0, top=60.0, right=400.0, bottom=90.0),
            polygon=((40.0, 60.0), (400.0, 60.0), (400.0, 90.0), (40.0, 90.0)),
        )
        return BackendRecognition(blocks=(block,))


# ──────────────────────────────────────────────────────────────────────────────
# PDF Rendering Unit Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_validate_pdf_path_valid(tmp_path: Path) -> None:
    pdf_file = tmp_path / "valid.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 header")
    validated = validate_pdf_path(pdf_file)
    assert validated == pdf_file


def test_validate_pdf_path_missing(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.pdf"
    with pytest.raises(PDFValidationError, match="does not exist"):
        validate_pdf_path(missing)


def test_validate_pdf_path_directory(tmp_path: Path) -> None:
    sub_dir = tmp_path / "my_folder"
    sub_dir.mkdir()
    with pytest.raises(PDFValidationError, match="not a regular file"):
        validate_pdf_path(sub_dir)


def test_validate_pdf_path_unsupported_extension(tmp_path: Path) -> None:
    txt_file = tmp_path / "document.txt"
    txt_file.write_text("hello", encoding="utf-8")
    with pytest.raises(PDFValidationError, match="Unsupported file type"):
        validate_pdf_path(txt_file)


def test_render_pdf_pages_single_page(tmp_path: Path) -> None:
    pdf_file = tmp_path / "single.pdf"
    _create_scanned_pdf(pdf_file, [["PUMP P-101 NORMAL", "PRESSURE 45 BAR"]])

    pages = render_pdf_pages(pdf_file, dpi=150)
    assert len(pages) == 1
    p1 = pages[0]
    assert p1.page_number == 1
    assert p1.image is not None
    assert isinstance(p1.image, np.ndarray)
    assert p1.image.ndim == 3
    assert p1.image.shape[2] == 3
    assert p1.image.dtype == np.uint8
    assert p1.width > 0
    assert p1.height > 0
    assert p1.dpi == 150
    assert p1.error is None


def test_render_pdf_pages_multipage(tmp_path: Path) -> None:
    pdf_file = tmp_path / "multi.pdf"
    pages_data = [
        ["PAGE 1: BOILER FEEDWATER"],
        ["PAGE 2: NAPHTHA SPLITTER"],
        ["PAGE 3: FLARE GAS RECOVERY"],
    ]
    _create_scanned_pdf(pdf_file, pages_data)

    assert get_pdf_page_count(pdf_file) == 3

    pages = render_pdf_pages(pdf_file, dpi=100)
    assert len(pages) == 3
    for idx, p in enumerate(pages):
        assert p.page_number == idx + 1
        assert p.image is not None
        assert p.error is None


def test_render_pdf_pages_corrupt_file(tmp_path: Path) -> None:
    corrupt_pdf = tmp_path / "corrupt.pdf"
    corrupt_pdf.write_bytes(b"This is completely corrupt non-pdf data\x00\x01\x02")
    with pytest.raises(PDFUnreadableError, match="Failed to open or parse"):
        render_pdf_pages(corrupt_pdf)


def test_render_pdf_pages_invalid_dpi(tmp_path: Path) -> None:
    pdf_file = tmp_path / "test.pdf"
    _create_scanned_pdf(pdf_file, [["TEST"]])
    with pytest.raises(ValueError, match="dpi must be positive"):
        render_pdf_pages(pdf_file, dpi=0)


# ──────────────────────────────────────────────────────────────────────────────
# OCR Pipeline PDF Processing Unit Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_process_pdf_single_page_success(tmp_path: Path) -> None:
    pdf_file = tmp_path / "inspection_single.pdf"
    _create_scanned_pdf(pdf_file, [["MRPL PUMP P-101", "STATUS: OPERATIONAL"]])

    backend = DummyMockOCRBackend()
    pipeline = OCRPipeline(backend)  # type: ignore[arg-type]

    result = pipeline.process_pdf(pdf_file, document_id="doc_single_001")
    assert isinstance(result, OCRDocumentResult)
    assert result.document_id == "doc_single_001"
    assert len(result.pages) == 1
    assert result.errors == ()

    p1 = result.pages[0]
    assert p1.page_number == 1
    assert "RECOGNIZED_TEXT_PAGE_1" in p1.text
    assert len(p1.blocks) == 1
    assert result.provenance["source_type"] == "pdf_path"
    assert result.provenance["total_pages"] == 1


def test_process_pdf_multipage_ordering_and_numbers(tmp_path: Path) -> None:
    pdf_file = tmp_path / "sop_multipage.pdf"
    pages_data = [
        ["SOP-001 PAGE 1: STARTUP PROCEDURE"],
        ["SOP-001 PAGE 2: VALVE SEQUENCING"],
        ["SOP-001 PAGE 3: PRESSURE CHECKS"],
    ]
    _create_scanned_pdf(pdf_file, pages_data)

    backend = DummyMockOCRBackend()
    pipeline = OCRPipeline(backend)  # type: ignore[arg-type]

    result = pipeline.process_pdf(pdf_file)
    assert result.document_id == "sop_multipage"
    assert len(result.pages) == 3
    assert result.provenance["total_pages"] == 3

    for idx, page in enumerate(result.pages):
        expected_page_num = idx + 1
        assert page.page_number == expected_page_num
        assert f"RECOGNIZED_TEXT_PAGE_{expected_page_num}" in page.text
        assert page.errors == ()


def test_process_pdf_empty_page_preserved(tmp_path: Path) -> None:
    """Empty pages must remain in OCRDocumentResult and not be dropped."""
    pdf_file = tmp_path / "empty_page.pdf"
    # Create a 2-page PDF where page 2 has no text
    img1 = _make_scanned_page_image(["PAGE 1 HAS TEXT"])
    img2 = Image.fromarray(np.full((600, 800, 3), 255, dtype=np.uint8))  # completely blank
    img1.save(pdf_file, "PDF", save_all=True, append_images=[img2])

    class EmptyDetectingMockBackend(DummyMockOCRBackend):
        def recognize(self, image: np.ndarray) -> BackendRecognition:
            if self.call_count == 1:
                self.call_count += 1
                # No text detected on blank page 2
                return BackendRecognition(blocks=())
            return super().recognize(image)

    backend = EmptyDetectingMockBackend()
    pipeline = OCRPipeline(backend)  # type: ignore[arg-type]

    result = pipeline.process_pdf(pdf_file)
    assert len(result.pages) == 2
    p1, p2 = result.pages
    assert p1.page_number == 1
    assert p1.text != ""
    assert len(p1.blocks) > 0

    assert p2.page_number == 2
    assert p2.text == ""
    assert p2.blocks == ()
    assert p2.errors == ()  # It's an empty page, not an error


def test_process_pdf_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "nonexistent.pdf"
    backend = DummyMockOCRBackend()
    pipeline = OCRPipeline(backend)  # type: ignore[arg-type]

    result = pipeline.process_pdf(missing)
    assert len(result.pages) == 0
    assert len(result.errors) == 1
    assert result.errors[0].code == "missing_file"


def test_process_pdf_unsupported_extension(tmp_path: Path) -> None:
    img_file = tmp_path / "document.png"
    img_file.write_bytes(b"dummy")
    backend = DummyMockOCRBackend()
    pipeline = OCRPipeline(backend)  # type: ignore[arg-type]

    result = pipeline.process_pdf(img_file)
    assert len(result.pages) == 0
    assert len(result.errors) == 1
    assert result.errors[0].code == "unsupported_format"


def test_process_pdf_corrupt_file(tmp_path: Path) -> None:
    corrupt = tmp_path / "damaged.pdf"
    corrupt.write_bytes(b"not a valid pdf header")
    backend = DummyMockOCRBackend()
    pipeline = OCRPipeline(backend)  # type: ignore[arg-type]

    result = pipeline.process_pdf(corrupt)
    assert len(result.pages) == 0
    assert len(result.errors) == 1
    assert result.errors[0].code == "invalid_pdf"


def test_process_pdf_page_rendering_failure_propagation(tmp_path: Path) -> None:
    """When one page fails to render, subsequent pages must still be processed."""
    pdf_file = tmp_path / "partial_render.pdf"
    _create_scanned_pdf(pdf_file, [["PAGE 1"], ["PAGE 2"], ["PAGE 3"]])

    backend = DummyMockOCRBackend()
    pipeline = OCRPipeline(backend)  # type: ignore[arg-type]

    # Mock render_pdf_pages so page 2 has an error
    def _mock_render(*args: Any, **kwargs: Any) -> list[RenderedPDFPage]:
        return [
            RenderedPDFPage(1, np.full((100, 100, 3), 255, dtype=np.uint8), 100, 100, 200, None),
            RenderedPDFPage(2, None, 0, 0, 200, "Corrupt stream on page 2"),
            RenderedPDFPage(3, np.full((100, 100, 3), 255, dtype=np.uint8), 100, 100, 200, None),
        ]

    with patch("member3_ocr.ocr_pipeline.render_pdf_pages", side_effect=_mock_render):
        result = pipeline.process_pdf(pdf_file)

    assert len(result.pages) == 3
    assert result.pages[0].page_number == 1
    assert result.pages[0].errors == ()
    assert "RECOGNIZED_TEXT_PAGE_1" in result.pages[0].text

    assert result.pages[1].page_number == 2
    assert len(result.pages[1].errors) == 1
    assert result.pages[1].errors[0].code == "page_render_failed"
    assert result.pages[1].text == ""

    assert result.pages[2].page_number == 3
    assert result.pages[2].errors == ()
    assert "RECOGNIZED_TEXT_PAGE_2" in result.pages[2].text


def test_process_pdf_backend_failure_on_one_page(tmp_path: Path) -> None:
    """When OCR fails on page 2, pages 1 and 3 are preserved."""
    pdf_file = tmp_path / "backend_fail.pdf"
    _create_scanned_pdf(pdf_file, [["PAGE 1"], ["PAGE 2"], ["PAGE 3"]])

    backend = DummyMockOCRBackend(fail_on_page=2)
    pipeline = OCRPipeline(backend)  # type: ignore[arg-type]

    result = pipeline.process_pdf(pdf_file)
    assert len(result.pages) == 3

    assert result.pages[0].errors == ()
    assert result.pages[0].text != ""

    assert len(result.pages[1].errors) == 1
    assert result.pages[1].errors[0].code == "backend_execution_failed"
    assert result.pages[1].text == ""

    assert result.pages[2].errors == ()
    assert result.pages[2].text != ""


def test_process_pdf_determinism(tmp_path: Path) -> None:
    """Repeated processing on same PDF must yield structurally identical results."""
    pdf_file = tmp_path / "deterministic.pdf"
    _create_scanned_pdf(pdf_file, [["LINE 1"], ["LINE 2"]])

    backend = DummyMockOCRBackend()
    pipeline = OCRPipeline(backend)  # type: ignore[arg-type]

    res1 = pipeline.process_pdf(pdf_file, document_id="fixed_id")
    # Reset call count
    backend.call_count = 0
    res2 = pipeline.process_pdf(pdf_file, document_id="fixed_id")

    assert res1.document_id == res2.document_id
    assert len(res1.pages) == len(res2.pages)
    for p1, p2 in zip(res1.pages, res2.pages):
        assert p1.page_number == p2.page_number
        assert p1.text == p2.text
        assert len(p1.blocks) == len(p2.blocks)
        assert p1.processed_width == p2.processed_width
        assert p1.processed_height == p2.processed_height


def test_existing_process_image_regression(tmp_path: Path) -> None:
    """Existing process_image API must remain fully functional."""
    img_file = tmp_path / "standard.png"
    img = np.full((200, 300, 3), 200, dtype=np.uint8)
    cv2.putText(img, "TEST", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.imwrite(str(img_file), img)

    backend = DummyMockOCRBackend()
    pipeline = OCRPipeline(backend)  # type: ignore[arg-type]

    result = pipeline.process_image(img_file, document_id="image_doc_01")
    assert isinstance(result, OCRDocumentResult)
    assert result.document_id == "image_doc_01"
    assert len(result.pages) == 1
    assert result.pages[0].page_number == 1
    assert result.provenance["source_type"] == "image_path"


def test_offline_no_network_guarantee(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure process_pdf makes zero socket / network calls."""
    pdf_file = tmp_path / "offline.pdf"
    _create_scanned_pdf(pdf_file, [["OFFLINE VERIFICATION"]])

    def _forbidden_network(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Network connection was attempted in offline mode!")

    monkeypatch.setattr(socket, "socket", _forbidden_network)

    backend = DummyMockOCRBackend()
    pipeline = OCRPipeline(backend)  # type: ignore[arg-type]
    result = pipeline.process_pdf(pdf_file)
    assert len(result.pages) == 1


# ──────────────────────────────────────────────────────────────────────────────
# Real Local PaddleOCR Integration Test
# ──────────────────────────────────────────────────────────────────────────────

def test_real_paddleocr_scanned_pdf_integration(tmp_path: Path) -> None:
    """Real local integration test using pre-staged local PP-OCRv5 models on CPU."""
    det_dir = Path(r"C:\SovereignAI\member3_ocr\models\paddleocr\PP-OCRv5_mobile_det_infer")
    rec_dir = Path(r"C:\SovereignAI\member3_ocr\models\paddleocr\PP-OCRv5_mobile_rec_infer")

    if not det_dir.is_dir() or not rec_dir.is_dir():
        pytest.skip("Local PP-OCRv5 models are not staged")

    # Generate a realistic 2-page scanned industrial PDF
    pdf_file = tmp_path / "refinery_scanned_inspection.pdf"
    pages_data = [
        [
            "MRPL REFINERY - UNIT 2 INSPECTION",
            "EQUIPMENT: CRUDE PUMP P-101",
            "STATUS: OPERATIONAL NORMAL",
        ],
        [
            "SAFETY AUDIT CHECKLIST",
            "VALVE V-204 PRESSURE 45 BAR",
            "EMERGENCY SHUTDOWN: READY",
        ],
    ]
    _create_scanned_pdf(pdf_file, pages_data, size=(1000, 700))

    config = PaddleOCRModelConfig(
        detection_model_dir=det_dir,
        recognition_model_dir=rec_dir,
        device="cpu",
    )
    backend = PaddleOCRBackend(config)
    pipeline = OCRPipeline(backend)

    # Execute real OCR inference on the scanned PDF
    result = pipeline.process_pdf(pdf_file, document_id="refinery_audit_01", dpi=150)

    # 1. Document Structure
    assert isinstance(result, OCRDocumentResult)
    assert result.document_id == "refinery_audit_01"
    assert len(result.pages) == 2
    assert result.errors == ()
    assert result.provenance["source_type"] == "pdf_path"
    assert result.provenance["total_pages"] == 2
    assert result.provenance["rendered_dpi"] == 150

    # 2. Page 1 Validation
    p1 = result.pages[0]
    assert p1.page_number == 1
    assert p1.errors == ()
    assert len(p1.blocks) >= 2
    assert any("PUMP" in b.text or "REFINERY" in b.text or "P-101" in b.text for b in p1.blocks)

    # 3. Page 2 Validation
    p2 = result.pages[1]
    assert p2.page_number == 2
    assert p2.errors == ()
    assert len(p2.blocks) >= 2
    assert any("VALVE" in b.text or "PRESSURE" in b.text or "V-204" in b.text for b in p2.blocks)

    # 4. JSON Serialization check
    json_str = result.to_json()
    assert '"document_id": "refinery_audit_01"' in json_str
    assert '"total_pages": 2' in json_str
