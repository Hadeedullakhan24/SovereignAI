"""Three-module integration test for Member 3:
Image Preprocessing -> OCR Pipeline -> Document Parser.

Verifies end-to-end flow using synthetic test images and a deterministic mock OCR backend.
Completely offline, air-gapped, dataset-safe, deterministic.
"""
from __future__ import annotations


import hashlib
import socket
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from member3_ocr.core.document_parser import DocumentParser
from member3_ocr.core.image_preprocessing import (
    PreprocessingOptions,
    PreprocessingResult,
    preprocess_image,
)
from member3_ocr.core.ocr_pipeline import (
    BackendCapabilities,
    BackendInfo,
    BackendRecognition,
    BoundingBox,
    Issue,
    OCRDocumentResult,
    OCRPipeline,
    TextBlock,
    TextLine,
    Word,
)
from rag_engine.schemas.document import DocumentLifecycleState
from rag_engine.schemas.parsed_document import ParsedDocument


class DeterministicReportMockBackend:
    """Deterministic OCR backend representing an industrial equipment report."""

    def __init__(self) -> None:
        self.initialized = False

    @property
    def backend_info(self) -> BackendInfo:
        return BackendInfo(
            name="mock_paddle_report",
            version="1.0",
            model_ids=("local/det", "local/rec"),
            device="cpu",
            capabilities=BackendCapabilities(),
        )

    def initialize(self) -> None:
        self.initialized = True

    def recognize(self, image: np.ndarray) -> BackendRecognition:
        # Construct blocks for:
        # Equipment Inspection Report
        # Equipment Tag: P-203
        # Pressure: 10 bar
        # Temperature: 80 C
        # Status: APPROVED
        def _make_item(idx: int, text: str, top: float, height: float = 24.0) -> TextBlock:
            bbox = BoundingBox(left=20.0, top=top, right=350.0, bottom=top + height)
            words = tuple(Word(w, 0.96, bbox) for w in text.split())
            line = TextLine(text, 0.96, bbox, words)
            return TextBlock(
                id=f"text-{idx}",
                text=text,
                confidence=0.96,
                bbox=bbox,
                lines=(line,),
                polygon=((20.0, top), (350.0, top), (350.0, top + height), (20.0, top + height)),
            )

        blocks = (
            _make_item(1, "EQUIPMENT INSPECTION REPORT", top=20.0, height=30.0),
            _make_item(2, "Equipment Tag: P-203", top=60.0),
            _make_item(3, "Pressure: 10 bar", top=90.0),
            _make_item(4, "Temperature: 80 C", top=120.0),
            _make_item(5, "Status: APPROVED", top=150.0),
        )
        return BackendRecognition(blocks=blocks)


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def test_three_module_end_to_end_integration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Execute end-to-end integration:

    Image Preprocessing -> OCR Pipeline -> Document Parser
    Validating all 12 integration requirements.
    """
    # 12. Confirm no network/download behavior
    def _blocked_socket(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Network socket call attempted in offline integration test!")

    monkeypatch.setattr(socket, "socket", _blocked_socket)

    # 1. Create synthetic image in pytest tmp_path
    image_path = tmp_path / "inspection_card.png"
    arr = np.full((300, 500, 3), 255, dtype=np.uint8)
    # Add simulated dark markings
    arr[20:50, 20:300] = 30
    arr[60:85, 20:250] = 40
    arr[90:115, 20:200] = 40
    arr[120:145, 20:210] = 40
    arr[150:175, 20:180] = 40
    Image.fromarray(arr).save(image_path)

    initial_sha = _file_sha256(image_path)
    initial_stat = image_path.stat()

    # 2. Run image preprocessing
    prep_out_dir = tmp_path / "preprocessed_out"
    prep_options = PreprocessingOptions(
        max_width=800,
        max_height=800,
        grayscale=True,
        denoise=True,
        contrast_enhancement=True,
    )
    prep_result = preprocess_image(
        image_path,
        options=prep_options,
        save_output=True,
        output_dir=prep_out_dir,
    )

    # 3. Confirm PreprocessingResult is valid
    assert isinstance(prep_result, PreprocessingResult)
    assert prep_result.image is not None
    assert prep_result.processed_width > 0
    assert prep_result.processed_height > 0
    assert "grayscale" in prep_result.operations_applied
    assert "denoise" in prep_result.operations_applied
    assert "clahe" in prep_result.operations_applied
    assert prep_result.output_path is not None
    assert prep_result.output_path.exists()
    assert prep_result.output_path.parent == prep_out_dir

    # 4. Feed PreprocessingResult into OCR pipeline/mock backend
    backend = DeterministicReportMockBackend()
    pipeline = OCRPipeline(backend)
    ocr_result = pipeline.process_image(prep_result, document_id="report-P203")

    # 5. Confirm OCRDocumentResult contains expected text
    assert isinstance(ocr_result, OCRDocumentResult)
    assert ocr_result.document_id == "report-P203"
    assert len(ocr_result.pages) == 1
    page = ocr_result.pages[0]
    assert "EQUIPMENT INSPECTION REPORT" in page.text
    assert "Equipment Tag: P-203" in page.text
    assert "Pressure: 10 bar" in page.text
    assert "Temperature: 80 C" in page.text
    assert "Status: APPROVED" in page.text
    assert len(page.blocks) == 5

    # 6. Feed OCRDocumentResult into DocumentParser
    parser = DocumentParser()
    parsed_doc = parser.parse_ocr_result(ocr_result)

    # 7. Confirm ParsedDocument is produced
    assert isinstance(parsed_doc, ParsedDocument)
    assert parsed_doc.document_id == "parsed_report-P203"
    assert parsed_doc.raw_document_id == "report-P203"

    canonical_text = DocumentParser.build_canonical_text(parsed_doc)

    # 8. Confirm:
    #    - title/section information where applicable
    #    - paragraph text
    #    - key-value information/provenance
    #    - page map
    #    - statistics
    #    - lifecycle state
    assert parsed_doc.metadata.title == "EQUIPMENT INSPECTION REPORT"
    assert len(parsed_doc.sections) >= 1
    assert parsed_doc.sections[0].title == "EQUIPMENT INSPECTION REPORT"

    # Check paragraph content in sections
    all_paragraphs = [p for s in parsed_doc.sections for p in s.paragraphs]
    combined_paragraphs = "\n".join(all_paragraphs)
    assert "Equipment Tag: P-203" in combined_paragraphs
    assert "Pressure: 10 bar" in combined_paragraphs
    assert "Temperature: 80 C" in combined_paragraphs
    assert "Status: APPROVED" in combined_paragraphs

    # Key-value information / provenance in processing history
    assert len(parsed_doc.processing_history) > 0
    hist = parsed_doc.processing_history[0]
    assert hist["stage"] == "document_parser"
    assert "key_value_fields" in hist
    assert "ocr_geometry" in hist

    # Page map
    assert 1 in parsed_doc.page_map
    page_entry = parsed_doc.page_map[1]
    assert page_entry.page_number == 1
    assert page_entry.char_start == 0
    assert page_entry.char_end == len(canonical_text)

    # Statistics
    assert parsed_doc.statistics.total_pages == 1
    assert parsed_doc.statistics.total_sections >= 1
    assert parsed_doc.statistics.total_words > 0
    assert parsed_doc.statistics.total_characters == len(canonical_text)

    # Lifecycle state
    assert parsed_doc.lifecycle_state == DocumentLifecycleState.PARSED

    # 9. Confirm final canonical text contains:
    #    P-203, 10 bar, 80 C, APPROVED
    assert "P-203" in canonical_text
    assert "10 bar" in canonical_text
    assert "80 C" in canonical_text
    assert "APPROVED" in canonical_text

    # Also verify get_full_text() works
    full_text = parsed_doc.get_full_text()
    assert "P-203" in full_text
    assert "10 bar" in full_text
    assert "80 C" in full_text
    assert "APPROVED" in full_text

    # 10. Confirm no source file was modified
    assert image_path.exists()
    assert _file_sha256(image_path) == initial_sha
    final_stat = image_path.stat()
    assert final_stat.st_mtime_ns == initial_stat.st_mtime_ns
    assert final_stat.st_size == initial_stat.st_size

    # 11. Confirm deterministic result on repeated runs
    ocr_result_2 = pipeline.process_image(prep_result, document_id="report-P203")
    parsed_doc_2 = parser.parse_ocr_result(ocr_result_2)
    canonical_text_2 = DocumentParser.build_canonical_text(parsed_doc_2)
    assert canonical_text == canonical_text_2
    assert parsed_doc.statistics.total_words == parsed_doc_2.statistics.total_words
    assert parsed_doc.statistics.total_characters == parsed_doc_2.statistics.total_characters
    assert [s.section_id for s in parsed_doc.sections] == [s.section_id for s in parsed_doc_2.sections]
    assert [s.title for s in parsed_doc.sections] == [s.title for s in parsed_doc_2.sections]
    assert [s.paragraphs for s in parsed_doc.sections] == [s.paragraphs for s in parsed_doc_2.sections]

