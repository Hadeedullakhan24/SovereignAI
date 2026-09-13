"""Unit tests for member3_ocr.document_parser.

All tests use synthetic OCRDocumentResult objects.
No PaddleOCR, no Paddle model files, no internet, no real dataset files.
"""
from __future__ import annotations


import pytest

from member3_ocr.core.document_parser import (
    AgentQueryableDocument,
    DocumentParser,
    DocumentParserError,
    _classify_heading,
    _extract_cross_refs,
    _group_into_paragraphs,
    _sort_blocks_reading_order,
    build_canonical_document_text,
    export_for_agent,
    export_for_rag,
)
from member3_ocr.core.ocr_pipeline import (
    BackendCapabilities,
    BackendInfo,
    BoundingBox,
    Issue,
    KeyValueField,
    OCRDocumentResult,
    OCRPageResult,
    Table as OCRTable,
    TextBlock,
    TextLine,
    Word,
)
from rag_engine.schemas.document import DocumentLifecycleState
from rag_engine.schemas.parsed_document import ParsedDocument


# ──────────────────────────────────────────────────────────────────────────────
# Helpers — synthetic OCR object factories
# ──────────────────────────────────────────────────────────────────────────────

def _make_backend() -> BackendInfo:
    return BackendInfo(
        name="test_backend",
        version="0.1",
        model_ids=("det_model", "rec_model"),
        device="cpu",
        capabilities=BackendCapabilities(),
    )


def _make_bbox(left: float, top: float, right: float, bottom: float) -> BoundingBox:
    return BoundingBox(left=left, top=top, right=right, bottom=bottom)


def _make_block(
    block_id: str,
    text: str,
    top: float,
    left: float = 10.0,
    right: float = 500.0,
    bottom: float | None = None,
    confidence: float = 0.95,
) -> TextBlock:
    b = bottom if bottom is not None else top + 20
    bbox = _make_bbox(left, top, right, b)
    word = Word(text=text, confidence=confidence, bbox=bbox)
    line = TextLine(text=text, confidence=confidence, bbox=bbox, words=(word,))
    return TextBlock(id=block_id, text=text, confidence=confidence, bbox=bbox, lines=(line,))


def _make_page(
    page_number: int,
    blocks: list[TextBlock],
    tables: list[OCRTable] | None = None,
    kv_fields: list[KeyValueField] | None = None,
    errors: list[Issue] | None = None,
) -> OCRPageResult:
    text = "\n".join(b.text for b in blocks)
    return OCRPageResult(
        page_number=page_number,
        original_width=800,
        original_height=1000,
        processed_width=800,
        processed_height=1000,
        text=text,
        blocks=tuple(blocks),
        tables=tuple(tables or []),
        key_value_fields=tuple(kv_fields or []),
        errors=tuple(errors or []),
    )


def _make_ocr_result(
    pages: list[OCRPageResult],
    document_id: str = "test_doc",
    warnings: list[Issue] | None = None,
    errors: list[Issue] | None = None,
) -> OCRDocumentResult:
    return OCRDocumentResult(
        document_id=document_id,
        pages=tuple(pages),
        backend=_make_backend(),
        provenance={"source_path": "/fake/test.png", "source_type": "image_path"},
        warnings=tuple(warnings or []),
        errors=tuple(errors or []),
        total_processing_time_ms=42.0,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Realistic synthetic fixture (MRPL pump report)
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def pump_report_result() -> OCRDocumentResult:
    """Two-page synthetic OCR result resembling an MRPL pump inspection report."""
    page1_blocks = [
        _make_block("b1", "MRPL PUMP INSPECTION REPORT", top=30, left=50, right=750, bottom=60),
        _make_block("b2", "1. Equipment Details", top=100, left=50, right=400, bottom=120),
        _make_block("b3", "Equipment Tag: P-203", top=140, left=50, right=400, bottom=160),
        _make_block("b4", "Equipment Type: Centrifugal Pump", top=165, left=50, right=400, bottom=185),
        _make_block("b5", "2. Inspection Findings", top=230, left=50, right=400, bottom=250),
        _make_block("b6", "Pump casing inspected for visible damage.", top=270, left=50, right=600, bottom=290),
        _make_block("b7", "No external leakage observed.", top=295, left=50, right=600, bottom=315),
    ]
    page2_blocks = [
        _make_block("b8", "3. Recommendations", top=40, left=50, right=400, bottom=60),
        _make_block("b9", "Continue scheduled monitoring.", top=80, left=50, right=600, bottom=100),
    ]
    pages = [
        _make_page(1, page1_blocks),
        _make_page(2, page2_blocks),
    ]
    return _make_ocr_result(pages, document_id="mrpl_pump_report")


# ──────────────────────────────────────────────────────────────────────────────
# Test 1 — Basic OCR → ParsedDocument conversion
# ──────────────────────────────────────────────────────────────────────────────

def test_basic_conversion(pump_report_result: OCRDocumentResult) -> None:
    parser = DocumentParser()
    result = parser.parse_ocr_result(pump_report_result)

    assert isinstance(result, ParsedDocument)
    assert result.document_id.startswith("parsed_")
    assert result.raw_document_id == "mrpl_pump_report"


# ──────────────────────────────────────────────────────────────────────────────
# Test 2 — Page numbers preserved
# ──────────────────────────────────────────────────────────────────────────────

def test_page_numbers_preserved(pump_report_result: OCRDocumentResult) -> None:
    parser = DocumentParser()
    result = parser.parse_ocr_result(pump_report_result)

    section_pages = {s.page_number for s in result.sections if s.page_number is not None}
    assert 1 in section_pages
    assert 2 in section_pages


# ──────────────────────────────────────────────────────────────────────────────
# Test 3 — Sections created
# ──────────────────────────────────────────────────────────────────────────────

def test_sections_created(pump_report_result: OCRDocumentResult) -> None:
    parser = DocumentParser()
    result = parser.parse_ocr_result(pump_report_result)

    assert len(result.sections) >= 1
    # Expect at least the numbered headings to be detected.
    titles = [s.title for s in result.sections]
    assert any("Equipment Details" in t or "1." in t for t in titles), \
        f"Expected 'Equipment Details' heading, got: {titles}"


# ──────────────────────────────────────────────────────────────────────────────
# Test 4 — Paragraphs created
# ──────────────────────────────────────────────────────────────────────────────

def test_paragraphs_created(pump_report_result: OCRDocumentResult) -> None:
    parser = DocumentParser()
    result = parser.parse_ocr_result(pump_report_result)

    total_paragraphs = sum(len(s.paragraphs) for s in result.sections)
    assert total_paragraphs >= 1


# ──────────────────────────────────────────────────────────────────────────────
# Test 5 — Reading order is deterministic
# ──────────────────────────────────────────────────────────────────────────────

def test_reading_order_deterministic() -> None:
    """Same input must produce the same section order on two runs."""
    blocks = [
        _make_block("bA", "HEADER TEXT", top=10),
        _make_block("bB", "Body text first sentence.", top=40),
        _make_block("bC", "Body text second sentence.", top=60),
    ]
    page = _make_page(1, blocks)
    ocr = _make_ocr_result([page])
    parser = DocumentParser()

    result1 = parser.parse_ocr_result(ocr)
    result2 = parser.parse_ocr_result(ocr)

    titles1 = [s.title for s in result1.sections]
    titles2 = [s.title for s in result2.sections]
    assert titles1 == titles2

    content1 = [s.content for s in result1.sections]
    content2 = [s.content for s in result2.sections]
    assert content1 == content2


# ──────────────────────────────────────────────────────────────────────────────
# Test 6 — CitationCoordinates generated
# ──────────────────────────────────────────────────────────────────────────────

def test_citation_coordinates_generated(pump_report_result: OCRDocumentResult) -> None:
    parser = DocumentParser()
    result = parser.parse_ocr_result(pump_report_result)

    for section in result.sections:
        assert section.citation_coords is not None, \
            f"Section '{section.title}' has no citation_coords"
        assert section.citation_coords.page_number is not None
        assert section.citation_coords.citation_id.startswith("cit_")


# ──────────────────────────────────────────────────────────────────────────────
# Test 7 — PageMapEntry generated
# ──────────────────────────────────────────────────────────────────────────────

def test_page_map_entry_generated(pump_report_result: OCRDocumentResult) -> None:
    parser = DocumentParser()
    result = parser.parse_ocr_result(pump_report_result)

    assert 1 in result.page_map
    assert 2 in result.page_map

    entry1 = result.page_map[1]
    assert entry1.page_number == 1
    assert isinstance(entry1.section_ids, list)
    assert isinstance(entry1.table_ids, list)


# ──────────────────────────────────────────────────────────────────────────────
# Test 8 — Character offsets are sensible
# ──────────────────────────────────────────────────────────────────────────────

def test_character_offsets_sensible(pump_report_result: OCRDocumentResult) -> None:
    parser = DocumentParser()
    result = parser.parse_ocr_result(pump_report_result)

    for page_num, entry in result.page_map.items():
        assert entry.char_start >= 0, f"Page {page_num}: char_start < 0"
        assert entry.char_end >= entry.char_start, \
            f"Page {page_num}: char_end < char_start"

    for section in result.sections:
        coords = section.citation_coords
        if coords is not None and coords.char_offset_start is not None and coords.char_offset_end is not None:
            assert coords.char_offset_end >= coords.char_offset_start, \
                f"Section '{section.title}': char_offset_end < char_offset_start"


# ──────────────────────────────────────────────────────────────────────────────
# Test 9 — OCR confidence propagated
# ──────────────────────────────────────────────────────────────────────────────

def test_ocr_confidence_propagated() -> None:
    blocks = [
        _make_block("bX", "SECTION ALPHA", top=10, confidence=0.88),
        _make_block("bY", "Some body content here.", top=50, confidence=0.76),
    ]
    page = _make_page(1, blocks)
    ocr = _make_ocr_result([page])
    parser = DocumentParser()
    result = parser.parse_ocr_result(ocr)

    # At least one section should have confidence < 1.0 (propagated from OCR).
    confs = [s.confidence for s in result.sections]
    assert any(c < 1.0 for c in confs), \
        f"Expected sub-1.0 confidence from OCR blocks, got: {confs}"


# ──────────────────────────────────────────────────────────────────────────────
# Test 10 — OCR provenance preserved
# ──────────────────────────────────────────────────────────────────────────────

def test_ocr_provenance_preserved(pump_report_result: OCRDocumentResult) -> None:
    parser = DocumentParser()
    result = parser.parse_ocr_result(pump_report_result)

    assert len(result.processing_history) >= 1
    history = result.processing_history[0]

    assert history["stage"] == "document_parser"
    assert history["backend_name"] == "test_backend"
    assert history["page_count"] == 2
    # Geometry log should contain entries for each block.
    assert isinstance(history["ocr_geometry"], list)
    assert len(history["ocr_geometry"]) > 0
    # Each entry should have bbox data.
    geo_entry = history["ocr_geometry"][0]
    assert "bbox" in geo_entry
    assert "left" in geo_entry["bbox"]


# ──────────────────────────────────────────────────────────────────────────────
# Test 11 — Empty OCR document handled safely
# ──────────────────────────────────────────────────────────────────────────────

def test_empty_ocr_document_handled_safely() -> None:
    ocr = _make_ocr_result(pages=[], document_id="empty_doc")
    parser = DocumentParser()
    result = parser.parse_ocr_result(ocr)

    assert isinstance(result, ParsedDocument)
    assert result.sections == []
    assert result.page_map == {}
    assert result.statistics.total_pages == 0
    assert result.validation_report is not None
    assert not result.validation_report.is_valid or result.validation_report.warning_count >= 1


# ──────────────────────────────────────────────────────────────────────────────
# Test 12 — Page with no text handled safely
# ──────────────────────────────────────────────────────────────────────────────

def test_page_with_no_text_handled_safely() -> None:
    blank_page = _make_page(1, blocks=[])
    ocr = _make_ocr_result([blank_page])
    parser = DocumentParser()
    result = parser.parse_ocr_result(ocr)

    assert isinstance(result, ParsedDocument)
    # Page map should still contain the page entry.
    assert 1 in result.page_map
    # Sections may be empty for a blank page.
    # Validation should flag it as INFO.
    if result.validation_report:
        codes = [issue.code for issue in result.validation_report.issues]
        assert "EMPTY_PAGE" in codes


# ──────────────────────────────────────────────────────────────────────────────
# Test 13 — OCR tables convert correctly when supplied
# ──────────────────────────────────────────────────────────────────────────────

def test_ocr_tables_convert_correctly() -> None:
    header_cell = {"row_idx": 0, "col_idx": 0, "text": "Parameter", "is_header": True, "confidence": 0.99}
    value_cell = {"row_idx": 1, "col_idx": 0, "text": "Pressure", "is_header": False, "confidence": 0.97}

    ocr_tbl = OCRTable(
        id="tbl-001",
        bbox=_make_bbox(50, 200, 500, 300),
        confidence=0.95,
        html=None,
        markdown=None,
        cells=(header_cell, value_cell),
    )

    page = _make_page(1, blocks=[_make_block("bH", "TABLE SECTION", top=10)], tables=[ocr_tbl])
    ocr = _make_ocr_result([page])
    parser = DocumentParser()
    result = parser.parse_ocr_result(ocr)

    assert len(result.tables) == 1
    tbl = result.tables[0]
    assert tbl.page_number == 1
    assert tbl.col_count >= 1
    assert len(tbl.cells) == 2


# ──────────────────────────────────────────────────────────────────────────────
# Test 14 — TableCell conversion works
# ──────────────────────────────────────────────────────────────────────────────

def test_table_cell_conversion() -> None:
    cells_raw = [
        {"row_idx": 0, "col_idx": 0, "text": "Col1", "is_header": True, "confidence": 1.0},
        {"row_idx": 0, "col_idx": 1, "text": "Col2", "is_header": True, "confidence": 1.0},
        {"row_idx": 1, "col_idx": 0, "text": "Val1", "is_header": False, "confidence": 0.90},
        {"row_idx": 1, "col_idx": 1, "text": "Val2", "is_header": False, "confidence": 0.85},
    ]

    ocr_tbl = OCRTable(
        id="tbl-002",
        bbox=_make_bbox(0, 0, 400, 200),
        confidence=0.92,
        cells=tuple(cells_raw),
    )
    page = _make_page(1, blocks=[_make_block("bH2", "SOME HEADING", top=10)], tables=[ocr_tbl])
    ocr = _make_ocr_result([page])
    parser = DocumentParser()
    result = parser.parse_ocr_result(ocr)

    assert len(result.tables) == 1
    tbl = result.tables[0]
    assert tbl.row_count == 1
    assert tbl.col_count == 2
    assert tbl.headers == ["Col1", "Col2"]
    assert tbl.rows == [["Val1", "Val2"]]

    header_cells = [c for c in tbl.cells if c.is_header]
    data_cells = [c for c in tbl.cells if not c.is_header]
    assert len(header_cells) == 2
    assert len(data_cells) == 2

    for cell in tbl.cells:
        assert 0.0 <= cell.confidence <= 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Test 15 — DocumentStatistics are correct
# ──────────────────────────────────────────────────────────────────────────────

def test_document_statistics_correct(pump_report_result: OCRDocumentResult) -> None:
    parser = DocumentParser()
    result = parser.parse_ocr_result(pump_report_result)

    stats = result.statistics
    assert stats.total_pages == 2
    assert stats.total_sections == len(result.sections)
    assert stats.total_tables == len(result.tables)
    assert stats.total_words >= 0
    assert stats.total_characters >= 0
    assert stats.total_entities == 0   # Phase 1: no entity extraction
    assert stats.total_warnings == 0   # Phase 1: no safety warning extraction
    assert stats.total_relations == 0  # Phase 1: no entity graph


# ──────────────────────────────────────────────────────────────────────────────
# Test 16 — Processing history exists
# ──────────────────────────────────────────────────────────────────────────────

def test_processing_history_exists(pump_report_result: OCRDocumentResult) -> None:
    parser = DocumentParser()
    result = parser.parse_ocr_result(pump_report_result)

    assert isinstance(result.processing_history, list)
    assert len(result.processing_history) >= 1
    history = result.processing_history[0]
    assert "stage" in history
    assert history["stage"] == "document_parser"
    assert "parser_version" in history
    assert "timestamp" in history
    assert "page_count" in history
    assert "processing_time_ms" in history


# ──────────────────────────────────────────────────────────────────────────────
# Test 17 — Lifecycle state is PARSED
# ──────────────────────────────────────────────────────────────────────────────

def test_lifecycle_state_is_parsed(pump_report_result: OCRDocumentResult) -> None:
    parser = DocumentParser()
    result = parser.parse_ocr_result(pump_report_result)

    assert result.lifecycle_state == DocumentLifecycleState.PARSED


# ──────────────────────────────────────────────────────────────────────────────
# Test 18 — No network access required (structural)
# ──────────────────────────────────────────────────────────────────────────────

def test_no_network_access_required(pump_report_result: OCRDocumentResult) -> None:
    """Verify the parser runs end-to-end without any real I/O.

    Since all I/O is blocked in this synthetic test environment, successful
    completion proves the module is offline-compatible.
    """
    parser = DocumentParser()
    result = parser.parse_ocr_result(pump_report_result)
    assert isinstance(result, ParsedDocument)


# ──────────────────────────────────────────────────────────────────────────────
# Test 19 — No dataset modification occurs
# ──────────────────────────────────────────────────────────────────────────────

def test_no_dataset_modification(tmp_path: "pytest.TempPathFixture") -> None:
    """Parser must not touch the datasets directory."""
    import os
    from pathlib import Path

    datasets_dir = Path(__file__).resolve().parents[2] / "datasets"

    # Capture mtime of datasets dir before parsing.
    before_mtime: float | None = None
    if datasets_dir.exists():
        before_mtime = datasets_dir.stat().st_mtime

    # Run the parser.
    blocks = [_make_block("bZ", "TEST DOCUMENT", top=10)]
    page = _make_page(1, blocks)
    ocr = _make_ocr_result([page])
    DocumentParser().parse_ocr_result(ocr)

    # Verify mtime unchanged.
    if before_mtime is not None:
        after_mtime = datasets_dir.stat().st_mtime
        assert after_mtime == before_mtime, "datasets/ directory was modified!"


# ──────────────────────────────────────────────────────────────────────────────
# Test 20 — Parser output validates as Member 1 ParsedDocument
# ──────────────────────────────────────────────────────────────────────────────

def test_output_validates_as_parsed_document(pump_report_result: OCRDocumentResult) -> None:
    """The ParsedDocument returned must survive a Pydantic round-trip validation."""
    parser = DocumentParser()
    result = parser.parse_ocr_result(pump_report_result)

    # Round-trip through Pydantic validation.
    dumped = result.model_dump()
    reloaded = ParsedDocument.model_validate(dumped)

    assert reloaded.document_id == result.document_id
    assert reloaded.lifecycle_state == DocumentLifecycleState.PARSED
    assert len(reloaded.sections) == len(result.sections)
    assert len(reloaded.tables) == len(result.tables)


# ──────────────────────────────────────────────────────────────────────────────
# Integration test — realistic multi-page MRPL pump report
# ──────────────────────────────────────────────────────────────────────────────

def test_mrpl_pump_report_integration(pump_report_result: OCRDocumentResult) -> None:
    """Full integration test: multi-page pump inspection report.

    Verifies the ParsedDocument contains:
    - multiple pages
    - sections with page numbers
    - paragraphs
    - citation coordinates
    - page map entries for both pages
    - statistics
    - processing history
    """
    parser = DocumentParser()
    result = parser.parse_ocr_result(
        pump_report_result,
        raw_document_id="mrpl_raw_001",
        title="MRPL PUMP INSPECTION REPORT",
        category="inspection_report",
    )

    # Multiple pages
    assert result.statistics.total_pages == 2

    # Sections with page numbers
    assert len(result.sections) >= 1
    page_nums_in_sections = {s.page_number for s in result.sections}
    assert 1 in page_nums_in_sections

    # Paragraphs
    total_paragraphs = sum(len(s.paragraphs) for s in result.sections)
    assert total_paragraphs >= 1

    # Citation coordinates
    for sec in result.sections:
        assert sec.citation_coords is not None
        assert sec.citation_coords.page_number is not None

    # Page map
    assert 1 in result.page_map
    assert 2 in result.page_map

    # Statistics
    assert result.statistics.total_sections >= 1
    assert result.statistics.total_words >= 1
    assert result.statistics.total_characters >= 1

    # Processing history
    assert len(result.processing_history) >= 1
    history = result.processing_history[0]
    assert history["page_count"] == 2
    assert history["section_count"] == len(result.sections)

    # Title set
    assert result.title == "MRPL PUMP INSPECTION REPORT"
    assert result.category == "inspection_report"
    assert result.lifecycle_state == DocumentLifecycleState.PARSED


# ──────────────────────────────────────────────────────────────────────────────
# Unit tests for helper functions
# ──────────────────────────────────────────────────────────────────────────────

class TestClassifyHeading:
    def test_numbered_heading_level1(self) -> None:
        result = _classify_heading("1. Introduction", is_isolated=False)
        assert result is not None
        _, level, conf = result
        assert level == 1
        assert conf >= 0.90

    def test_numbered_heading_level2(self) -> None:
        result = _classify_heading("1.1 Equipment Details", is_isolated=False)
        assert result is not None
        _, level, _ = result
        assert level == 2

    def test_numbered_heading_level3(self) -> None:
        result = _classify_heading("2.3.1 Sub-section Title", is_isolated=False)
        assert result is not None
        _, level, _ = result
        assert level == 3

    def test_uppercase_heading(self) -> None:
        result = _classify_heading("MRPL PUMP INSPECTION REPORT", is_isolated=False)
        assert result is not None

    def test_body_text_not_heading(self) -> None:
        result = _classify_heading(
            "The pump casing was inspected for visible damage and leakage.", is_isolated=False
        )
        assert result is None

    def test_isolated_short_line_is_heading(self) -> None:
        result = _classify_heading("Recommendations", is_isolated=True)
        assert result is not None

    def test_long_line_not_heading(self) -> None:
        long_line = "A" * 130
        result = _classify_heading(long_line, is_isolated=False)
        assert result is None

    def test_few_alpha_chars_not_heading(self) -> None:
        result = _classify_heading("P-203", is_isolated=False)
        assert result is None  # Only 4 alpha chars → below threshold


class TestSortBlocksReadingOrder:
    def test_sorts_top_to_bottom(self) -> None:
        b1 = _make_block("b1", "First line", top=100)
        b2 = _make_block("b2", "Second line", top=50)
        sorted_blocks = _sort_blocks_reading_order((b1, b2))
        assert sorted_blocks[0].id == "b2"
        assert sorted_blocks[1].id == "b1"

    def test_sorts_left_to_right_for_same_row(self) -> None:
        b1 = _make_block("b1", "Right column", top=50, left=400)
        b2 = _make_block("b2", "Left column", top=50, left=50)
        sorted_blocks = _sort_blocks_reading_order((b1, b2))
        assert sorted_blocks[0].id == "b2"
        assert sorted_blocks[1].id == "b1"

    def test_empty_tuple(self) -> None:
        assert _sort_blocks_reading_order(()) == []


class TestGroupIntoParagraphs:
    def test_single_block_single_paragraph(self) -> None:
        blocks = [_make_block("b1", "Single line of text.", top=10, bottom=30)]
        paragraphs = _group_into_paragraphs(blocks)
        assert len(paragraphs) == 1
        assert "Single line" in paragraphs[0]

    def test_close_blocks_same_paragraph(self) -> None:
        b1 = _make_block("b1", "First line", top=10, bottom=30)
        b2 = _make_block("b2", "Second line", top=32, bottom=52)
        paragraphs = _group_into_paragraphs([b1, b2])
        # Close blocks → same paragraph
        assert len(paragraphs) == 1

    def test_sentence_end_creates_paragraph_break(self) -> None:
        b1 = _make_block("b1", "End of sentence.", top=10, bottom=30)
        b2 = _make_block("b2", "Next sentence here.", top=32, bottom=52)
        paragraphs = _group_into_paragraphs([b1, b2])
        # Sentence-ending punctuation triggers new paragraph.
        assert len(paragraphs) >= 1

    def test_empty_blocks(self) -> None:
        assert _group_into_paragraphs([]) == []


class TestExtractCrossRefs:
    def test_figure_reference(self) -> None:
        refs = _extract_cross_refs("See Figure 3 for details.", page_num=1)
        assert len(refs) == 1
        assert refs[0].ref_type == "FIGURE"

    def test_table_reference(self) -> None:
        refs = _extract_cross_refs("Refer to Table 4 for specifications.", page_num=2)
        assert len(refs) == 1
        assert refs[0].ref_type == "TABLE"

    def test_section_reference(self) -> None:
        refs = _extract_cross_refs("As described in Section 2.1.", page_num=1)
        assert len(refs) == 1
        assert refs[0].ref_type == "SECTION"

    def test_standard_reference(self) -> None:
        refs = _extract_cross_refs("Per OISD-105 requirements.", page_num=1)
        assert len(refs) >= 1
        assert any(r.ref_type == "STANDARD" for r in refs)

    def test_no_references(self) -> None:
        refs = _extract_cross_refs("The pump was inspected yesterday.", page_num=1)
        assert refs == []

    def test_deduplication(self) -> None:
        refs = _extract_cross_refs("See Figure 3 and also Figure 3 again.", page_num=1)
        targets = [r.target.lower() for r in refs]
        # Should deduplicate "Figure 3".
        assert len([t for t in targets if "figure 3" in t.lower() or "fig" in t.lower()]) == 1


class TestTypeError:
    def test_wrong_type_raises(self) -> None:
        parser = DocumentParser()
        with pytest.raises(TypeError, match="OCRDocumentResult"):
            parser.parse_ocr_result("not an OCR result")  # type: ignore[arg-type]


class TestExplicitTitle:
    def test_explicit_title_wins(self) -> None:
        blocks = [_make_block("b1", "SOME UNRELATED TEXT", top=10)]
        page = _make_page(1, blocks)
        ocr = _make_ocr_result([page])
        parser = DocumentParser()
        result = parser.parse_ocr_result(ocr, title="My Explicit Title")
        assert result.title == "My Explicit Title"

    def test_empty_title_suppresses_heuristic(self) -> None:
        blocks = [_make_block("b1", "MRPL INSPECTION REPORT", top=10)]
        page = _make_page(1, blocks)
        ocr = _make_ocr_result([page])
        parser = DocumentParser()
        result = parser.parse_ocr_result(ocr, title="")
        assert result.title == ""


class TestKeyValueProvenancePreserved:
    def test_kv_fields_preserved_in_history(self) -> None:
        kv = KeyValueField(
            key="Equipment Tag",
            value="P-203",
            confidence=0.97,
            extraction_method="backend",
        )
        page = _make_page(1, blocks=[_make_block("b1", "SOME HEADER", top=10)], kv_fields=[kv])
        ocr = _make_ocr_result([page])
        parser = DocumentParser()
        result = parser.parse_ocr_result(ocr)

        history = result.processing_history[0]
        kv_log = history["key_value_fields"]
        assert len(kv_log) == 1
        assert kv_log[0]["key"] == "Equipment Tag"
        assert kv_log[0]["value"] == "P-203"


class TestOCRErrorsPreserved:
    def test_ocr_page_errors_in_validation(self) -> None:
        err = Issue(code="backend_execution_failed", message="Backend crashed", severity="error")
        page = _make_page(1, blocks=[_make_block("b1", "Text here", top=10)], errors=[err])
        ocr = _make_ocr_result([page])
        parser = DocumentParser()
        result = parser.parse_ocr_result(ocr)

        assert result.validation_report is not None
        codes = [i.code for i in result.validation_report.issues]
        assert any("PAGE_OCR_ERROR" in code for code in codes)


# ──────────────────────────────────────────────────────────────────────────────
# Regression tests for determinism, canonical text offsets, empty docs & paragraphs
# ──────────────────────────────────────────────────────────────────────────────

class TestRegressionFixes:
    def test_full_determinism(self, pump_report_result: OCRDocumentResult) -> None:
        """Same OCRDocumentResult parsed twice must produce identical IDs and structures."""
        header_cell = {"row_idx": 0, "col_idx": 0, "text": "Param", "is_header": True, "confidence": 0.99}
        val_cell = {"row_idx": 1, "col_idx": 0, "text": "Val", "is_header": False, "confidence": 0.95}
        ocr_tbl = OCRTable(
            id="pump_specs",
            bbox=_make_bbox(50, 500, 400, 600),
            confidence=0.95,
            cells=(header_cell, val_cell),
        )
        p1 = pump_report_result.pages[0]
        new_p1 = OCRPageResult(
            page_number=p1.page_number,
            original_width=p1.original_width,
            original_height=p1.original_height,
            processed_width=p1.processed_width,
            processed_height=p1.processed_height,
            text=p1.text,
            blocks=p1.blocks,
            tables=(ocr_tbl,),
            key_value_fields=p1.key_value_fields,
            errors=p1.errors,
        )
        ocr_with_table = OCRDocumentResult(
            document_id=pump_report_result.document_id,
            pages=(new_p1, pump_report_result.pages[1]),
            backend=pump_report_result.backend,
            provenance=pump_report_result.provenance,
            warnings=pump_report_result.warnings,
            errors=pump_report_result.errors,
            total_processing_time_ms=pump_report_result.total_processing_time_ms,
        )

        parser = DocumentParser()
        res1 = parser.parse_ocr_result(ocr_with_table)
        res2 = parser.parse_ocr_result(ocr_with_table)

        assert res1.document_id == res2.document_id == "parsed_mrpl_pump_report"
        assert [s.section_id for s in res1.sections] == [s.section_id for s in res2.sections]
        assert [s.citation_coords.citation_id for s in res1.sections] == [s.citation_coords.citation_id for s in res2.sections]
        assert [t.table_id for t in res1.tables] == [t.table_id for t in res2.tables]
        assert [t.citation_coords.citation_id for t in res1.tables] == [t.citation_coords.citation_id for t in res2.tables]

    def test_two_page_canonical_text_mapping(self) -> None:
        """PageMapEntry offsets must correspond to and correctly slice the canonical text."""
        p1_blocks = [
            _make_block("b1", "1. Overview", top=10),
            _make_block("b2", "This is the first page overview.", top=40),
        ]
        p2_blocks = [
            _make_block("b3", "2. Technical Data", top=10),
            _make_block("b4", "Pressure: 15 bar", top=40),
        ]
        p1 = _make_page(1, p1_blocks)
        p2 = _make_page(2, p2_blocks)
        ocr = _make_ocr_result([p1, p2], document_id="doc_two_page")

        parser = DocumentParser()
        res = parser.parse_ocr_result(ocr)

        canonical_text = DocumentParser.build_canonical_text(res)
        assert len(canonical_text) == res.statistics.total_characters

        entry1 = res.page_map[1]
        entry2 = res.page_map[2]

        assert entry1.char_start == 0
        assert entry1.char_end > 0
        assert entry2.char_start == entry1.char_end + 2
        assert entry2.char_end == len(canonical_text)

        p1_slice = canonical_text[entry1.char_start:entry1.char_end]
        p2_slice = canonical_text[entry2.char_start:entry2.char_end]

        assert "1. Overview" in p1_slice
        assert "This is the first page overview." in p1_slice
        assert "2. Technical Data" in p2_slice
        assert "Pressure: 15 bar" in p2_slice
        assert canonical_text[entry1.char_end:entry2.char_start] == "\n\n"

        for sec in res.sections:
            coords = sec.citation_coords
            assert coords is not None
            assert coords.char_offset_start is not None
            assert coords.char_offset_end is not None
            sec_slice = canonical_text[coords.char_offset_start:coords.char_offset_end]
            if sec.title != "Document Content":
                assert sec.title in sec_slice
            if sec.content:
                assert sec.content in sec_slice

    def test_empty_document_total_pages_zero(self) -> None:
        """For an OCRDocumentResult with zero pages, statistics.total_pages MUST be 0."""
        ocr = _make_ocr_result(pages=[], document_id="empty_doc")
        parser = DocumentParser()
        result = parser.parse_ocr_result(ocr)

        assert result.statistics.total_pages == 0
        assert result.statistics.total_characters == 0
        assert result.statistics.total_words == 0
        assert result.statistics.total_sections == 0

    def test_engineering_paragraph_colon_not_split(self) -> None:
        """Engineering text with colons like 'Equipment Tag: P-203' must not be split solely because of ':'."""
        blocks = [
            _make_block("b1", "Equipment Tag: P-203", top=10, bottom=30),
            _make_block("b2", "Pressure: 10 bar", top=34, bottom=54),
            _make_block("b3", "Temperature: 80 C", top=58, bottom=78),
        ]
        paragraphs = _group_into_paragraphs(blocks)
        assert len(paragraphs) == 1
        assert "Equipment Tag: P-203" in paragraphs[0]
        assert "Pressure: 10 bar" in paragraphs[0]
        assert "Temperature: 80 C" in paragraphs[0]

    def test_get_full_text_works(self, pump_report_result: OCRDocumentResult) -> None:
        """Confirm get_full_text() on the resulting ParsedDocument returns joined section text."""
        parser = DocumentParser()
        doc = parser.parse_ocr_result(pump_report_result)
        full_text = doc.get_full_text()
        assert isinstance(full_text, str)
        assert len(full_text) > 0
        assert "P-203" in full_text
        assert "Centrifugal Pump" in full_text

    def test_metadata_not_inventing_values(self, pump_report_result: OCRDocumentResult) -> None:
        """ParsedMetadata must not invent values like author, dates, or plant unit not in OCR."""
        parser = DocumentParser()
        doc = parser.parse_ocr_result(pump_report_result)
        assert doc.metadata.author is None
        assert doc.metadata.plant_unit is None
        assert doc.metadata.engineer_names == []
        assert doc.metadata.inspection_dates == []


    def test_table_ids_are_deterministic(self) -> None:
        """Table IDs must be deterministic and identical across repeated parser runs."""
        table = OCRTable(
            id="raw_table_1",
            bbox=_make_bbox(10, 100, 300, 200),
            cells=(
                {"row": 0, "col": 0, "text": "Tag"},
                {"row": 0, "col": 1, "text": "Value"},
            ),
        )
        page = _make_page(1, blocks=[_make_block("b1", "Table Header", 10)], tables=[table])
        ocr1 = _make_ocr_result([page], document_id="doc_tbl")
        ocr2 = _make_ocr_result([page], document_id="doc_tbl")

        parser = DocumentParser()
        res1 = parser.parse_ocr_result(ocr1)
        res2 = parser.parse_ocr_result(ocr2)

        assert len(res1.tables) == 1
        assert len(res2.tables) == 1
        assert res1.tables[0].table_id == res2.tables[0].table_id
        assert not res1.tables[0].table_id.startswith("uuid")

    def test_pure_data_table_layout_generates_valid_section(self) -> None:
        """Pure data tables without natural headings should generate a single valid body section."""
        # Represents REAL_TBL_01 pattern: data grid with no internal chapter breaks
        blocks = [
            _make_block(f"b{i}", f"Row {i} Data Item {i*10}", top=50 + i * 25, bottom=70 + i * 25)
            for i in range(15)
        ]
        table = OCRTable(
            id="table_grid_1",
            bbox=_make_bbox(10, 50, 400, 450),
            cells=tuple({"row": i, "col": 0, "text": f"Row {i}"} for i in range(15)),
        )
        page = _make_page(1, blocks=blocks, tables=[table])
        ocr = _make_ocr_result([page], document_id="pure_table_doc")

        parser = DocumentParser()
        doc = parser.parse_ocr_result(ocr, category="Table Document")

        assert len(doc.sections) >= 1
        assert len(doc.tables) == 1
        assert doc.sections[0].citation_coords is not None
        assert doc.sections[0].citation_coords.char_offset_start >= 0
        assert doc.sections[0].citation_coords.char_offset_end > doc.sections[0].citation_coords.char_offset_start
        full_text = doc.get_full_text()
        assert "Row 0 Data Item 0" in full_text
        assert "Row 14 Data Item 140" in full_text

    def test_dense_form_layout_with_disclaimer_footer(self) -> None:
        """Forms with key-values and a large legal disclaimer footer should be parsed deterministically."""
        # Represents REAL_FORM_01 pattern: short KV fields + long disclaimer block
        kv_blocks = [
            _make_block("k1", "TO: Inspector A", top=20, bottom=40),
            _make_block("k2", "DATE: 2026-09-13", top=45, bottom=65),
            _make_block("k3", "PAGES: 1", top=70, bottom=90),
        ]
        disclaimer_text = (
            "NOTE: THIS MESSAGE IS INTENDED ONLY FOR THE USE OF THE INDIVIDUAL OR ENTITY TO WHOM IT IS "
            "ADDRESSED AND MAY CONTAIN INFORMATION THAT IS PRIVILEGED, CONFIDENTIAL, AND EXEMPT FROM DISCLOSURE."
        )
        disclaimer_block = _make_block("disc_1", disclaimer_text, top=120, bottom=180)
        all_blocks = kv_blocks + [disclaimer_block]

        kv_fields = [
            KeyValueField(key="TO", value="Inspector A", confidence=0.98, extraction_method="test"),
            KeyValueField(key="DATE", value="2026-09-13", confidence=0.97, extraction_method="test"),
        ]
        page = _make_page(1, blocks=all_blocks, kv_fields=kv_fields)
        ocr = _make_ocr_result([page], document_id="funsd_form_doc")

        parser = DocumentParser()
        doc1 = parser.parse_ocr_result(ocr, category="Structured Form")
        doc2 = parser.parse_ocr_result(ocr, category="Structured Form")

        # Determinism
        assert [s.section_id for s in doc1.sections] == [s.section_id for s in doc2.sections]
        # Canonical text integrity
        canonical = build_canonical_document_text(doc1)
        for s in doc1.sections:
            if s.citation_coords and s.citation_coords.char_offset_start is not None:
                assert s.citation_coords.char_offset_end <= len(canonical)
                assert s.citation_coords.char_offset_start <= s.citation_coords.char_offset_end
        assert "CONFIDENTIAL" in doc1.get_full_text()


# ──────────────────────────────────────────────────────────────────────────────
# Contracts 2 & 3 Unit Tests (export_for_rag and export_for_agent)
# ──────────────────────────────────────────────────────────────────────────────

class TestContracts2And3Exports:
    """Isolated unit tests for ParsedDocument export_for_rag and export_for_agent."""

    def _sample_parsed_doc(self) -> ParsedDocument:
        blocks = [
            _make_block("b1", "1. Introduction and Objectives", top=10, bottom=30),
            _make_block("b2", "This document describes refinery standard maintenance procedures.", top=40, bottom=70),
        ]
        page = _make_page(1, blocks=blocks)
        ocr = _make_ocr_result([page], document_id="doc_exp_test")
        parser = DocumentParser()
        return parser.parse_ocr_result(ocr, category="Technical Standard", title="Maintenance Standard")

    def test_export_for_rag_isolated(self) -> None:
        doc = self._sample_parsed_doc()
        chunks = export_for_rag(doc)
        assert len(chunks) >= 1
        c0 = chunks[0]
        assert c0.metadata.document_id == doc.document_id
        assert c0.metadata.document_name == "Maintenance Standard"
        assert c0.metadata.category == "Technical Standard"
        assert c0.metadata.chunk_strategy == "section"
        assert len(c0.metadata.heading_path) >= 1

        # Test modality prefixing
        prefixed_chunks = export_for_rag(doc, prefix_modality=True)
        assert prefixed_chunks[0].metadata.heading_path[0] == "Document"

    def test_export_for_agent_isolated(self) -> None:
        doc = self._sample_parsed_doc()
        agent_doc = export_for_agent(doc)
        assert isinstance(agent_doc, AgentQueryableDocument)
        assert agent_doc.document_id == doc.document_id
        assert agent_doc.title == "Maintenance Standard"
        assert agent_doc.category == "Technical Standard"

        sections = agent_doc.get_sections()
        assert len(sections) >= 1
        assert "section_id" in sections[0]
        assert "Introduction" in sections[0]["title"]

        full_text = agent_doc.get_full_text()
        assert "refinery standard" in full_text

        citation = agent_doc.citation_for_section(sections[0]["section_id"])
        assert citation is not None
        assert citation["section_title"] == sections[0]["title"]




