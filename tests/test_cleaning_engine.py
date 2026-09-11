"""Comprehensive test suite for Milestone 4 — Cleaning & Normalization Engine.

Covers:
- Stage 1: Unicode normalization (NFKC)
- Stage 2: Encoding normalization
- Stage 3: Whitespace cleanup
- Stage 4: Line ending normalization
- Stage 5: Paragraph reconstruction
- Stage 6: Hyphen repair
- Stage 7: Header removal
- Stage 7: Footer removal
- Stage 8: Page mapping & citation preservation
- Stage 9: Table whitespace cleanup
- Stage 10: Bullet normalization
- Stage 11: List normalization
- Stage 12: Engineering token protection
- Thread safety (RLock across concurrent threads)
- Failure handling
- Plugin loading & dynamic execution
- Health & Observability metrics and events
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import pytest

from rag_engine.preprocessing import (
    BaseCleaner,
    BulletNormalizer,
    CleanerFactory,
    CleanerRegistry,
    CleaningError,
    CleaningEventBus,
    CleaningFailed,
    CleaningFinished,
    CleaningHealthReport,
    CleaningMetricsCollector,
    CleaningPipeline,
    CleaningStarted,
    CorruptedDocumentError,
    EncodingNormalizer,
    EngineeringTokenProtector,
    HeaderFooterDetector,
    HeaderRemoved,
    ListNormalizer,
    NormalizationApplied,
    PageMapper,
    PluginCleanerManager,
    ProtectedTokenDetected,
    TableCleaner,
    TextCleaner,
    UnicodeNormalizer,
    WhitespaceCleaner,
    get_cleaner_factory,
    get_cleaner_registry,
    get_cleaning_event_bus,
    get_cleaning_metrics_collector,
    get_plugin_cleaner_manager,
    health,
    supported_features,
    version,
)
from rag_engine.schemas.parsed_document import (
    CitationCoordinates,
    CleanParsedDocument,
    DocumentStatistics,
    ParsedDocument,
    Section,
    Table,
    TableCell,
)


def _create_sample_parsed_doc(
    doc_id: str = "doc_test_001",
    raw_text: str = "Sample content for testing.",
    page_num: int = 1,
) -> ParsedDocument:
    """Helper to create a standard ParsedDocument fixture."""
    sec = Section(
        section_id=f"sec_{doc_id}_1",
        title="Introduction",
        level=1,
        content=raw_text,
        page_number=page_num,
    )
    return ParsedDocument(
        document_id=doc_id,
        raw_document_id=f"raw_{doc_id}",
        title="Test Document",
        category="manuals",
        sections=[sec],
        statistics=DocumentStatistics(
            total_pages=1,
            total_sections=1,
            total_characters=len(raw_text),
            total_words=len(raw_text.split()),
        ),
    )


# ---------------------------------------------------------------------------
# Stage 1: Unicode Normalization (NFKC)
# ---------------------------------------------------------------------------
def test_unicode_nfkc_normalization() -> None:
    # Ligature 'ﬁ' -> 'fi', fullwidth digits '１２３' -> '123'
    raw = "The ﬁlter efﬁciency is １２３% with café."
    normalized = UnicodeNormalizer.normalize(raw)
    assert "filter" in normalized
    assert "efficiency" in normalized
    assert "123%" in normalized
    assert "café" in normalized


# ---------------------------------------------------------------------------
# Stage 2: Encoding Normalization
# ---------------------------------------------------------------------------
def test_encoding_normalization_control_chars_and_quotes() -> None:
    # Control character \x00, zero-width space \u200b, smart quotes “ ”
    raw = "Clean\x00 this\u200b “text” with ‘quotes’\x07."
    cleaned = EncodingNormalizer.normalize(raw)
    assert cleaned == 'Clean this "text" with \'quotes\'.'
    assert "\x00" not in cleaned
    assert "\u200b" not in cleaned
    assert "\x07" not in cleaned


# ---------------------------------------------------------------------------
# Stage 3: Whitespace Normalization
# ---------------------------------------------------------------------------
def test_whitespace_cleanup() -> None:
    raw = "  Line 1   with   multiple    spaces.   \n\t  Line 2   with\ttabs.  "
    cleaned = WhitespaceCleaner.clean_whitespace(raw)
    expected = "Line 1 with multiple spaces.\nLine 2 with tabs."
    assert cleaned == expected


# ---------------------------------------------------------------------------
# Stage 4: Line Ending Normalization
# ---------------------------------------------------------------------------
def test_line_ending_normalization() -> None:
    raw = "Line 1\r\n\r\n\r\n\r\nLine 2\rLine 3\r\n\r\nLine 4"
    cleaned = WhitespaceCleaner.normalize_line_endings(raw)
    # Consecutive newlines should collapse to maximum 2
    assert "\r" not in cleaned
    assert "\n\n\n" not in cleaned
    assert "Line 1\n\nLine 2\nLine 3\n\nLine 4" in cleaned


# ---------------------------------------------------------------------------
# Stage 5: Broken Paragraph Reconstruction
# ---------------------------------------------------------------------------
def test_broken_paragraph_reconstruction() -> None:
    raw = (
        "This is an operational sentence that was split\n"
        "across lines due to PDF layout wrapping.\n\n"
        "# Section Header\n"
        "Header must not merge with next line.\n\n"
        "- Bullet item 1\n"
        "- Bullet item 2"
    )
    reconstructed = WhitespaceCleaner.reconstruct_broken_paragraphs(raw)
    assert "This is an operational sentence that was split across lines due to PDF layout wrapping." in reconstructed
    assert "# Section Header" in reconstructed
    assert "- Bullet item 1" in reconstructed


# ---------------------------------------------------------------------------
# Stage 6: Hyphenated Word Reconstruction
# ---------------------------------------------------------------------------
def test_hyphen_repair() -> None:
    raw = (
        "The pump is cur-\nrently oper-\nating at maximum capacity.\n"
        "Refinery standard engi-\nneering procedure."
    )
    repaired = WhitespaceCleaner.reconstruct_hyphenated_words(raw)
    assert "currently" in repaired
    assert "operating" in repaired
    assert "engineering" in repaired


# ---------------------------------------------------------------------------
# Stage 7: Header and Footer Detection & Removal
# ---------------------------------------------------------------------------
def test_header_footer_removal() -> None:
    detector = HeaderFooterDetector(min_occurrences=2)

    # 3 sections simulating 3 pages with repeated header & footer
    sec1 = Section(
        section_id="sec_1",
        title="Page 1",
        content="MRPL REFINERY STANDARD PROCEDURE\nActual unique content for section 1.\nPage 1 of 3",
        page_number=1,
    )
    sec2 = Section(
        section_id="sec_2",
        title="Page 2",
        content="MRPL REFINERY STANDARD PROCEDURE\nActual unique content for section 2.\nPage 2 of 3",
        page_number=2,
    )
    sec3 = Section(
        section_id="sec_3",
        title="Page 3",
        content="MRPL REFINERY STANDARD PROCEDURE\nActual unique content for section 3.\nPage 3 of 3",
        page_number=3,
    )

    cleaned_secs, removed_hdrs, removed_ftrs = detector.detect_and_remove([sec1, sec2, sec3])
    assert "MRPL REFINERY STANDARD PROCEDURE" in removed_hdrs
    assert len(removed_ftrs) > 0

    for s in cleaned_secs:
        assert "MRPL REFINERY STANDARD PROCEDURE" not in s.content
        assert "Actual unique content" in s.content


# ---------------------------------------------------------------------------
# Stage 8: Page Mapping & Citation Preservation
# ---------------------------------------------------------------------------
def test_page_mapping_preservation() -> None:
    sec1 = Section(section_id="sec_1", title="S1", content="Page 1 text.", page_number=1)
    sec2 = Section(section_id="sec_2", title="S2", content="Page 2 text.", page_number=2)
    tbl1 = Table(
        table_id="tbl_1",
        caption="T1",
        headers=["Col1"],
        rows=[["Val1"]],
        page_number=2,
    )

    page_map = PageMapper.build_page_map([sec1, sec2], [tbl1], total_pages=2)
    assert 1 in page_map
    assert 2 in page_map
    assert "sec_1" in page_map[1].section_ids
    assert "sec_2" in page_map[2].section_ids
    assert "tbl_1" in page_map[2].table_ids

    # Coordinate update
    updated_secs = PageMapper.update_citation_coordinates([sec1, sec2])
    assert updated_secs[0].citation_coords is not None
    assert updated_secs[0].citation_coords.char_offset_start == 0


# ---------------------------------------------------------------------------
# Stage 9: Table Whitespace Cleanup
# ---------------------------------------------------------------------------
def test_table_whitespace_cleanup() -> None:
    raw_table = Table(
        table_id="tbl_test",
        caption="Pump Specifications",
        headers=["  Equipment Tag  ", "  Operating Pressure  \t"],
        rows=[
            ["   Pump P-203  ", "   10 bar   \n  "],
            ["   MOV-101   ", "   250°C   "],
        ],
        cells=[
            TableCell(row_idx=0, col_idx=0, value="   Pump P-203  "),
            TableCell(row_idx=0, col_idx=1, value="   10 bar   "),
        ],
    )

    cleaned_tbl = TableCleaner.clean_table(raw_table)
    assert cleaned_tbl.headers == ["Equipment Tag", "Operating Pressure"]
    assert cleaned_tbl.rows[0] == ["Pump P-203", "10 bar"]
    assert cleaned_tbl.rows[1] == ["MOV-101", "250°C"]
    assert cleaned_tbl.cells[0].value == "Pump P-203"
    assert "| Equipment Tag | Operating Pressure |" in cleaned_tbl.normalized_text


# ---------------------------------------------------------------------------
# Stage 10: Bullet Normalization
# ---------------------------------------------------------------------------
def test_bullet_normalization() -> None:
    raw = (
        "• First bullet point\n"
        "* Second bullet point\n"
        "▪ Third bullet point\n"
        "► Fourth bullet point"
    )
    cleaned = BulletNormalizer.normalize(raw)
    assert cleaned == (
        "- First bullet point\n"
        "- Second bullet point\n"
        "- Third bullet point\n"
        "- Fourth bullet point"
    )


# ---------------------------------------------------------------------------
# Stage 11: List Normalization
# ---------------------------------------------------------------------------
def test_list_normalization() -> None:
    raw = (
        "1) First item\n"
        "(2) Second item\n"
        "3 - Third item\n"
        "4.   Fourth item"
    )
    cleaned = ListNormalizer.normalize(raw)
    assert "1. First item" in cleaned
    assert "2. Second item" in cleaned
    assert "3. Third item" in cleaned
    assert "4. Fourth item" in cleaned


# ---------------------------------------------------------------------------
# Stage 12: Engineering Token Protection
# ---------------------------------------------------------------------------
def test_engineering_token_protection() -> None:
    protector = EngineeringTokenProtector()

    text = (
        "Inspect Pump P-203 and valve MOV-101 according to API-610 and OISD-105 standards. "
        "The ASME and PNGRB codes specify max limit 10 bar at 250°C. "
        "Differential pressure 15 kg/cm², 2.5 MPa, 150 psi, and flow rate 45 m³/hr. "
        "Check ISO-9001 quality loop Line #4-CDU-101 under Rev. 3."
    )

    tokens = protector.extract_tokens(text)
    expected_tokens = [
        "Pump P-203",
        "MOV-101",
        "API-610",
        "OISD-105",
        "ASME",
        "PNGRB",
        "10 bar",
        "250°C",
        "kg/cm²",
        "MPa",
        "psi",
        "m³/hr",
    ]

    for expected in expected_tokens:
        assert any(expected in t for t in tokens), f"Missing expected token: {expected}"

    # Test masking and unmasking roundtrip
    masked, mapping = protector.mask(text)
    assert "Pump P-203" not in masked
    assert "__ENG_TOKEN_" in masked
    unmasked = protector.unmask(masked, mapping)
    assert unmasked == text

    # Verification
    missing = protector.verify_tokens_preserved(text, text, strict=True)
    assert len(missing) == 0


# ---------------------------------------------------------------------------
# End-to-End Pipeline Execution
# ---------------------------------------------------------------------------
def test_complete_cleaning_pipeline_e2e() -> None:
    pipeline = CleaningPipeline(strict_token_verification=True)

    raw_text = (
        "   REFINERY STANDARD DOCUMENT   \r\n\r\n"
        "The main feed is handled by Pump P-203 and valve MOV-101.\r\n"
        "Oper-\nating pressure must be 10 bar at 250°C per API-610 and OISD-105.\r\n\r\n"
        "• Inspect ASME safety valves\r\n"
        "• Verify PNGRB compliance at 15 kg/cm² or 2.5 MPa\r\n\r\n"
        "1) Check differential pressure of 50 psi\r\n"
        "2) Maintain flow rate at 120 m³/hr\r\n\r\n"
        "REFINERY STANDARD DOCUMENT"
    )

    doc = _create_sample_parsed_doc("doc_e2e_001", raw_text)
    cleaned_doc = pipeline.clean(doc)

    assert isinstance(cleaned_doc, CleanParsedDocument)
    assert cleaned_doc.cleaning_status == "CLEANED"
    assert cleaned_doc.normalization_version == "1.0.0"
    assert cleaned_doc.cleaning_statistics is not None
    assert cleaned_doc.cleaning_statistics.execution_time_ms >= 0.0

    full_cleaned_text = cleaned_doc.get_full_text()
    # Check token preservation
    assert "Pump P-203" in full_cleaned_text
    assert "MOV-101" in full_cleaned_text
    assert "API-610" in full_cleaned_text
    assert "OISD-105" in full_cleaned_text
    assert "10 bar" in full_cleaned_text
    assert "250°C" in full_cleaned_text
    assert "Operating" in full_cleaned_text
    assert "- Inspect ASME safety valves" in full_cleaned_text
    assert "1. Check differential pressure" in full_cleaned_text
    assert len(cleaned_doc.page_map) > 0
    assert len(cleaned_doc.processing_history) >= 12


# ---------------------------------------------------------------------------
# Thread Safety (Concurrent execution with RLock)
# ---------------------------------------------------------------------------
def test_cleaning_pipeline_thread_safety() -> None:
    pipeline = CleaningPipeline()
    docs = [
        _create_sample_parsed_doc(
            f"doc_thread_{i}",
            f"Unit {i} with Pump P-{200 + i} operating at {10 + i} bar and 250°C per API-610.",
        )
        for i in range(20)
    ]

    def _run_clean(d: ParsedDocument) -> CleanParsedDocument:
        return pipeline.clean(d)

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(_run_clean, docs))

    assert len(results) == 20
    for idx, r in enumerate(results):
        assert r.cleaning_status == "CLEANED"
        assert f"Pump P-{200 + idx}" in r.get_full_text()
        assert f"{10 + idx} bar" in r.get_full_text()


# ---------------------------------------------------------------------------
# Failure Handling
# ---------------------------------------------------------------------------
def test_cleaning_failure_handling() -> None:
    pipeline = CleaningPipeline()
    with pytest.raises(CorruptedDocumentError):
        pipeline.clean(None)  # type: ignore


# ---------------------------------------------------------------------------
# Plugin Cleaner Loading
# ---------------------------------------------------------------------------
def test_plugin_cleaner_loading() -> None:
    plugin_mgr = PluginCleanerManager()

    class CustomTestCleaner(BaseCleaner):
        @property
        def name(self) -> str:
            return "CustomTestCleaner"

        def clean(self, document: ParsedDocument) -> CleanParsedDocument:
            return CleanParsedDocument(**document.model_dump(), cleaning_status="CLEANED_BY_PLUGIN")

    plugin_mgr.register_plugin("custom_cleaner", CustomTestCleaner)
    registry = plugin_mgr._registry
    cleaner_cls = registry.get("custom_cleaner")
    assert cleaner_cls is not None
    instance = cleaner_cls()
    assert instance.name == "CustomTestCleaner"


# ---------------------------------------------------------------------------
# Health & Diagnostics
# ---------------------------------------------------------------------------
def test_cleaning_health_and_diagnostics() -> None:
    h = health()
    assert isinstance(h, CleaningHealthReport)
    assert h.status in ("HEALTHY", "DEGRADED")
    assert version() == "1.0.0"
    assert len(supported_features()) >= 12
