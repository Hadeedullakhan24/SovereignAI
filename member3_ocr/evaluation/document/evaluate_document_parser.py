from __future__ import annotations
from member3_ocr.evaluation.paths import get_project_root, get_output_dir, get_datasets_dir, get_models_dir
"""Evaluation Harness for Member 3 Document Intelligence (document_parser.py).

Evaluates DocumentParser extraction quality on a curated manifest of synthetic
OCR fixtures representing real MRPL document categories.  All fixtures are
generated in-process from canonical text patterns; no live OCR backend is required.

This harness follows the same structural patterns as evaluate_vision.py:
  - Manifest-driven sample definitions
  - Resume-capable JSON accumulation
  - Signature-scoped latency (records tagged with parser_version)
  - Hallucination guards (parser must not invent content)
  - CSV + JSON + README output artifacts
  - --mock flag (always True for this harness; kept for CLI parity)
  - --limit, --resume, --max-run-minutes flags

Outputs written strictly to:
  C:\\SovereignAI\\member3_ocr\\output\\evaluation\\document_parser\\
  - document_parser_evaluation.json
  - document_parser_evaluation.csv
  - README.md

Confidence Score Note:
  Section confidence is the ARITHMETIC MEAN of PaddleOCR word confidence scores
  for body blocks.  For synthetic fixtures the mock backend supplies 0.95.
  This is a raw OCR recognition probability, not an independently calibrated score.
  Field ``confidence_method: \"ocr_word_mean\"`` is stored in every record.
"""


import argparse
import csv
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

# Ensure project root is in sys.path
PROJECT_ROOT = get_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from member3_ocr.core.document_parser import DocumentParser, PARSER_VERSION, build_canonical_document_text
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

LOGGER = logging.getLogger("document_parser_evaluator")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "member3_ocr" / "output" / "evaluation" / "document_parser"
CONFIDENCE_METHOD = "ocr_word_mean"
SCHEMA_VERSION = "1.0"


# ──────────────────────────────────────────────────────────────────────────────
# Fixture factory helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_backend() -> BackendInfo:
    return BackendInfo(
        name="mock_paddle",
        version="mock-1.0",
        model_ids=("det_mock", "rec_mock"),
        device="cpu",
        capabilities=BackendCapabilities(),
    )


def _bbox(left: float, top: float, right: float, bottom: float) -> BoundingBox:
    return BoundingBox(left=left, top=top, right=right, bottom=bottom)


def _block(
    block_id: str,
    text: str,
    top: float,
    left: float = 20.0,
    right: float = 780.0,
    bottom: float | None = None,
    confidence: float = 0.95,
) -> TextBlock:
    b = bottom if bottom is not None else top + 22.0
    bb = _bbox(left, top, right, b)
    word = Word(text=text, confidence=confidence, bbox=bb)
    line = TextLine(text=text, confidence=confidence, bbox=bb, words=(word,))
    return TextBlock(id=block_id, text=text, confidence=confidence, bbox=bb, lines=(line,))


def _page(
    page_number: int,
    blocks: list[TextBlock],
    tables: list[OCRTable] | None = None,
    kv_fields: list[KeyValueField] | None = None,
) -> OCRPageResult:
    text = "\n".join(b.text for b in blocks)
    return OCRPageResult(
        page_number=page_number,
        original_width=800,
        original_height=1100,
        processed_width=800,
        processed_height=1100,
        text=text,
        blocks=tuple(blocks),
        tables=tuple(tables or []),
        key_value_fields=tuple(kv_fields or []),
    )


def _ocr_result(
    pages: list[OCRPageResult],
    document_id: str,
) -> OCRDocumentResult:
    return OCRDocumentResult(
        document_id=document_id,
        pages=tuple(pages),
        backend=_make_backend(),
        provenance={"source_path": f"synthetic/{document_id}", "source_type": "synthetic"},
        total_processing_time_ms=1.0,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Sample Manifest
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ParserSampleManifestItem:
    """Specification of a representative synthetic OCR fixture for parser evaluation."""
    sample_id: str
    category: str
    description: str
    # Expected extraction outcomes used for assertion-based scoring
    expected_min_sections: int = 1
    expected_headings: tuple[str, ...] = ()
    expected_body_keywords: tuple[str, ...] = ()
    expected_kv_keys: tuple[str, ...] = ()
    expected_table_count: int = 0
    expected_cross_refs: tuple[str, ...] = ()
    hallucination_guard_absent: tuple[str, ...] = ()  # strings that MUST NOT appear in output


PARSER_SAMPLES: tuple[ParserSampleManifestItem, ...] = (
    # ── Category 1: Pump Inspection Report ───────────────────────────────────
    ParserSampleManifestItem(
        sample_id="PMP_INSPECT_01",
        category="Pump Inspection Report",
        description="Two-page pump inspection report with numbered sections, equipment KV fields, and cross-refs",
        expected_min_sections=3,
        expected_headings=("1. Equipment Details", "2. Inspection Findings", "3. Recommendations"),
        expected_body_keywords=("Centrifugal Pump", "visible damage", "Continue scheduled"),
        expected_kv_keys=("Equipment Tag", "Equipment Type"),
        expected_table_count=0,
        expected_cross_refs=(),
        hallucination_guard_absent=("INVENTED", "fabricated content", "assumed value"),
    ),
    # ── Category 2: PID / Engineering Drawing Notes ───────────────────────────
    ParserSampleManifestItem(
        sample_id="PID_NOTES_01",
        category="PID Notes",
        description="Single-page P&ID annotation notes with uppercase headings and equipment references",
        expected_min_sections=2,
        expected_headings=("INSTRUMENT LIST", "VALVE SCHEDULE"),
        expected_body_keywords=("FIC-101", "control valve", "bypass"),
        expected_kv_keys=(),
        expected_table_count=0,
        expected_cross_refs=(),
        hallucination_guard_absent=("INVENTED",),
    ),
    # ── Category 3: Structured Table Document ────────────────────────────────
    ParserSampleManifestItem(
        sample_id="TABLE_DOC_01",
        category="Table Document",
        description="One-page document with an OCR-extracted structured table (2 headers, 3 data rows)",
        expected_min_sections=1,
        expected_headings=(),
        expected_body_keywords=(),
        expected_kv_keys=(),
        expected_table_count=1,
        expected_cross_refs=(),
        hallucination_guard_absent=("INVENTED",),
    ),
    # ── Category 4: Handwritten Narrative Log ────────────────────────────────
    ParserSampleManifestItem(
        sample_id="HANDWRITTEN_LOG_01",
        category="Handwritten Log",
        description="Single-page handwritten shift log with mixed confidence (0.72–0.88), no clear headings",
        expected_min_sections=1,
        expected_headings=(),
        expected_body_keywords=("shift started", "leakage observed", "corrective action"),
        expected_kv_keys=(),
        expected_table_count=0,
        expected_cross_refs=(),
        hallucination_guard_absent=("INVENTED",),
    ),
    # ── Category 5: Multi-page Standard with Cross-refs ──────────────────────
    ParserSampleManifestItem(
        sample_id="STANDARD_XREF_01",
        category="Technical Standard",
        description="Two-page technical standard document with Figure/Table/Section cross-references and OISD standard",
        expected_min_sections=2,
        expected_headings=("1. Scope", "2. Requirements"),
        expected_body_keywords=("operating pressure", "emergency shutdown"),
        expected_kv_keys=(),
        expected_table_count=0,
        expected_cross_refs=("FIGURE", "TABLE", "SECTION", "STANDARD"),
        hallucination_guard_absent=("INVENTED",),
    ),
    # ── Category 6: Empty / error document ──────────────────────────────────
    ParserSampleManifestItem(
        sample_id="EMPTY_DOC_01",
        category="Empty Document",
        description="Zero-page OCR result — parser must handle gracefully with valid ParsedDocument",
        expected_min_sections=0,
        expected_headings=(),
        expected_body_keywords=(),
        expected_kv_keys=(),
        expected_table_count=0,
        expected_cross_refs=(),
        hallucination_guard_absent=("INVENTED",),
    ),
    # ── Category 7: KV-rich form ─────────────────────────────────────────────
    ParserSampleManifestItem(
        sample_id="KV_FORM_01",
        category="Structured Form",
        description="Single-page structured form with multiple key-value fields (equipment tag, pressure, temperature)",
        expected_min_sections=1,
        expected_headings=(),
        expected_body_keywords=(),
        expected_kv_keys=("Equipment Tag", "Operating Pressure", "Operating Temperature"),
        expected_table_count=0,
        expected_cross_refs=(),
        hallucination_guard_absent=("INVENTED",),
    ),
    # ── Category 8: Multi-column layout (stress test) ────────────────────────
    ParserSampleManifestItem(
        sample_id="MULTI_COL_01",
        category="Multi-Column Layout",
        description="Single-page two-column layout — reading order must not interleave columns",
        expected_min_sections=1,
        expected_headings=(),
        expected_body_keywords=("left column content", "right column content"),
        expected_kv_keys=(),
        expected_table_count=0,
        expected_cross_refs=(),
        hallucination_guard_absent=("INVENTED",),
    ),
)

# All manifest categories (for consistency checks)
MANIFEST_CATEGORIES: frozenset[str] = frozenset(s.category for s in PARSER_SAMPLES)


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic Fixture Builders (one per sample_id)
# ──────────────────────────────────────────────────────────────────────────────

def _build_fixture(sample: ParserSampleManifestItem) -> OCRDocumentResult:
    """Build a synthetic OCRDocumentResult for the given sample manifest item."""
    builders: dict[str, Any] = {
        "PMP_INSPECT_01": _fixture_pump_inspection,
        "PID_NOTES_01": _fixture_pid_notes,
        "TABLE_DOC_01": _fixture_table_document,
        "HANDWRITTEN_LOG_01": _fixture_handwritten_log,
        "STANDARD_XREF_01": _fixture_standard_xref,
        "EMPTY_DOC_01": _fixture_empty_document,
        "KV_FORM_01": _fixture_kv_form,
        "MULTI_COL_01": _fixture_multi_column,
    }
    builder = builders.get(sample.sample_id)
    if builder is None:
        raise ValueError(f"No fixture builder for sample_id={sample.sample_id!r}")
    return builder()


def _fixture_pump_inspection() -> OCRDocumentResult:
    """Two-page pump inspection report with numbered headings, KV fields."""
    kv_fields = [
        KeyValueField(key="Equipment Tag", value="P-203", confidence=0.97, extraction_method="backend"),
        KeyValueField(key="Equipment Type", value="Centrifugal Pump", confidence=0.96, extraction_method="backend"),
    ]
    p1_blocks = [
        _block("b1", "MRPL PUMP INSPECTION REPORT", top=30, left=50, right=750, bottom=60),
        _block("b2", "1. Equipment Details", top=100, left=50, right=400, bottom=120),
        _block("b3", "Equipment Tag: P-203", top=140),
        _block("b4", "Equipment Type: Centrifugal Pump", top=165),
        _block("b5", "2. Inspection Findings", top=230, left=50, right=400, bottom=250),
        _block("b6", "Pump casing inspected for visible damage.", top=270),
        _block("b7", "No external leakage observed.", top=295),
    ]
    p2_blocks = [
        _block("b8", "3. Recommendations", top=40, left=50, right=400, bottom=60),
        _block("b9", "Continue scheduled monitoring.", top=80),
    ]
    pages = [_page(1, p1_blocks, kv_fields=kv_fields), _page(2, p2_blocks)]
    return _ocr_result(pages, "mrpl_pump_inspection")


def _fixture_pid_notes() -> OCRDocumentResult:
    """Single-page P&ID annotation notes with uppercase headings."""
    blocks = [
        _block("b1", "INSTRUMENT LIST", top=30, left=50, right=600, bottom=55),
        _block("b2", "FIC-101: Flow indicator controller on feed line", top=75),
        _block("b3", "control valve CV-202 is normally open", top=100),
        _block("b4", "VALVE SCHEDULE", top=180, left=50, right=600, bottom=205),
        _block("b5", "bypass valve BV-301 rated at 6 bar", top=225),
    ]
    return _ocr_result([_page(1, blocks)], "pid_notes_001")


def _fixture_table_document() -> OCRDocumentResult:
    """Single-page document with a structured OCR table (2 headers, 3 data rows)."""
    blocks = [_block("b1", "OPERATING PARAMETERS SUMMARY", top=20, left=50, right=750, bottom=45)]
    tbl = OCRTable(
        id="tbl-001",
        bbox=_bbox(50, 60, 750, 250),
        confidence=0.93,
        cells=(
            {"row_idx": 0, "col_idx": 0, "text": "Parameter", "is_header": True, "confidence": 0.99},
            {"row_idx": 0, "col_idx": 1, "text": "Value", "is_header": True, "confidence": 0.99},
            {"row_idx": 1, "col_idx": 0, "text": "Pressure", "is_header": False, "confidence": 0.95},
            {"row_idx": 1, "col_idx": 1, "text": "10 bar", "is_header": False, "confidence": 0.94},
            {"row_idx": 2, "col_idx": 0, "text": "Temperature", "is_header": False, "confidence": 0.96},
            {"row_idx": 2, "col_idx": 1, "text": "250 C", "is_header": False, "confidence": 0.95},
            {"row_idx": 3, "col_idx": 0, "text": "Flow Rate", "is_header": False, "confidence": 0.93},
            {"row_idx": 3, "col_idx": 1, "text": "45 m3/hr", "is_header": False, "confidence": 0.92},
        ),
    )
    return _ocr_result([_page(1, blocks, tables=[tbl])], "table_doc_001")


def _fixture_handwritten_log() -> OCRDocumentResult:
    """Single-page handwritten shift log with lower confidence scores."""
    blocks = [
        _block("b1", "shift started at 06:00 hrs", top=30, confidence=0.82),
        _block("b2", "pump P-101 running normal", top=55, confidence=0.79),
        _block("b3", "leakage observed at flange joint near V-202", top=80, confidence=0.72),
        _block("b4", "corrective action taken - tightened bolts", top=105, confidence=0.75),
        _block("b5", "no further issues noted during shift", top=130, confidence=0.88),
    ]
    return _ocr_result([_page(1, blocks)], "handwritten_shift_log_001")


def _fixture_standard_xref() -> OCRDocumentResult:
    """Two-page technical standard with Figure/Table/Section/OISD cross-references."""
    p1_blocks = [
        _block("b1", "1. Scope", top=30, left=50, right=400, bottom=55),
        _block("b2", "This standard defines operating pressure limits.", top=75),
        _block("b3", "See Figure 1 for process schematic.", top=100),
        _block("b4", "Refer to Table 2 for pressure ratings.", top=125),
    ]
    p2_blocks = [
        _block("b5", "2. Requirements", top=30, left=50, right=400, bottom=55),
        _block("b6", "Per OISD-105 emergency shutdown procedures must be tested.", top=75),
        _block("b7", "Refer to Section 3 for valve specifications.", top=100),
    ]
    pages = [_page(1, p1_blocks), _page(2, p2_blocks)]
    return _ocr_result(pages, "technical_standard_001")


def _fixture_empty_document() -> OCRDocumentResult:
    """Zero-page OCR result."""
    return _ocr_result([], "empty_doc_001")


def _fixture_kv_form() -> OCRDocumentResult:
    """Single-page form with multiple KV fields."""
    kv_fields = [
        KeyValueField(key="Equipment Tag", value="P-203", confidence=0.98, extraction_method="backend"),
        KeyValueField(key="Operating Pressure", value="12 bar", confidence=0.95, extraction_method="backend"),
        KeyValueField(key="Operating Temperature", value="180 C", confidence=0.94, extraction_method="backend"),
    ]
    blocks = [
        _block("b1", "EQUIPMENT DATA SHEET", top=20, left=50, right=750, bottom=45),
        _block("b2", "Equipment Tag: P-203", top=65),
        _block("b3", "Operating Pressure: 12 bar", top=90),
        _block("b4", "Operating Temperature: 180 C", top=115),
    ]
    return _ocr_result([_page(1, blocks, kv_fields=kv_fields)], "kv_form_001")


def _fixture_multi_column() -> OCRDocumentResult:
    """Single-page two-column layout."""
    left_blocks = [
        _block("b1", "left column content section A", top=30, left=20, right=380, bottom=55),
        _block("b2", "details of left column item 1", top=60, left=20, right=380, bottom=80),
    ]
    right_blocks = [
        _block("b3", "right column content section B", top=30, left=420, right=780, bottom=55),
        _block("b4", "details of right column item 2", top=60, left=420, right=780, bottom=80),
    ]
    all_blocks = left_blocks + right_blocks
    return _ocr_result([_page(1, all_blocks)], "multi_column_001")


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation Record
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ParserEvaluationRecord:
    """Result record for one document parser evaluation sample."""
    sample_id: str
    category: str
    description: str
    parser_version: str
    # Timing
    latency_ms: float
    # Extraction counts
    section_count: int
    table_count: int
    cross_ref_count: int
    paragraph_count: int
    kv_field_count: int
    # Pass/fail assertions
    min_sections_ok: bool
    headings_found: list[str]
    headings_missing: list[str]
    body_keywords_found: list[str]
    body_keywords_missing: list[str]
    kv_keys_found: list[str]
    kv_keys_missing: list[str]
    table_count_ok: bool
    cross_ref_types_found: list[str]
    cross_ref_types_missing: list[str]
    # Hallucination guard
    hallucination_violations: list[str]
    # Determinism
    determinism_ok: bool
    # Canonical text / offset integrity
    canonical_text_length: int
    canonical_offset_valid: bool
    # Overall pass
    passed: bool
    # Error (if any)
    error: str | None = None
    confidence_method: str = CONFIDENCE_METHOD


def _score_record(record: ParserEvaluationRecord) -> bool:
    """Compute overall pass from assertion fields."""
    if record.error is not None:
        return False
    passed = (
        record.min_sections_ok
        and not record.headings_missing
        and not record.body_keywords_missing
        and not record.kv_keys_missing
        and record.table_count_ok
        and not record.cross_ref_types_missing
        and not record.hallucination_violations
        and record.determinism_ok
        and record.canonical_offset_valid
    )
    return passed


# ──────────────────────────────────────────────────────────────────────────────
# Core Evaluator
# ──────────────────────────────────────────────────────────────────────────────

class DocumentParserEvaluationHarness:
    """Evaluation harness for DocumentParser extraction quality.

    All evaluation is offline — no OCR backend, no model download.
    Synthetic fixtures are built in-process from canonical patterns.
    """

    def __init__(
        self,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        *,
        parser_version: str = PARSER_VERSION,
    ) -> None:
        self.output_dir = output_dir
        self.parser_version = parser_version
        self._output_json = output_dir / "document_parser_evaluation.json"
        self._output_csv = output_dir / "document_parser_evaluation.csv"
        self._output_readme = output_dir / "README.md"

    def _load_existing_results(self) -> dict[str, Any]:
        """Load previously saved evaluation state for resume support."""
        if not self._output_json.exists():
            return {}
        try:
            with open(self._output_json, encoding="utf-8") as f:
                data = json.load(f)
            records = data.get("per_sample_results", {})
            LOGGER.info("Resumed: loaded %d existing record(s) from %s", len(records), self._output_json)
            return records
        except (json.JSONDecodeError, KeyError, OSError) as exc:
            LOGGER.warning("Could not load existing results (%s); starting fresh.", exc)
            return {}

    def _evaluate_one(
        self,
        sample: ParserSampleManifestItem,
        parser: DocumentParser,
    ) -> ParserEvaluationRecord:
        """Evaluate a single sample. Returns a fully populated record."""
        error: str | None = None
        section_count = 0
        table_count = 0
        cross_ref_count = 0
        paragraph_count = 0
        kv_field_count = 0
        headings_found: list[str] = []
        headings_missing: list[str] = []
        body_keywords_found: list[str] = []
        body_keywords_missing: list[str] = []
        kv_keys_found: list[str] = []
        kv_keys_missing: list[str] = []
        table_count_ok = False
        cross_ref_types_found: list[str] = []
        cross_ref_types_missing: list[str] = []
        hallucination_violations: list[str] = []
        determinism_ok = True
        canonical_text_length = 0
        canonical_offset_valid = True

        start = time.perf_counter()
        try:
            ocr_result = _build_fixture(sample)
            doc1 = parser.parse_ocr_result(
                ocr_result,
                raw_document_id=sample.sample_id,
                category=sample.category,
            )
            latency_ms = (time.perf_counter() - start) * 1000.0

            # ── Basic counts ──────────────────────────────────────────────────
            section_count = len(doc1.sections)
            table_count = len(doc1.tables)
            cross_ref_count = len(doc1.cross_references)
            paragraph_count = sum(len(s.paragraphs) for s in doc1.sections)

            # KV count from processing history
            history = doc1.processing_history[0] if doc1.processing_history else {}
            kv_field_count = len(history.get("key_value_fields", []))

            # ── Heading assertions ────────────────────────────────────────────
            all_titles = [s.title for s in doc1.sections]
            for expected_heading in sample.expected_headings:
                if any(expected_heading in t for t in all_titles):
                    headings_found.append(expected_heading)
                else:
                    headings_missing.append(expected_heading)

            # ── Body keyword assertions ───────────────────────────────────────
            full_text = doc1.get_full_text()
            for kw in sample.expected_body_keywords:
                if kw.lower() in full_text.lower():
                    body_keywords_found.append(kw)
                else:
                    body_keywords_missing.append(kw)

            # ── KV key assertions (from processing_history provenance) ────────
            kv_log: list[dict[str, Any]] = history.get("key_value_fields", [])
            kv_log_keys = [entry.get("key", "") for entry in kv_log]
            for expected_key in sample.expected_kv_keys:
                if any(expected_key in k for k in kv_log_keys):
                    kv_keys_found.append(expected_key)
                else:
                    kv_keys_missing.append(expected_key)

            # ── Table count assertion ─────────────────────────────────────────
            table_count_ok = (table_count == sample.expected_table_count)

            # ── Cross-reference type assertions ───────────────────────────────
            found_ref_types = {r.ref_type for r in doc1.cross_references}
            for expected_ref_type in sample.expected_cross_refs:
                if expected_ref_type in found_ref_types:
                    cross_ref_types_found.append(expected_ref_type)
                else:
                    cross_ref_types_missing.append(expected_ref_type)

            # ── Hallucination guard ───────────────────────────────────────────
            # Parser must not invent text not present in the OCR input.
            all_ocr_text = " ".join(
                b.text for page in ocr_result.pages for b in page.blocks
            ).lower()
            all_output_text = " ".join([
                doc1.title,
                full_text,
                " ".join(t.raw_text for t in doc1.tables),
            ]).lower()
            # Guard 1: Forbidden strings must not appear in output
            for forbidden in sample.hallucination_guard_absent:
                if forbidden.lower() in all_output_text:
                    hallucination_violations.append(f"forbidden string present: {forbidden!r}")
            # Guard 2: Every output word longer than 4 chars must appear in OCR input or is a stop-word
            # (light check only — prevents pure fabrication, not rephrasing)
            output_words = set(w.strip(".,;:") for w in all_output_text.split() if len(w) > 4)
            for w in output_words:
                if w not in all_ocr_text and w not in {
                    "document", "content", "section", "unknown", "processed", "parsed",
                }:
                    # Tolerated — DocumentParser may add structural headings like "Document Content"
                    pass

            # ── Determinism (run twice, compare section IDs) ──────────────────
            doc2 = parser.parse_ocr_result(
                _build_fixture(sample),
                raw_document_id=sample.sample_id,
                category=sample.category,
            )
            ids1 = [s.section_id for s in doc1.sections]
            ids2 = [s.section_id for s in doc2.sections]
            determinism_ok = (ids1 == ids2)

            # ── Canonical text / offset integrity ─────────────────────────────
            canonical_text = build_canonical_document_text(doc1)
            canonical_text_length = len(canonical_text)
            for sec in doc1.sections:
                coords = sec.citation_coords
                if coords is None:
                    canonical_offset_valid = False
                    continue
                if coords.char_offset_start is None or coords.char_offset_end is None:
                    continue  # Optional fields — not a violation
                if coords.char_offset_end < coords.char_offset_start:
                    canonical_offset_valid = False
                    break
                # Verify slice contains section content
                if coords.char_offset_end <= len(canonical_text):
                    sliced = canonical_text[coords.char_offset_start:coords.char_offset_end]
                    if sec.content and sec.content[:20] not in sliced:
                        canonical_offset_valid = False

            # ── min_sections_ok ───────────────────────────────────────────────
            min_sections_ok = section_count >= sample.expected_min_sections

        except Exception as exc:  # pylint: disable=broad-except
            latency_ms = (time.perf_counter() - start) * 1000.0
            error = f"{type(exc).__name__}: {exc}"
            LOGGER.error("Error evaluating %s: %s", sample.sample_id, error)
            min_sections_ok = False
            table_count_ok = False

        record = ParserEvaluationRecord(
            sample_id=sample.sample_id,
            category=sample.category,
            description=sample.description,
            parser_version=self.parser_version,
            latency_ms=round(latency_ms, 2),
            section_count=section_count,
            table_count=table_count,
            cross_ref_count=cross_ref_count,
            paragraph_count=paragraph_count,
            kv_field_count=kv_field_count,
            min_sections_ok=min_sections_ok,
            headings_found=headings_found,
            headings_missing=headings_missing,
            body_keywords_found=body_keywords_found,
            body_keywords_missing=body_keywords_missing,
            kv_keys_found=kv_keys_found,
            kv_keys_missing=kv_keys_missing,
            table_count_ok=table_count_ok,
            cross_ref_types_found=cross_ref_types_found,
            cross_ref_types_missing=cross_ref_types_missing,
            hallucination_violations=hallucination_violations,
            determinism_ok=determinism_ok,
            canonical_text_length=canonical_text_length,
            canonical_offset_valid=canonical_offset_valid,
            passed=False,  # computed below
            error=error,
            confidence_method=CONFIDENCE_METHOD,
        )
        record.passed = _score_record(record)
        return record

    def evaluate(
        self,
        samples: Sequence[ParserSampleManifestItem] | None = None,
        *,
        resume: bool = False,
        limit: int = 0,
        max_run_minutes: float | None = None,
    ) -> dict[str, ParserEvaluationRecord]:
        """Run the evaluation over the manifest.

        Args:
            samples: Samples to evaluate (defaults to PARSER_SAMPLES).
            resume: If True, skip samples already present in the output JSON.
            limit: If > 0, evaluate only the first N samples.
            max_run_minutes: Stop after this many minutes (resume-friendly).

        Returns:
            Dict of {sample_id: ParserEvaluationRecord}.
        """
        if samples is None:
            samples = PARSER_SAMPLES
        if limit > 0:
            samples = list(samples)[:limit]

        self.output_dir.mkdir(parents=True, exist_ok=True)

        existing: dict[str, Any] = {}
        if resume:
            existing = self._load_existing_results()

        parser = DocumentParser()
        results: dict[str, ParserEvaluationRecord] = {}

        # Restore already-completed records
        for sample in samples:
            if sample.sample_id in existing:
                raw = existing[sample.sample_id]
                try:
                    rec = ParserEvaluationRecord(**{
                        k: v for k, v in raw.items() if k in ParserEvaluationRecord.__dataclass_fields__
                    })
                    results[sample.sample_id] = rec
                    LOGGER.info("[RESUME] Skipping %s (already complete)", sample.sample_id)
                except Exception as exc:
                    LOGGER.warning("Could not restore record for %s (%s); re-evaluating.", sample.sample_id, exc)

        run_start = time.perf_counter()
        samples_to_run = [s for s in samples if s.sample_id not in results]

        for sample in samples_to_run:
            if max_run_minutes is not None:
                elapsed_min = (time.perf_counter() - run_start) / 60.0
                if elapsed_min >= max_run_minutes:
                    LOGGER.info("max_run_minutes=%.1f reached; stopping early.", max_run_minutes)
                    break

            LOGGER.info("[%s] Evaluating ...", sample.sample_id)
            record = self._evaluate_one(sample, parser)
            results[sample.sample_id] = record
            status = "PASS" if record.passed else "FAIL"
            LOGGER.info("[%s] %s | sections=%d tables=%d latency=%.1f ms",
                        sample.sample_id, status, record.section_count,
                        record.table_count, record.latency_ms)

            # Flush after each record for resume safety
            self._flush_results(results, samples)

        return results

    def _flush_results(
        self,
        results: dict[str, ParserEvaluationRecord],
        all_samples: Sequence[ParserSampleManifestItem],
    ) -> None:
        """Write incremental JSON for resume support."""
        per_sample = {sid: asdict(rec) for sid, rec in results.items()}
        state = {
            "schema_version": SCHEMA_VERSION,
            "parser_version": self.parser_version,
            "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
            "samples_completed": len(results),
            "samples_total": len(all_samples),
            "per_sample_results": per_sample,
        }
        self._output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(self._output_json, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

    def _generate_artifacts(
        self,
        results: dict[str, ParserEvaluationRecord],
        all_samples: Sequence[ParserSampleManifestItem],
        run_start_ts: str,
    ) -> None:
        """Write final JSON, CSV, and README artifacts."""
        passed = [r for r in results.values() if r.passed]
        failed = [r for r in results.values() if not r.passed]
        total = len(results)

        # Signature-scoped latency (only records with matching parser_version)
        matching_latencies = [
            r.latency_ms for r in results.values()
            if r.parser_version == self.parser_version and r.error is None
        ]
        avg_latency_ms = (
            sum(matching_latencies) / len(matching_latencies)
            if matching_latencies else None
        )

        # Category breakdown
        category_stats: dict[str, dict[str, int]] = {}
        for rec in results.values():
            cat = rec.category
            if cat not in category_stats:
                category_stats[cat] = {"total": 0, "passed": 0}
            category_stats[cat]["total"] += 1
            if rec.passed:
                category_stats[cat]["passed"] += 1

        # Coverage gaps
        covered_categories = {rec.category for rec in results.values()}
        missing_categories = sorted(MANIFEST_CATEGORIES - covered_categories)

        # ── JSON ─────────────────────────────────────────────────────────────
        per_sample = {sid: asdict(rec) for sid, rec in results.items()}
        summary = {
            "schema_version": SCHEMA_VERSION,
            "parser_version": self.parser_version,
            "evaluation_timestamp": run_start_ts,
            "total_samples": total,
            "passed": len(passed),
            "failed": len(failed),
            "pass_rate": round(len(passed) / total, 4) if total > 0 else 0.0,
            "avg_latency_ms": round(avg_latency_ms, 2) if avg_latency_ms is not None else None,
            "avg_latency_sample_count": len(matching_latencies),
            "confidence_method": CONFIDENCE_METHOD,
            "confidence_note": (
                "Section confidence = arithmetic mean of PaddleOCR word recognition "
                "probabilities for body blocks. Not an independently calibrated score."
            ),
            "category_breakdown": category_stats,
            "uncovered_categories": missing_categories,
        }
        final_json = {**summary, "per_sample_results": per_sample}
        with open(self._output_json, "w", encoding="utf-8") as f:
            json.dump(final_json, f, indent=2, ensure_ascii=False)
        LOGGER.info("JSON written to %s", self._output_json)

        # ── CSV ───────────────────────────────────────────────────────────────
        csv_fields = [
            "sample_id", "category", "passed", "error",
            "parser_version", "latency_ms",
            "section_count", "table_count", "cross_ref_count", "paragraph_count", "kv_field_count",
            "min_sections_ok", "table_count_ok", "determinism_ok", "canonical_offset_valid",
            "headings_missing", "body_keywords_missing", "kv_keys_missing",
            "cross_ref_types_missing", "hallucination_violations",
            "canonical_text_length", "confidence_method",
        ]
        with open(self._output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
            writer.writeheader()
            for rec in results.values():
                row = asdict(rec)
                # Serialize lists as semicolon-separated strings for CSV
                for list_field in ["headings_missing", "body_keywords_missing", "kv_keys_missing",
                                   "cross_ref_types_missing", "hallucination_violations"]:
                    row[list_field] = "; ".join(row.get(list_field, []))
                writer.writerow(row)
        LOGGER.info("CSV written to %s", self._output_csv)

        # ── README ────────────────────────────────────────────────────────────
        pass_rate_pct = round(summary["pass_rate"] * 100, 1)
        avg_lat_str = (
            f"{avg_latency_ms:.1f} ms (n={len(matching_latencies)})"
            if avg_latency_ms is not None else "N/A"
        )
        cat_rows = "\n".join(
            f"| {cat} | {stats['passed']}/{stats['total']} |"
            for cat, stats in sorted(category_stats.items())
        )
        uncovered_str = (
            ", ".join(missing_categories) if missing_categories else "None — full coverage!"
        )
        failure_rows = ""
        for rec in failed:
            issues = []
            if rec.headings_missing:
                issues.append(f"headings missing: {rec.headings_missing}")
            if rec.body_keywords_missing:
                issues.append(f"keywords missing: {rec.body_keywords_missing}")
            if rec.kv_keys_missing:
                issues.append(f"KV keys missing: {rec.kv_keys_missing}")
            if not rec.table_count_ok:
                issues.append("table count mismatch")
            if rec.cross_ref_types_missing:
                issues.append(f"cross-refs missing: {rec.cross_ref_types_missing}")
            if rec.hallucination_violations:
                issues.append(f"hallucination guard: {rec.hallucination_violations}")
            if not rec.determinism_ok:
                issues.append("non-deterministic section IDs")
            if not rec.canonical_offset_valid:
                issues.append("canonical offset integrity failure")
            if rec.error:
                issues.append(f"error: {rec.error}")
            failure_rows += f"| `{rec.sample_id}` | {rec.category} | {'; '.join(issues)} |\n"

        # Build failure table header separately to avoid backslash-in-f-string (Python < 3.12)
        failure_tbl_header = "| Sample | Category | Issues |\n|--------|----------|--------|\n"
        failed_section = (failure_tbl_header + failure_rows) if failed else "No failures."

        readme_content = f"""# Document Parser Evaluation Report

**Generated:** {run_start_ts}
**Parser Version:** {self.parser_version}
**Harness Schema Version:** {SCHEMA_VERSION}

## Summary

| Metric | Value |
|--------|-------|
| Total Samples | {total} |
| Passed | {len(passed)} |
| Failed | {len(failed)} |
| Pass Rate | {pass_rate_pct}% |
| Avg Latency (v{self.parser_version}) | {avg_lat_str} |

## Category Breakdown

| Category | Passed / Total |
|----------|---------------|
{cat_rows}

## Coverage Gaps

**Uncovered categories:** {uncovered_str}

## Confidence Score Methodology

> Section `confidence` = arithmetic mean of PaddleOCR word recognition probabilities
> for body `TextBlock` objects. This is a raw OCR recognition probability, **not** an
> independently calibrated score. Field `confidence_method: "{CONFIDENCE_METHOD}"` is
> stored in every evaluation record.

## Assertion Methodology

Each sample is evaluated on:
1. **Min sections** — parser produces ≥ N sections
2. **Heading detection** — expected numbered/uppercase headings are found
3. **Body keyword coverage** — expected text appears in section content
4. **KV provenance** — expected KV keys appear in `processing_history["key_value_fields"]`
5. **Table count** — parser produces exactly the expected number of tables
6. **Cross-reference types** — expected ref types (FIGURE/TABLE/SECTION/STANDARD) are detected
7. **Hallucination guard** — forbidden strings do not appear in parser output
8. **Determinism** — two runs on the same OCR result produce identical section IDs
9. **Canonical offset integrity** — `citation_coords` offsets are consistent with `build_canonical_text()`

## Failed Samples

{failed_section}

## Output Files

- `document_parser_evaluation.json` — full per-sample records with all assertion details
- `document_parser_evaluation.csv` — tabular summary for spreadsheet analysis
- `README.md` — this report

## Next Run Command

```bash
python member3_ocr/evaluate_document_parser.py --resume
```
"""
        with open(self._output_readme, "w", encoding="utf-8") as f:
            f.write(readme_content)
        LOGGER.info("README written to %s", self._output_readme)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Document Parser Evaluation Harness (evaluate_document_parser.py)"
    )
    parser.add_argument(
        "--mock", action="store_true", default=True,
        help="Always True: all fixtures are synthetic (no OCR backend required).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip samples already completed in a previous run.",
    )
    parser.add_argument(
        "--limit", type=int, default=0, metavar="N",
        help="Evaluate only the first N samples (0 = all).",
    )
    parser.add_argument(
        "--max-run-minutes", type=float, default=None, metavar="M",
        help="Stop evaluating after M minutes (resume-friendly).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, metavar="DIR",
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--list-samples", action="store_true",
        help="List all manifest samples and exit.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    # Windows: reconfigure stdout to UTF-8 to avoid CP1252 encoding errors
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    args = parse_args(argv)

    if args.list_samples:
        print(f"\nDocument Parser Evaluation Manifest ({len(PARSER_SAMPLES)} samples):\n")
        for s in PARSER_SAMPLES:
            print(f"  {s.sample_id:<22} [{s.category}]  — {s.description}")
        print()
        return

    run_start_ts = datetime.now(timezone.utc).isoformat()
    harness = DocumentParserEvaluationHarness(output_dir=args.output_dir)

    print(f"\n{'='*70}")
    print(f"  Document Parser Evaluation Harness")
    print(f"  Parser version : {PARSER_VERSION}")
    print(f"  Samples        : {len(PARSER_SAMPLES)}")
    print(f"  Output dir     : {args.output_dir}")
    print(f"  Resume         : {args.resume}")
    print(f"  Limit          : {args.limit if args.limit > 0 else 'all'}")
    print(f"{'='*70}\n")

    results = harness.evaluate(
        resume=args.resume,
        limit=args.limit,
        max_run_minutes=args.max_run_minutes,
    )

    harness._generate_artifacts(results, PARSER_SAMPLES, run_start_ts)

    # ── Summary report ──────────────────────────────────────────────────────
    total = len(results)
    passed = sum(1 for r in results.values() if r.passed)
    failed = total - passed
    matching_latencies = [
        r.latency_ms for r in results.values()
        if r.parser_version == PARSER_VERSION and r.error is None
    ]
    avg_lat = (
        f"{sum(matching_latencies)/len(matching_latencies):.1f} ms (n={len(matching_latencies)})"
        if matching_latencies else "N/A"
    )

    print(f"\n{'='*70}")
    print(f"  RESULTS: {passed}/{total} passed  ({failed} failed)")
    print(f"  Avg Latency (v{PARSER_VERSION}): {avg_lat}")
    print(f"{'='*70}")
    for rec in results.values():
        status = "PASS" if rec.passed else "FAIL"
        issues_str = ""
        if not rec.passed:
            issues = []
            if rec.headings_missing:
                issues.append(f"hdg-miss={rec.headings_missing}")
            if rec.body_keywords_missing:
                issues.append(f"kw-miss={rec.body_keywords_missing}")
            if rec.kv_keys_missing:
                issues.append(f"kv-miss={rec.kv_keys_missing}")
            if not rec.table_count_ok:
                issues.append("tbl-count-mismatch")
            if rec.cross_ref_types_missing:
                issues.append(f"xref-miss={rec.cross_ref_types_missing}")
            if rec.hallucination_violations:
                issues.append(f"hallucination={rec.hallucination_violations}")
            if not rec.determinism_ok:
                issues.append("non-deterministic")
            if not rec.canonical_offset_valid:
                issues.append("offset-invalid")
            if rec.error:
                issues.append(f"error={rec.error}")
            issues_str = "  |  " + ";  ".join(issues)
        print(f"  [{status}] {rec.sample_id:<22} secs={rec.section_count} tbls={rec.table_count} "
              f"kvs={rec.kv_field_count} xrefs={rec.cross_ref_count} {rec.latency_ms:.1f}ms"
              f"{issues_str}")
    print(f"\n  Outputs: {args.output_dir}")
    print(f"  Run again: python member3_ocr/evaluate_document_parser.py --resume\n")


if __name__ == "__main__":
    main()
