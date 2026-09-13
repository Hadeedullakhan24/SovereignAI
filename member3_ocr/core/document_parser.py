"""OCR → ParsedDocument bridge for Member 3.

Converts OCRDocumentResult (from .ocr_pipeline) into Member 1's
ParsedDocument (from rag_engine.schemas.parsed_document).

This module is the sole integration point between OCR output and the
downstream cleaning → chunking → embedding → RAG pipeline.

Design principles:
- Completely offline: no network, no model downloads, no cloud APIs.
- Deterministic: same input always produces the same structural output.
- Conservative: never invents semantic content not present in OCR output.
- Provenance-preserving: OCR geometry is retained in processing_history
  because CitationCoordinates has no bbox field.
- Dataset-safe: never reads or writes inside datasets/.
"""
from __future__ import annotations


import re
import time
from datetime import datetime, timezone
from typing import Any

from rag_engine.schemas.document import DocumentLifecycleState
from rag_engine.schemas.parsed_document import (
    CitationCoordinates,
    CrossReference,
    DocumentStatistics,
    EntityGraph,
    PageMapEntry,
    ParsedDocument,
    ParsedMetadata,
    Section,
    Table,
    TableCell,
    ValidationIssue,
    ValidationReport,
)
try:
    from rag_engine.schemas.chunk import Chunk, ChunkMetadata
except ImportError:
    Chunk = None  # type: ignore[assignment, misc]
    ChunkMetadata = None  # type: ignore[assignment, misc]

from .ocr_pipeline import (
    KeyValueField,
    OCRDocumentResult,
    OCRPageResult,
    Table as OCRTable,
    TextBlock,
)

# ──────────────────────────────────────────────────────────────────────────────
# Module constants
# ──────────────────────────────────────────────────────────────────────────────

PARSER_VERSION = "1.0.0"

INTER_PAGE_SEPARATOR = "\n\n"
SECTION_SEPARATOR = "\n\n"

# Heading heuristics — conservative thresholds chosen to avoid mis-classifying
# ordinary engineering text.
_MAX_HEADING_CHARS = 120       # Longer lines are almost never headings.
_MIN_HEADING_ALPHA = 3         # Need at least 3 alphabetic chars.
_UPPERCASE_RATIO_THRESHOLD = 0.75  # For "all-caps" heading detection.

# Numbered heading pattern: "1.", "1.1 Title", "2.3.1 Sub-section" at line start.
_NUMERIC_HEADING_RE = re.compile(r"^(\d+(?:\.\d+)*)(?:[.\s]\s*|\s+)(.+)$")

# Cross-reference pattern (kept local to avoid importing rag_engine.parsers
# which has its own heavy dependency tree).
_CROSS_REF_RE = re.compile(
    r"\b(Fig(?:ure)?\.?\s*\d+(?:\.\d+)*"
    r"|Table\s*\d+(?:\.\d+)*"
    r"|Section\s*\d+(?:\.\d+)*"
    r"|OISD(?:-STD)?-\d+"
    r"|API-\d+"
    r"|ASME\s+[A-Z0-9.]+)\b",
    re.IGNORECASE,
)

# Paragraph separation punctuation: sentence-ending punctuation only.
# Colons (":") and semicolons (";") are explicitly excluded so engineering text
# like "Equipment Tag: P-203" is not split prematurely.
_PARAGRAPH_END_PUNCT = frozenset({".", "!", "?"})

# Rounding tolerance used by PaddleOCRBackend reading-order sort (px).
_LINE_HEIGHT_TOLERANCE_PX = 10


# ──────────────────────────────────────────────────────────────────────────────
# Exceptions
# ──────────────────────────────────────────────────────────────────────────────

class DocumentParserError(RuntimeError):
    """Base exception for document parser errors."""


class EmptyOCRResultError(DocumentParserError):
    """Raised when the OCR result contains no pages and no usable content."""


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

class DocumentParser:
    """Convert an OCRDocumentResult into a Member 1 ParsedDocument.

    This class is the bridge between the OCR pipeline and the RAG pipeline.
    It does NOT perform OCR itself, does NOT call any model, and does NOT
    access the network.  All structural extraction is deterministic.

    Usage::

        parser = DocumentParser()
        parsed = parser.parse_ocr_result(
            ocr_result,
            raw_document_id="doc-abc123",
            title="MRPL Pump Inspection Report",
            category="inspection_report",
        )
    """

    def __init__(self, *, version: str = PARSER_VERSION) -> None:
        self.version = version

    # ── Main entry point ──────────────────────────────────────────────────────

    def parse_ocr_result(
        self,
        ocr_result: OCRDocumentResult,
        *,
        raw_document_id: str | None = None,
        title: str | None = None,
        category: str = "Unknown",
    ) -> ParsedDocument:
        """Convert an OCRDocumentResult into a ParsedDocument.

        Parameters
        ----------
        ocr_result:
            The normalized OCR output produced by OCRPipeline.process_image()
            or an equivalent synthetic OCRDocumentResult.
        raw_document_id:
            The ID of the upstream source Document (Milestone 3A).  Defaults
            to ocr_result.document_id when not supplied.
        title:
            Explicit document title.  When omitted, a conservative heuristic
            is applied.  Pass an empty string to suppress heuristic detection.
        category:
            Document category string for ParsedMetadata and ParsedDocument.

        Returns
        -------
        ParsedDocument
            A fully populated ParsedDocument with lifecycle_state=PARSED.

        Raises
        ------
        TypeError
            If ocr_result is not an OCRDocumentResult.
        """
        if not isinstance(ocr_result, OCRDocumentResult):
            raise TypeError(
                f"ocr_result must be an OCRDocumentResult, got {type(ocr_result).__name__}"
            )

        start_ns = time.perf_counter_ns()

        resolved_raw_id = raw_document_id or ocr_result.document_id
        document_id = f"parsed_{ocr_result.document_id}"

        # ── Per-page extraction ───────────────────────────────────────────────
        all_sections: list[Section] = []
        all_tables: list[Table] = []
        all_cross_refs: list[CrossReference] = []
        page_map: dict[int, PageMapEntry] = {}
        ocr_geometry_log: list[dict[str, Any]] = []
        kv_fields_log: list[dict[str, Any]] = []
        page_texts: list[str] = []

        # Running character cursor across all pages (for page_map char offsets).
        char_cursor = 0
        total_page_count = len(ocr_result.pages)

        for page_idx, page in enumerate(ocr_result.pages):
            is_last_page = (page_idx == total_page_count - 1)
            (
                page_sections,
                page_tables,
                page_refs,
                page_entry,
                page_geom,
                page_kvs,
                page_text,
                char_cursor,
            ) = self._process_page(page, char_cursor, is_last_page=is_last_page)

            all_sections.extend(page_sections)
            all_tables.extend(page_tables)
            all_cross_refs.extend(page_refs)
            page_map[page.page_number] = page_entry
            ocr_geometry_log.extend(page_geom)
            kv_fields_log.extend(page_kvs)
            page_texts.append(page_text)

        # ── Title detection ───────────────────────────────────────────────────
        if title is None:
            resolved_title = self._detect_title(ocr_result.pages, all_sections)
        else:
            resolved_title = title

        # ── Metadata ─────────────────────────────────────────────────────────
        metadata = ParsedMetadata(
            title=resolved_title,
            category=category,
        )

        # ── Statistics ───────────────────────────────────────────────────────
        full_text = INTER_PAGE_SEPARATOR.join(page_texts)
        statistics = _compute_statistics(
            full_text=full_text,
            sections=all_sections,
            tables=all_tables,
            pages=ocr_result.pages,
        )

        # ── Validation report ─────────────────────────────────────────────────
        validation_report = self._validate(all_sections, all_tables, ocr_result)

        # ── Processing history ────────────────────────────────────────────────
        elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
        processing_history_entry: dict[str, Any] = {
            "stage": "document_parser",
            "parser_version": self.version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source_ocr_schema_version": ocr_result.schema_version,
            "backend_name": ocr_result.backend.name,
            "backend_version": ocr_result.backend.version,
            "backend_device": ocr_result.backend.device,
            "page_count": len(ocr_result.pages),
            "section_count": len(all_sections),
            "table_count": len(all_tables),
            "cross_ref_count": len(all_cross_refs),
            "ocr_warnings": [
                {"code": w.code, "message": w.message, "severity": w.severity}
                for w in ocr_result.warnings
            ],
            "ocr_errors": [
                {"code": e.code, "message": e.message, "severity": e.severity}
                for e in ocr_result.errors
            ],
            "processing_time_ms": round(elapsed_ms, 3),
            # OCR geometry is preserved here because CitationCoordinates does
            # not have a bbox field.  This allows future stages to recover
            # bounding-box data for highlighting and citation navigation.
            "ocr_geometry": ocr_geometry_log,
            # Key-value fields are preserved here since ParsedDocument has no
            # dedicated kv_fields list; this is documented as a known limitation.
            "key_value_fields": kv_fields_log,
        }

        return ParsedDocument(
            document_id=document_id,
            raw_document_id=resolved_raw_id,
            title=resolved_title,
            category=category,
            sections=all_sections,
            tables=all_tables,
            equipment=[],
            entity_graph=EntityGraph(),
            warnings=[],
            cross_references=all_cross_refs,
            drawing_metadata=None,
            metadata=metadata,
            statistics=statistics,
            processing_history=[processing_history_entry],
            validation_report=validation_report,
            lifecycle_state=DocumentLifecycleState.PARSED,
            cleaning_status="RAW",
            cleaning_statistics=None,
            normalization_version="1.0.0",
            page_map=page_map,
            removed_headers=[],
            removed_footers=[],
            protected_tokens=[],
            cleaning_warnings=[],
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Per-page processing
    # ──────────────────────────────────────────────────────────────────────────

    def _process_page(
        self,
        page: OCRPageResult,
        char_cursor: int,
        *,
        is_last_page: bool = False,
    ) -> tuple[
        list[Section],
        list[Table],
        list[CrossReference],
        PageMapEntry,
        list[dict[str, Any]],
        list[dict[str, Any]],
        str,
        int,
    ]:
        """Extract all content from one OCR page."""
        page_num = page.page_number

        # Sort blocks into reading order.
        sorted_blocks = _sort_blocks_reading_order(page.blocks)

        # Build geometry provenance log (bbox data not storable in CitationCoordinates).
        geometry_log = _build_geometry_log(page_num, sorted_blocks)

        # Build key-value provenance log.
        kv_log = _build_kv_log(page_num, page.key_value_fields)

        # Extract sections from blocks with consistent canonical text construction.
        page_char_start = char_cursor
        sections, page_sec_texts, char_cursor = self._extract_sections_from_blocks(
            sorted_blocks, page_num, char_cursor
        )
        page_char_end = char_cursor
        page_text = SECTION_SEPARATOR.join(page_sec_texts)

        # Account for inter-page separator if not the last page.
        if not is_last_page:
            char_cursor += len(INTER_PAGE_SEPARATOR)

        # Convert OCR tables if present.
        tables = [
            _convert_ocr_table(ocr_tbl, page_num, tbl_idx)
            for tbl_idx, ocr_tbl in enumerate(page.tables)
        ]

        # Extract cross-references from page canonical text.
        cross_refs = _extract_cross_refs(page_text, page_num)

        # Build page map entry.
        page_entry = PageMapEntry(
            page_number=page_num,
            char_start=page_char_start,
            char_end=page_char_end,
            section_ids=[s.section_id for s in sections],
            table_ids=[t.table_id for t in tables],
        )

        return sections, tables, cross_refs, page_entry, geometry_log, kv_log, page_text, char_cursor

    # ──────────────────────────────────────────────────────────────────────────
    # Section extraction
    # ──────────────────────────────────────────────────────────────────────────

    def _extract_sections_from_blocks(
        self,
        blocks: list[TextBlock],
        page_num: int,
        start_char_cursor: int,
    ) -> tuple[list[Section], list[str], int]:
        """Convert a reading-order list of TextBlocks into Section objects.

        Strategy
        --------
        1. Classify each block as a heading candidate or body text using
           conservative deterministic rules.
        2. Accumulate body blocks under the current heading.
        3. Flush the accumulated body when a new heading is detected.
        4. If no heading is detected, all content falls under a default section.
        5. Build deterministic section IDs and aligned citation coordinates.
        """
        if not blocks:
            return [], [], start_char_cursor

        # Estimate typical line height on this page to detect isolated blocks.
        line_heights = [
            b.bbox.bottom - b.bbox.top
            for b in blocks
            if b.bbox.bottom > b.bbox.top
        ]
        median_line_h = _median(line_heights) if line_heights else 20.0

        raw_sections: list[dict[str, Any]] = []
        current_heading: str | None = None
        current_heading_level: int = 1
        current_heading_confidence: float = 0.90
        current_body_blocks: list[TextBlock] = []

        def flush() -> None:
            nonlocal current_heading, current_body_blocks

            has_explicit_heading = (current_heading is not None)
            heading = current_heading if has_explicit_heading else "Document Content"
            body_texts = [b.text for b in current_body_blocks]
            paragraphs = _group_into_paragraphs(current_body_blocks)
            body = "\n".join(body_texts)

            # Skip empty default sections where neither heading nor body exists.
            if not has_explicit_heading and not body.strip() and not paragraphs:
                current_heading = None
                current_body_blocks = []
                return

            confidences = [
                b.confidence for b in current_body_blocks if b.confidence is not None
            ]
            avg_conf = (
                sum(confidences) / len(confidences)
                if confidences
                else current_heading_confidence
            )

            raw_sections.append({
                "title": heading,
                "level": current_heading_level,
                "content": body,
                "paragraphs": paragraphs,
                "has_explicit_heading": has_explicit_heading,
                "confidence": round(avg_conf, 4),
            })

            current_heading = None
            current_body_blocks = []

        for i, block in enumerate(blocks):
            text = block.text.strip()
            if not text:
                continue

            prev_bottom = blocks[i - 1].bbox.bottom if i > 0 else block.bbox.top
            gap_above = max(0.0, block.bbox.top - prev_bottom)
            is_isolated = gap_above > median_line_h * 1.5

            heading_result = _classify_heading(text, is_isolated)

            if heading_result is not None:
                heading_text, heading_level, heading_conf = heading_result
                flush()
                current_heading = heading_text
                current_heading_level = heading_level
                current_heading_confidence = heading_conf
            else:
                current_body_blocks.append(block)

        flush()

        sections: list[Section] = []
        page_sec_texts: list[str] = []
        char_cursor = start_char_cursor

        for sec_idx, raw_sec in enumerate(raw_sections):
            sec_text = _section_canonical_text(
                raw_sec["title"],
                raw_sec["content"],
                raw_sec["has_explicit_heading"],
            )
            sec_char_start = char_cursor
            sec_char_end = sec_char_start + len(sec_text)
            sec_id = f"sec_{page_num}_{sec_idx}"
            cit_id = f"cit_{sec_id}"

            section = Section(
                section_id=sec_id,
                title=raw_sec["title"],
                level=raw_sec["level"],
                content=raw_sec["content"],
                raw_text=raw_sec["content"],
                normalized_text=raw_sec["content"],
                paragraphs=raw_sec["paragraphs"],
                bullet_points=[],
                numbered_items=[],
                page_number=page_num,
                citation_coords=CitationCoordinates(
                    page_number=page_num,
                    section_title=raw_sec["title"],
                    paragraph_index=0,
                    char_offset_start=sec_char_start,
                    char_offset_end=sec_char_end,
                    citation_id=cit_id,
                ),
                confidence=raw_sec["confidence"],
            )
            sections.append(section)
            page_sec_texts.append(sec_text)
            char_cursor = sec_char_end
            if sec_idx < len(raw_sections) - 1:
                char_cursor += len(SECTION_SEPARATOR)

        return sections, page_sec_texts, char_cursor

    # ──────────────────────────────────────────────────────────────────────────
    # Title detection
    # ──────────────────────────────────────────────────────────────────────────

    def _detect_title(
        self,
        pages: tuple[OCRPageResult, ...],
        sections: list[Section],
    ) -> str:
        """Heuristically detect the document title from the first page.

        Candidates (in order of preference):
        1. First section heading from page 1 if it looks like a document title
           (1-12 words, not just a number).
        2. First non-empty uppercase-heavy line from page 1.
        3. Empty string (uncertain — callers should supply explicit title).
        """
        if not pages:
            return ""

        first_page = pages[0]
        first_blocks = _sort_blocks_reading_order(first_page.blocks)

        # Prefer first detected heading from page 1.
        for sec in sections:
            if sec.page_number == first_page.page_number and sec.title != "Document Content":
                words = sec.title.split()
                if 1 <= len(words) <= 12:
                    return sec.title

        # Fall back to first prominent line on page 1.
        for block in first_blocks:
            text = block.text.strip()
            if not text or len(text) > _MAX_HEADING_CHARS:
                continue
            alpha_chars = sum(1 for c in text if c.isalpha())
            if alpha_chars < _MIN_HEADING_ALPHA:
                continue
            upper_ratio = sum(1 for c in text if c.isupper()) / max(alpha_chars, 1)
            if upper_ratio >= _UPPERCASE_RATIO_THRESHOLD:
                return text

        return ""

    # ──────────────────────────────────────────────────────────────────────────
    # Validation
    # ──────────────────────────────────────────────────────────────────────────

    def _validate(
        self,
        sections: list[Section],
        tables: list[Table],
        ocr_result: OCRDocumentResult,
    ) -> ValidationReport:
        """Produce a lightweight structural validation report."""
        issues: list[ValidationIssue] = []

        if not ocr_result.pages:
            issues.append(ValidationIssue(
                severity="WARNING",
                code="NO_PAGES",
                message="OCR result contains no pages.",
                location="document",
            ))

        for page in ocr_result.pages:
            if page.errors:
                for err in page.errors:
                    issues.append(ValidationIssue(
                        severity="WARNING",
                        code=f"PAGE_OCR_ERROR_{err.code.upper()}",
                        message=err.message,
                        location=f"page:{page.page_number}",
                    ))
            if not page.text.strip():
                issues.append(ValidationIssue(
                    severity="INFO",
                    code="EMPTY_PAGE",
                    message=f"Page {page.page_number} produced no text.",
                    location=f"page:{page.page_number}",
                ))

        if not sections:
            issues.append(ValidationIssue(
                severity="INFO",
                code="NO_SECTIONS",
                message="Document produced no extractable sections.",
                location="document",
            ))

        error_count = sum(1 for i in issues if i.severity == "ERROR")
        warning_count = sum(1 for i in issues if i.severity == "WARNING")

        return ValidationReport(
            is_valid=error_count == 0,
            issues=issues,
            error_count=error_count,
            warning_count=warning_count,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Canonical text reconstruction
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def build_canonical_text(document: ParsedDocument) -> str:
        """Reconstruct the canonical document text matching page_map and citation offsets."""
        return build_canonical_document_text(document)


# ──────────────────────────────────────────────────────────────────────────────
# Canonical document text helpers
# ──────────────────────────────────────────────────────────────────────────────

def _section_canonical_text(title: str, content: str, has_explicit_heading: bool) -> str:
    """Return the canonical text representation of a section."""
    if has_explicit_heading:
        if content:
            return f"{title}\n{content}"
        return title
    return content


def build_canonical_document_text(document: ParsedDocument) -> str:
    """Reconstruct the canonical document text matching page_map and citation offsets."""
    page_texts: list[str] = []
    sorted_page_nums = sorted(document.page_map.keys())
    for page_num in sorted_page_nums:
        page_entry = document.page_map[page_num]
        sec_id_set = set(page_entry.section_ids)
        page_sections = [s for s in document.sections if s.section_id in sec_id_set]
        sec_texts = [
            _section_canonical_text(s.title, s.content, s.title != "Document Content")
            for s in page_sections
        ]
        page_texts.append(SECTION_SEPARATOR.join(sec_texts))
    return INTER_PAGE_SEPARATOR.join(page_texts)


# ──────────────────────────────────────────────────────────────────────────────
# Heading classification (module-level helper, deterministic)
# ──────────────────────────────────────────────────────────────────────────────

def _classify_heading(
    text: str,
    is_isolated: bool,
) -> tuple[str, int, float] | None:
    """Return (heading_text, level, confidence) or None if not a heading.

    Rules (in priority order):
    1. Numbered heading pattern: "1.", "1.1 Title", "2.3.1 Sub-section"
       — level = dot count + 1
    2. Short, uppercase-heavy line (>=75% uppercase alpha, <=80 chars).
    3. Isolated block (large vertical gap) that is short (<=10 words) and has
       mixed/title case, and does not end with ordinary punctuation.
    """
    stripped = text.strip()
    if not stripped:
        return None

    alpha_chars = sum(1 for c in stripped if c.isalpha())
    if alpha_chars < _MIN_HEADING_ALPHA:
        return None

    # Rule 1: numbered heading
    num_match = _NUMERIC_HEADING_RE.match(stripped)
    if num_match and len(stripped) <= _MAX_HEADING_CHARS and not stripped.endswith("."):
        number_part = num_match.group(1)
        level = min(number_part.count(".") + 1, 6)
        return stripped, level, 0.95

    # Rule 2: uppercase-heavy short line
    if len(stripped) <= 80:
        upper_ratio = sum(1 for c in stripped if c.isupper()) / max(alpha_chars, 1)
        if upper_ratio >= _UPPERCASE_RATIO_THRESHOLD and not stripped.endswith("."):
            return stripped, 1, 0.90

    # Rule 3: isolated and short (<=10 words), does not end with prose punctuation
    if is_isolated:
        words = stripped.split()
        if (
            1 <= len(words) <= 10
            and stripped[-1] not in {".", ",", ";"}
            and not stripped[-1].isdigit()
        ):
            return stripped, 2, 0.80

    return None


# ──────────────────────────────────────────────────────────────────────────────
# Reading order
# ──────────────────────────────────────────────────────────────────────────────

def _sort_blocks_reading_order(blocks: tuple[TextBlock, ...]) -> list[TextBlock]:
    """Sort TextBlocks in top-to-bottom, left-to-right reading order.

    Uses the same 10-pixel rounding that PaddleOCRBackend.normalize_raw_output
    applies, so the order is consistent with OCR pipeline output.
    """
    return sorted(
        blocks,
        key=lambda b: (
            round(b.bbox.top / _LINE_HEIGHT_TOLERANCE_PX) * _LINE_HEIGHT_TOLERANCE_PX,
            b.bbox.left,
        ),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Paragraph grouping
# ──────────────────────────────────────────────────────────────────────────────

def _group_into_paragraphs(blocks: list[TextBlock]) -> list[str]:
    """Group consecutive body blocks into paragraph strings.

    A new paragraph starts when:
    - There is a significant vertical gap between consecutive blocks.
    - The previous block's text ends with sentence-terminating punctuation (. ! ?).
    """
    if not blocks:
        return []

    paragraphs: list[str] = []
    current_lines: list[str] = []

    # Estimate typical inter-line gap.
    gaps: list[float] = []
    for i in range(1, len(blocks)):
        gap = max(0.0, blocks[i].bbox.top - blocks[i - 1].bbox.bottom)
        gaps.append(gap)
    typical_gap = _median(gaps) if gaps else 5.0
    gap_threshold = max(typical_gap * 1.8, 8.0)

    for i, block in enumerate(blocks):
        text = block.text.strip()
        if not text:
            continue

        current_lines.append(text)

        flush_paragraph = False
        if i < len(blocks) - 1:
            gap = max(0.0, blocks[i + 1].bbox.top - block.bbox.bottom)
            if gap >= gap_threshold:
                flush_paragraph = True
            elif text and text[-1] in _PARAGRAPH_END_PUNCT:
                flush_paragraph = True
        else:
            flush_paragraph = True

        if flush_paragraph and current_lines:
            paragraphs.append(" ".join(current_lines))
            current_lines = []

    if current_lines:
        paragraphs.append(" ".join(current_lines))

    return paragraphs


# ──────────────────────────────────────────────────────────────────────────────
# Table conversion
# ──────────────────────────────────────────────────────────────────────────────

def _convert_ocr_table(ocr_tbl: OCRTable, page_num: int, table_index: int = 0) -> Table:
    """Convert an OCRTable dataclass into a Member 1 Table model.

    The current PaddleOCR basic backend does not produce table data, so this
    function handles the reserved Table schema that future backends may fill.
    """
    cells: list[TableCell] = []
    headers: list[str] = []
    rows: list[list[str]] = []

    for raw_cell in ocr_tbl.cells:
        if not isinstance(raw_cell, dict):
            try:
                raw_cell = dict(raw_cell)
            except (TypeError, ValueError):
                continue

        row_idx = int(raw_cell.get("row_idx", 0))
        col_idx = int(raw_cell.get("col_idx", 0))
        value = str(raw_cell.get("text", raw_cell.get("value", "")))
        is_header = bool(raw_cell.get("is_header", row_idx == 0))
        conf_raw = raw_cell.get("confidence", ocr_tbl.confidence)
        conf = float(conf_raw) if conf_raw is not None else 1.0

        cells.append(TableCell(
            row_idx=row_idx,
            col_idx=col_idx,
            value=value,
            raw_value=value,
            is_header=is_header,
            confidence=min(1.0, max(0.0, conf)),
        ))

        if is_header:
            while len(headers) <= col_idx:
                headers.append("")
            headers[col_idx] = value
        else:
            while len(rows) <= row_idx - 1:
                rows.append([])
            row = rows[row_idx - 1]
            while len(row) <= col_idx:
                row.append("")
            row[col_idx] = value

    col_count = max((c.col_idx + 1 for c in cells), default=0)
    row_count = len(rows)

    tbl_suffix = f"_{ocr_tbl.id}" if ocr_tbl.id else ""
    table_id = f"tbl_{page_num}_{table_index}{tbl_suffix}"
    cit_id = f"cit_{table_id}"

    raw_text = ocr_tbl.markdown or ocr_tbl.html or ""

    dataframe_dict: dict[str, list[Any]] = {}
    for c in range(len(headers)):
        col_key = headers[c] if headers[c] else f"col_{c}"
        dataframe_dict[col_key] = [row[c] if c < len(row) else "" for row in rows]

    return Table(
        table_id=table_id,
        caption="",
        headers=headers,
        rows=rows,
        row_count=row_count,
        col_count=col_count,
        page_number=page_num,
        dataframe_dict=dataframe_dict,
        raw_text=raw_text,
        normalized_text=raw_text,
        cells=cells,
        citation_coords=CitationCoordinates(
            page_number=page_num,
            citation_id=cit_id,
        ),
        confidence=min(1.0, max(0.0, ocr_tbl.confidence or 1.0)),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Cross-reference extraction
# ──────────────────────────────────────────────────────────────────────────────

def _extract_cross_refs(text: str, page_num: int) -> list[CrossReference]:
    """Extract figure/table/section/standard cross-references from text."""
    refs: list[CrossReference] = []
    seen: set[str] = set()
    ref_idx = 0

    for match in _CROSS_REF_RE.finditer(text):
        val = match.group(1).strip()
        val_lower = val.lower()
        if val_lower in seen:
            continue
        seen.add(val_lower)

        val_upper = val.upper()
        if "FIG" in val_upper:
            ref_type = "FIGURE"
        elif "TABLE" in val_upper:
            ref_type = "TABLE"
        elif "SECTION" in val_upper:
            ref_type = "SECTION"
        else:
            ref_type = "STANDARD"

        refs.append(CrossReference(
            ref_type=ref_type,
            target=val,
            raw_text=val,
            page_number=page_num,
            citation_id=f"cit_ref_{page_num}_{ref_idx}",
            confidence=0.90,
        ))
        ref_idx += 1

    return refs


# ──────────────────────────────────────────────────────────────────────────────
# Statistics
# ──────────────────────────────────────────────────────────────────────────────

def _compute_statistics(
    full_text: str,
    sections: list[Section],
    tables: list[Table],
    pages: tuple[OCRPageResult, ...],
) -> DocumentStatistics:
    """Compute DocumentStatistics from extracted content."""
    total_paragraphs = sum(len(s.paragraphs) for s in sections)
    words = len(full_text.split()) if full_text.strip() else 0
    chars = len(full_text)
    total_pages = len(pages)

    return DocumentStatistics(
        total_pages=total_pages,
        total_sections=len(sections),
        total_paragraphs=total_paragraphs,
        total_words=words,
        total_characters=chars,
        total_tables=len(tables),
        total_entities=0,
        total_warnings=0,
        total_relations=0,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Geometry / KV provenance helpers
# ──────────────────────────────────────────────────────────────────────────────

def _build_geometry_log(page_num: int, blocks: list[TextBlock]) -> list[dict[str, Any]]:
    """Serialize OCR bounding-box geometry into a JSON-safe provenance record.

    CitationCoordinates does not carry a bbox field, so this log in
    processing_history preserves the data needed for future bounding-box
    citation navigation.
    """
    return [
        {
            "page_number": page_num,
            "block_id": block.id,
            "text_snippet": block.text[:60],
            "bbox": {
                "left": block.bbox.left,
                "top": block.bbox.top,
                "right": block.bbox.right,
                "bottom": block.bbox.bottom,
                "coordinate_space": block.bbox.coordinate_space,
            },
            "confidence": block.confidence,
        }
        for block in blocks
    ]


def _build_kv_log(page_num: int, kv_fields: tuple[KeyValueField, ...]) -> list[dict[str, Any]]:
    """Serialize key-value fields into a provenance record.

    ParsedDocument has no dedicated kv_fields list, so key-value data is
    preserved in processing_history until Member 1 extends the schema.
    """
    records: list[dict[str, Any]] = []
    for kv in kv_fields:
        record: dict[str, Any] = {
            "page_number": page_num,
            "key": kv.key,
            "value": kv.value,
            "confidence": kv.confidence,
            "extraction_method": kv.extraction_method,
        }
        if kv.key_bbox is not None:
            record["key_bbox"] = {
                "left": kv.key_bbox.left,
                "top": kv.key_bbox.top,
                "right": kv.key_bbox.right,
                "bottom": kv.key_bbox.bottom,
            }
        if kv.value_bbox is not None:
            record["value_bbox"] = {
                "left": kv.value_bbox.left,
                "top": kv.value_bbox.top,
                "right": kv.value_bbox.right,
                "bottom": kv.value_bbox.bottom,
            }
        records.append(record)
    return records


# ──────────────────────────────────────────────────────────────────────────────
# Numeric utilities
# ──────────────────────────────────────────────────────────────────────────────

def _median(values: list[float]) -> float:
    """Return the median of a non-empty list of floats."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    mid = len(sorted_vals) // 2
    if len(sorted_vals) % 2 == 0:
        return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0
    return sorted_vals[mid]


# ──────────────────────────────────────────────────────────────────────────────
# Contracts 2 & 3: Concrete RAG Chunk Export & Agent Queryable Interface
# ──────────────────────────────────────────────────────────────────────────────

class AgentQueryableDocument:
    """Read-only view of a ParsedDocument for agent query access (Contract 3 implementation)."""

    def __init__(self, parsed_doc: ParsedDocument) -> None:
        self._doc = parsed_doc

    @property
    def document_id(self) -> str:
        return self._doc.document_id

    @property
    def title(self) -> str:
        return self._doc.title

    @property
    def category(self) -> str:
        return self._doc.category

    def get_full_text(self) -> str:
        return self._doc.get_full_text()

    def get_sections(self) -> list[dict[str, Any]]:
        return [
            {
                "section_id": s.section_id,
                "title": s.title,
                "level": s.level,
                "content": s.content,
                "page_number": s.page_number,
                "confidence": s.confidence,
            }
            for s in self._doc.sections
        ]

    def get_tables(self) -> list[dict[str, Any]]:
        return [
            {
                "table_id": t.table_id,
                "headers": list(t.headers),
                "rows": [list(r) for r in t.rows],
                "row_count": t.row_count,
                "col_count": t.col_count,
                "page_number": t.page_number,
                "confidence": t.confidence,
            }
            for t in self._doc.tables
        ]

    def get_key_value_fields(self) -> list[dict[str, Any]]:
        if self._doc.processing_history:
            for hist in self._doc.processing_history:
                if "key_value_fields" in hist:
                    return list(hist["key_value_fields"])
        return []

    def get_cross_references(self) -> list[dict[str, Any]]:
        return [
            {
                "ref_type": r.ref_type,
                "target": r.target,
                "page_number": r.page_number,
                "confidence": r.confidence,
            }
            for r in self._doc.cross_references
        ]

    def citation_for_section(self, section_id: str) -> dict[str, Any] | None:
        for s in self._doc.sections:
            if s.section_id == section_id:
                c = s.citation_coords
                return {
                    "page_number": c.page_number,
                    "section_title": c.section_title,
                    "char_offset_start": c.char_offset_start,
                    "char_offset_end": c.char_offset_end,
                    "citation_id": c.citation_id,
                }
        return None


def export_for_rag(parsed_doc: ParsedDocument, *, prefix_modality: bool = False) -> list[Chunk]:
    """Convert a ParsedDocument into retrievable Chunk objects (Contract 2 mapping)."""
    if Chunk is None:
        raise RuntimeError("rag_engine.schemas.chunk.Chunk is required for RAG chunk export")

    chunks: list[Chunk] = []
    doc_id = parsed_doc.document_id
    doc_title = parsed_doc.title or doc_id
    category = parsed_doc.category or "Unknown"

    chunk_idx = 0

    # 1. Section chunks
    for sec in parsed_doc.sections:
        content = sec.content.strip()
        if not content and not sec.title.strip():
            continue
        body_text = content if content else sec.title
        heading = [doc_title, sec.title]
        if prefix_modality:
            heading.insert(0, "Document")

        c = Chunk.create(
            document_id=doc_id,
            content=body_text,
            chunk_index=chunk_idx,
            document_name=doc_title,
            page_number=sec.page_number,
            section_title=sec.title,
            section_id=sec.section_id,
            heading_path=heading,
            category=category,
            char_start=sec.citation_coords.char_offset_start,
            char_end=sec.citation_coords.char_offset_end,
            chunk_strategy="section",
        )
        chunks.append(c)
        chunk_idx += 1

    # 2. Table chunks
    for tbl in parsed_doc.tables:
        table_text = (tbl.normalized_text or tbl.raw_text or "").strip()
        if not table_text and tbl.headers:
            table_text = " | ".join(tbl.headers)
        if not table_text:
            continue

        heading = [doc_title, f"Table {tbl.table_id}"]
        if prefix_modality:
            heading.insert(0, "Document")

        c = Chunk.create(
            document_id=doc_id,
            content=table_text,
            chunk_index=chunk_idx,
            document_name=doc_title,
            page_number=tbl.page_number,
            section_title=f"Table {tbl.table_id}",
            table_id=tbl.table_id,
            is_table_chunk=True,
            heading_path=heading,
            category=category,
            chunk_strategy="table",
        )
        chunks.append(c)
        chunk_idx += 1

    return chunks


def export_for_agent(parsed_doc: ParsedDocument) -> AgentQueryableDocument:
    """Create an agent-queryable interface for a ParsedDocument (Contract 3)."""
    return AgentQueryableDocument(parsed_doc)


export_document_for_rag = export_for_rag
export_document_for_agent = export_for_agent
