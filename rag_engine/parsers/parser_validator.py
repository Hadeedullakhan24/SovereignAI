"""Structural validation and consistency checking for parsed documents."""

from __future__ import annotations

import re
from typing import Optional

from rag_engine.schemas.parsed_document import (
    ParsedDocument,
    Section,
    Table,
    ValidationIssue,
    ValidationReport,
)


class ParserValidator:
    """Deterministic structural validator for ParsedDocument entities."""

    def validate(self, doc: ParsedDocument) -> ValidationReport:
        """Execute comprehensive structural validation rules."""
        issues: list[ValidationIssue] = []

        # 1. Document Title Validation
        if not doc.title or not doc.title.strip():
            issues.append(
                ValidationIssue(
                    severity="WARNING",
                    code="VAL_MISSING_TITLE",
                    message="Document has no detected or declared title.",
                    location="metadata.title",
                )
            )

        # 2. Section Hierarchy and Heading Validation
        self._validate_section_hierarchy(doc.sections, issues)

        # 3. Table Structural Integrity Validation
        self._validate_tables(doc.tables, issues)

        # 4. Equipment Entity Consistency Validation
        self._validate_entities(doc, issues)

        # 5. Cross-Reference Integrity Validation
        self._validate_references(doc, issues)

        # 6. Page Number Integrity Validation
        if doc.statistics.total_pages > 1:
            missing_page_sections = [
                s.title for s in doc.sections if s.page_number is None
            ]
            if missing_page_sections:
                issues.append(
                    ValidationIssue(
                        severity="INFO",
                        code="VAL_MISSING_PAGE_NUMBERS",
                        message=f"{len(missing_page_sections)} section(s) missing explicit page numbers in multi-page document.",
                        location="sections",
                    )
                )

        error_count = sum(1 for i in issues if i.severity == "ERROR")
        warning_count = sum(1 for i in issues if i.severity == "WARNING")
        is_valid = error_count == 0

        return ValidationReport(
            is_valid=is_valid,
            issues=issues,
            error_count=error_count,
            warning_count=warning_count,
        )

    def _validate_section_hierarchy(
        self, sections: list[Section], issues: list[ValidationIssue]
    ) -> None:
        """Validate heading levels, nesting sequence, and duplicate titles."""
        prev_level = 0
        seen_titles_by_level: dict[int, set[str]] = {}

        for sec in sections:
            # Check duplicate headings at same level
            level_titles = seen_titles_by_level.setdefault(sec.level, set())
            title_clean = sec.title.strip().lower()
            if title_clean in level_titles:
                issues.append(
                    ValidationIssue(
                        severity="WARNING",
                        code="VAL_DUPLICATE_HEADING",
                        message=f"Duplicate heading '{sec.title}' detected at level {sec.level}.",
                        location=f"section:{sec.section_id}",
                    )
                )
            else:
                level_titles.add(title_clean)

            # Check broken hierarchy jump (e.g. Level 1 followed by Level 3 without Level 2)
            if prev_level > 0 and (sec.level - prev_level) > 1:
                issues.append(
                    ValidationIssue(
                        severity="WARNING",
                        code="VAL_BROKEN_HIERARCHY",
                        message=f"Heading level jumped from H{prev_level} directly to H{sec.level} without intermediate heading.",
                        location=f"section:{sec.section_id}",
                    )
                )
            prev_level = sec.level

            # Recursively check nested subsections if any
            if sec.subsections:
                self._validate_section_hierarchy(sec.subsections, issues)

    def _validate_tables(self, tables: list[Table], issues: list[ValidationIssue]) -> None:
        """Validate table columns, row lengths, and header alignment."""
        for tbl in tables:
            if tbl.row_count == 0 and not tbl.headers:
                issues.append(
                    ValidationIssue(
                        severity="WARNING",
                        code="VAL_EMPTY_TABLE",
                        message=f"Table '{tbl.table_id}' has zero rows and zero headers.",
                        location=f"table:{tbl.table_id}",
                    )
                )
                continue

            expected_cols = tbl.col_count
            if tbl.headers and len(tbl.headers) != expected_cols:
                issues.append(
                    ValidationIssue(
                        severity="WARNING",
                        code="VAL_TABLE_HEADER_MISMATCH",
                        message=f"Table '{tbl.table_id}' header count ({len(tbl.headers)}) does not match declared column count ({expected_cols}).",
                        location=f"table:{tbl.table_id}",
                    )
                )

            # Validate each row length
            for idx, row in enumerate(tbl.rows):
                if len(row) != expected_cols and expected_cols > 0:
                    issues.append(
                        ValidationIssue(
                            severity="WARNING",
                            code="VAL_MALFORMED_TABLE_ROW",
                            message=f"Table '{tbl.table_id}' row {idx} has {len(row)} cells, expected {expected_cols}.",
                            location=f"table:{tbl.table_id}:row_{idx}",
                        )
                    )

    def _validate_entities(self, doc: ParsedDocument, issues: list[ValidationIssue]) -> None:
        """Detect duplicate equipment tags with conflicting types."""
        tag_types: dict[str, str] = {}
        for eq in doc.equipment:
            tag_upper = eq.tag.upper().strip()
            if tag_upper in tag_types:
                prev_type = tag_types[tag_upper]
                if prev_type != eq.equipment_type:
                    issues.append(
                        ValidationIssue(
                            severity="WARNING",
                            code="VAL_CONFLICTING_ENTITY_TYPE",
                            message=f"Equipment tag '{eq.tag}' assigned conflicting types: '{prev_type}' vs '{eq.equipment_type}'.",
                            location=f"equipment:{eq.entity_id}",
                        )
                    )
            else:
                tag_types[tag_upper] = eq.equipment_type

    def _validate_references(self, doc: ParsedDocument, issues: list[ValidationIssue]) -> None:
        """Check for broken cross-references to non-existent tables or figures."""
        table_captions = {t.caption.lower() for t in doc.tables if t.caption}
        table_ids = {t.table_id.lower() for t in doc.tables}

        for ref in doc.cross_references:
            if ref.ref_type.upper() == "TABLE":
                target_clean = ref.target.lower().strip()
                # Check if referenced table exists by id or caption
                matched = any(
                    target_clean in caption or target_clean in tbl_id
                    for caption in table_captions
                    for tbl_id in table_ids
                )
                if not matched and not doc.tables:
                    issues.append(
                        ValidationIssue(
                            severity="INFO",
                            code="VAL_BROKEN_TABLE_REFERENCE",
                            message=f"Document references '{ref.target}' but no matching table was parsed.",
                            location="cross_references",
                        )
                    )
