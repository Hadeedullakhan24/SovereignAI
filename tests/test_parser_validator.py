"""Unit tests for ParserValidator structural checking and error reporting."""

import pytest

from rag_engine.parsers.parser_validator import ParserValidator
from rag_engine.schemas.document import DocumentLifecycleState
from rag_engine.schemas.parsed_document import (
    CrossReference,
    DocumentStatistics,
    EntityGraph,
    EquipmentEntity,
    EquipmentType,
    ParsedDocument,
    ParsedMetadata,
    Section,
    Table,
)


def test_validator_clean_document() -> None:
    validator = ParserValidator()
    doc = ParsedDocument(
        document_id="doc_valid_01",
        raw_document_id="raw_01",
        title="Valid Refinery Manual",
        category="Manual",
        sections=[
            Section(section_id="sec_1", title="Introduction", level=1, content="Text"),
            Section(section_id="sec_2", title="Overview", level=2, content="Subtext"),
        ],
        tables=[
            Table(
                table_id="tbl_1",
                caption="Parameters",
                headers=["A", "B"],
                rows=[["1", "2"]],
                row_count=1,
                col_count=2,
            )
        ],
        equipment=[
            EquipmentEntity(
                entity_id="eq_1",
                tag="P-101",
                name="Pump P-101",
                equipment_type=EquipmentType.PUMP,
            )
        ],
        metadata=ParsedMetadata(title="Valid Refinery Manual"),
        statistics=DocumentStatistics(),
    )
    report = validator.validate(doc)
    assert report.is_valid is True
    assert report.error_count == 0
    assert report.warning_count == 0


def test_validator_broken_hierarchy() -> None:
    validator = ParserValidator()
    doc = ParsedDocument(
        document_id="doc_hier_01",
        raw_document_id="raw_01",
        title="Hierarchy Test",
        category="Manual",
        sections=[
            Section(section_id="sec_1", title="Level 1", level=1, content="Text"),
            # Direct jump from level 1 to level 3
            Section(section_id="sec_2", title="Level 3", level=3, content="Subtext"),
        ],
        metadata=ParsedMetadata(title="Hierarchy Test"),
        statistics=DocumentStatistics(),
    )
    report = validator.validate(doc)
    codes = [issue.code for issue in report.issues]
    assert "VAL_BROKEN_HIERARCHY" in codes


def test_validator_duplicate_heading() -> None:
    validator = ParserValidator()
    doc = ParsedDocument(
        document_id="doc_dup_01",
        raw_document_id="raw_01",
        title="Duplicate Heading Test",
        category="Manual",
        sections=[
            Section(section_id="sec_1", title="System Operations", level=1, content="Text 1"),
            Section(section_id="sec_2", title="System Operations", level=1, content="Text 2"),
        ],
        metadata=ParsedMetadata(title="Duplicate Heading Test"),
        statistics=DocumentStatistics(),
    )
    report = validator.validate(doc)
    codes = [issue.code for issue in report.issues]
    assert "VAL_DUPLICATE_HEADING" in codes


def test_validator_malformed_table() -> None:
    validator = ParserValidator()
    doc = ParsedDocument(
        document_id="doc_tbl_01",
        raw_document_id="raw_01",
        title="Malformed Table Test",
        category="Manual",
        tables=[
            Table(
                table_id="tbl_bad_1",
                caption="Corrupted",
                headers=["Col1", "Col2", "Col3"],
                # Row only has 2 cells while col_count is 3
                rows=[["val1", "val2"]],
                row_count=1,
                col_count=3,
            )
        ],
        metadata=ParsedMetadata(title="Malformed Table Test"),
        statistics=DocumentStatistics(),
    )
    report = validator.validate(doc)
    codes = [issue.code for issue in report.issues]
    assert "VAL_MALFORMED_TABLE_ROW" in codes


def test_validator_conflicting_entity_type() -> None:
    validator = ParserValidator()
    doc = ParsedDocument(
        document_id="doc_ent_01",
        raw_document_id="raw_01",
        title="Conflicting Entity Test",
        category="Manual",
        equipment=[
            EquipmentEntity(
                entity_id="eq_1",
                tag="P-203",
                name="Pump P-203",
                equipment_type=EquipmentType.PUMP,
            ),
            # Same tag P-203 declared as VALVE
            EquipmentEntity(
                entity_id="eq_2",
                tag="P-203",
                name="Valve P-203",
                equipment_type=EquipmentType.VALVE,
            ),
        ],
        metadata=ParsedMetadata(title="Conflicting Entity Test"),
        statistics=DocumentStatistics(),
    )
    report = validator.validate(doc)
    codes = [issue.code for issue in report.issues]
    assert "VAL_CONFLICTING_ENTITY_TYPE" in codes
