"""Unit tests for document parsing framework, classifier, profiles, and observability."""

import pytest
from pathlib import Path

from rag_engine.parsers.document_classifier import (
    DocumentCategory,
    DocumentClassifier,
)
from rag_engine.parsers.exceptions import UnsupportedDocumentError
from rag_engine.parsers.generic_text_parser import GenericTextParser
from rag_engine.parsers.parser_events import DocumentParsed, ParserEventBus
from rag_engine.parsers.parser_factory import ParserFactory
from rag_engine.parsers.parser_metrics import ParseMetrics, ParserMetricsCollector
from rag_engine.parsers.parser_registry import ParserRegistry
from rag_engine.parsers.parsing_context import ParsingContext
from rag_engine.parsers.profiles import (
    GenericRefineryProfile,
    MRPLProfile,
    RefineryProfile,
    get_profile,
    register_profile,
)
from rag_engine.schemas.document import Document, DocumentMetadata
from rag_engine.schemas.parsed_document import EquipmentType


def test_document_classifier_deterministic() -> None:
    classifier = DocumentClassifier()

    # Manual
    doc_manual = Document(
        doc_id="doc_1",
        content="Operation manual for centrifugal pumps in CDU-1 refinery complex.",
        metadata=DocumentMetadata(
            source_path="/data/manuals/cdu_pump_manual.txt",
            file_name="cdu_pump_manual.txt",
            file_format=".txt",
            category="manuals",
        ),
    )
    res_manual = classifier.classify(doc_manual)
    assert res_manual.category == DocumentCategory.MANUAL
    assert res_manual.confidence >= 0.50

    # Inspection Report
    doc_inspect = Document(
        doc_id="doc_2",
        content="Visual inspection and ultrasonic testing report for pipe thickness.",
        metadata=DocumentMetadata(
            source_path="/data/inspections/ndt_report_v101.txt",
            file_name="ndt_report_v101.txt",
            file_format=".txt",
            category="inspection_reports",
        ),
    )
    res_inspect = classifier.classify(doc_inspect)
    assert res_inspect.category == DocumentCategory.INSPECTION_REPORT

    # Safety Document
    doc_safety = Document(
        doc_id="doc_3",
        content="Job Safety Analysis and OISD-105 compliance checklist. Wear PPE.",
        metadata=DocumentMetadata(
            source_path="/data/safety/sop_fire_safety.txt",
            file_name="sop_fire_safety.txt",
            file_format=".txt",
            category="safety_docs",
        ),
    )
    res_safety = classifier.classify(doc_safety)
    assert res_safety.category == DocumentCategory.SAFETY_DOCUMENT

    # Email
    doc_email = Document(
        doc_id="doc_4",
        content="From: plant.manager@mrpl.co.in\nTo: shift.engineer@mrpl.co.in\nSubject: Shutdown Schedule",
        metadata=DocumentMetadata(
            source_path="/data/emails/memo_01.eml",
            file_name="memo_01.eml",
            file_format=".eml",
        ),
    )
    res_email = classifier.classify(doc_email)
    assert res_email.category == DocumentCategory.EMAIL

    # Engineering Drawing
    doc_dwg = Document(
        doc_id="doc_5",
        content="Title Block: Drawing No: DWG-MRPL-P&ID-201. Scale 1:100.",
        metadata=DocumentMetadata(
            source_path="/data/drawings/dwg_pid_201.dwg",
            file_name="dwg_pid_201.dwg",
            file_format=".dwg",
            category="engineering_drawings",
        ),
    )
    res_dwg = classifier.classify(doc_dwg)
    assert res_dwg.category == DocumentCategory.ENGINEERING_DRAWING


def test_refinery_profiles() -> None:
    generic = get_profile("generic")
    assert isinstance(generic, GenericRefineryProfile)
    assert generic.resolve_equipment_type("P-101A") == EquipmentType.PUMP
    assert generic.resolve_equipment_type("V-201") == EquipmentType.VALVE
    assert generic.resolve_equipment_type("HX-301") == EquipmentType.HEAT_EXCHANGER

    mrpl = get_profile("mrpl")
    assert isinstance(mrpl, MRPLProfile)
    assert "CDU-1" in mrpl.plant_units
    assert "OISD-105" in mrpl.standard_names
    assert mrpl.resolve_equipment_type("PSV-201") == EquipmentType.VALVE
    assert mrpl.resolve_equipment_type("K-601") == EquipmentType.COMPRESSOR


def test_parser_registry_and_factory() -> None:
    registry = ParserRegistry()
    registry.register(GenericTextParser, extensions=[".txt"])

    assert registry.get_by_extension(".txt") == GenericTextParser
    assert registry.get_by_extension("txt") == GenericTextParser
    assert len(registry.list_parsers()) == 1

    factory = ParserFactory(registry=registry)
    doc = Document(
        doc_id="doc_factory_test",
        content="Simple pump P-101 text.",
        metadata=DocumentMetadata(
            source_path="/test/file.txt",
            file_name="file.txt",
            file_format=".txt",
        ),
    )
    parser, classification = factory.get_parser(doc)
    assert isinstance(parser, GenericTextParser)
    assert classification.category in [DocumentCategory.UNKNOWN, DocumentCategory.MANUAL]


def test_parser_metrics_collector() -> None:
    collector = ParserMetricsCollector()
    metrics = ParseMetrics(
        document_id="doc_test",
        parser_name="GenericTextParser",
        duration_ms=45.2,
        pages_parsed=2,
        sections_extracted=4,
        tables_parsed=1,
        entities_extracted=3,
        warnings_count=1,
        success=True,
    )
    collector.record(metrics)

    summary = collector.get_summary()
    assert summary["total_documents_processed"] == 1
    assert summary["total_successful"] == 1
    assert summary["total_entities_extracted"] == 3
    assert summary["total_tables_parsed"] == 1
    assert summary["average_duration_ms"] == 45.2

    collector.clear()
    assert collector.get_summary()["total_documents_processed"] == 0


def test_parser_event_bus() -> None:
    bus = ParserEventBus()
    received_events = []

    def on_doc_parsed(event):
        received_events.append(event)

    bus.subscribe("DocumentParsed", on_doc_parsed)

    event = DocumentParsed(
        document_id="doc_ev_1",
        payload={"title": "Test Manual"},
    )
    bus.publish(event)

    assert len(received_events) == 1
    assert received_events[0].document_id == "doc_ev_1"
    assert received_events[0].payload["title"] == "Test Manual"

    bus.clear()
