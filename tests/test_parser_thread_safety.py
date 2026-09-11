"""Stress tests and thread safety validation for document parsing engine."""

import concurrent.futures
import pytest

from rag_engine.parsers.generic_text_parser import GenericTextParser
from rag_engine.parsers.parser_events import ParserEventBus
from rag_engine.parsers.parser_metrics import ParserMetricsCollector
from rag_engine.parsers.parser_validator import ParserValidator
from rag_engine.schemas.document import Document, DocumentMetadata


def test_concurrent_parsing_thread_safety() -> None:
    collector = ParserMetricsCollector()
    bus = ParserEventBus()
    validator = ParserValidator()

    received_events = []
    bus.subscribe("DocumentParsed", lambda e: received_events.append(e))

    num_threads = 20
    num_docs = 40

    def parse_worker(idx: int):
        parser = GenericTextParser(
            metrics_collector=collector,
            event_bus=bus,
            validator=validator,
        )
        content = (
            f"1.{idx} Operations Section {idx}\n"
            f"Centrifugal pump P-{100 + idx} connects to valve V-{200 + idx} at {10 + (idx % 5)} bar and {120 + idx} °C.\n"
            f"WARNING: Flammable line requiring strict OISD-105 compliance.\n"
            f"| Metric | Value |\n"
            f"| --- | --- |\n"
            f"| Loop | LIC-{300 + idx} |\n"
        )
        doc = Document(
            doc_id=f"doc_concurrent_{idx}",
            content=content,
            metadata=DocumentMetadata(
                source_path=f"/test/concurrent_{idx}.txt",
                file_name=f"concurrent_{idx}.txt",
                file_format=".txt",
            ),
        )
        return parser.parse(doc)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(parse_worker, i) for i in range(num_docs)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    # All documents must successfully parse
    assert len(results) == num_docs
    for res in results:
        assert len(res.sections) >= 1
        assert len(res.equipment) >= 2
        assert len(res.tables) == 1
        assert len(res.warnings) >= 1

    # Metrics summary must accurately match 40 processed docs
    summary = collector.get_summary()
    assert summary["total_documents_processed"] == num_docs
    assert summary["total_successful"] == num_docs
    assert summary["total_failed"] == 0
    assert summary["total_entities_extracted"] >= num_docs * 2
    assert summary["total_tables_parsed"] == num_docs

    # All events should have been captured
    assert len(received_events) == num_docs
