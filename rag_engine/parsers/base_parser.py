"""Base parser abstraction and template execution workflow."""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, Iterator, Optional

from rag_engine.parsers.exceptions import (
    ParserError,
    ParsingFailedError,
    UnsupportedDocumentError,
)
from rag_engine.parsers.parser_events import (
    DocumentParsed,
    ParserEventBus,
    ParsingFailed,
    ValidationCompleted,
)
from rag_engine.parsers.parser_health import ParserHealthReport, ParserHealthStatus
from rag_engine.parsers.parser_metrics import ParseMetrics, ParserMetricsCollector
from rag_engine.parsers.parser_validator import ParserValidator
from rag_engine.parsers.parsing_context import ParsingContext
from rag_engine.schemas.document import Document, DocumentLifecycleState
from rag_engine.schemas.parsed_document import (
    DocumentStatistics,
    ParsedDocument,
    ParsedMetadata,
    Section,
)


class BaseParser(ABC):
    """Abstract Base Class defining standard document parsing template workflow."""

    def __init__(
        self,
        metrics_collector: Optional[ParserMetricsCollector] = None,
        event_bus: Optional[ParserEventBus] = None,
        validator: Optional[ParserValidator] = None,
    ) -> None:
        self.metrics_collector = metrics_collector or ParserMetricsCollector()
        self.event_bus = event_bus or ParserEventBus()
        self.validator = validator or ParserValidator()

    def parse(
        self,
        document: Document,
        context: Optional[ParsingContext] = None,
    ) -> ParsedDocument:
        """Template method executing full document parsing lifecycle."""
        ctx = context or ParsingContext()
        start_time = time.perf_counter()

        if not self.can_parse(document):
            raise UnsupportedDocumentError(
                f"Parser '{self.__class__.__name__}' cannot parse document '{document.metadata.file_name}'",
                document_id=document.doc_id,
                source_path=document.metadata.source_path,
            )

        try:
            # Delegate format-specific parsing to concrete subclass
            parsed_doc = self._parse_document(document, ctx)

            # Calculate and populate statistics
            stats = self._calculate_statistics(parsed_doc)
            object.__setattr__(parsed_doc, "statistics", stats)

            # Run structural validation if requested
            if ctx.run_validation:
                report = self.validator.validate(parsed_doc)
                object.__setattr__(parsed_doc, "validation_report", report)
                self.event_bus.publish(
                    ValidationCompleted(
                        document_id=parsed_doc.document_id,
                        payload={
                            "is_valid": report.is_valid,
                            "errors": report.error_count,
                            "warnings": report.warning_count,
                        },
                    )
                )

            # Update lifecycle state
            object.__setattr__(parsed_doc, "lifecycle_state", DocumentLifecycleState.PARSED)

            # Record telemetry
            duration_ms = (time.perf_counter() - start_time) * 1000
            metrics = ParseMetrics(
                document_id=parsed_doc.document_id,
                parser_name=self.__class__.__name__,
                driver_used=self.driver_name(),
                duration_ms=round(duration_ms, 2),
                pages_parsed=stats.total_pages,
                sections_extracted=stats.total_sections,
                tables_parsed=stats.total_tables,
                entities_extracted=stats.total_entities,
                warnings_count=stats.total_warnings,
                errors_count=0 if not parsed_doc.validation_report else parsed_doc.validation_report.error_count,
                success=True,
            )
            self.metrics_collector.record(metrics)

            # Publish event
            self.event_bus.publish(
                DocumentParsed(
                    document_id=parsed_doc.document_id,
                    payload={
                        "title": parsed_doc.title,
                        "category": parsed_doc.category,
                        "duration_ms": duration_ms,
                        "sections": stats.total_sections,
                        "tables": stats.total_tables,
                        "entities": stats.total_entities,
                    },
                )
            )

            return parsed_doc

        except Exception as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self.metrics_collector.record(
                ParseMetrics(
                    document_id=document.doc_id,
                    parser_name=self.__class__.__name__,
                    driver_used=self.driver_name(),
                    duration_ms=round(duration_ms, 2),
                    success=False,
                    error_message=str(exc),
                )
            )
            self.event_bus.publish(
                ParsingFailed(
                    document_id=document.doc_id,
                    payload={"error": str(exc), "parser": self.__class__.__name__},
                )
            )
            if isinstance(exc, ParserError):
                raise
            raise ParsingFailedError(
                f"Parsing failed for '{document.metadata.file_name}': {exc}",
                document_id=document.doc_id,
                source_path=document.metadata.source_path,
                cause=exc,
            ) from exc

    def parse_stream(
        self,
        document: Document,
        context: Optional[ParsingContext] = None,
    ) -> Iterator[Section]:
        """Lazy stream/generator yielding sections sequentially for large manuals."""
        parsed_doc = self.parse(document, context)
        for section in parsed_doc.sections:
            yield section

    def _calculate_statistics(self, doc: ParsedDocument) -> DocumentStatistics:
        """Aggregate statistical metrics for the parsed document."""
        full_text = doc.get_full_text()
        words = len(full_text.split())
        chars = len(full_text)
        paragraphs_count = sum(len(s.paragraphs) for s in doc.sections)

        pages = set()
        for s in doc.sections:
            if s.page_number is not None:
                pages.add(s.page_number)
        for t in doc.tables:
            if t.page_number is not None:
                pages.add(t.page_number)
        total_pages = max(len(pages), 1)

        return DocumentStatistics(
            total_pages=total_pages,
            total_sections=len(doc.sections),
            total_paragraphs=paragraphs_count,
            total_words=words,
            total_characters=chars,
            total_tables=len(doc.tables),
            total_entities=len(doc.equipment),
            total_warnings=len(doc.warnings),
            total_relations=len(doc.entity_graph.edges),
        )

    @abstractmethod
    def _parse_document(self, document: Document, context: ParsingContext) -> ParsedDocument:
        """Internal subclass implementation of format-specific parsing."""

    @abstractmethod
    def can_parse(self, document: Document) -> bool:
        """Return True if this parser can process the document."""

    @abstractmethod
    def supported_formats(self) -> list[str]:
        """Return list of supported file extensions (e.g. ['.pdf', '.docx'])."""

    @abstractmethod
    def supported_categories(self) -> list[str]:
        """Return list of supported document categories."""

    def driver_name(self) -> str:
        """Return name of driver/engine used."""
        return "native"

    def version(self) -> str:
        """Return parser implementation version."""
        return "1.0.0"

    def health(self) -> ParserHealthReport:
        """Generate operational health status report for this parser."""
        return ParserHealthReport(
            parser_name=self.__class__.__name__,
            status=ParserHealthStatus.HEALTHY,
            supported_formats=self.supported_formats(),
            supported_categories=self.supported_categories(),
            available_drivers=[self.driver_name()],
            missing_dependencies=[],
            details={"version": self.version()},
        )
