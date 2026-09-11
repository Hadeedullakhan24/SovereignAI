"""Factory for resolving and instantiating appropriate document parsers."""

from __future__ import annotations

from typing import Optional, Type

from rag_engine.parsers.base_parser import BaseParser
from rag_engine.parsers.document_classifier import (
    ClassificationResult,
    DocumentCategory,
    DocumentClassifier,
)
from rag_engine.parsers.exceptions import UnsupportedDocumentError
from rag_engine.parsers.parser_events import ParserEventBus
from rag_engine.parsers.parser_metrics import ParserMetricsCollector
from rag_engine.parsers.parser_registry import ParserRegistry, get_global_registry
from rag_engine.parsers.parser_validator import ParserValidator
from rag_engine.schemas.document import Document


class ParserFactory:
    """Factory resolving parser based on deterministic classification and format."""

    def __init__(
        self,
        registry: Optional[ParserRegistry] = None,
        classifier: Optional[DocumentClassifier] = None,
        metrics_collector: Optional[ParserMetricsCollector] = None,
        event_bus: Optional[ParserEventBus] = None,
        validator: Optional[ParserValidator] = None,
    ) -> None:
        self.registry = registry or get_global_registry()
        self.classifier = classifier or DocumentClassifier()
        self.metrics_collector = metrics_collector or ParserMetricsCollector()
        self.event_bus = event_bus or ParserEventBus()
        self.validator = validator or ParserValidator()

    def get_parser(
        self,
        document: Document,
    ) -> tuple[BaseParser, ClassificationResult]:
        """Resolve, instantiate, and return the optimal parser after deterministic classification."""
        # 1. First invoke DocumentClassifier
        classification = self.classifier.classify(document)

        parser_cls: Optional[Type[BaseParser]] = None
        ext = (document.metadata.file_format or "").lower().strip()
        format_parser_cls = self.registry.get_by_extension(ext)

        # 2. Check for category-specialized parser if high-confidence domain category detected
        if classification.confidence >= 0.60:
            cat_str = classification.category.value.lower()
            cat_parser_cls = self.registry.get_by_category(cat_str)
            if cat_parser_cls:
                try:
                    if cat_parser_cls().can_parse(document):
                        parser_cls = cat_parser_cls
                except Exception:
                    pass

        # 3. If no valid category parser or it cannot parse this document, use format parser
        if not parser_cls and format_parser_cls:
            try:
                if format_parser_cls().can_parse(document):
                    parser_cls = format_parser_cls
            except Exception:
                pass

        # 4. Universal fallback: GenericTextParser
        if not parser_cls:
            from rag_engine.parsers.generic_text_parser import GenericTextParser
            parser_cls = GenericTextParser

        # Instantiate parser with shared observability components
        parser_instance = parser_cls(
            metrics_collector=self.metrics_collector,
            event_bus=self.event_bus,
            validator=self.validator,
        )

        return parser_instance, classification

