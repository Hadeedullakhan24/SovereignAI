"""Refinery deep document parsing engine for structured knowledge extraction."""

from rag_engine.parsers.base_parser import BaseParser
from rag_engine.parsers.csv_parser import CSVParser
from rag_engine.parsers.docx_parser import DOCXParser
from rag_engine.parsers.document_classifier import (
    ClassificationResult,
    DocumentCategory,
    DocumentClassifier,
)
from rag_engine.parsers.email_parser import EmailParser
from rag_engine.parsers.engineering_parser import EngineeringParser
from rag_engine.parsers.exceptions import (
    CorruptedContentError,
    ParserError,
    ParserInitializationError,
    ParserTimeoutError,
    ParsingFailedError,
    ProfileError,
    UnsupportedDocumentError,
    ValidationError,
)
from rag_engine.parsers.generic_text_parser import GenericTextParser
from rag_engine.parsers.image_metadata_parser import ImageMetadataParser
from rag_engine.parsers.inspection_parser import InspectionParser
from rag_engine.parsers.markdown_parser import MarkdownParser
from rag_engine.parsers.parser_events import (
    DocumentParsed,
    EntityParsed,
    MetadataParsed,
    ParserEvent,
    ParserEventBus,
    ParserFallbackActivated,
    ParsingFailed,
    SectionParsed,
    TableParsed,
    ValidationCompleted,
)
from rag_engine.parsers.parser_factory import ParserFactory
from rag_engine.parsers.parser_health import ParserHealthReport, ParserHealthStatus
from rag_engine.parsers.parser_metrics import ParseMetrics, ParserMetricsCollector
from rag_engine.parsers.parser_registry import (
    ParserRegistry,
    get_global_registry,
    register_parser,
)
from rag_engine.parsers.parser_utils import ParserUtils
from rag_engine.parsers.parser_validator import ParserValidator
from rag_engine.parsers.parsing_context import ParsingContext
from rag_engine.parsers.pdf_parser import PDFParser
from rag_engine.parsers.plugin_parser import PluginParserManager
from rag_engine.parsers.pptx_parser import PPTXParser
from rag_engine.parsers.profiles import (
    GenericRefineryProfile,
    MRPLProfile,
    RefineryProfile,
    get_profile,
    register_profile,
)
from rag_engine.parsers.safety_parser import SafetyParser

__all__ = [
    "BaseParser",
    "GenericTextParser",
    "PDFParser",
    "DOCXParser",
    "PPTXParser",
    "CSVParser",
    "MarkdownParser",
    "ImageMetadataParser",
    "EmailParser",
    "EngineeringParser",
    "InspectionParser",
    "SafetyParser",
    "DocumentClassifier",
    "DocumentCategory",
    "ClassificationResult",
    "ParserRegistry",
    "register_parser",
    "get_global_registry",
    "ParserFactory",
    "ParsingContext",
    "ParserValidator",
    "ParserUtils",
    "ParseMetrics",
    "ParserMetricsCollector",
    "ParserEvent",
    "ParserEventBus",
    "DocumentParsed",
    "SectionParsed",
    "EntityParsed",
    "TableParsed",
    "MetadataParsed",
    "ValidationCompleted",
    "ParsingFailed",
    "ParserFallbackActivated",
    "ParserHealthReport",
    "ParserHealthStatus",
    "PluginParserManager",
    "RefineryProfile",
    "GenericRefineryProfile",
    "MRPLProfile",
    "get_profile",
    "register_profile",
    "ParserError",
    "ParserInitializationError",
    "UnsupportedDocumentError",
    "ParsingFailedError",
    "CorruptedContentError",
    "ParserTimeoutError",
    "ValidationError",
    "ProfileError",
]
