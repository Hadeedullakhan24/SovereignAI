"""
Loaders Package — Universal Document Loading Framework.

Sovereign On-Premise Agentic AI Workbench (SIH26117 / MRPL)
Member 1: Knowledge Base / RAG Engine

Converts every supported refinery document into a unified, extensible Document schema.
"""

from rag_engine.loaders.base_loader import BaseLoader
from rag_engine.loaders.csv_loader import CSVLoader
from rag_engine.loaders.docx_loader import DOCXLoader
from rag_engine.loaders.exceptions import (
    CorruptedDocumentError,
    LoaderError,
    LoaderInitializationError,
    UnsupportedFormatError,
    ValidationFailedError,
)
from rag_engine.loaders.image_loader import ImageLoader
from rag_engine.loaders.loader_events import (
    CorruptedDocument,
    DocumentFailed,
    DocumentLoaded,
    EventBus,
    FallbackActivated,
    LoaderEvent,
    UnsupportedDocument,
    ValidationFailed,
    global_event_bus,
)
from rag_engine.loaders.loader_factory import LoaderFactory, global_loader_factory
from rag_engine.loaders.loader_health import LoaderHealthReport, LoaderHealthStatus
from rag_engine.loaders.loader_metrics import LoadMetrics, MetricsCollector, global_metrics
from rag_engine.loaders.loader_registry import (
    LoaderRegistry,
    global_loader_registry,
    register_loader,
)
from rag_engine.loaders.markdown_loader import MarkdownLoader
from rag_engine.loaders.pdf_loader import PDFLoader
from rag_engine.loaders.plugin_loader import PluginLoaderManager
from rag_engine.loaders.pptx_loader import PPTXLoader
from rag_engine.loaders.processing_context import ProcessingContext
from rag_engine.loaders.txt_loader import TXTLoader
from rag_engine.loaders.xlsx_loader import XLSXLoader

__all__ = [
    # Contracts
    "BaseLoader",
    "ProcessingContext",
    # Exceptions
    "LoaderError",
    "UnsupportedFormatError",
    "CorruptedDocumentError",
    "LoaderInitializationError",
    "ValidationFailedError",
    # Registry & Factory
    "LoaderRegistry",
    "global_loader_registry",
    "register_loader",
    "LoaderFactory",
    "global_loader_factory",
    "PluginLoaderManager",
    # Concrete Loaders
    "PDFLoader",
    "DOCXLoader",
    "PPTXLoader",
    "XLSXLoader",
    "CSVLoader",
    "TXTLoader",
    "MarkdownLoader",
    "ImageLoader",
    # Telemetry & Events
    "LoadMetrics",
    "MetricsCollector",
    "global_metrics",
    "LoaderEvent",
    "DocumentLoaded",
    "DocumentFailed",
    "FallbackActivated",
    "ValidationFailed",
    "UnsupportedDocument",
    "CorruptedDocument",
    "EventBus",
    "global_event_bus",
    "LoaderHealthStatus",
    "LoaderHealthReport",
]
