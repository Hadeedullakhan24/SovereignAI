"""Preprocessing — Production-grade Cleaning & Normalization Engine.

Prepares ParsedDocument objects from Milestone 3B for the Chunking Engine (Milestone 5).
Features a 12-stage deterministic pipeline:
  Stage 1:  Unicode normalization (NFKC)
  Stage 2:  Encoding normalization
  Stage 3:  Whitespace normalization
  Stage 4:  Line ending normalization
  Stage 5:  Broken paragraph reconstruction
  Stage 6:  Hyphenated word reconstruction
  Stage 7:  Header/Footer detection & removal
  Stage 8:  Page number preservation & citation mapping
  Stage 9:  Table whitespace cleanup
  Stage 10: Bullet normalization
  Stage 11: List normalization
  Stage 12: Engineering token protection

Completely offline, deterministic, thread-safe (RLock), and highly observable.
"""

from rag_engine.preprocessing.base_cleaner import BaseCleaner
from rag_engine.preprocessing.cleaning_events import (
    CleaningEvent,
    CleaningEventBus,
    CleaningFailed,
    CleaningFinished,
    CleaningStarted,
    FooterRemoved,
    HeaderRemoved,
    NormalizationApplied,
    ProtectedTokenDetected,
    get_cleaning_event_bus,
)
from rag_engine.preprocessing.cleaning_health import (
    CleaningHealthReport,
    dependencies,
    health,
    supported_features,
    version,
)
from rag_engine.preprocessing.cleaning_metrics import (
    CleaningMetrics,
    CleaningMetricsCollector,
    get_cleaning_metrics_collector,
)
from rag_engine.preprocessing.cleaning_pipeline import CleaningPipeline
from rag_engine.preprocessing.engineering_token_protector import EngineeringTokenProtector
from rag_engine.preprocessing.exceptions import (
    CleaningError,
    CorruptedDocumentError,
    PipelineStageError,
    PluginError,
    TokenProtectionError,
)
from rag_engine.preprocessing.factory import CleanerFactory, get_cleaner_factory
from rag_engine.preprocessing.header_footer_detector import HeaderFooterDetector
from rag_engine.preprocessing.normalizers import (
    BulletNormalizer,
    EncodingNormalizer,
    ListNormalizer,
    UnicodeNormalizer,
)
from rag_engine.preprocessing.page_mapper import PageMapper
from rag_engine.preprocessing.plugin_cleaner import (
    PluginCleanerManager,
    get_plugin_cleaner_manager,
)
from rag_engine.preprocessing.processing_history import (
    ProcessingHistoryRecorder,
    ProcessingStageRecord,
)
from rag_engine.preprocessing.registry import (
    CleanerRegistry,
    get_cleaner_registry,
    register_cleaner,
)
from rag_engine.preprocessing.table_cleaner import TableCleaner
from rag_engine.preprocessing.text_cleaner import TextCleaner
from rag_engine.preprocessing.whitespace_cleaner import WhitespaceCleaner

__all__ = [
    "BaseCleaner",
    "CleaningPipeline",
    "CleanerRegistry",
    "get_cleaner_registry",
    "register_cleaner",
    "CleanerFactory",
    "get_cleaner_factory",
    "PluginCleanerManager",
    "get_plugin_cleaner_manager",
    "EngineeringTokenProtector",
    "HeaderFooterDetector",
    "TableCleaner",
    "PageMapper",
    "TextCleaner",
    "UnicodeNormalizer",
    "EncodingNormalizer",
    "BulletNormalizer",
    "ListNormalizer",
    "WhitespaceCleaner",
    "CleaningMetrics",
    "CleaningMetricsCollector",
    "get_cleaning_metrics_collector",
    "CleaningEvent",
    "CleaningStarted",
    "CleaningFinished",
    "CleaningFailed",
    "HeaderRemoved",
    "FooterRemoved",
    "ProtectedTokenDetected",
    "NormalizationApplied",
    "CleaningEventBus",
    "get_cleaning_event_bus",
    "CleaningHealthReport",
    "health",
    "version",
    "dependencies",
    "supported_features",
    "ProcessingStageRecord",
    "ProcessingHistoryRecorder",
    "CleaningError",
    "PipelineStageError",
    "TokenProtectionError",
    "PluginError",
    "CorruptedDocumentError",
]
