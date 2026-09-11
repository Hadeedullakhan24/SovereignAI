"""Milestone 8: Enterprise Hybrid Retrieval Engine.

Universal sovereign retrieval platform for SIH26117 / MRPL, integrating multi-channel
Dense vector search via Qdrant, Sparse BM25 indexing, Reciprocal Rank Fusion, local Cross-Encoder
neural reranking, relational context expansion, deterministic citation provenance, and
token-aware context packing.
"""

from rag_engine.retrieval.adaptive_retriever import AdaptiveRetriever
from rag_engine.retrieval.base_retriever import (
    BaseRetriever,
    CitationBundle,
    RetrievalResult,
    ScoredRetrievalChunk,
)
from rag_engine.retrieval.bm25_retriever import BM25Index, BM25Retriever
from rag_engine.retrieval.citation_builder import CitationBuilder
from rag_engine.retrieval.context_expander import ContextExpander
from rag_engine.retrieval.context_packer import (
    ContextPacker,
    ContextWindowBuilder,
    TokenBudgetManager,
)
from rag_engine.retrieval.dense_retriever import DenseRetriever
from rag_engine.retrieval.duplicate_remover import DuplicateRemover
from rag_engine.retrieval.hybrid_retriever import HybridRetriever
from rag_engine.retrieval.metadata_booster import MetadataBooster
from rag_engine.retrieval.metadata_filter import (
    MetadataFilterBuilder,
    MetadataFilterPlanner,
)
from rag_engine.retrieval.query_analyzer import (
    ExtractedEntities,
    IntentType,
    QueryAnalyzer,
    QueryIntent,
)
from rag_engine.retrieval.query_cache import QueryCache
from rag_engine.retrieval.query_classifier import QueryClassifier
from rag_engine.retrieval.query_expander import QueryExpander
from rag_engine.retrieval.query_normalizer import QueryNormalizer
from rag_engine.retrieval.query_rewriter import QueryRewriter
from rag_engine.retrieval.reranker import (
    BaseReranker,
    DeterministicTestReranker,
    LocalCrossEncoderReranker,
)
from rag_engine.retrieval.retrieval_events import (
    RetrievalEvent,
    RetrievalEventBus,
)
from rag_engine.retrieval.retrieval_exceptions import (
    ContextExpansionError,
    ContextPackingError,
    DenseRetrievalError,
    DuplicateRemovalError,
    QueryAnalysisError,
    QueryCacheError,
    QueryNormalizationError,
    RerankerError,
    RetrievalError,
    RetrievalValidationError,
    RRFError,
    ScoreNormalizationError,
    SparseRetrievalError,
    StrategyNotFoundError,
)
from rag_engine.retrieval.retrieval_factory import (
    RetrievalFactory,
    get_retriever,
)
from rag_engine.retrieval.retrieval_health import (
    RetrievalHealthChecker,
    RetrievalHealthReport,
)
from rag_engine.retrieval.retrieval_metrics import (
    RetrievalMetrics,
    RetrievalMetricsCollector,
)
from rag_engine.retrieval.retrieval_pipeline import RetrievalPipeline
from rag_engine.retrieval.retrieval_registry import (
    RetrievalRegistry,
    register_retriever,
)
from rag_engine.retrieval.retrieval_utils import (
    estimate_token_count,
    format_cache_key,
    normalize_whitespace,
    tokenize_refinery_text,
)
from rag_engine.retrieval.retrieval_validator import RetrievalValidator
from rag_engine.retrieval.rrf_fusion import ReciprocalRankFusion
from rag_engine.retrieval.score_normalizer import (
    NormalizationMethod,
    ScoreNormalizer,
)

__all__ = [
    "BaseRetriever",
    "DenseRetriever",
    "BM25Retriever",
    "BM25Index",
    "HybridRetriever",
    "AdaptiveRetriever",
    "RetrievalPipeline",
    "RetrievalResult",
    "ScoredRetrievalChunk",
    "CitationBundle",
    "QueryAnalyzer",
    "QueryClassifier",
    "QueryIntent",
    "IntentType",
    "ExtractedEntities",
    "QueryNormalizer",
    "QueryRewriter",
    "QueryExpander",
    "MetadataFilterBuilder",
    "MetadataFilterPlanner",
    "ScoreNormalizer",
    "NormalizationMethod",
    "ReciprocalRankFusion",
    "MetadataBooster",
    "BaseReranker",
    "LocalCrossEncoderReranker",
    "DeterministicTestReranker",
    "ContextExpander",
    "DuplicateRemover",
    "CitationBuilder",
    "ContextPacker",
    "ContextWindowBuilder",
    "TokenBudgetManager",
    "QueryCache",
    "RetrievalFactory",
    "get_retriever",
    "RetrievalRegistry",
    "register_retriever",
    "RetrievalMetrics",
    "RetrievalMetricsCollector",
    "RetrievalEventBus",
    "RetrievalEvent",
    "RetrievalHealthChecker",
    "RetrievalHealthReport",
    "RetrievalValidator",
    "RetrievalError",
    "QueryAnalysisError",
    "QueryNormalizationError",
    "DenseRetrievalError",
    "SparseRetrievalError",
    "ScoreNormalizationError",
    "RRFError",
    "RerankerError",
    "ContextExpansionError",
    "DuplicateRemovalError",
    "ContextPackingError",
    "QueryCacheError",
    "RetrievalValidationError",
    "StrategyNotFoundError",
    "estimate_token_count",
    "format_cache_key",
    "normalize_whitespace",
    "tokenize_refinery_text",
]
