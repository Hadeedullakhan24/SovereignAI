"""Retrieval Engine Exceptions.

Domain-specific exceptions for Milestone 8 Hybrid Retrieval Engine.
"""

from __future__ import annotations


class RetrievalError(Exception):
    """Base exception for all retrieval engine operations."""


class QueryAnalysisError(RetrievalError):
    """Raised when query intent parsing or entity extraction fails."""


class QueryNormalizationError(RetrievalError):
    """Raised when query rewriting or normalization fails."""


class DenseRetrievalError(RetrievalError):
    """Raised when vector similarity search via VectorRepository fails."""


class SparseRetrievalError(RetrievalError):
    """Raised when BM25 inverted index retrieval or update fails."""


class ScoreNormalizationError(RetrievalError):
    """Raised when score calibration or normalization fails."""


class RRFError(RetrievalError):
    """Raised when Reciprocal Rank Fusion calculation fails."""


class RerankerError(RetrievalError):
    """Raised when cross-encoder neural reranking fails."""


class ContextExpansionError(RetrievalError):
    """Raised when sequential or hierarchical neighbor expansion fails."""


class DuplicateRemovalError(RetrievalError):
    """Raised when deduplication of chunks or citations fails."""


class CitationError(RetrievalError):
    """Raised when building citations or provenance bundles fails."""


class ContextPackingError(RetrievalError):
    """Raised when packing context or token budget enforcement fails."""


class QueryCacheError(RetrievalError):
    """Raised when SQLite query caching, invalidation, or lookup fails."""


class RetrievalValidationError(RetrievalError):
    """Raised when retrieval result fails post-retrieval validation."""


class StrategyNotFoundError(RetrievalError):
    """Raised when a requested retrieval strategy is not registered."""
