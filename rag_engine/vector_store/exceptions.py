"""Vector Store Exception Hierarchy.

Authoritative exception classes for all vector storage, indexing, routing,
validation, transaction, and snapshot operations within Milestone 7.
"""

from __future__ import annotations


class VectorStoreError(Exception):
    """Base exception for all vector database operations."""
    pass


class CollectionNotFoundError(VectorStoreError):
    """Raised when a requested collection or alias does not exist."""
    pass


class CollectionAlreadyExistsError(VectorStoreError):
    """Raised when attempting to create a collection that already exists."""
    pass


class PayloadValidationError(VectorStoreError):
    """Raised when payload fields violate schema constraints, types, lengths, or enums."""
    pass


class VectorDimensionMismatchError(VectorStoreError):
    """Raised when an embedding vector dimensionality does not match collection config."""
    pass


class VectorQualityError(VectorStoreError):
    """Raised when an embedding vector contains NaN, Inf, or fails normalization."""
    pass


class RoutingError(VectorStoreError):
    """Raised when a chunk cannot be deterministically mapped to a collection."""
    pass


class VersioningError(VectorStoreError):
    """Raised on invalid collection version creation, rollback, or migration."""
    pass


class TransactionError(VectorStoreError):
    """Raised on transaction journal failure, staging conflict, or rollback failure."""
    pass


class SnapshotError(VectorStoreError):
    """Raised on snapshot archive creation, SHA-256 mismatch, or restoration failure."""
    pass


class OptimizationError(VectorStoreError):
    """Raised when segment merge, vacuum, or HNSW rebalancing fails."""
    pass


class BackendNotSupportedError(VectorStoreError):
    """Raised when a requested backend driver is not registered in VectorRegistry."""
    pass
