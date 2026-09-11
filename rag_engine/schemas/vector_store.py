"""Schema: VectorStore — Pydantic v2 models for enterprise vector storage and indexing.

Defines schemas for collection configurations, metadata filtering, indexing results,
versioning, health diagnostics, storage statistics, and optimization telemetry.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from rag_engine.schemas.chunk import ChunkMetadata


class DistanceMetric(str, Enum):
    """Distance metrics supported for vector similarity search."""

    COSINE = "Cosine"
    EUCLIDEAN = "Euclid"
    DOT = "Dot"


class PayloadSchemaType(str, Enum):
    """Data types for payload schema indexing in vector storage."""

    KEYWORD = "keyword"
    INTEGER = "integer"
    FLOAT = "float"
    BOOL = "bool"
    TEXT = "text"
    GEO = "geo"


class FilterOperator(str, Enum):
    """Operators supported in payload metadata filters."""

    EQUALS = "eq"
    NOT_EQUALS = "ne"
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"
    GREATER_THAN = "gt"
    GREATER_EQUAL = "gte"
    LESS_THAN = "lt"
    LESS_EQUAL = "lte"


class FieldFilter(BaseModel):
    """Filter condition for a single payload field."""

    model_config = ConfigDict(frozen=True)

    field: str = Field(description="Payload metadata field name to filter on")
    operator: FilterOperator = Field(default=FilterOperator.EQUALS, description="Comparison operator")
    value: Any = Field(description="Value to compare against")


class MetadataFilter(BaseModel):
    """Structured query filter combining multiple field conditions."""

    model_config = ConfigDict(frozen=True)

    must: list[FieldFilter] = Field(default_factory=list, description="All conditions must match (AND)")
    should: list[FieldFilter] = Field(default_factory=list, description="At least one condition should match (OR)")
    must_not: list[FieldFilter] = Field(default_factory=list, description="None of these conditions must match (NOT)")


class ScoredVectorChunk(BaseModel):
    """Result of a vector similarity search with score and lineage metadata."""

    model_config = ConfigDict(extra="allow")

    chunk_id: str = Field(description="Deterministic chunk identifier")
    score: float = Field(description="Similarity score (e.g. cosine similarity)")
    rank: int = Field(default=0, ge=0, description="0-indexed result rank")
    vector: Optional[list[float]] = Field(default=None, description="Vector values if requested")
    text_preview: Optional[str] = Field(default=None, description="Text preview or chunk content")
    document_id: Optional[str] = Field(default=None, description="Parent document ID")
    category: Optional[str] = Field(default=None, description="Refinery document category")
    plant_unit: Optional[str] = Field(default=None, description="Refinery plant unit")
    page_number: Optional[int] = Field(default=None, description="Source page number")
    metadata: ChunkMetadata = Field(default_factory=ChunkMetadata, description="Full inherited metadata")
    payload: dict[str, Any] = Field(default_factory=dict, description="Raw underlying payload dictionary")


class IndexingResult(BaseModel):
    """Summary of a vector indexing or batch ingestion operation."""

    model_config = ConfigDict(frozen=True)

    collection_name: str = Field(description="Collection modified")
    total_chunks: int = Field(default=0, description="Total chunks submitted")
    inserted: int = Field(default=0, description="New chunks indexed")
    updated: int = Field(default=0, description="Modified chunks updated in-place")
    deleted: int = Field(default=0, description="Orphan chunks purged")
    skipped: int = Field(default=0, description="Unchanged chunks skipped")
    errors: list[str] = Field(default_factory=list, description="Error messages encountered")
    duration_ms: float = Field(default=0.0, description="Ingestion elapsed time in milliseconds")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of completion",
    )


class DiffPlan(BaseModel):
    """Reconciliation plan produced by 3-way incremental diff."""

    model_config = ConfigDict(frozen=True)

    new_chunk_ids: list[str] = Field(default_factory=list, description="IDs of chunks not in store")
    modified_chunk_ids: list[str] = Field(default_factory=list, description="IDs of chunks with changed hashes")
    unchanged_chunk_ids: list[str] = Field(default_factory=list, description="IDs of chunks identical to store")
    deleted_chunk_ids: list[str] = Field(default_factory=list, description="IDs of previously stored chunks now absent")


class CollectionStats(BaseModel):
    """Statistics for an individual vector collection."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Collection name")
    vector_count: int = Field(default=0, description="Total vectors indexed")
    indexed_vectors_count: int = Field(default=0, description="Vectors indexed in HNSW graph")
    points_count: int = Field(default=0, description="Total points in collection")
    segments_count: int = Field(default=0, description="Number of storage segments")
    payload_indexes_count: int = Field(default=0, description="Active payload indexes count")
    disk_size_bytes: int = Field(default=0, description="Estimated disk consumption in bytes")
    status: str = Field(default="green", description="Health status (green, yellow, red)")


class CollectionVersionInfo(BaseModel):
    """Metadata describing a versioned collection instance."""

    model_config = ConfigDict(frozen=True)

    base_name: str = Field(description="Logical domain name (e.g. engineering_docs)")
    version: str = Field(description="Version tag (e.g. v1, v2)")
    collection_name: str = Field(description="Physical collection name in database (e.g. engineering_docs_v1)")
    is_active: bool = Field(default=False, description="Whether this version is currently the active target")
    vector_count: int = Field(default=0, description="Number of vectors in this version")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp",
    )


class VersionDiffReport(BaseModel):
    """Comparison report between two collection versions."""

    model_config = ConfigDict(frozen=True)

    source_version: str = Field(description="Source collection version name")
    target_version: str = Field(description="Target collection version name")
    source_vector_count: int = Field(description="Vectors in source")
    target_vector_count: int = Field(description="Vectors in target")
    difference_count: int = Field(description="Net difference in vectors")
    missing_in_target: list[str] = Field(default_factory=list, description="Chunk IDs in source but not target")
    extra_in_target: list[str] = Field(default_factory=list, description="Chunk IDs in target but not source")


class MigrationReport(BaseModel):
    """Report detailing batch migration between collection versions."""

    model_config = ConfigDict(frozen=True)

    source_collection: str = Field(description="Source collection name")
    target_collection: str = Field(description="Target collection name")
    total_migrated: int = Field(default=0, description="Total points copied/migrated")
    errors_count: int = Field(default=0, description="Failed points count")
    duration_ms: float = Field(default=0.0, description="Migration duration in ms")
    success: bool = Field(default=True, description="Whether migration finished cleanly")


class PayloadIndexHealthReport(BaseModel):
    """Audit report for payload indexes on a collection."""

    model_config = ConfigDict(frozen=True)

    collection_name: str = Field(description="Collection audited")
    total_expected_indexes: int = Field(description="Total indexes configured")
    active_indexes: list[str] = Field(default_factory=list, description="Names of currently active indexes")
    missing_indexes: list[str] = Field(default_factory=list, description="Configured indexes missing in backend")
    is_healthy: bool = Field(description="True if all expected indexes are active")


class OptimizationMetrics(BaseModel):
    """Metrics captured during an individual vector optimization task."""

    model_config = ConfigDict(frozen=True)

    collection_name: str = Field(description="Collection optimized")
    optimization_type: str = Field(description="Type: segment_merge, vacuum, payload_compact, hnsw_rebalance")
    segments_before: int = Field(default=0, description="Segments before optimization")
    segments_after: int = Field(default=0, description="Segments after optimization")
    purged_points: int = Field(default=0, description="Number of soft-deleted points purged")
    reclaimed_bytes: int = Field(default=0, description="Estimated disk bytes reclaimed")
    duration_ms: float = Field(default=0.0, description="Optimization duration in ms")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Execution timestamp",
    )


class CompleteOptimizationReport(BaseModel):
    """Comprehensive report for all optimization passes executed on a collection."""

    model_config = ConfigDict(frozen=True)

    collection_name: str = Field(description="Collection optimized")
    tasks_executed: list[OptimizationMetrics] = Field(default_factory=list, description="Individual task reports")
    total_reclaimed_bytes: int = Field(default=0, description="Total disk bytes reclaimed across all tasks")
    total_duration_ms: float = Field(default=0.0, description="Total elapsed time in ms")


class VectorDBHealthReport(BaseModel):
    """Health diagnostic report for the vector storage engine."""

    model_config = ConfigDict(frozen=True)

    status: str = Field(description="Overall status: HEALTHY, DEGRADED, UNHEALTHY")
    backend_type: str = Field(description="Underlying engine (e.g. qdrant_local)")
    is_airgapped: bool = Field(default=True, description="Strictly offline with zero network leakage")
    total_collections: int = Field(default=0, description="Number of active collections")
    total_vectors: int = Field(default=0, description="Sum of vectors across all collections")
    canary_write_latency_ms: float = Field(default=0.0, description="Canary probe write latency")
    canary_read_latency_ms: float = Field(default=0.0, description="Canary probe read latency")
    disk_free_bytes: int = Field(default=0, description="Available disk space on storage drive")
    warnings: list[str] = Field(default_factory=list, description="Operational warnings if any")


class StorageStatsReport(BaseModel):
    """Comprehensive storage telemetry report across all collections."""

    model_config = ConfigDict(frozen=True)

    total_collections: int = Field(default=0, description="Total registered collections")
    active_aliases: int = Field(default=0, description="Number of active collection aliases")
    total_vectors: int = Field(default=0, description="Total vectors indexed")
    average_payload_size_bytes: float = Field(default=0.0, description="Average chunk payload byte size")
    total_disk_bytes: int = Field(default=0, description="Total disk footprint of vector_db/ in bytes")
    collections_detail: dict[str, CollectionStats] = Field(default_factory=dict, description="Detailed stats per collection")
    optimization_history_count: int = Field(default=0, description="Total optimization events recorded")
    snapshot_history_count: int = Field(default=0, description="Total snapshots in backup catalog")
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of telemetry collection",
    )
