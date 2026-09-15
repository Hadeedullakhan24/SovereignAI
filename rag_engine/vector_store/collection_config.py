"""Collection Configuration Descriptors.

Provides backend-agnostic, declarative configuration models for vector collections,
HNSW indexing parameters, segment optimizers, future replication, and quantization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from rag_engine.schemas.vector_store import DistanceMetric, PayloadSchemaType
from rag_engine.config.runtime_paths import runtime_file


def _runtime_root() -> Path:
    """Return a writable runtime root without mutating source/index data.

    Embedded Qdrant requires exclusive write access to its lock and SQLite
    files.  A checked-out repository is often read-only (or shared by several
    developers), so it is never a safe default persistence location.  An
    operator may set ``SOVEREIGNAI_RUNTIME_DIR`` to a managed local volume.
    """
    return runtime_file()


def default_vector_storage_path() -> Path:
    """Return the location for mutable embedded Qdrant state.

    A source checkout can contain a seeded ``vector_db/qdrant`` directory, but
    it must never become the implicit runtime database: that both mutates
    repository data and makes test/process state leak across runs.
    """
    return _runtime_root() / "vector_db" / "qdrant"


def default_journal_path() -> Path:
    return _runtime_root() / "vector_db" / "journal"


def default_backup_path() -> Path:
    return _runtime_root() / "vector_db" / "backups"


def default_telemetry_path() -> Path:
    return _runtime_root() / "vector_db" / "telemetry"


class HNSWConfig(BaseModel):
    """Hierarchical Navigable Small World (HNSW) graph indexing configuration."""

    model_config = ConfigDict(frozen=True)

    m: int = Field(default=16, ge=4, le=128, description="Number of bidirectional edges per node")
    ef_construct: int = Field(default=100, ge=16, le=1024, description="Search depth during index construction")
    full_scan_threshold: int = Field(default=10000, description="Payload filtering threshold for exact scan")
    max_indexing_threads: int = Field(default=0, description="Threads for indexing; 0 means auto")
    on_disk: bool = Field(default=True, description="Store HNSW graph links on disk with memory mapping")


class OptimizerConfig(BaseModel):
    """Background segment compaction and vacuuming settings."""

    model_config = ConfigDict(frozen=True)

    deleted_threshold: float = Field(default=0.2, ge=0.0, le=1.0, description="Ratio of deleted points to trigger vacuum")
    vacuum_min_vector_number: int = Field(default=1000, description="Minimum vectors in segment to trigger vacuum")
    default_segment_number: int = Field(default=2, ge=1, description="Target number of active storage segments")
    max_segment_size: Optional[int] = Field(default=None, description="Max vectors per segment in KB")
    indexing_threshold: int = Field(default=20000, description="Minimum points before background indexing triggers")


class ReplicationConfig(BaseModel):
    """Distributed replication settings for future cluster deployment."""

    model_config = ConfigDict(frozen=True)

    replication_factor: int = Field(default=1, ge=1, description="Number of segment copies across cluster nodes")
    shard_number: int = Field(default=1, ge=1, description="Number of sharding partitions across cluster")
    write_consistency_factor: int = Field(default=1, ge=1, description="Write ACKs required before commit")


class QuantizationConfig(BaseModel):
    """Vector compression settings for high-density deployment."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = Field(default=False, description="Whether scalar/product quantization is active")
    quantization_type: str = Field(default="scalar", description="'scalar' or 'product'")
    always_ram: bool = Field(default=False, description="Keep quantized vectors in RAM for instant lookup")


class CollectionConfig(BaseModel):
    """Declarative, backend-independent descriptor for a vector collection."""

    model_config = ConfigDict(extra="allow")

    name: str = Field(description="Collection name or identifier (e.g. mrpl_manuals_v1)")
    vector_size: int = Field(default=384, ge=1, description="Dimensionality of vectors in this collection")
    distance: DistanceMetric = Field(default=DistanceMetric.COSINE, description="Distance metric for similarity")
    payload_indexes: dict[str, PayloadSchemaType] = Field(
        default_factory=dict,
        description="Fields to index automatically with their payload types",
    )
    hnsw_config: HNSWConfig = Field(default_factory=HNSWConfig, description="HNSW graph parameters")
    optimizer_config: OptimizerConfig = Field(default_factory=OptimizerConfig, description="Segment optimizer settings")
    replication_config: ReplicationConfig = Field(default_factory=ReplicationConfig, description="Replication settings")
    quantization_config: QuantizationConfig = Field(default_factory=QuantizationConfig, description="Quantization settings")
    on_disk_payload: bool = Field(default=True, description="Store chunk metadata payloads on disk")


class VectorStoreConfig(BaseModel):
    """Global configuration for the vector storage engine."""

    model_config = ConfigDict(extra="allow")

    backend: str = Field(default="qdrant", description="Vector database engine: 'qdrant', 'milvus', etc.")
    storage_path: Path = Field(default_factory=default_vector_storage_path, description="Local embedded filesystem directory")
    url: Optional[str] = Field(default=None, description="Network URL if running in distributed Qdrant Server mode")
    api_key: Optional[str] = Field(default=None, description="Optional API key for distributed server mode")
    timeout_seconds: float = Field(default=30.0, description="Default operational timeout in seconds")
    batch_size: int = Field(default=500, description="Default chunk slicing batch size for ingestion")
    enable_wal_journal: bool = Field(default=True, description="Maintain transaction write-ahead logging journal")
    journal_path: Path = Field(default_factory=default_journal_path, description="Directory for transaction WAL records")
    backup_path: Path = Field(default_factory=default_backup_path, description="Directory for tar.gz snapshots")
    telemetry_path: Path = Field(default_factory=default_telemetry_path, description="Directory for storage stats")

    def model_post_init(self, __context: Any) -> None:
        """Co-locate implicit operational state with an explicit store path.

        Callers that set only ``storage_path`` expect a self-contained local
        store.  Leaving the WAL at a process-relative default instead can
        write into a checkout and makes otherwise isolated stores interfere.
        Explicit journal/backup/telemetry settings are retained unchanged.
        """
        root = self.storage_path.parent
        if "journal_path" not in self.model_fields_set:
            self.journal_path = root / "journal"
        if "backup_path" not in self.model_fields_set:
            self.backup_path = root / "backups"
        if "telemetry_path" not in self.model_fields_set:
            self.telemetry_path = root / "telemetry"
