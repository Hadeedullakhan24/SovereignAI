"""Vector Store Package.

Enterprise-Grade Vector Storage & Indexing Platform for the Sovereign AI Workbench.
Decoupled, 100% offline, air-gapped, supporting Qdrant Local with seamless cluster readiness.
"""

from rag_engine.interfaces.base_vector_store import BaseVectorStore
from rag_engine.vector_store.collection_config import (
    CollectionConfig,
    HNSWConfig,
    OptimizerConfig,
    QuantizationConfig,
    ReplicationConfig,
    VectorStoreConfig,
)
from rag_engine.vector_store.collection_manager import CollectionManager
from rag_engine.vector_store.collection_router import CollectionRouter
from rag_engine.vector_store.collection_version_manager import CollectionVersionManager
from rag_engine.vector_store.exceptions import (
    BackendNotSupportedError,
    CollectionAlreadyExistsError,
    CollectionNotFoundError,
    OptimizationError,
    PayloadValidationError,
    RoutingError,
    SnapshotError,
    TransactionError,
    VectorDimensionMismatchError,
    VectorQualityError,
    VectorStoreError,
    VersioningError,
)
from rag_engine.vector_store.incremental_indexer import IncrementalIndexer
from rag_engine.vector_store.index_manager import IndexManager
from rag_engine.vector_store.metadata_serializer import MetadataSerializer
from rag_engine.vector_store.payload_index_manager import PayloadIndexManager
from rag_engine.vector_store.payload_validator import PayloadValidator
from rag_engine.vector_store.qdrant_store import QdrantVectorStore
from rag_engine.vector_store.schema_validator import SchemaValidator
from rag_engine.vector_store.snapshot_manager import SnapshotManager
from rag_engine.vector_store.storage_stats import StorageStatsCollector
from rag_engine.vector_store.transaction_manager import TransactionManager
from rag_engine.vector_store.vector_events import (
    ChunksIndexedEvent,
    CollectionCreatedEvent,
    SnapshotCreatedEvent,
    VectorEvent,
    VectorEventBus,
    VersionSwappedEvent,
    global_vector_event_bus,
)
from rag_engine.vector_store.vector_factory import VectorFactory, get_vector_store, global_vector_factory
from rag_engine.vector_store.vector_health import VectorHealthMonitor
from rag_engine.vector_store.vector_lifecycle_manager import VectorLifecycleManager
from rag_engine.vector_store.vector_metrics import VectorMetricsCollector, global_vector_metrics
from rag_engine.vector_store.vector_optimizer import VectorOptimizer
from rag_engine.vector_store.vector_registry import VectorRegistry, global_vector_registry
from rag_engine.vector_store.vector_repository import (
    VectorRepository,
    get_vector_repository,
)
from rag_engine.vector_store.vector_utils import (
    chunk_id_to_uuid,
    compute_payload_checksum,
    compute_vector_checksum,
)

__all__ = [
    "BaseVectorStore",
    "QdrantVectorStore",
    "VectorStoreConfig",
    "CollectionConfig",
    "HNSWConfig",
    "OptimizerConfig",
    "ReplicationConfig",
    "QuantizationConfig",
    "VectorRegistry",
    "global_vector_registry",
    "VectorFactory",
    "global_vector_factory",
    "get_vector_store",
    "VectorRepository",
    "get_vector_repository",
    "CollectionRouter",
    "CollectionVersionManager",
    "CollectionManager",
    "PayloadIndexManager",
    "VectorOptimizer",
    "VectorLifecycleManager",
    "PayloadValidator",
    "SchemaValidator",
    "MetadataSerializer",
    "TransactionManager",
    "IncrementalIndexer",
    "IndexManager",
    "SnapshotManager",
    "StorageStatsCollector",
    "VectorHealthMonitor",
    "VectorMetricsCollector",
    "global_vector_metrics",
    "VectorEventBus",
    "global_vector_event_bus",
    "VectorEvent",
    "CollectionCreatedEvent",
    "ChunksIndexedEvent",
    "VersionSwappedEvent",
    "SnapshotCreatedEvent",
    "chunk_id_to_uuid",
    "compute_payload_checksum",
    "compute_vector_checksum",
    # Exceptions
    "VectorStoreError",
    "CollectionNotFoundError",
    "CollectionAlreadyExistsError",
    "PayloadValidationError",
    "VectorDimensionMismatchError",
    "VectorQualityError",
    "RoutingError",
    "VersioningError",
    "TransactionError",
    "SnapshotError",
    "OptimizationError",
    "BackendNotSupportedError",
]
