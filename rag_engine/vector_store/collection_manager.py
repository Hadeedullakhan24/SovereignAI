"""Collection Manager.

Manages collection lifecycles, predefined catalogs, creation of collections with
domain-specific HNSW parameters, drop, reset, and alias mapping operations.
"""

from __future__ import annotations

import logging
from typing import Optional

from rag_engine.interfaces.base_vector_store import BaseVectorStore
from rag_engine.schemas.vector_store import CollectionStats, DistanceMetric, PayloadSchemaType
from rag_engine.vector_store.collection_config import (
    CollectionConfig,
    HNSWConfig,
    OptimizerConfig,
)
from rag_engine.vector_store.exceptions import CollectionNotFoundError, VectorStoreError

logger = logging.getLogger(__name__)


class CollectionManager:
    """Orchestrates collection lifecycle and predefined domain catalog management."""

    def __init__(self, store: BaseVectorStore) -> None:
        self.store = store

    def get_default_catalog(self, vector_dim: int = 384) -> list[CollectionConfig]:
        """Return predefined enterprise collection configurations for MRPL refinery domains."""
        common_indexes = {
            "document_id": PayloadSchemaType.KEYWORD,
            "document_type": PayloadSchemaType.KEYWORD,
            "category": PayloadSchemaType.KEYWORD,
            "plant_unit": PayloadSchemaType.KEYWORD,
            "equipment_entities": PayloadSchemaType.KEYWORD,
            "safety_entities": PayloadSchemaType.KEYWORD,
            "page_number": PayloadSchemaType.INTEGER,
            "section_title": PayloadSchemaType.TEXT,
            "revision": PayloadSchemaType.KEYWORD,
            "version": PayloadSchemaType.KEYWORD,
            "source_file": PayloadSchemaType.KEYWORD,
            "language": PayloadSchemaType.KEYWORD,
        }

        catalog = [
            # 1. Engineering Manuals
            CollectionConfig(
                name="mrpl_manuals_v1",
                vector_size=vector_dim,
                distance=DistanceMetric.COSINE,
                payload_indexes=common_indexes,
                hnsw_config=HNSWConfig(m=24, ef_construct=200, on_disk=True),
                optimizer_config=OptimizerConfig(default_segment_number=2),
            ),
            # 2. Inspection Reports
            CollectionConfig(
                name="mrpl_inspection_v1",
                vector_size=vector_dim,
                distance=DistanceMetric.COSINE,
                payload_indexes=common_indexes,
                hnsw_config=HNSWConfig(m=16, ef_construct=100, on_disk=True),
            ),
            # 3. Maintenance Documents
            CollectionConfig(
                name="mrpl_maintenance_v1",
                vector_size=vector_dim,
                distance=DistanceMetric.COSINE,
                payload_indexes=common_indexes,
                hnsw_config=HNSWConfig(m=16, ef_construct=100, on_disk=True),
            ),
            # 4. Safety Standards
            CollectionConfig(
                name="mrpl_safety_v1",
                vector_size=vector_dim,
                distance=DistanceMetric.COSINE,
                payload_indexes=common_indexes,
                hnsw_config=HNSWConfig(m=16, ef_construct=100, on_disk=True),
            ),
            # 5. Technical Emails & Correspondence
            CollectionConfig(
                name="mrpl_emails_v1",
                vector_size=vector_dim,
                distance=DistanceMetric.COSINE,
                payload_indexes=common_indexes,
                hnsw_config=HNSWConfig(m=16, ef_construct=100, on_disk=True),
            ),
            # 6. Engineering Drawings
            CollectionConfig(
                name="mrpl_drawings_v1",
                vector_size=vector_dim,
                distance=DistanceMetric.COSINE,
                payload_indexes=common_indexes,
                hnsw_config=HNSWConfig(m=32, ef_construct=250, on_disk=True),
            ),
            # 7. Piping & Instrumentation Diagrams (P&IDs)
            CollectionConfig(
                name="mrpl_pids_v1",
                vector_size=vector_dim,
                distance=DistanceMetric.COSINE,
                payload_indexes=common_indexes,
                hnsw_config=HNSWConfig(m=32, ef_construct=250, on_disk=True),
            ),
            # 8. Standard Operating Procedures (SOPs)
            CollectionConfig(
                name="mrpl_sops_v1",
                vector_size=vector_dim,
                distance=DistanceMetric.COSINE,
                payload_indexes=common_indexes,
                hnsw_config=HNSWConfig(m=16, ef_construct=100, on_disk=True),
            ),
            # 9. General Refinery Documents Fallback
            CollectionConfig(
                name="mrpl_general_v1",
                vector_size=vector_dim,
                distance=DistanceMetric.COSINE,
                payload_indexes=common_indexes,
                hnsw_config=HNSWConfig(m=16, ef_construct=100, on_disk=True),
            ),
        ]
        return catalog

    def ensure_default_collections(self, vector_dim: int = 384) -> dict[str, bool]:
        """Ensure that all standard domain collections exist in the vector store."""
        results: dict[str, bool] = {}
        for config in self.get_default_catalog(vector_dim):
            if not self.store.collection_exists(config.name):
                try:
                    self.store.create_collection(config)
                    results[config.name] = True
                    logger.info("Created default collection '%s'", config.name)
                except Exception as e:
                    logger.error("Failed to create collection '%s': %s", config.name, e)
                    results[config.name] = False
            else:
                results[config.name] = True
        return results

    def reset_collection(self, name: str, config: Optional[CollectionConfig] = None) -> bool:
        """Drop and recreate an existing collection cleanly."""
        if not self.store.collection_exists(name):
            raise CollectionNotFoundError(f"Cannot reset non-existent collection '{name}'.")

        dim = 384
        try:
            stats = self.store.get_collection_stats(name)
            dim = stats.vector_count  # fallback
        except Exception:
            pass

        self.store.delete_collection(name)
        recreate_cfg = config or CollectionConfig(name=name, vector_size=dim)
        return self.store.create_collection(recreate_cfg)

    def get_all_stats(self) -> dict[str, CollectionStats]:
        """Retrieve statistics for all existing collections."""
        stats: dict[str, CollectionStats] = {}
        for name in self.store.list_collections():
            try:
                stats[name] = self.store.get_collection_stats(name)
            except Exception as e:
                logger.warning("Could not fetch stats for '%s': %s", name, e)
        return stats
