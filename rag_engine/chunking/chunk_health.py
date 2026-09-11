"""Health, version, and diagnostic reporting for the Chunking Engine."""

from __future__ import annotations

import sys
from typing import Any
from pydantic import BaseModel, Field

from rag_engine.chunking.chunk_metrics import get_chunk_metrics_collector

CHUNKING_ENGINE_VERSION = "1.0.0"

SUPPORTED_CHUNK_STRATEGIES = [
    "fixed_size_token_sliding_window",
    "fixed_size_character_sliding_window",
    "recursive_hierarchical_splitting",
    "section_aware_boundary_preservation",
    "table_aware_row_cohesion",
    "list_aware_procedure_preservation",
    "deterministic_sha256_chunk_ids",
    "content_hash_deduplication",
    "metadata_inheritance_cascading",
    "parent_child_hierarchy_linking",
    "bpe_heuristic_token_estimation",
    "thread_safe_rlock_architecture",
    "100_percent_offline_zero_api_calls",
]


class ChunkHealthReport(BaseModel):
    """Health diagnostic report for the chunking engine."""

    status: str = Field(default="HEALTHY", description="HEALTHY | DEGRADED | UNHEALTHY")
    version: str = Field(default=CHUNKING_ENGINE_VERSION)
    python_version: str = Field(default=sys.version.split()[0])
    dependencies: dict[str, str] = Field(default_factory=dict)
    supported_strategies: list[str] = Field(default_factory=lambda: list(SUPPORTED_CHUNK_STRATEGIES))
    registered_strategies: list[str] = Field(default_factory=list)
    thread_safe: bool = Field(default=True)
    aggregate_statistics: dict[str, Any] = Field(default_factory=dict)


def version() -> str:
    """Return the chunking engine version string."""
    return CHUNKING_ENGINE_VERSION


def dependencies() -> dict[str, str]:
    """Return dictionary of runtime dependencies and versions."""
    deps: dict[str, str] = {
        "python": sys.version.split()[0],
        "pydantic": "2.x",
    }
    try:
        import pydantic

        deps["pydantic"] = getattr(pydantic, "__version__", "2.x")
    except ImportError:
        deps["pydantic"] = "missing"
    return deps


def supported_features() -> list[str]:
    """Return list of supported chunking engine capabilities."""
    return list(SUPPORTED_CHUNK_STRATEGIES)


def health() -> ChunkHealthReport:
    """Generate comprehensive health report of the chunking engine."""
    collector = get_chunk_metrics_collector()
    stats = collector.get_global_statistics()
    status = "HEALTHY"
    if stats.get("total_failures", 0) > 0 and stats.get("total_documents_chunked", 0) > 0:
        failure_rate = stats["total_failures"] / stats["total_documents_chunked"]
        if failure_rate > 0.5:
            status = "DEGRADED"

    from rag_engine.chunking.chunk_registry import get_chunk_registry

    return ChunkHealthReport(
        status=status,
        version=CHUNKING_ENGINE_VERSION,
        python_version=sys.version.split()[0],
        dependencies=dependencies(),
        supported_strategies=supported_features(),
        registered_strategies=get_chunk_registry().list_strategies(),
        aggregate_statistics=stats,
    )
