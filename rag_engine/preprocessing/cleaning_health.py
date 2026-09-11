"""Health, version, dependencies, and feature diagnostics for the Cleaning Engine."""

from __future__ import annotations

import sys
from typing import Any
from pydantic import BaseModel, Field

from rag_engine.preprocessing.cleaning_metrics import get_cleaning_metrics_collector

CLEANING_ENGINE_VERSION = "1.0.0"

SUPPORTED_FEATURES = [
    "stage1_unicode_nfkc_normalization",
    "stage2_encoding_control_char_stripping",
    "stage3_whitespace_normalization",
    "stage4_line_ending_normalization",
    "stage5_broken_paragraph_reconstruction",
    "stage6_hyphenated_word_reconstruction",
    "stage7_repeated_header_footer_detection",
    "stage8_page_number_citation_preservation",
    "stage9_table_whitespace_structure_cleanup",
    "stage10_bullet_normalization",
    "stage11_numbered_list_normalization",
    "stage12_engineering_token_protection",
    "runtime_cleaner_plugin_system",
    "thread_safe_rlock_architecture",
    "observability_metrics_and_events",
    "100_percent_offline_zero_api_calls",
]


class CleaningHealthReport(BaseModel):
    """Health diagnostic report for the cleaning engine."""

    status: str = Field(default="HEALTHY", description="HEALTHY | DEGRADED | UNHEALTHY")
    version: str = Field(default=CLEANING_ENGINE_VERSION)
    python_version: str = Field(default=sys.version.split()[0])
    dependencies: dict[str, str] = Field(default_factory=dict)
    supported_features: list[str] = Field(default_factory=lambda: list(SUPPORTED_FEATURES))
    aggregate_statistics: dict[str, Any] = Field(default_factory=dict)


def version() -> str:
    """Return the cleaning engine version string."""
    return CLEANING_ENGINE_VERSION


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
    """Return list of supported cleaning engine features."""
    return list(SUPPORTED_FEATURES)


def health() -> CleaningHealthReport:
    """Generate comprehensive health report of the cleaning engine."""
    collector = get_cleaning_metrics_collector()
    stats = collector.get_aggregate_statistics()
    status = "HEALTHY"
    if stats.get("total_failures", 0) > 0 and stats.get("total_documents", 0) > 0:
        failure_rate = stats["total_failures"] / stats["total_documents"]
        if failure_rate > 0.5:
            status = "DEGRADED"

    return CleaningHealthReport(
        status=status,
        version=CLEANING_ENGINE_VERSION,
        python_version=sys.version.split()[0],
        dependencies=dependencies(),
        supported_features=supported_features(),
        aggregate_statistics=stats,
    )
