"""
Loader Health — Operational Health Monitoring & Diagnostics.

Sovereign On-Premise Agentic AI Workbench (SIH26117 / MRPL)
Tracks driver availability, installed vs missing optional dependencies,
and version metadata for every loader.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class LoaderHealthStatus(StrEnum):
    """Health classification for a loader."""

    HEALTHY = "HEALTHY"          # Primary driver and all dependencies available
    DEGRADED = "DEGRADED"        # Operating in air-gapped fallback driver mode
    UNAVAILABLE = "UNAVAILABLE"  # Cannot load any files


@dataclass(frozen=True)
class LoaderHealthReport:
    """Detailed health report for a document loader."""

    loader_name: str
    status: LoaderHealthStatus
    version: str
    supported_formats: list[str]
    primary_driver_available: bool
    fallback_driver_available: bool
    dependencies: dict[str, bool] = field(default_factory=dict)
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "loader_name": self.loader_name,
            "status": self.status.value,
            "version": self.version,
            "supported_formats": self.supported_formats,
            "primary_driver_available": self.primary_driver_available,
            "fallback_driver_available": self.fallback_driver_available,
            "dependencies": dict(self.dependencies),
            "message": self.message,
        }
