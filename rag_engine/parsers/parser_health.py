"""Health monitoring and diagnostics for document parsers."""

from __future__ import annotations

from enum import StrEnum
from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class ParserHealthStatus(StrEnum):
    """Health states for a parser component."""

    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"


class ParserHealthReport(BaseModel):
    """Health diagnostic report for a parser."""

    model_config = ConfigDict(frozen=True)

    parser_name: str = Field(description="Name of the parser class")
    status: ParserHealthStatus = Field(description="Current operational health status")
    supported_formats: list[str] = Field(default_factory=list, description="Supported file extensions")
    supported_categories: list[str] = Field(default_factory=list, description="Supported document categories")
    available_drivers: list[str] = Field(default_factory=list, description="Currently installed drivers")
    missing_dependencies: list[str] = Field(default_factory=list, description="Missing optional drivers")
    details: dict[str, Any] = Field(default_factory=dict, description="Diagnostic attributes")
