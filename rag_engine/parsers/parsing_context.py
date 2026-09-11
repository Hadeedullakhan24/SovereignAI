"""Execution context and configuration for document parsing operations."""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, ConfigDict, Field

from rag_engine.parsers.profiles.base_profile import RefineryProfile
from rag_engine.parsers.profiles.mrpl_profile import MRPLProfile


class ParsingContext(BaseModel):
    """Immutable context configuring parser behavior and feature extraction."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    profile: RefineryProfile = Field(
        default_factory=MRPLProfile,
        description="Active refinery nomenclature and standards profile",
    )
    extract_sections: bool = Field(default=True, description="Extract hierarchical H1-H6 sections")
    extract_tables: bool = Field(default=True, description="Parse tabular layouts into structured Tables")
    extract_equipment: bool = Field(default=True, description="Extract equipment tags and operating limits")
    extract_relationships: bool = Field(default=True, description="Construct EntityGraph relationships")
    extract_safety: bool = Field(default=True, description="Isolate DANGER/WARNING notices & standards")
    extract_metadata: bool = Field(default=True, description="Extract dates, revisions, and engineers")
    extract_cross_references: bool = Field(default=True, description="Extract figure/table/section references")
    run_validation: bool = Field(default=True, description="Execute ParserValidator on parsed output")
    streaming_mode: bool = Field(default=False, description="Enable lazy stream/generator processing")
    page_batch_size: int = Field(default=50, ge=1, description="Page batch size for streaming/memory control")
    max_pages_to_parse: Optional[int] = Field(default=None, description="Optional limit on parsed pages")
    confidence_threshold: float = Field(default=0.50, ge=0.0, le=1.0, description="Minimum confidence cutoff")
    extra_params: dict[str, Any] = Field(default_factory=dict, description="Format-specific custom flags")
