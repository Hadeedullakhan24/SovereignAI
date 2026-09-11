"""Schema: Citation — Represents a source reference for retrieved content."""

from __future__ import annotations

from enum import StrEnum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class SourceType(StrEnum):
    """Enumeration of supported source document types."""

    PDF = "pdf"
    DOCX = "docx"
    MARKDOWN = "markdown"
    CSV = "csv"
    TEXT = "text"
    IMAGE = "image"
    UNKNOWN = "unknown"


class Source(BaseModel):
    """A reference to the original source document."""

    model_config = ConfigDict(frozen=True)

    file_name: str = Field(description="Source file name")
    file_path: str = Field(description="Source file path")
    source_type: SourceType = Field(
        default=SourceType.UNKNOWN, description="Type of source document"
    )
    page_number: Optional[int] = Field(default=None, description="Page number")
    section: Optional[str] = Field(default=None, description="Section heading")


class Citation(BaseModel):
    """A citation linking a retrieved chunk back to its source."""

    model_config = ConfigDict(frozen=True)

    citation_id: str = Field(description="Unique citation identifier")
    source: Source = Field(description="Source document reference")
    chunk_id: str = Field(description="ID of the cited chunk")
    excerpt: str = Field(default="", description="Relevant text excerpt")
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Confidence score"
    )
