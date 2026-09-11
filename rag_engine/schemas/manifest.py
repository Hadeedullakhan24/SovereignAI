"""Schema: Manifest — Data models for dataset scanning, validation, and indexing."""

from __future__ import annotations

from enum import StrEnum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class ValidationStatus(StrEnum):
    """Status classification of a scanned document."""

    VALID = "VALID"
    EMPTY = "EMPTY"
    CORRUPTED = "CORRUPTED"
    UNSUPPORTED = "UNSUPPORTED"
    DUPLICATE = "DUPLICATE"


class DocumentRecord(BaseModel):
    """Detailed record for an indexed document in the dataset manifest."""

    model_config = ConfigDict(frozen=True)

    doc_id: str = Field(description="Unique deterministic identifier for the document")
    file_path: str = Field(description="Absolute file path on disk")
    relative_path: str = Field(description="Relative path from the datasets root")
    category: str = Field(description="Refinery dataset category, e.g. 'manuals', 'safety_docs'")
    extension: str = Field(description="Normalized file extension in lowercase, e.g. '.pdf'")
    size_bytes: int = Field(default=0, ge=0, description="File size in bytes")
    hash_sha256: str = Field(description="SHA-256 hex digest of file contents")
    last_modified: str = Field(description="Last modified timestamp in ISO 8601 format")
    status: ValidationStatus = Field(
        default=ValidationStatus.VALID, description="Validation status"
    )
    error_message: Optional[str] = Field(
        default=None, description="Detailed validation error message if invalid"
    )
    is_duplicate: bool = Field(
        default=False, description="Whether this document has an exact content duplicate"
    )
    duplicate_of: Optional[str] = Field(
        default=None, description="doc_id or path of the canonical primary document"
    )


class DatasetStatistics(BaseModel):
    """Aggregate statistics for a scanned and validated dataset."""

    model_config = ConfigDict(frozen=True)

    total_files: int = Field(default=0, ge=0, description="Total files scanned")
    valid_files: int = Field(default=0, ge=0, description="Number of valid files")
    empty_files: int = Field(default=0, ge=0, description="Number of zero-byte files")
    corrupted_files: int = Field(default=0, ge=0, description="Number of corrupted/unreadable files")
    unsupported_files: int = Field(default=0, ge=0, description="Number of unsupported files ignored")
    duplicate_files: int = Field(default=0, ge=0, description="Total duplicate occurrences")
    category_counts: dict[str, int] = Field(
        default_factory=dict, description="File counts per dataset category"
    )
    extension_counts: dict[str, int] = Field(
        default_factory=dict, description="File counts per extension"
    )
    total_size_bytes: int = Field(default=0, ge=0, description="Total size in bytes")
    total_size_formatted: str = Field(
        default="0 B", description="Human-readable formatted size (e.g. '450.2 MB')"
    )
    exact_duplicates_count: int = Field(
        default=0, ge=0, description="Number of duplicate groups with identical SHA-256"
    )
    filename_duplicates_count: int = Field(
        default=0, ge=0, description="Number of files sharing the same filename in different folders"
    )
    exact_duplicate_groups: dict[str, list[str]] = Field(
        default_factory=dict, description="Mapping of SHA-256 hash to list of duplicate file paths"
    )
    filename_duplicate_groups: dict[str, list[str]] = Field(
        default_factory=dict, description="Mapping of filename to list of duplicate file paths"
    )


class Manifest(BaseModel):
    """Top-level dataset manifest containing indexed documents and statistics."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = Field(default="1.0.0", description="Manifest schema version")
    generated_at: str = Field(description="Generation timestamp in ISO 8601 format")
    total_documents: int = Field(default=0, ge=0, description="Total documents indexed")
    statistics: DatasetStatistics = Field(description="Aggregate dataset statistics")
    documents: list[DocumentRecord] = Field(
        default_factory=list, description="List of all indexed document records"
    )
