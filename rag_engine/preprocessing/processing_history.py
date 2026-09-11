"""Processing history and audit logging for document transformations."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional
from pydantic import BaseModel, ConfigDict, Field


class ProcessingStageRecord(BaseModel):
    """Immutable audit record representing the execution of a cleaning stage."""

    model_config = ConfigDict(frozen=True)

    stage_name: str = Field(description="Identifier of the pipeline stage")
    stage_order: int = Field(description="1-based execution order in the pipeline")
    status: str = Field(default="SUCCESS", description="SUCCESS | MODIFIED | SKIPPED | FAILED")
    characters_before: int = Field(default=0, ge=0)
    characters_after: int = Field(default=0, ge=0)
    details: dict[str, Any] = Field(default_factory=dict, description="Stage-specific metadata")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 execution timestamp",
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize record to dictionary for embedding in ParsedDocument."""
        return self.model_dump()


class ProcessingHistoryRecorder:
    """Helper to record stage operations into a document's processing_history list."""

    @staticmethod
    def create_record(
        stage_name: str,
        stage_order: int,
        status: str,
        characters_before: int,
        characters_after: int,
        details: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Construct a serializable dictionary record for document history."""
        record = ProcessingStageRecord(
            stage_name=stage_name,
            stage_order=stage_order,
            status=status,
            characters_before=characters_before,
            characters_after=characters_after,
            details=details or {},
        )
        return record.to_dict()
