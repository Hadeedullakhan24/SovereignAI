"""
Processing Context — Pipeline Execution Context Object.

Sovereign On-Premise Agentic AI Workbench (SIH26117 / MRPL)
Carries execution metadata, refinery operational category, user session,
and execution modes throughout the RAG pipeline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
import uuid


@dataclass(frozen=True)
class ProcessingContext:
    """Immutable context object accompanying documents throughout all pipeline stages."""

    dataset: str = "datasets"
    category: str = "general"
    subcategory: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    ingestion_mode: str = "batch"
    dry_run: bool = False
    user: str = "system"
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    extra_context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert processing context into a dictionary."""
        return asdict(self)

    @classmethod
    def create_default(
        cls,
        category: str = "general",
        subcategory: str = "",
        dataset: str = "datasets",
    ) -> ProcessingContext:
        """Convenience constructor for default execution context."""
        return cls(
            dataset=dataset,
            category=category,
            subcategory=subcategory,
        )
