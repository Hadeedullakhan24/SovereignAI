"""Configuration models for Prompt Engineering and Context Assembly."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PromptConfig(BaseModel):
    """Central configuration for Prompt Builder and Context Assembly Gateway."""

    model_config = ConfigDict(frozen=True)

    default_archetype: str = Field(default="general_qa", description="Default domain archetype")
    persona: str = Field(default="Senior Refinery Operations Specialist", description="Engineering persona")
    enable_compression: bool = Field(default=True, description="Enable smart context compression when over budget")
    compression_strategy: str = Field(default="selective_extraction", description="Strategy: truncate, selective_extraction, compact")
    conversation_style: str = Field(default="markdown", description="Format: markdown, chatml, plain")
    citation_style: str = Field(default="inline_anchors", description="Format: inline_anchors, footnotes, tabular")
    strict_validation: bool = Field(default=True, description="Reject prompt on any critical validation failure")
    prompt_version: str = Field(default="v1.0.0", description="Semantic prompt engine version")
    retrieval_version: str = Field(default="m8_v1.0", description="Retrieval pipeline contract version")
    enforce_deterministic_sorting: bool = Field(default=True, description="Enforce stable candidate sorting")
