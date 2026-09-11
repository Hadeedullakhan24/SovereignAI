"""Pydantic v2 schemas and immutable dataclasses for Milestone 9 Prompt Engineering.

Defines:
- RetrievedPrompt: Full end-to-end prompt container produced by prompt pipeline.
- PromptPayload: Core generation payload (backwards-compatible).
- SystemPrompt: Structured system prompt with persona and safety hashes.
- ConversationTurn: Single multi-turn conversation turn.
- ContextWindow: Structured context window with packed citations.
- PromptValidationResult: Validation diagnostics and safety report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
from typing import Any, Optional, Sequence
from pydantic import BaseModel, ConfigDict, Field


class ConversationTurn(BaseModel):
    """Structured representation of a single conversation turn."""

    model_config = ConfigDict(frozen=True)

    role: str = Field(description="Role: 'user', 'assistant', or 'system'")
    content: str = Field(description="Turn message content")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    token_count: int = Field(default=0, description="Pre-computed token count of content")


class SystemPrompt(BaseModel):
    """Structured system prompt with persona constraints and cryptographic hash."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(description="Formatted system instruction text")
    archetype: str = Field(description="Refinery domain archetype name")
    persona: str = Field(default="Refinery Engineering Specialist")
    safety_rules_applied: list[str] = Field(default_factory=list)
    system_hash: str = Field(description="SHA-256 hash of system prompt text")

    @classmethod
    def create(
        cls,
        text: str,
        archetype: str,
        persona: str = "Refinery Engineering Specialist",
        safety_rules: Optional[list[str]] = None,
    ) -> SystemPrompt:
        shash = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
        return cls(
            text=text.strip(),
            archetype=archetype,
            persona=persona,
            safety_rules_applied=safety_rules or [],
            system_hash=shash,
        )


class ContextWindow(BaseModel):
    """Structured packed context window ready for prompt injection."""

    model_config = ConfigDict(frozen=True)

    formatted_text: str = Field(description="Packed context text with citation markers")
    token_count: int = Field(description="Total token consumption of context")
    chunk_count: int = Field(description="Number of chunks successfully packed")
    chunk_to_anchor_map: dict[str, str] = Field(default_factory=dict)
    dropped_chunk_ids: list[str] = Field(default_factory=list)
    compression_ratio: float = Field(default=1.0, description="Original tokens / Packed tokens")


class PromptValidationResult(BaseModel):
    """Result of pre-generation prompt validation."""

    model_config = ConfigDict(frozen=True)

    is_valid: bool = Field(description="True if all prompt safety and syntax gates pass")
    total_tokens: int = Field(description="Total prompt token consumption")
    max_allowed_tokens: int = Field(description="Budget ceiling for target model")
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    contains_injection_attempt: bool = Field(default=False)
    has_unanchored_citations: bool = Field(default=False)


@dataclass(frozen=True)
class PromptPayload:
    """Core prompt payload compatible with all downstream generation pipeline stages."""

    prompt_text: str
    prompt_hash: str
    prompt_version: str
    retrieval_version: str
    model_name: str
    archetype: str
    timestamp: str
    chunk_to_anchor_map: dict[str, str]
    token_count: int

    @property
    def estimated_tokens(self) -> int:
        """Backward compatibility alias for token_count."""
        return self.token_count

    @property
    def full_prompt(self) -> str:
        """Backward compatibility alias for prompt_text."""
        return self.prompt_text



@dataclass(frozen=True)
class RetrievedPrompt(PromptPayload):
    """Master production prompt contract uniting retrieval evidence, context, and cryptographic lineage."""

    query: str = ""
    system_prompt: Optional[str] = None
    context_window: Optional[str] = None
    conversation_history: Optional[str] = None
    citation_ids: list[str] = field(default_factory=list)
    system_tokens: int = 0
    context_tokens: int = 0
    history_tokens: int = 0
    query_tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        query: str,
        prompt_text: str,
        prompt_hash: str,
        model_name: str,
        archetype: str,
        chunk_to_anchor_map: dict[str, str],
        token_count: int,
        prompt_version: str = "v1.0.0",
        retrieval_version: str = "m8_v1.0",
        system_prompt: Optional[str] = None,
        context_window: Optional[str] = None,
        conversation_history: Optional[str] = None,
        citation_ids: Optional[list[str]] = None,
        system_tokens: int = 0,
        context_tokens: int = 0,
        history_tokens: int = 0,
        query_tokens: int = 0,
        metadata: Optional[dict[str, Any]] = None,
    ) -> RetrievedPrompt:
        now_ts = datetime.now(timezone.utc).isoformat()
        return cls(
            prompt_text=prompt_text,
            prompt_hash=prompt_hash,
            prompt_version=prompt_version,
            retrieval_version=retrieval_version,
            model_name=model_name,
            archetype=archetype,
            timestamp=now_ts,
            chunk_to_anchor_map=chunk_to_anchor_map,
            token_count=token_count,
            query=query,
            system_prompt=system_prompt,
            context_window=context_window,
            conversation_history=conversation_history,
            citation_ids=citation_ids or [],
            system_tokens=system_tokens,
            context_tokens=context_tokens,
            history_tokens=history_tokens,
            query_tokens=query_tokens,
            metadata=metadata or {},
        )
