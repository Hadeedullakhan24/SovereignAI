"""Generation Engine — Centralized Configuration & Parameters.

Defines Pydantic v2 configuration models for local in-process LLM inference,
token budget partitions, prompt templates, anti-hallucination thresholds,
safety guardrails, and persistent generation caching.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field
from rag_engine.config.runtime_paths import runtime_file


class TokenBudgetConfig(BaseModel):
    """Dynamic token budget allocation across prompt sections."""

    model_config = ConfigDict(frozen=True)

    max_context_window: int = Field(
        default=4096,
        description="Total context window limit for the active model in tokens",
    )
    system_prompt_budget: int = Field(
        default=400,
        description="Maximum tokens allocated for system prompt & instructions",
    )
    memory_budget: int = Field(
        default=800,
        description="Maximum tokens allocated for conversation history turns",
    )
    retrieved_context_budget: int = Field(
        default=2000,
        description="Maximum tokens allocated for retrieved ground truth chunks",
    )
    generation_budget: int = Field(
        default=896,
        description="Maximum tokens allocated for LLM response generation",
    )


class ModelInferenceConfig(BaseModel):
    """Inference hyperparameter controls for local in-process generation."""

    model_config = ConfigDict(frozen=True)

    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Sampling temperature; 0.1 for high determinism in refinery QA",
    )
    top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling threshold",
    )
    top_k: int = Field(
        default=50,
        ge=0,
        description="Top-k sampling threshold; 0 disables top-k",
    )
    max_new_tokens: int = Field(
        default=400,
        ge=1,
        le=4096,
        description="Upper bound on generated tokens",
    )
    repetition_penalty: float = Field(
        default=1.1,
        ge=1.0,
        le=2.0,
        description="Penalty for repetitive tokens",
    )
    stop_sequences: list[str] = Field(
        default_factory=lambda: [
            "<|im_end|>",
            "<|endoftext|>",
            "\n\nHuman:",
            "\n\nUser:",
            "=== END ===",
        ],
        description="Sequence markers that halt generation",
    )


class GuardrailConfig(BaseModel):
    """Thresholds and toggles for generation guardrails."""

    model_config = ConfigDict(frozen=True)

    enable_citation_validation: bool = Field(
        default=True,
        description="Strictly verify that [n] citations map to authentic chunks",
    )
    strip_phantom_citations: bool = Field(
        default=True,
        description="Automatically remove unanchored citations from generated text",
    )
    enable_hallucination_guard: bool = Field(
        default=True,
        description="Cross-check technical numerical values and equipment tags against context",
    )
    hallucination_tolerance_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Minimum grounding overlap ratio required before triggering warning",
    )
    enable_safety_validation: bool = Field(
        default=True,
        description="Screen inputs for prompt injection and outputs for credential leaks",
    )


class GenerationConfig(BaseModel):
    """Master configuration container for the Generation Gateway."""

    model_config = ConfigDict(frozen=True)

    default_model_name: str = Field(
        default="auto",
        description="Active model identifier, path, or 'auto' for dynamic routing",
    )
    models_dir: Path = Field(
        default=Path("models/llm"),
        description="Local directory storing open-weight models for 100% offline loading",
    )
    device: str = Field(
        default="auto",
        description="Target compute device: 'cpu', 'cuda', or 'auto'",
    )
    torch_dtype: str = Field(
        default="float16",
        description="Precision: 'float16', 'bfloat16', or 'float32'",
    )
    num_threads: Optional[int] = Field(
        default=None,
        description="Optional CPU inference thread count for PyTorch (None uses system/env default)",
    )
    budgets: TokenBudgetConfig = Field(default_factory=TokenBudgetConfig)
    inference: ModelInferenceConfig = Field(default_factory=ModelInferenceConfig)
    guardrails: GuardrailConfig = Field(default_factory=GuardrailConfig)
    cache_enabled: bool = Field(
        default=True,
        description="Enable persistent SQLite generation cache",
    )
    cache_db_path: Path = Field(
        default_factory=lambda: runtime_file("cache", "generation_cache.db"),
        description="Path to SQLite WAL generation cache database",
    )
    cache_ttl_seconds: int = Field(
        default=604800,  # 7 days
        description="Time-to-live for generation cache records in seconds",
    )
    strict_offline: bool = Field(
        default=True,
        description="Strict offline air-gapped execution mode with local_files_only enforcement",
    )
    telemetry_enabled: bool = Field(
        default=False,
        description="Strict zero external network telemetry / cloud reporting flag",
    )
