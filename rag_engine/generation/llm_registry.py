"""LLM Registry re-export for top-level generation access."""

from rag_engine.generation.models.model_registry import (
    LLMRegistry,
    MODEL_ALIASES,
    MODEL_SPECIFICATIONS,
    register_llm,
)

__all__ = [
    "LLMRegistry",
    "MODEL_SPECIFICATIONS",
    "MODEL_ALIASES",
    "register_llm",
]
