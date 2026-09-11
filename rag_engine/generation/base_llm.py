"""Base LLM interface re-export for top-level generation access."""

from rag_engine.generation.models.base_model import (
    BaseLLM,
    BaseLocalLLM,
    GGUFLocalLLM,
    LLMGenerationOutput,
    QuantizationType,
)

__all__ = [
    "BaseLLM",
    "BaseLocalLLM",
    "GGUFLocalLLM",
    "LLMGenerationOutput",
    "QuantizationType",
]
