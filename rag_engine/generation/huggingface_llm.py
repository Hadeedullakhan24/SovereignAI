"""HuggingFace Local LLM re-export for top-level generation access."""

from rag_engine.generation.models.hf_causal_lm import (
    HFLocalLLM,
    HuggingFaceCausalLM,
)

__all__ = [
    "HuggingFaceCausalLM",
    "HFLocalLLM",
]
