"""Local Model Management — In-Process Air-Gapped Model Runtimes."""

from rag_engine.generation.models.base_model import (
    BaseLLM,
    BaseLocalLLM,
    GGUFLocalLLM,
    LLMGenerationOutput,
    QuantizationType,
)
from rag_engine.generation.models.deterministic_test_llm import DeterministicTestLLM
from rag_engine.generation.models.hf_causal_lm import (
    HFLocalLLM,
    HuggingFaceCausalLM,
)
from rag_engine.generation.models.model_factory import LLMFactory
from rag_engine.generation.models.model_registry import (
    LLMRegistry,
    register_llm,
)

__all__ = [
    "BaseLLM",
    "BaseLocalLLM",
    "HFLocalLLM",
    "HuggingFaceCausalLM",
    "GGUFLocalLLM",
    "DeterministicTestLLM",
    "QuantizationType",
    "LLMGenerationOutput",
    "LLMRegistry",
    "LLMFactory",
    "register_llm",
]
