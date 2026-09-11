"""Base LLM Interface — Provider-Agnostic Contract for In-Process Language Models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterator

from rag_engine.generation.generation_config import ModelInferenceConfig


class QuantizationType(str, Enum):
    """Supported model quantization and precision levels."""

    FP16 = "fp16"
    BF16 = "bf16"
    FP32 = "fp32"
    INT8 = "int8"
    INT4 = "int4"
    NONE = "none"


@dataclass(frozen=True)
class LLMGenerationOutput:
    """Standardized output container from a local LLM generation run."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str = "stop"


class BaseLLM(ABC):
    """Abstract base class for provider-agnostic, in-process, air-gapped language models."""

    def __init__(
        self,
        model_name: str,
        context_window_size: int = 4096,
        quantization: QuantizationType = QuantizationType.FP16,
    ) -> None:
        self._model_name = model_name
        self._context_window_size = context_window_size
        self._quantization = quantization

    @property
    def model_name(self) -> str:
        """Name or identifier of the active model."""
        return self._model_name

    @property
    def context_window_size(self) -> int:
        """Maximum context length supported by the model."""
        return self._context_window_size

    @property
    def quantization(self) -> QuantizationType:
        """Configured precision / quantization level."""
        return self._quantization

    @abstractmethod
    def generate(
        self,
        prompt: str,
        config: ModelInferenceConfig | None = None,
    ) -> LLMGenerationOutput:
        """Generate a complete text response synchronously.
        
        Args:
            prompt: Formatted, token-bounded prompt string.
            config: Inference hyperparameters.
            
        Returns:
            LLMGenerationOutput containing generated text and token counts.
        """
        ...

    @abstractmethod
    def stream_generate(
        self,
        prompt: str,
        config: ModelInferenceConfig | None = None,
    ) -> Iterator[str]:
        """Stream generated response tokens incrementally.
        
        Args:
            prompt: Formatted, token-bounded prompt string.
            config: Inference hyperparameters.
            
        Yields:
            Incremental string tokens as they are produced.
        """
        ...

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Calculate token count for text using the model's native tokenizer."""
        ...


# Backward compatibility alias
BaseLocalLLM = BaseLLM


class GGUFLocalLLM(BaseLLM):
    """Architectural placeholder for future GGUF/llama.cpp binary model execution."""

    def __init__(
        self,
        model_path: Path | str,
        context_window_size: int = 4096,
        quantization: QuantizationType = QuantizationType.INT4,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name=Path(model_path).name,
            context_window_size=context_window_size,
            quantization=quantization,
        )
        self.model_path = Path(model_path)

    def generate(
        self,
        prompt: str,
        config: ModelInferenceConfig | None = None,
    ) -> LLMGenerationOutput:
        raise NotImplementedError(
            "GGUFLocalLLM is an architectural extension placeholder for future GGUF weights. "
            "Use HFLocalLLM for current production in-process execution."
        )

    def stream_generate(
        self,
        prompt: str,
        config: ModelInferenceConfig | None = None,
    ) -> Iterator[str]:
        raise NotImplementedError(
            "GGUFLocalLLM is an architectural extension placeholder for future GGUF weights. "
            "Use HFLocalLLM for current production in-process execution."
        )

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)
