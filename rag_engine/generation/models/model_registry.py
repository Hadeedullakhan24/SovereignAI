"""LLM Registry — Dynamic Catalog and Plugin Architecture for Local Language Models."""

from __future__ import annotations

import logging
from pathlib import Path
import threading
from typing import Any, Callable, Dict, Optional, Type

from rag_engine.generation.models.base_model import BaseLocalLLM

logger = logging.getLogger(__name__)

# Standard open-weight local model specifications for sovereign offline deployment
MODEL_SPECIFICATIONS: Dict[str, Dict[str, Any]] = {
    "deterministic_test": {
        "family": "test",
        "description": "Deterministic in-memory testing model for unit tests and CI",
        "context_window": 4096,
        "default_quantization": "fp16",
    },
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": {
        "family": "tinyllama",
        "description": "Compact 1.1B conversational model for resource-constrained edge inference",
        "context_window": 2048,
        "default_quantization": "fp16",
        "local_folder": "TinyLlama_TinyLlama-1.1B-Chat-v1.0",
    },
    "microsoft/Phi-3.5-mini-instruct": {
        "family": "phi",
        "description": "Phi 3.5 Mini Instruct (3.8B, Reasoning-optimized, 128k context)",
        "context_window": 131072,
        "default_quantization": "fp16",
        "local_folder": "phi-3.5-mini-instruct",
    },
    "microsoft/Phi-3-mini-4k-instruct": {
        "family": "phi3",
        "description": "High-efficiency 3.8B model with strong technical reasoning",
        "context_window": 4096,
        "default_quantization": "fp16",
        "local_folder": "microsoft_Phi-3-mini-4k-instruct",
    },
    "HuggingFaceTB/SmolLM2-1.7B-Instruct": {
        "family": "smollm",
        "description": "SmolLM2 1.7B Instruct (Compact, Fast, 8k context)",
        "context_window": 8192,
        "default_quantization": "fp16",
        "local_folder": "smollm2-1.7b-instruct",
    },
    "Qwen/Qwen2.5-1.5B-Instruct": {
        "family": "qwen",
        "description": "Lightweight 1.5B model with structured output capability",
        "context_window": 32768,
        "default_quantization": "fp16",
        "local_folder": "qwen2.5-1.5b-instruct",
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "family": "qwen",
        "description": "Production-grade 7B model for complex refinery troubleshooting",
        "context_window": 8192,
        "default_quantization": "int8",
        "local_folder": "Qwen_Qwen2.5-7B-Instruct",
    },
    "mistralai/Mistral-7B-Instruct-v0.3": {
        "family": "mistral",
        "description": "Advanced 7B instruction model with strong document synthesis",
        "context_window": 8192,
        "default_quantization": "int8",
        "local_folder": "mistralai_Mistral-7B-Instruct-v0.3",
    },
    "meta-llama/Llama-3.2-1B-Instruct": {
        "family": "llama",
        "description": "Lightweight 1B Llama-3.2 model for rapid local QA",
        "context_window": 4096,
        "default_quantization": "fp16",
        "local_folder": "meta-llama_Llama-3.2-1B-Instruct",
    },
    "meta-llama/Llama-3.2-3B-Instruct": {
        "family": "llama",
        "description": "Balanced 3B Llama-3.2 model with strong instruction following",
        "context_window": 8192,
        "default_quantization": "fp16",
        "local_folder": "meta-llama_Llama-3.2-3B-Instruct",
    },
    "gguf_model": {
        "family": "gguf",
        "description": "Architectural placeholder for future GGUF/llama.cpp binary weights",
        "context_window": 4096,
        "default_quantization": "int4",
    },
}

MODEL_ALIASES: Dict[str, str] = {
    "test": "deterministic_test",
    "mock": "deterministic_test",
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "phi3": "microsoft/Phi-3-mini-4k-instruct",
    "phi-3": "microsoft/Phi-3-mini-4k-instruct",
    "phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
    "phi3.5": "microsoft/Phi-3.5-mini-instruct",
    "phi-3.5": "microsoft/Phi-3.5-mini-instruct",
    "phi-3.5-mini": "microsoft/Phi-3.5-mini-instruct",
    "phi-3.5-mini-instruct": "microsoft/Phi-3.5-mini-instruct",
    "smollm": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "smollm2": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "smollm2-1.7b": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "smollm2-1.7b-instruct": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "qwen": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5-1.5b-instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen-1.5b-instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "llama": "meta-llama/Llama-3.2-1B-Instruct",
    "llama3": "meta-llama/Llama-3.2-1B-Instruct",
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "gguf": "gguf_model",
    "llama_cpp": "gguf_model",
}


class LLMRegistry:
    """Thread-safe registry maintaining catalog specifications and model class bindings."""

    _instance: LLMRegistry | None = None
    _lock = threading.RLock()

    def __init__(self) -> None:
        self._registry: Dict[str, Type[BaseLocalLLM]] = {}
        self._specs: Dict[str, Dict[str, Any]] = dict(MODEL_SPECIFICATIONS)
        self._aliases: Dict[str, str] = dict(MODEL_ALIASES)
        self._reg_lock = threading.RLock()
        self._register_defaults()

    @classmethod
    def get_instance(cls) -> LLMRegistry:
        """Get or create singleton LLMRegistry."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def _register_defaults(self) -> None:
        """Register built-in model implementations."""
        from rag_engine.generation.models.base_model import GGUFLocalLLM
        from rag_engine.generation.models.deterministic_test_llm import DeterministicTestLLM
        from rag_engine.generation.models.hf_causal_lm import HFLocalLLM

        self.register("deterministic_test", DeterministicTestLLM)
        self.register("mock", DeterministicTestLLM)
        self.register("huggingface", HFLocalLLM)
        self.register("hf", HFLocalLLM)
        self.register("hf_local", HFLocalLLM)
        self.register("gguf", GGUFLocalLLM)
        self.register("llama_cpp", GGUFLocalLLM)

    def register(self, name: str, model_cls: Type[BaseLocalLLM]) -> None:
        """Register a model implementation class under a key."""
        with self._reg_lock:
            key = name.lower().strip()
            self._registry[key] = model_cls
            logger.debug("Registered LLM implementation '%s' -> %s", key, model_cls.__name__)

    def register_model_spec(
        self,
        name: str,
        spec: Dict[str, Any],
        aliases: list[str] | None = None,
    ) -> None:
        """Register or override a model specification in the catalog."""
        with self._reg_lock:
            self._specs[name] = spec
            if aliases:
                for a in aliases:
                    self._aliases[a.lower().strip()] = name

    def resolve_alias(self, name_or_alias: str) -> str:
        """Resolve a friendly alias to its canonical model name or folder."""
        with self._reg_lock:
            clean = name_or_alias.strip()
            return self._aliases.get(clean.lower(), clean)

    def get_specification(self, name_or_alias: str) -> Dict[str, Any] | None:
        """Retrieve model metadata dictionary from the catalog."""
        with self._reg_lock:
            canonical = self.resolve_alias(name_or_alias)
            return self._specs.get(canonical)

    def get(self, name: str) -> Type[BaseLocalLLM] | None:
        """Retrieve model class by name or alias."""
        with self._reg_lock:
            canonical = self.resolve_alias(name).lower()
            if canonical in self._registry:
                return self._registry[canonical]
            return self._registry.get(name.lower().strip())

    def list_models(self) -> list[str]:
        """Return list of all registered model identifiers."""
        with self._reg_lock:
            return sorted(set(list(self._registry.keys()) + list(self._specs.keys()) + list(self._aliases.keys())))

    def contains(self, name: str) -> bool:
        """Check if model name is registered."""
        with self._reg_lock:
            clean = name.strip()
            canonical = self.resolve_alias(clean)
            return (
                canonical in self._registry
                or canonical in self._specs
                or clean.lower() in self._registry
                or clean.lower() in self._aliases
            )

    def find_local_model_path(
        self,
        name_or_alias: str,
        models_dirs: list[Path] | None = None,
    ) -> Path | None:
        """Locate physically present model directory on local disk with valid config and weights."""
        with self._reg_lock:
            clean = name_or_alias.strip()
            canonical = self.resolve_alias(clean)
            if canonical.lower() in ("deterministic_test", "mock", "test"):
                return None

            spec = self.get_specification(canonical) or {}
            local_folder = spec.get("local_folder", canonical.replace("/", "_"))

            base_dirs = list(models_dirs) if models_dirs else [
                Path("models/llm"),
                Path("models/llms"),
                Path("models"),
            ]

            candidates: list[Path] = []
            for b in base_dirs:
                candidates.extend([
                    b / local_folder,
                    b / canonical,
                    b / clean,
                ])
                if canonical == "Qwen/Qwen2.5-1.5B-Instruct":
                    candidates.append(b / "qwen2.5-1.5b-instruct")

            candidates.extend([
                Path(local_folder),
                Path(canonical),
                Path(clean),
            ])
            if canonical == "Qwen/Qwen2.5-1.5B-Instruct":
                candidates.extend([
                    Path("models/llm/qwen2.5-1.5b-instruct"),
                    Path("models/llms/qwen2.5-1.5b-instruct"),
                ])

            for p in candidates:
                if p.exists() and p.is_dir():
                    has_config = (p / "config.json").exists()
                    has_weights = bool(
                        list(p.glob("*.safetensors"))
                        or list(p.glob("*.bin"))
                        or list(p.glob("*.index.json"))
                        or (list((p / "onnx").glob("*.onnx")) if (p / "onnx").exists() else [])
                    )
                    if has_config and has_weights:
                        return p.resolve()
            return None

    def is_model_installed(
        self,
        name_or_alias: str,
        models_dirs: list[Path] | None = None,
    ) -> bool:
        """Check if model weights and config are physically installed on disk."""
        clean = name_or_alias.strip()
        canonical = self.resolve_alias(clean)
        if canonical.lower() in ("deterministic_test", "mock", "test"):
            return True
        return self.find_local_model_path(name_or_alias, models_dirs) is not None


def register_llm(name: str) -> Callable[[Type[BaseLocalLLM]], Type[BaseLocalLLM]]:
    """Decorator for registering custom BaseLocalLLM subclasses."""

    def decorator(cls: Type[BaseLocalLLM]) -> Type[BaseLocalLLM]:
        LLMRegistry.get_instance().register(name, cls)
        return cls

    return decorator
