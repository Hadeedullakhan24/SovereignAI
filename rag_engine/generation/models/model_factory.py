"""LLM Factory — Thread-Safe Construction and Lifecycle Management."""

from __future__ import annotations

import logging
from pathlib import Path
import threading
from typing import Any, Dict

from rag_engine.generation.generation_config import GenerationConfig
from rag_engine.generation.generation_exceptions import ModelNotFoundError
from rag_engine.generation.models.base_model import BaseLocalLLM
from rag_engine.generation.models.model_registry import LLMRegistry

logger = logging.getLogger(__name__)


class LLMFactory:
    """Thread-safe factory creating and caching local model instances."""

    _instance: LLMFactory | None = None
    _lock = threading.Lock()

    def __init__(self, registry: LLMRegistry | None = None) -> None:
        self.registry = registry or LLMRegistry.get_instance()
        self._cache: Dict[str, BaseLocalLLM] = {}
        self._cache_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> LLMFactory:
        """Get or create singleton LLMFactory."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def create(
        self,
        model_name: str | None = None,
        config: GenerationConfig | None = None,
        **kwargs: Any,
    ) -> BaseLocalLLM:
        """Create or retrieve a cached local LLM instance.
        
        Args:
            model_name: Name of model architecture, alias, or local folder name.
            config: Central GenerationConfig.
            kwargs: Additional kwargs passed to constructor.
        """
        cfg = config or GenerationConfig()
        raw_name = (model_name or cfg.default_model_name).strip()
        if not raw_name or raw_name.lower() == "auto":
            import sys
            if "pytest" in sys.modules or "unittest" in sys.modules:
                raw_name = "deterministic_test"
            else:
                raw_name = "deterministic_test"
        canonical_name = self.registry.resolve_alias(raw_name)

        with self._cache_lock:
            if canonical_name in self._cache:
                return self._cache[canonical_name]

            # 1. Deterministic test / mock models
            if canonical_name.lower() in ("deterministic_test", "mock", "test"):
                cls = self.registry.get("deterministic_test")
                if cls:
                    instance = cls(model_name=canonical_name, **kwargs)
                    self._cache[canonical_name] = instance
                    return instance

            # 2. Check for specification in registry
            spec = self.registry.get_specification(canonical_name) or {}
            local_folder = spec.get("local_folder", canonical_name.replace("/", "_"))

            # Search in possible local model directories
            found_path = self.registry.find_local_model_path(
                canonical_name,
                [cfg.models_dir, Path("models/llm"), Path("models/llms")],
            )

            if not found_path:
                search_paths = [
                    cfg.models_dir / local_folder,
                    cfg.models_dir / canonical_name,
                    Path("models/llm") / local_folder,
                    Path("models/llm") / canonical_name,
                    Path("models/llms") / local_folder,
                    Path("models/llms") / canonical_name,
                    Path(local_folder),
                    Path(canonical_name),
                ]
                if canonical_name == "Qwen/Qwen2.5-1.5B-Instruct":
                    search_paths.extend([
                        Path("models/llm/qwen2.5-1.5b-instruct"),
                        Path("models/llms/qwen2.5-1.5b-instruct"),
                    ])
                for p in search_paths:
                    if p.exists() and p.is_dir():
                        found_path = p
                        break

            if not self.registry.contains(canonical_name) and found_path is None:
                raise ModelNotFoundError(
                    f"Model '{raw_name}' (canonical: '{canonical_name}') not found in registry and local files do not exist."
                )

            # 3. If local path found or direct class mapping exists
            cls = self.registry.get(canonical_name) or self.registry.get("huggingface")
            if cls is not None:
                target_path = found_path or (Path("models/llm") / local_folder)
                if not target_path.exists():
                    raise ModelNotFoundError(
                        f"Local model path does not exist for '{raw_name}': {target_path.resolve()}."
                    )
                instance = cls(
                    model_path=target_path,
                    device=cfg.device,
                    torch_dtype=cfg.torch_dtype,
                    context_window_size=spec.get("context_window", cfg.budgets.max_context_window),
                    num_threads=getattr(cfg, "num_threads", None),
                    **kwargs,
                )
                self._cache[canonical_name] = instance
                return instance

            raise ModelNotFoundError(
                f"Model '{raw_name}' (canonical: '{canonical_name}') not found in registry and local files do not exist."
            )

    def clear(self) -> None:
        """Clear cached model instances."""
        with self._cache_lock:
            self._cache.clear()
