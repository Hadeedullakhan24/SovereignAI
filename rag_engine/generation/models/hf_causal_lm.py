"""Hugging Face In-Process Causal LM — 100% Offline Local Model Inference.

Loads open-weight LLMs from the local filesystem (`models/llms/<name>/`) using
Hugging Face Transformers with strict `local_files_only=True` air-gapped guarantees.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
import threading
from typing import Any, Iterator, Optional

from rag_engine.generation.generation_config import ModelInferenceConfig
from rag_engine.generation.generation_exceptions import (
    GenerationInferenceError,
    ModelLoadingError,
    ModelNotFoundError,
)
from rag_engine.generation.models.base_model import (
    BaseLLM,
    LLMGenerationOutput,
    QuantizationType,
)

logger = logging.getLogger(__name__)


class HFLocalLLM(BaseLLM):
    """In-process Causal Language Model running on local CPU/GPU hardware."""

    def __init__(
        self,
        model_path: Path | str,
        device: str = "auto",
        torch_dtype: str = "float16",
        quantization: QuantizationType = QuantizationType.FP16,
        context_window_size: int = 4096,
        num_threads: Optional[int] = None,
    ) -> None:
        path = Path(model_path)
        if not path.exists():
            raise ModelNotFoundError(
                f"Local model weights not found at: {path.resolve()}. "
                "Download open weights locally via `scripts/download_llm_models.py`."
            )

        super().__init__(
            model_name=path.name,
            context_window_size=context_window_size,
            quantization=quantization,
        )
        self.model_path = path
        self.device = device
        self.torch_dtype = torch_dtype
        self.num_threads = num_threads

        self._tokenizer: Any = None
        self._model: Any = None
        self._load_lock = threading.Lock()
        self._load_time_ms: float = 0.0

    @property
    def model_load_time_ms(self) -> float:
        """Time taken to load model into memory in milliseconds."""
        return self._load_time_ms

    def _lazy_load(self) -> None:
        """Load tokenizer and model weights strictly from local disk."""
        with self._load_lock:
            if self._model is not None and self._tokenizer is not None:
                return

            import time
            start = time.perf_counter()
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer

                logger.info("Loading local LLM tokenizer from %s [local_files_only=True]", self.model_path)
                self._tokenizer = AutoTokenizer.from_pretrained(
                    str(self.model_path),
                    local_files_only=True,
                    trust_remote_code=False,
                )

                if self._tokenizer.pad_token is None:
                    self._tokenizer.pad_token = self._tokenizer.eos_token

                target_device = self.device if self.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

                # Map dtype and quantization
                dtype_map = {
                    "float16": torch.float16,
                    "bfloat16": torch.bfloat16,
                    "float32": torch.float32,
                    QuantizationType.FP16: torch.float16,
                    QuantizationType.BF16: torch.bfloat16,
                    QuantizationType.FP32: torch.float32,
                }
                dtype = dtype_map.get(self.quantization, dtype_map.get(self.torch_dtype, torch.float16))
                if target_device == "cpu":
                    # On CPU without hardware bfloat16/fp16 acceleration, float32 ensures AVX2 GEMM support
                    dtype = torch.float32
                    import os
                    configured_threads = self.num_threads
                    if configured_threads is None:
                        env_t = os.environ.get("TORCH_NUM_THREADS")
                        if env_t and env_t.strip().isdigit():
                            configured_threads = int(env_t.strip())
                    if configured_threads is not None and configured_threads > 0:
                        logger.info("Setting PyTorch CPU inference threads to %d", configured_threads)
                        torch.set_num_threads(configured_threads)

                load_kwargs: dict[str, Any] = {
                    "local_files_only": True,
                    "dtype": dtype,
                    "device_map": target_device,
                    "trust_remote_code": False,
                    "low_cpu_mem_usage": True,
                }

                # Optional 8-bit / 4-bit quantization support via bitsandbytes
                if self.quantization == QuantizationType.INT8:
                    load_kwargs["load_in_8bit"] = True
                elif self.quantization == QuantizationType.INT4:
                    load_kwargs["load_in_4bit"] = True

                logger.info(
                    "Loading local LLM weights from %s [device=%s, dtype=%s, quantization=%s]",
                    self.model_path,
                    self.device,
                    dtype,
                    self.quantization,
                )
                self._model = AutoModelForCausalLM.from_pretrained(
                    str(self.model_path),
                    **load_kwargs,
                )
                self._model.eval()
                self._load_time_ms = (time.perf_counter() - start) * 1000.0
                logger.info(
                    "Local LLM %s loaded successfully in %.2f ms",
                    self.model_name,
                    self._load_time_ms,
                )

            except Exception as e:
                raise ModelLoadingError(
                    f"Failed to load local model weights from {self.model_path}: {e}"
                ) from e

    def count_tokens(self, text: str) -> int:
        """Calculate exact token count using the active tokenizer."""
        self._lazy_load()
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    def generate(
        self,
        prompt: str,
        config: ModelInferenceConfig | None = None,
    ) -> LLMGenerationOutput:
        """Generate full completion synchronously."""
        self._lazy_load()
        cfg = config or ModelInferenceConfig()

        try:
            import torch

            inputs = self._tokenizer(prompt, return_tensors="pt")
            prompt_tokens = inputs.input_ids.shape[1]

            device = next(self._model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            gen_kwargs = {
                "max_new_tokens": cfg.max_new_tokens,
                "temperature": max(cfg.temperature, 1e-5),
                "top_p": cfg.top_p,
                "top_k": getattr(cfg, "top_k", 50),
                "repetition_penalty": cfg.repetition_penalty,
                "do_sample": cfg.temperature > 0.0,
                "pad_token_id": self._tokenizer.pad_token_id,
                "eos_token_id": self._tokenizer.eos_token_id,
            }

            with torch.no_grad():
                outputs = self._model.generate(**inputs, **gen_kwargs)

            generated_ids = outputs[0][prompt_tokens:]
            completion_tokens = len(generated_ids)
            text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)

            return LLMGenerationOutput(
                text=text.strip(),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                finish_reason="stop",
            )
        except Exception as e:
            raise GenerationInferenceError(f"In-process generation failed: {e}") from e

    def stream_generate(
        self,
        prompt: str,
        config: ModelInferenceConfig | None = None,
    ) -> Iterator[str]:
        """Stream generated tokens iteratively using TextIteratorStreamer."""
        self._lazy_load()
        cfg = config or ModelInferenceConfig()

        try:
            from transformers import TextIteratorStreamer

            inputs = self._tokenizer(prompt, return_tensors="pt")
            device = next(self._model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            streamer = TextIteratorStreamer(
                self._tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )

            gen_kwargs = {
                **inputs,
                "streamer": streamer,
                "max_new_tokens": cfg.max_new_tokens,
                "temperature": max(cfg.temperature, 1e-5),
                "top_p": cfg.top_p,
                "top_k": getattr(cfg, "top_k", 50),
                "repetition_penalty": cfg.repetition_penalty,
                "do_sample": cfg.temperature > 0.0,
                "pad_token_id": self._tokenizer.pad_token_id,
                "eos_token_id": self._tokenizer.eos_token_id,
            }

            thread = threading.Thread(target=self._model.generate, kwargs=gen_kwargs)
            thread.start()

            for new_text in streamer:
                yield new_text

            thread.join()
        except Exception as e:
            raise GenerationInferenceError(f"In-process streaming generation failed: {e}") from e

    def unload(self) -> None:
        """Unload model and free GPU/CPU memory."""
        with self._load_lock:
            self._model = None
            self._tokenizer = None
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass


# Backward compatibility alias
HuggingFaceCausalLM = HFLocalLLM
