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
                raw_tok: Any = AutoTokenizer.from_pretrained(
                    str(self.model_path),
                    local_files_only=True,
                    trust_remote_code=False,
                )
                if raw_tok is None:
                    raise ModelLoadingError(f"Failed to load tokenizer from {self.model_path}")

                tok: Any = raw_tok
                if tok.pad_token is None:
                    tok.pad_token = tok.eos_token

                # Ensure <|im_end|> and <|endoftext|> are included in eos_token_id
                eos_ids = set()
                if tok.eos_token_id is not None:
                    if isinstance(tok.eos_token_id, list):
                        eos_ids.update(tok.eos_token_id)
                    else:
                        eos_ids.add(tok.eos_token_id)

                for tok_str in ("<|im_end|>", "<|endoftext|>"):
                    tok_id = tok.convert_tokens_to_ids(tok_str)
                    if tok_id is not None and isinstance(tok_id, int) and tok_id != getattr(tok, "unk_token_id", None):
                        eos_ids.add(tok_id)

                self._tokenizer = tok
                self._eos_token_ids = list(eos_ids) if eos_ids else tok.eos_token_id

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
                        else:
                            cpu_cnt = os.cpu_count() or 4
                            configured_threads = min(8, max(2, cpu_cnt // 2))
                    if configured_threads is not None and configured_threads > 0:
                        logger.info("Setting PyTorch CPU inference threads to %d", configured_threads)
                        torch.set_num_threads(configured_threads)

                try:
                    import accelerate  # noqa: F401
                    _has_accelerate = True
                except ImportError:
                    _has_accelerate = False

                load_kwargs: dict[str, Any] = {
                    "local_files_only": True,
                    "dtype": dtype,
                    "trust_remote_code": False,
                    "low_cpu_mem_usage": True,
                }
                if _has_accelerate:
                    load_kwargs["device_map"] = target_device

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
                if not _has_accelerate:
                    self._model = self._model.to(target_device)
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

    def _format_prompt(self, prompt: str) -> str:
        """Format prompt using tokenizer chat template if available and not already formatted."""
        if not prompt or "<|im_start|>" in prompt or not getattr(self._tokenizer, "chat_template", None):
            return prompt

        if "=== SYSTEM INSTRUCTIONS ===" in prompt:
            parts = prompt.split("=== SYSTEM INSTRUCTIONS ===", 1)[1]
            split_markers = [
                "\n=== PREVIOUS CONVERSATION TURNS ===",
                "=== PREVIOUS CONVERSATION TURNS ===",
                "\n=== VERIFIED MRPL REFINERY GROUND TRUTH CONTEXT ===",
                "=== VERIFIED MRPL REFINERY GROUND TRUTH CONTEXT ===",
                "\n=== USER QUERY ===",
                "=== USER QUERY ===",
            ]
            split_at = None
            for m in split_markers:
                if m in parts:
                    split_at = m
                    break
            if split_at:
                sys_part, rest = parts.split(split_at, 1)
                user_msg = (split_at.strip("\n") + "\n" + rest).replace("\nASSISTANT: ", "").strip()
            else:
                sys_part, user_msg = parts.replace("\nASSISTANT: ", "").strip(), ""
            messages = [
                {"role": "system", "content": sys_part.strip()},
                {"role": "user", "content": user_msg},
            ]
            return str(self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        elif "ASSISTANT: " in prompt:
            user_msg = prompt.replace("\nASSISTANT: ", "").strip()
            messages = [{"role": "user", "content": user_msg}]
            return str(self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

        return prompt

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

            formatted_prompt = self._format_prompt(prompt)
            inputs = self._tokenizer(formatted_prompt, return_tensors="pt")
            prompt_tokens = inputs.input_ids.shape[1]

            device = next(self._model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            effective_eos = getattr(self, "_eos_token_ids", None) or self._tokenizer.eos_token_id
            gen_kwargs = {
                "max_new_tokens": cfg.max_new_tokens,
                "temperature": max(cfg.temperature, 1e-5),
                "top_p": cfg.top_p,
                "top_k": getattr(cfg, "top_k", 50),
                "repetition_penalty": cfg.repetition_penalty,
                "do_sample": cfg.temperature > 0.0,
                "pad_token_id": self._tokenizer.pad_token_id,
                "eos_token_id": effective_eos,
            }
            if cfg.stop_sequences:
                gen_kwargs["stop_strings"] = cfg.stop_sequences
                gen_kwargs["tokenizer"] = self._tokenizer

            with torch.no_grad():
                outputs = self._model.generate(**inputs, **gen_kwargs)

            generated_ids = outputs[0][prompt_tokens:]
            completion_tokens = len(generated_ids)
            text = self._tokenizer.decode(generated_ids, skip_special_tokens=True).replace("\r\n", "\n").replace("\r", "\n")

            # Clean trailing stop markers if any leaked through decoding
            if cfg.stop_sequences:
                for stop_seq in cfg.stop_sequences:
                    if stop_seq in text:
                        text = text.split(stop_seq)[0]

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

            formatted_prompt = self._format_prompt(prompt)
            inputs = self._tokenizer(formatted_prompt, return_tensors="pt")
            device = next(self._model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            streamer = TextIteratorStreamer(
                self._tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )

            effective_eos = getattr(self, "_eos_token_ids", None) or self._tokenizer.eos_token_id
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
                "eos_token_id": effective_eos,
            }
            if cfg.stop_sequences:
                gen_kwargs["stop_strings"] = cfg.stop_sequences
                gen_kwargs["tokenizer"] = self._tokenizer

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
