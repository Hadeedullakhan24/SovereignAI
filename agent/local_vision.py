"""Local-only Qwen2.5-VL runtime used by the agent vision tool.

The adapter intentionally performs no model download and loads lazily, because
the model is large.  It is a real inference adapter: callers receive a
structured unavailable/error condition if the installed runtime or device
cannot load the locally staged weights; it never substitutes generated text.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional


class LocalVisionUnavailable(RuntimeError):
    """Raised when local Qwen2.5-VL inference cannot be performed."""


class QwenVisionRuntime:
    def __init__(self, model_path: Path, device: Optional[str] = None) -> None:
        self.model_path = Path(model_path).resolve()
        self.device = device or os.environ.get("AGENT_VISION_DEVICE", "auto")
        self._model: Any = None
        self._processor: Any = None

    def availability(self) -> Dict[str, Any]:
        config = self.model_path / "config.json"
        weights = list(self.model_path.glob("*.safetensors")) + list(self.model_path.glob("pytorch_model*.bin"))
        return {"available": config.is_file() and bool(weights), "path": str(self.model_path)}

    def _load(self) -> None:
        if self._model is not None:
            return
        if not self.availability()["available"]:
            raise LocalVisionUnavailable(f"Qwen2.5-VL weights are not available at {self.model_path}.")
        try:
            import torch
            from transformers import AutoModelForVision2Seq, AutoProcessor
        except ImportError as exc:
            raise LocalVisionUnavailable("Local VLM runtime requires installed torch and transformers packages.") from exc

        if self.device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            resolved_device = self.device
        if resolved_device == "cuda" and not torch.cuda.is_available():
            raise LocalVisionUnavailable("AGENT_VISION_DEVICE=cuda but CUDA is not available.")
        dtype = torch.float16 if resolved_device == "cuda" else torch.float32
        try:
            self._processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True)
            self._model = AutoModelForVision2Seq.from_pretrained(
                self.model_path, local_files_only=True, torch_dtype=dtype
            ).to(resolved_device).eval()
            self.device = resolved_device
        except Exception as exc:
            self._model = None
            self._processor = None
            raise LocalVisionUnavailable(f"Unable to load local Qwen2.5-VL model: {exc}") from exc

    def answer(self, image_path: Path, prompt: str, max_new_tokens: int = 256) -> Dict[str, Any]:
        self._load()
        try:
            from PIL import Image
            import torch
            image = Image.open(image_path).convert("RGB")
            messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
            text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self._processor(text=[text], images=[image], return_tensors="pt", padding=True)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.inference_mode():
                generated = self._model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            generated = generated[:, inputs["input_ids"].shape[1]:]
            answer = self._processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
            if not answer:
                raise LocalVisionUnavailable("Local Qwen2.5-VL returned an empty response.")
            return {"status": "success", "answer": answer, "model": "Qwen/Qwen2.5-VL-3B-Instruct", "device": self.device}
        except LocalVisionUnavailable:
            raise
        except Exception as exc:
            raise LocalVisionUnavailable(f"Local Qwen2.5-VL inference failed: {exc}") from exc

    def unload(self) -> None:
        self._model = None
        self._processor = None
