"""Local shared image/text embeddings; never downloads models or contacts services."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
from PIL import Image


class VisionCapabilityUnavailable(RuntimeError):
    """Raised when vision embedding capabilities or weights are unavailable."""
    pass


@runtime_checkable
class VisionEmbedder(Protocol):
    def embed_image(self, image_path: str | Path) -> list[float]: ...
    def embed_text(self, text: str) -> list[float]: ...
    def get_dimension(self) -> int: ...
    def get_model_name(self) -> str: ...
    def get_metadata(self) -> dict[str, str | int]: ...


class OpenCLIPVisionEmbedder:
    """OpenCLIP ViT-B/32 adapter using only manually staged local weights."""

    MODEL_NAME: str = "ViT-B-32"
    MODEL_ID: str = "openclip-vit-b-32"
    DIMENSION: int = 512
    DEFAULT_WEIGHTS_FILENAME: str = "open_clip_model.safetensors"

    def __init__(
        self,
        model_path: str | Path = "models/vision/openclip-vit-b-32",
        device: str = "auto",
    ) -> None:
        self.model_path = Path(model_path).resolve()
        self.device_requested = device
        self.device = "cpu"
        self._model: Any = None
        self._preprocess: Any = None
        self._tokenizer: Any = None

    def _weights_path(self) -> Path:
        """Resolve the path to the local .safetensors model file."""
        if self.model_path.is_file():
            return self.model_path
        candidate = self.model_path / self.DEFAULT_WEIGHTS_FILENAME
        if candidate.is_file():
            return candidate
        alt_candidate = self.model_path / "model.safetensors"
        if alt_candidate.is_file():
            return alt_candidate
        return candidate

    def _load(self) -> None:
        """Load the local model weights without attempting any network downloads."""
        if self._model is not None:
            return

        weights_file = self._weights_path()
        if not weights_file.is_file():
            raise VisionCapabilityUnavailable(
                f"OpenCLIP ViT-B/32 weights are unavailable. Expected local file: {weights_file}. "
                "No model download is attempted."
            )

        try:
            import open_clip  # type: ignore[import-not-found]
            import safetensors.torch  # type: ignore[import-not-found]
            import torch
        except ImportError as exc:
            raise VisionCapabilityUnavailable(
                "OpenCLIP runtime is not installed. Install it separately and manually stage model weights; "
                "runtime installation/download is disabled."
            ) from exc

        self.device = (
            "cuda"
            if self.device_requested == "auto" and torch.cuda.is_available()
            else self.device_requested
        )
        if self.device == "cuda" and not torch.cuda.is_available():
            raise VisionCapabilityUnavailable(
                "Vision embedding requested CUDA, but CUDA is unavailable."
            )

        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                self.MODEL_NAME,
                pretrained=str(weights_file),
                device=self.device,
            )
            if int(getattr(model.visual, "output_dim", self.DIMENSION)) != self.DIMENSION:
                raise VisionCapabilityUnavailable(
                    f"Staged model visual output dimension is not {self.DIMENSION}-D ViT-B/32."
                )

            self._model = model.eval()
            self._preprocess = preprocess
            self._tokenizer = open_clip.get_tokenizer(self.MODEL_NAME)
        except VisionCapabilityUnavailable:
            raise
        except Exception as exc:
            raise VisionCapabilityUnavailable(
                f"Could not load local OpenCLIP model from {weights_file}: {exc}"
            ) from exc

    @staticmethod
    def _normalise(vector: np.ndarray) -> list[float]:
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            raise VisionCapabilityUnavailable("OpenCLIP returned a zero embedding vector.")
        return [float(x) for x in (vector / norm)]

    def embed_image(self, image_path: str | Path) -> list[float]:
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Vision source image not found: {path}")

        self._load()
        if self._model is None or self._preprocess is None:
            raise VisionCapabilityUnavailable("OpenCLIP model was not initialized.")
        import torch

        with Image.open(path) as image:
            image_tensor = self._preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            vector = self._model.encode_image(image_tensor)[0].float().cpu().numpy()

        return self._normalise(vector)

    def embed_text(self, text: str) -> list[float]:
        if not text or not text.strip():
            raise ValueError("Vision text query must not be empty.")

        self._load()
        if self._model is None or self._tokenizer is None:
            raise VisionCapabilityUnavailable("OpenCLIP model was not initialized.")
        import torch

        with torch.inference_mode():
            tokens = self._tokenizer([text.strip()]).to(self.device)
            vector = self._model.encode_text(tokens)[0].float().cpu().numpy()

        return self._normalise(vector)

    def get_dimension(self) -> int:
        return self.DIMENSION

    def get_model_name(self) -> str:
        return self.MODEL_ID

    def get_metadata(self) -> dict[str, Any]:
        return {
            "model_name": self.MODEL_ID,
            "model_path": str(self.model_path),
            "dimension": self.DIMENSION,
            "device": self.device,
        }
