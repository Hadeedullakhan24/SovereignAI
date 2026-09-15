"""Offline local image generation component using Stable Diffusion 1.5.

Operates 100% offline; never attempts network downloads or connects to external APIs.
Optimized for 6 GB VRAM GPUs (NVIDIA RTX 3050 Laptop GPU) using FP16,
attention slicing, and explicit VRAM release.
"""
from __future__ import annotations

import gc
import hashlib
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Enforce offline guarantees and quiet TensorFlow backend in sandboxed environments
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("DIFFUSERS_OFFLINE", "1")

import sys
import torch
from PIL import Image

def _apply_diffusers_onnx_fix() -> None:
    """Ensure Diffusers routes OnnxRuntimeModel to its built-in dummy implementation.

    On Windows / Python 3.13, onnxruntime may be installed on disk (e.g. by chromadb)
    but fail to load its native C++ DLL (onnxruntime_pybind11_state.pyd).
    Diffusers' import-time detection uses `find_spec`, creating a false-positive availability
    signal that causes Diffusers to attempt importing `diffusers.pipelines.onnx_utils`
    during StableDiffusionPipeline sub-model type resolution.

    Routing `OnnxRuntimeModel` to Diffusers' own official `dummy_onnx_objects`
    allows the pure PyTorch StableDiffusionPipeline to load without requiring a
    working onnxruntime installation.
    """
    try:
        import diffusers
        from diffusers.utils import dummy_onnx_objects

        diffusers.OnnxRuntimeModel = dummy_onnx_objects.OnnxRuntimeModel
        if hasattr(diffusers, "_class_to_module"):
            diffusers._class_to_module["OnnxRuntimeModel"] = "utils.dummy_onnx_objects"
        sys.modules["diffusers.pipelines.onnx_utils"] = dummy_onnx_objects
    except Exception as exc:
        logging.getLogger(__name__).debug("Diffusers ONNX fix application error: %s", exc)

# Apply fix on module import
_apply_diffusers_onnx_fix()

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIFFUSION_MODEL_PATH = _PROJECT_ROOT / "models" / "diffusion" / "stable-diffusion-v1-5"
DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "workspace_sandbox"


class DiffusionCapabilityUnavailable(RuntimeError):
    """Raised when the local diffusion model weights or offline pipeline cannot be loaded."""
    pass


class ImageGenerationError(RuntimeError):
    """Raised when image generation fails during execution."""
    pass


@dataclass
class ImageGenerationResult:
    """Structured result container returned by image generation operations.

    Fields
    ------
    success                 : True if image was successfully generated and saved.
    image_path              : Absolute filesystem path to the saved PNG image, or None.
    prompt                  : The textual prompt used for generation.
    negative_prompt         : Optional negative prompt used.
    width                   : Image width in pixels.
    height                  : Image height in pixels.
    steps                   : Number of denoising inference steps executed.
    guidance_scale          : Classifier-Free Guidance (CFG) scale.
    seed                    : Random seed used, or None if non-deterministic.
    model_name              : Identifier or name of the diffusion model.
    device                  : Hardware device ('cuda' or 'cpu').
    generation_time_seconds : Total latency for generation in seconds.
    peak_vram_mb            : Peak CUDA VRAM allocated during generation (MB), or None.
    error                   : Error message if generation failed, else None.
    metadata                : Additional contextual or diagnostic metadata.
    """

    success: bool
    image_path: Optional[str]
    prompt: str
    negative_prompt: Optional[str] = None
    width: int = 512
    height: int = 512
    steps: int = 30
    guidance_scale: float = 7.5
    seed: Optional[int] = None
    model_name: str = "stable-diffusion-v1-5"
    device: str = "cuda"
    generation_time_seconds: float = 0.0
    peak_vram_mb: Optional[float] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return result as a plain dictionary."""
        return asdict(self)

    def summary(self) -> str:
        """Return human-readable summary string."""
        if self.success:
            vram_str = f" | peak_vram={self.peak_vram_mb:.1f}MB" if self.peak_vram_mb is not None else ""
            return (
                f"[ImageGenerationResult: SUCCESS] path={self.image_path} | "
                f"size={self.width}x{self.height} | steps={self.steps} | "
                f"time={self.generation_time_seconds:.2f}s{vram_str}"
            )
        return f"[ImageGenerationResult: FAILED] error={self.error!r}"


class DiffusionImageGenerator:
    """Local, offline text-to-image generator using Stable Diffusion 1.5.

    Guarantees:
    - Never downloads weights or hits network APIs at runtime (`local_files_only=True`).
    - Uses FP16 precision on CUDA to minimize VRAM.
    - Slices attention to fit comfortably on 6 GB VRAM GPUs.
    - Lazily loads models and exposes explicit `unload()` for memory hygiene.
    """

    MODEL_NAME: str = "stable-diffusion-v1-5"
    MODEL_ID: str = "models/diffusion/stable-diffusion-v1-5"

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: str = "auto",
        torch_dtype: Optional[torch.dtype] = None,
        enable_attention_slicing: bool = True,
        enable_vae_slicing: bool = True,
        default_output_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        if model_path is None:
            self.model_path = DEFAULT_DIFFUSION_MODEL_PATH.resolve()
        else:
            p = Path(model_path)
            self.model_path = p.resolve() if p.is_absolute() else (_PROJECT_ROOT / p).resolve()

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device.lower()

        if torch_dtype is None:
            self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        else:
            self.torch_dtype = torch_dtype

        self.enable_attention_slicing = enable_attention_slicing
        self.enable_vae_slicing = enable_vae_slicing
        self.default_output_dir = Path(default_output_dir).resolve() if default_output_dir else DEFAULT_OUTPUT_DIR.resolve()

        self._pipe: Optional[Any] = None

    @property
    def is_loaded(self) -> bool:
        """Check if pipeline is currently instantiated in memory."""
        return self._pipe is not None

    def is_available(self) -> bool:
        """Check whether local model checkpoint files exist without loading them."""
        if not self.model_path.exists() or not self.model_path.is_dir():
            return False
        model_index = self.model_path / "model_index.json"
        unet_dir = self.model_path / "unet"
        vae_dir = self.model_path / "vae"
        return model_index.is_file() and unet_dir.is_dir() and vae_dir.is_dir()

    def validate_parameters(
        self,
        prompt: str,
        width: int = 512,
        height: int = 512,
        steps: int = 30,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
    ) -> List[str]:
        """Validate generation parameters, returning a list of error messages."""
        errors: List[str] = []

        if not isinstance(prompt, str) or not prompt.strip():
            errors.append("Prompt must be a non-empty string.")

        if not isinstance(width, int) or width <= 0:
            errors.append(f"Width must be a positive integer, got {width}.")
        elif width % 8 != 0:
            errors.append(f"Width must be divisible by 8, got {width}.")
        elif width < 64 or width > 2048:
            errors.append(f"Width must be between 64 and 2048 pixels, got {width}.")

        if not isinstance(height, int) or height <= 0:
            errors.append(f"Height must be a positive integer, got {height}.")
        elif height % 8 != 0:
            errors.append(f"Height must be divisible by 8, got {height}.")
        elif height < 64 or height > 2048:
            errors.append(f"Height must be between 64 and 2048 pixels, got {height}.")

        if not isinstance(steps, int) or steps < 1 or steps > 150:
            errors.append(f"Steps must be an integer between 1 and 150, got {steps}.")

        if not isinstance(guidance_scale, (int, float)) or guidance_scale < 1.0 or guidance_scale > 30.0:
            errors.append(f"Guidance scale must be a number between 1.0 and 30.0, got {guidance_scale}.")

        if seed is not None:
            if not isinstance(seed, int) or seed < 0 or seed > 2**32 - 1:
                errors.append(f"Seed must be an integer between 0 and 2^32 - 1, got {seed}.")

        return errors

    def load(self) -> Any:
        """Load the local Stable Diffusion 1.5 pipeline into memory.

        Raises
        ------
        DiffusionCapabilityUnavailable:
            If required local model weights or configuration files are missing.
        """
        if self._pipe is not None:
            return self._pipe

        if not self.is_available():
            raise DiffusionCapabilityUnavailable(
                f"Local Stable Diffusion 1.5 model not found at '{self.model_path}'. "
                f"Ensure offline weights are staged in '{self.model_path}'."
            )

        logger.info(
            "Loading local Stable Diffusion 1.5 pipeline from %s on device=%s, dtype=%s",
            self.model_path,
            self.device,
            self.torch_dtype,
        )

        try:
            _apply_diffusers_onnx_fix()
            from diffusers import StableDiffusionPipeline

            variant = "fp16" if self.torch_dtype == torch.float16 else None

            pipe = StableDiffusionPipeline.from_pretrained(
                str(self.model_path),
                torch_dtype=self.torch_dtype,
                variant=variant,
                local_files_only=True,
                safety_checker=None,
            )

            pipe = pipe.to(self.device)

            if self.enable_attention_slicing and hasattr(pipe, "enable_attention_slicing"):
                pipe.enable_attention_slicing()

            if self.enable_vae_slicing and hasattr(pipe, "enable_vae_slicing"):
                try:
                    pipe.enable_vae_slicing()
                except Exception as vae_err:
                    logger.debug("VAE slicing could not be enabled: %s", vae_err)

            self._pipe = pipe
            return self._pipe

        except Exception as exc:
            self._pipe = None
            raise DiffusionCapabilityUnavailable(
                f"Failed to load local Stable Diffusion pipeline from '{self.model_path}': {exc}"
            ) from exc

    def unload(self) -> None:
        """Release pipeline from memory and flush GPU cache."""
        if self._pipe is not None:
            logger.info("Unloading diffusion pipeline and reclaiming VRAM.")
            del self._pipe
            self._pipe = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def release_resources(self) -> None:
        """Alias for unload() to conform with workbench resource conventions."""
        self.unload()

    def __enter__(self) -> "DiffusionImageGenerator":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.unload()

    def _determine_output_path(
        self,
        output_dir: Optional[Union[str, Path]],
        filename: Optional[str],
        filename_prefix: Optional[str],
        prompt: str,
        seed: Optional[int],
    ) -> Path:
        """Resolve and ensure target output filepath."""
        target_dir = Path(output_dir).resolve() if output_dir else self.default_output_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        if filename:
            name = filename if filename.lower().endswith(".png") else f"{filename}.png"
            return target_dir / name

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        prefix = filename_prefix.strip() if filename_prefix else "sd15"
        seed_str = f"_s{seed}" if seed is not None else ""
        short_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8]
        generated_name = f"{prefix}_{timestamp}_{short_hash}{seed_str}.png"
        return target_dir / generated_name

    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 512,
        height: int = 512,
        steps: int = 30,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        output_dir: Optional[Union[str, Path]] = None,
        filename: Optional[str] = None,
        filename_prefix: Optional[str] = None,
    ) -> ImageGenerationResult:
        """Generate an image from a natural-language prompt and save it to disk.

        Parameters
        ----------
        prompt          : Natural language description of image to generate.
        negative_prompt : Optional guidance for elements to avoid.
        width           : Width in pixels (must be multiple of 8, default 512).
        height          : Height in pixels (must be multiple of 8, default 512).
        steps           : Denoising inference steps (default 30).
        guidance_scale  : Classifier-Free Guidance scale (default 7.5).
        seed            : Optional deterministic random seed.
        output_dir      : Target directory to save the image (default workspace_sandbox).
        filename        : Explicit filename for output image (default auto-generated).
        filename_prefix : Prefix for auto-generated filename (e.g. 'pump', 'pid').

        Returns
        -------
        ImageGenerationResult:
            Structured record with execution status, file path, timing, and VRAM metrics.
        """
        start_time = time.perf_counter()

        # 1. Parameter Validation
        val_errors = self.validate_parameters(
            prompt=prompt,
            width=width,
            height=height,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        if val_errors:
            error_msg = "; ".join(val_errors)
            logger.warning("DiffusionImageGenerator parameter validation failed: %s", error_msg)
            return ImageGenerationResult(
                success=False,
                image_path=None,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                steps=steps,
                guidance_scale=guidance_scale,
                seed=seed,
                model_name=self.MODEL_NAME,
                device=self.device,
                generation_time_seconds=round(time.perf_counter() - start_time, 4),
                peak_vram_mb=None,
                error=f"Validation error: {error_msg}",
            )

        # 2. Pipeline Loading
        try:
            pipe = self.load()
        except DiffusionCapabilityUnavailable as exc:
            logger.error("DiffusionCapabilityUnavailable: %s", exc)
            return ImageGenerationResult(
                success=False,
                image_path=None,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                steps=steps,
                guidance_scale=guidance_scale,
                seed=seed,
                model_name=self.MODEL_NAME,
                device=self.device,
                generation_time_seconds=round(time.perf_counter() - start_time, 4),
                peak_vram_mb=None,
                error=str(exc),
            )
        except Exception as exc:
            logger.exception("Unexpected error initializing diffusion pipeline: %s", exc)
            return ImageGenerationResult(
                success=False,
                image_path=None,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                steps=steps,
                guidance_scale=guidance_scale,
                seed=seed,
                model_name=self.MODEL_NAME,
                device=self.device,
                generation_time_seconds=round(time.perf_counter() - start_time, 4),
                peak_vram_mb=None,
                error=f"Pipeline initialization failed: {exc}",
            )

        # 3. Execution Setup (Deterministic Generator & Peak VRAM Tracking)
        generator: Optional[torch.Generator] = None
        if seed is not None:
            # CPU generator is reliable and fully portable across devices in diffusers
            generator = torch.Generator(device="cpu").manual_seed(seed)

        is_cuda = self.device == "cuda" and torch.cuda.is_available()
        if is_cuda:
            torch.cuda.reset_peak_memory_stats()

        # 4. Pipeline Inference
        try:
            call_kwargs: Dict[str, Any] = {
                "prompt": prompt,
                "width": width,
                "height": height,
                "num_inference_steps": steps,
                "guidance_scale": guidance_scale,
                "generator": generator,
            }
            if negative_prompt is not None:
                call_kwargs["negative_prompt"] = negative_prompt

            output = pipe(**call_kwargs)

            if not hasattr(output, "images") or not output.images:
                raise ImageGenerationError("Diffusion pipeline returned no output images.")

            pil_image: Image.Image = output.images[0]

            # 5. Output Path Resolution & Image Saving
            target_path = self._determine_output_path(
                output_dir=output_dir,
                filename=filename,
                filename_prefix=filename_prefix,
                prompt=prompt,
                seed=seed,
            )
            pil_image.save(target_path, format="PNG")

            generation_time = round(time.perf_counter() - start_time, 4)
            peak_vram: Optional[float] = None
            if is_cuda:
                peak_vram = round(torch.cuda.max_memory_allocated() / (1024**2), 2)

            logger.info(
                "Successfully generated image: path=%s, time=%.2fs, peak_vram=%s",
                target_path,
                generation_time,
                peak_vram,
            )

            return ImageGenerationResult(
                success=True,
                image_path=str(target_path.resolve()),
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                steps=steps,
                guidance_scale=guidance_scale,
                seed=seed,
                model_name=self.MODEL_NAME,
                device=self.device,
                generation_time_seconds=generation_time,
                peak_vram_mb=peak_vram,
                error=None,
                metadata={
                    "file_size_bytes": target_path.stat().st_size,
                    "model_path": str(self.model_path),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
            )

        except Exception as exc:
            generation_time = round(time.perf_counter() - start_time, 4)
            peak_vram = None
            if is_cuda:
                try:
                    peak_vram = round(torch.cuda.max_memory_allocated() / (1024**2), 2)
                except Exception:
                    pass

            logger.exception("Image generation failed: %s", exc)
            return ImageGenerationResult(
                success=False,
                image_path=None,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                steps=steps,
                guidance_scale=guidance_scale,
                seed=seed,
                model_name=self.MODEL_NAME,
                device=self.device,
                generation_time_seconds=generation_time,
                peak_vram_mb=peak_vram,
                error=str(exc),
            )
