"""Unit tests for offline local image generator (DiffusionImageGenerator).

All tests are unit-level and strictly offline; they do not perform live GPU diffusion
or download models over the network.
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

from rag_engine.vision.image_generator import (
    DEFAULT_DIFFUSION_MODEL_PATH,
    DiffusionCapabilityUnavailable,
    DiffusionImageGenerator,
    ImageGenerationError,
    ImageGenerationResult,
)


class TestImageGenerationResultStructure:
    """Test the structure, fields, and helper methods of ImageGenerationResult."""

    def test_result_success_fields(self, tmp_path: Path) -> None:
        dummy_file = tmp_path / "test.png"
        dummy_file.write_bytes(b"dummy png bytes")

        result = ImageGenerationResult(
            success=True,
            image_path=str(dummy_file),
            prompt="Centrifugal water pump in refinery",
            negative_prompt="blurry, distorted",
            width=512,
            height=512,
            steps=30,
            guidance_scale=7.5,
            seed=42,
            model_name="stable-diffusion-v1-5",
            device="cuda",
            generation_time_seconds=1.234,
            peak_vram_mb=2048.5,
            error=None,
            metadata={"file_size_bytes": dummy_file.stat().st_size},
        )

        assert result.success is True
        assert result.image_path == str(dummy_file)
        assert result.prompt == "Centrifugal water pump in refinery"
        assert result.negative_prompt == "blurry, distorted"
        assert result.width == 512
        assert result.height == 512
        assert result.steps == 30
        assert result.guidance_scale == 7.5
        assert result.seed == 42
        assert result.device == "cuda"
        assert result.generation_time_seconds == 1.234
        assert result.peak_vram_mb == 2048.5
        assert result.error is None
        assert result.metadata["file_size_bytes"] > 0

        # to_dict conversion
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["success"] is True
        assert d["seed"] == 42
        assert d["width"] == 512

        # summary string
        summary = result.summary()
        assert "SUCCESS" in summary
        assert "512x512" in summary
        assert "peak_vram=2048.5MB" in summary

    def test_result_failure_fields(self) -> None:
        result = ImageGenerationResult(
            success=False,
            image_path=None,
            prompt="Faulty prompt",
            error="CUDA out of memory",
        )
        assert result.success is False
        assert result.image_path is None
        assert "CUDA out of memory" in str(result.error)

        summary = result.summary()
        assert "FAILED" in summary
        assert "CUDA out of memory" in summary


class TestDiffusionImageGeneratorValidation:
    """Test parameter validation rules of DiffusionImageGenerator."""

    @pytest.fixture
    def generator(self, tmp_path: Path) -> DiffusionImageGenerator:
        return DiffusionImageGenerator(
            model_path=tmp_path / "mock_model",
            device="cpu",
            default_output_dir=tmp_path / "sandbox",
        )

    def test_validate_valid_parameters(self, generator: DiffusionImageGenerator) -> None:
        errors = generator.validate_parameters(
            prompt="Industrial gas turbine unit",
            width=512,
            height=512,
            steps=25,
            guidance_scale=8.0,
            seed=12345,
        )
        assert errors == []

    def test_validate_empty_prompt(self, generator: DiffusionImageGenerator) -> None:
        errors = generator.validate_parameters(prompt="   ")
        assert any("Prompt must be a non-empty string" in e for e in errors)

    def test_validate_dimensions_not_divisible_by_eight(self, generator: DiffusionImageGenerator) -> None:
        errors = generator.validate_parameters(prompt="test", width=515, height=512)
        assert any("Width must be divisible by 8" in e for e in errors)

        errors_h = generator.validate_parameters(prompt="test", width=512, height=500)
        assert any("Height must be divisible by 8" in e for e in errors_h)

    def test_validate_dimensions_out_of_range(self, generator: DiffusionImageGenerator) -> None:
        errors_low = generator.validate_parameters(prompt="test", width=32, height=512)
        assert any("between 64 and 2048" in e for e in errors_low)

        errors_high = generator.validate_parameters(prompt="test", width=512, height=4096)
        assert any("between 64 and 2048" in e for e in errors_high)

    def test_validate_steps_out_of_range(self, generator: DiffusionImageGenerator) -> None:
        errors_zero = generator.validate_parameters(prompt="test", steps=0)
        assert any("Steps must be an integer between 1 and 150" in e for e in errors_zero)

        errors_too_high = generator.validate_parameters(prompt="test", steps=200)
        assert any("Steps must be an integer between 1 and 150" in e for e in errors_too_high)

    def test_validate_guidance_scale_out_of_range(self, generator: DiffusionImageGenerator) -> None:
        errors_low = generator.validate_parameters(prompt="test", guidance_scale=0.5)
        assert any("Guidance scale must be a number between 1.0 and 30.0" in e for e in errors_low)

        errors_high = generator.validate_parameters(prompt="test", guidance_scale=35.0)
        assert any("Guidance scale must be a number between 1.0 and 30.0" in e for e in errors_high)

    def test_validate_negative_seed(self, generator: DiffusionImageGenerator) -> None:
        errors = generator.validate_parameters(prompt="test", seed=-5)
        assert any("Seed must be an integer between 0 and 2^32 - 1" in e for e in errors)


class TestMissingLocalModelHandling:
    """Test detection and error handling when local models or weights are missing."""

    def test_is_available_returns_false_for_missing_dir(self, tmp_path: Path) -> None:
        non_existent = tmp_path / "non_existent_model_dir"
        gen = DiffusionImageGenerator(model_path=non_existent)
        assert gen.is_available() is False

    def test_is_available_returns_false_for_incomplete_dir(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "incomplete_model"
        model_dir.mkdir(parents=True)
        # Missing model_index.json, unet, vae
        gen = DiffusionImageGenerator(model_path=model_dir)
        assert gen.is_available() is False

    def test_load_raises_capability_unavailable(self, tmp_path: Path) -> None:
        non_existent = tmp_path / "does_not_exist"
        gen = DiffusionImageGenerator(model_path=non_existent)
        with pytest.raises(DiffusionCapabilityUnavailable) as exc_info:
            gen.load()
        assert "Local Stable Diffusion 1.5 model not found" in str(exc_info.value)

    def test_generate_returns_failure_result_on_missing_model(self, tmp_path: Path) -> None:
        non_existent = tmp_path / "does_not_exist"
        gen = DiffusionImageGenerator(model_path=non_existent)
        result = gen.generate(prompt="Test prompt")
        assert result.success is False
        assert result.image_path is None
        assert "Local Stable Diffusion 1.5 model not found" in str(result.error)


class TestOutputPathHandling:
    """Test output path determination, prefixes, and directory creation."""

    def test_determine_output_path_explicit_filename(self, tmp_path: Path) -> None:
        gen = DiffusionImageGenerator(default_output_dir=tmp_path / "default_out")
        custom_dir = tmp_path / "custom_out"

        path1 = gen._determine_output_path(
            output_dir=custom_dir,
            filename="my_drawing.png",
            filename_prefix=None,
            prompt="sample prompt",
            seed=None,
        )
        assert path1 == custom_dir / "my_drawing.png"
        assert custom_dir.exists()

        path2 = gen._determine_output_path(
            output_dir=custom_dir,
            filename="no_extension",
            filename_prefix=None,
            prompt="sample prompt",
            seed=None,
        )
        assert path2 == custom_dir / "no_extension.png"

    def test_determine_output_path_with_prefix_and_seed(self, tmp_path: Path) -> None:
        gen = DiffusionImageGenerator(default_output_dir=tmp_path / "default_out")
        path = gen._determine_output_path(
            output_dir=None,
            filename=None,
            filename_prefix="pump",
            prompt="industrial pump drawing",
            seed=42,
        )
        assert path.parent == tmp_path / "default_out"
        assert path.name.startswith("pump_")
        assert "_s42.png" in path.name
        assert path.suffix == ".png"


class TestOfflineConfiguration:
    """Verify offline environment settings."""

    def test_offline_environment_variables(self) -> None:
        assert os.environ.get("HF_HUB_OFFLINE") == "1"
        assert os.environ.get("DIFFUSERS_OFFLINE") == "1"
        assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"


class TestMockedPipelineGeneration:
    """Test the full generate() flow with a mocked pipeline producing a real PIL image."""

    def test_generate_mocked_success(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "sandbox"
        gen = DiffusionImageGenerator(
            model_path=tmp_path / "mock_sd",
            device="cpu",
            default_output_dir=output_dir,
        )

        # Create a mock pipeline that returns a synthetic PIL image
        mock_pipe = MagicMock()
        mock_output = MagicMock()
        synthetic_img = Image.new("RGB", (512, 512), color=(0, 128, 255))
        mock_output.images = [synthetic_img]
        mock_pipe.return_value = mock_output

        # Patch load() to return our mock pipeline
        with patch.object(gen, "load", return_value=mock_pipe):
            result = gen.generate(
                prompt="Isometric view of an oil pipeline valve",
                negative_prompt="text, low quality",
                width=512,
                height=512,
                steps=20,
                guidance_scale=7.0,
                seed=999,
                filename="mock_valve.png",
            )

        assert result.success is True
        assert result.error is None
        assert result.image_path is not None
        assert Path(result.image_path).is_file()
        assert result.prompt == "Isometric view of an oil pipeline valve"
        assert result.negative_prompt == "text, low quality"
        assert result.seed == 999
        assert result.width == 512
        assert result.height == 512
        assert result.steps == 20
        assert result.guidance_scale == 7.0
        assert result.generation_time_seconds > 0

        # Verify image was saved as valid PNG and can be opened
        saved_img = Image.open(result.image_path)
        assert saved_img.size == (512, 512)
        assert saved_img.format == "PNG"

        # Verify metadata
        assert result.metadata["file_size_bytes"] > 0
        assert "created_at" in result.metadata

    def test_generate_validation_failure_short_circuit(self, tmp_path: Path) -> None:
        gen = DiffusionImageGenerator(default_output_dir=tmp_path)
        with patch.object(gen, "load") as mock_load:
            result = gen.generate(prompt="", width=512, height=512)
            assert result.success is False
            assert "Validation error" in str(result.error)
            mock_load.assert_not_called()

    def test_generate_pipeline_exception_handling(self, tmp_path: Path) -> None:
        gen = DiffusionImageGenerator(default_output_dir=tmp_path)
        mock_pipe = MagicMock()
        mock_pipe.side_effect = RuntimeError("Internal CUDA execution failure")

        with patch.object(gen, "load", return_value=mock_pipe):
            result = gen.generate(prompt="Test prompt")
            assert result.success is False
            assert result.image_path is None
            assert "Internal CUDA execution failure" in str(result.error)


class TestResourceManagement:
    """Test memory unloading and context management."""

    def test_unload_pipeline(self, tmp_path: Path) -> None:
        gen = DiffusionImageGenerator(model_path=tmp_path)
        gen._pipe = MagicMock()
        assert gen.is_loaded is True

        gen.unload()
        assert gen.is_loaded is False
        assert gen._pipe is None

        # Multiple unloads should be idempotent
        gen.unload()
        assert gen.is_loaded is False

    def test_context_manager_unloads(self, tmp_path: Path) -> None:
        with DiffusionImageGenerator(model_path=tmp_path) as gen:
            gen._pipe = MagicMock()
            assert gen.is_loaded is True

        assert gen.is_loaded is False
