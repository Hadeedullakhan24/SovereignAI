"""Unit tests for Stable Diffusion image generator integration in ToolExecutor.

Verifies:
1. Tool registration and schema discovery.
2. Parameter propagation from ToolExecutor to DiffusionImageGenerator.
3. Successful ToolResult formatting and metadata structure.
4. Artifact staging via stage_artifact_for_backend.
5. Structured error handling for capability unavailable and pipeline failures.
6. Memory hygiene (lazy loading and explicit unload).
7. Isolation: existing vision and document tools remain unaffected.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from agent.router import Capability, RoutingDecision
from agent.tool_executor import (
    DOCUMENT_TOOL_CONTRACTS,
    ImageGeneratorTool,
    ToolExecutor,
    ToolResult,
)
from rag_engine.vision.image_generator import (
    DiffusionCapabilityUnavailable,
    ImageGenerationResult,
)
from rag_engine.generation.prompt.prompt_templates import PromptArchetype


# ── Test Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def mock_gen_result(tmp_path):
    """Create a realistic ImageGenerationResult with a real mock image on disk."""
    img_file = tmp_path / "mock_generated_pump.png"
    img_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 200)

    return ImageGenerationResult(
        success=True,
        image_path=str(img_file),
        prompt="industrial centrifugal pump in refinery",
        negative_prompt="blurry, distorted",
        width=512,
        height=512,
        steps=25,
        guidance_scale=7.5,
        seed=42,
        model_name="stable-diffusion-v1-5",
        device="cuda",
        generation_time_seconds=1.23,
        peak_vram_mb=2688.0,
        error=None,
        metadata={"file_size_bytes": img_file.stat().st_size},
    )


@pytest.fixture
def mock_generator(mock_gen_result):
    """Mock DiffusionImageGenerator double."""
    gen = MagicMock()
    gen.is_available.return_value = True
    gen.generate.return_value = mock_gen_result
    gen.unload = MagicMock()
    return gen


@pytest.fixture
def executor(tmp_path, mock_generator):
    """ToolExecutor with injected mock image generator."""
    return ToolExecutor(sandbox_dir=tmp_path, image_generator=mock_generator)


# ── 1. Contract & Schema Registration ─────────────────────────────────────────

class TestImageToolRegistration:
    """Test tool discovery and schema registration."""

    def test_image_generator_contract_exists_in_document_tool_contracts(self):
        assert "image_generator" in DOCUMENT_TOOL_CONTRACTS
        contract = DOCUMENT_TOOL_CONTRACTS["image_generator"]
        assert contract["format"] == "png"
        assert contract["implemented"] is True
        assert "prompt" in contract["input"]
        assert "width" in contract["input"]
        assert "height" in contract["input"]
        assert "steps" in contract["input"]
        assert "seed" in contract["input"]
        assert "path" in contract["output"]
        assert "filename" in contract["output"]

    def test_image_generator_discoverable_via_tool_contracts(self, executor):
        contracts = executor.tool_contracts()
        assert "image_generator" in contracts
        assert contracts["image_generator"]["format"] == "png"


# ── 2. Tool Execution & Parameter Propagation ─────────────────────────────────

class TestImageToolExecution:
    """Test execution dispatch, argument forwarding, and return values."""

    def test_execute_image_generator_success(self, executor, mock_generator, tmp_path):
        res = executor.execute(
            "image_generator",
            prompt="centrifugal pump P-201",
            negative_prompt="low quality",
            width=512,
            height=512,
            steps=25,
            guidance_scale=7.5,
            seed=42,
            filename="pump_custom.png",
        )

        assert isinstance(res, ToolResult)
        assert res.tool_name == "image_generator"
        assert res.status == "success"
        assert res.is_verified is True
        assert res.is_success is True
        assert res.error is None
        assert res.execution_time_ms > 0

        # Verify generator received correct arguments
        mock_generator.generate.assert_called_once_with(
            prompt="centrifugal pump P-201",
            negative_prompt="low quality",
            width=512,
            height=512,
            steps=25,
            guidance_scale=7.5,
            seed=42,
            output_dir=tmp_path.resolve(),
            filename="pump_custom.png",
            filename_prefix=None,
        )

        # Output payload checks
        output = res.output
        assert output["status"] == "success"
        assert output["success"] is True
        assert output["filename"] == "mock_generated_pump.png"
        assert "mock_generated_pump.png" in output["path"]
        assert output["width"] == 512
        assert output["height"] == 512
        assert output["seed"] == 42
        assert output["generation_time_seconds"] == 1.23
        assert output["peak_vram_mb"] == 2688.0

        # Metadata checks
        assert res.metadata["filename"] == "mock_generated_pump.png"
        assert res.metadata["dimensions"] == {"width": 512, "height": 512}
        assert res.metadata["seed"] == 42
        assert res.metadata["model"] == "stable-diffusion-v1-5"

    def test_execute_via_routing_decision(self, executor, mock_generator):
        decision = RoutingDecision(
            capability=Capability.VISION,
            archetype=PromptArchetype.GENERAL_QA,
            model_record=None,
            capability_available=True,
            tool_name="image_generator",
            use_rag_context=False,
            reason="Routing to image generator",
        )

        res = executor.execute(
            decision,
            task="generate image of pump",
            prompt="centrifugal pump P-201",
            seed=100,
        )

        assert res.status == "success"
        assert res.is_verified is True
        mock_generator.generate.assert_called_once()

    def test_direct_generate_image_method(self, executor, mock_generator):
        out = executor.generate_image(
            prompt="distillation column",
            width=512,
            height=512,
            steps=20,
            seed=999,
        )

        assert out["status"] == "success"
        assert out["success"] is True
        assert out["width"] == 512
        assert out["seed"] == 42  # from mock fixture
        mock_generator.generate.assert_called_once()


# ── 3. Artifact Staging ───────────────────────────────────────────────────────

class TestImageArtifactStaging:
    """Test sandbox containment and backend artifact staging."""

    def test_artifact_staged_for_backend(self, executor, mock_generator, tmp_path):
        with patch("agent.tool_executor.stage_artifact_for_backend") as mock_stage:
            mock_stage.return_value = Path("backend/artifacts/mock_generated_pump.png")

            res = executor.execute(
                "image_generator",
                prompt="pipeline valve",
            )

            assert res.status == "success"
            mock_stage.assert_called_once()
            called_path = mock_stage.call_args[0][0]
            assert called_path.name == "mock_generated_pump.png"
            assert res.output["staged_path"] == "backend\\artifacts\\mock_generated_pump.png" or "backend/artifacts" in res.output["staged_path"]


# ── 4. Error Handling & Capability Unavailable ────────────────────────────────

class TestImageToolErrorHandling:
    """Test error handling and capability unavailability states."""

    def test_missing_model_capability_unavailable_result(self, tmp_path):
        mock_gen = MagicMock()
        mock_gen.is_available.return_value = False
        mock_gen.generate.return_value = ImageGenerationResult(
            success=False,
            image_path=None,
            prompt="pump",
            error="DiffusionCapabilityUnavailable: Model weights not found on disk.",
        )
        mock_gen.unload = MagicMock()

        exec_tool = ToolExecutor(sandbox_dir=tmp_path, image_generator=mock_gen)
        res = exec_tool.execute("image_generator", prompt="pump")

        assert res.status == "capability_unavailable"
        assert res.is_verified is False
        assert res.is_success is False
        assert "DiffusionCapabilityUnavailable" in (res.error or "") or "weights not found" in (res.error or "")

    def test_exception_during_load_returns_capability_unavailable(self, tmp_path):
        mock_gen = MagicMock()
        mock_gen.generate.side_effect = DiffusionCapabilityUnavailable("Weights corrupt or missing.")
        mock_gen.unload = MagicMock()

        exec_tool = ToolExecutor(sandbox_dir=tmp_path, image_generator=mock_gen)
        res = exec_tool.execute("image_generator", prompt="pump")

        assert res.status == "capability_unavailable"
        assert res.is_verified is False
        assert "Weights corrupt or missing" in (res.error or "")

    def test_general_runtime_exception_returns_error_status(self, tmp_path):
        mock_gen = MagicMock()
        mock_gen.generate.side_effect = RuntimeError("CUDA out of memory error during inference.")
        mock_gen.unload = MagicMock()

        exec_tool = ToolExecutor(sandbox_dir=tmp_path, image_generator=mock_gen)
        res = exec_tool.execute("image_generator", prompt="pump")

        assert res.status == "error"
        assert res.is_verified is False
        assert "CUDA out of memory" in (res.error or "")

    def test_parameter_validation_failure_returns_error_status(self, tmp_path):
        mock_gen = MagicMock()
        mock_gen.is_available.return_value = True
        mock_gen.generate.return_value = ImageGenerationResult(
            success=False,
            image_path=None,
            prompt="",
            error="Validation error: Prompt must be a non-empty string.",
        )
        mock_gen.unload = MagicMock()

        exec_tool = ToolExecutor(sandbox_dir=tmp_path, image_generator=mock_gen)
        res = exec_tool.execute("image_generator", prompt="")

        assert res.status == "error"
        assert res.is_verified is False
        assert "Validation error" in (res.error or "")


# ── 5. Memory Hygiene ─────────────────────────────────────────────────────────

class TestImageToolMemoryHygiene:
    """Test lazy loading and prompt-cycle unloading."""

    def test_unload_called_after_generation_by_default(self, executor, mock_generator):
        executor.execute("image_generator", prompt="cooling tower")
        mock_generator.unload.assert_called_once()

    def test_unload_called_even_if_generation_fails(self, tmp_path):
        mock_gen = MagicMock()
        mock_gen.generate.side_effect = ValueError("Inference failed.")
        mock_gen.unload = MagicMock()

        exec_tool = ToolExecutor(sandbox_dir=tmp_path, image_generator=mock_gen)
        exec_tool.execute("image_generator", prompt="cooling tower")

        mock_gen.unload.assert_called_once()


# ── 6. Isolation ──────────────────────────────────────────────────────────────

class TestImageToolIsolation:
    """Verify other tools remain unaffected."""

    def test_calculator_still_functions(self, executor):
        res = executor.execute("calculator", expression="25 * 4")
        assert res.status == "success"
        assert res.output["result"] == 100

    def test_file_manager_still_functions(self, executor, tmp_path):
        res = executor.execute("file_manager", action="write", filename="notes.txt", content="test")
        assert res.status == "success"
        assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "test"
