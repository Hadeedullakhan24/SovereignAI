"""Unit and integration tests for automatic image generation intent and routing.

Verifies:
1. Positive intent classification (routing natural-language image generation prompts).
2. Negative intent preservation (preventing OCR, inspection, RAG, and calculation queries from false-positive routing).
3. Parameter extraction (dimensions, steps, guidance scale, seed, negative prompt, filename, prompt cleanup).
4. SovereignAgent end-to-end dispatch and response formatting with mocked DiffusionImageGenerator.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from agent.agent import AgentResponse, SovereignAgent
from agent.model_registry import AgentModelRegistry, ModelRecord
from agent.router import (
    Capability,
    RoutingDecision,
    TaskRouter,
    extract_image_generation_params,
    get_router,
)
from agent.tool_executor import ToolExecutor, ToolResult
from rag_engine.vision.image_generator import ImageGenerationResult


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def router():
    """Default TaskRouter instance."""
    return TaskRouter()


@pytest.fixture
def mock_image_gen_result(tmp_path):
    """Realistic ImageGenerationResult for agent execution testing."""
    img_file = tmp_path / "mock_pump_output.png"
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
        seed=1234,
        model_name="stable-diffusion-v1-5",
        device="cuda",
        generation_time_seconds=1.45,
        peak_vram_mb=2685.0,
        error=None,
        metadata={"file_size_bytes": img_file.stat().st_size},
    )


@pytest.fixture
def mock_generator(mock_image_gen_result):
    """Mock DiffusionImageGenerator."""
    gen = MagicMock()
    gen.is_available.return_value = True
    gen.generate.return_value = mock_image_gen_result
    gen.unload = MagicMock()
    return gen


@pytest.fixture
def agent(tmp_path, mock_generator):
    """SovereignAgent with mock tool executor and image generator."""
    executor = ToolExecutor(sandbox_dir=tmp_path, image_generator=mock_generator)
    return SovereignAgent(sandbox_dir=tmp_path, tool_executor=executor)


# ── 1. Positive Routing Tests ─────────────────────────────────────────────────

class TestPositiveImageGenerationRouting:
    """Test queries that MUST route to Capability.IMAGE_GENERATION."""

    @pytest.mark.parametrize(
        "query, expected_tool",
        [
            ("Generate an image of an industrial centrifugal pump", "image_generator"),
            ("Create a diagram of a crude distillation column", "image_generator"),
            ("Make an image showing an oil refinery pipeline at sunset", "image_generator"),
            ("Render a visual representation of heat exchanger E-101", "image_generator"),
            ("Synthesize a picture of a high-pressure storage sphere", "image_generator"),
            ("Draw a schematic image of a boiler feed water system", "image_generator"),
            ("Produce an illustration of a flare stack", "image_generator"),
            ("Paint a rendering of refinery storage tanks", "image_generator"),
            ("Sketch an image of a distillation tray", "image_generator"),
            ("Illustrate a picture of a catalytic cracking unit", "image_generator"),
            ("Run text-to-image for an offshore drilling platform", "image_generator"),
            ("Generate a stable diffusion image of industrial piping", "image_generator"),
            ("txt2img: heavy crude oil pump in petrochemical plant", "image_generator"),
        ],
    )
    def test_positive_queries_route_to_image_generation(self, router, query, expected_tool):
        decision = router.route(query)
        assert decision.capability == Capability.IMAGE_GENERATION
        assert decision.tool_name == expected_tool
        assert decision.use_rag_context is False
        assert decision.is_actionable() is True


# ── 2. Negative Intent Preservation Tests ─────────────────────────────────────

class TestNegativeIntentPreservation:
    """Ensure non-generative queries do NOT route to image generation."""

    @pytest.mark.parametrize(
        "query, expected_capability, expected_tool",
        [
            ("Analyze this drawing datasets/engineering_drawings/PID/PID_002_Process_Example.jpg", Capability.VISION, "vision_inspector"),
            ("Read the nameplate from this image of the pump P-101.", Capability.VISION, "vision_inspector"),
            ("What does this P&ID show?", Capability.VISION, "vision_inspector"),
            ("Find images similar to this photo", Capability.VISION, "vision_inspector"),
            ("Inspect the scanned document for weld defects", Capability.VISION, "vision_inspector"),
            ("OCR extraction of table from image_01.png", Capability.VISION, "vision_inspector"),
            ("Calculate the relief valve sizing for vessel V-305.", Capability.CALCULATION, "calculator"),
            ("Verify MAWP calculation for column C-101", Capability.CALCULATION, "calculator"),
            ("Export inspection report as actual pdf artifact", Capability.DOCUMENT_GENERATION, "pdf_generator"),
            ("Generate excel spreadsheet report for pump tags", Capability.DOCUMENT_GENERATION, "xlsx_generator"),
            ("Write a Python script to parse the inspection report.", Capability.CODING, "code_interpreter"),
            ("What is the design operating pressure of centrifugal pump P-203?", Capability.RAG, "rag_pipeline"),
        ],
    )
    def test_negative_queries_do_not_route_to_image_generation(self, router, query, expected_capability, expected_tool):
        decision = router.route(query)
        assert decision.capability == expected_capability
        assert decision.tool_name == expected_tool
        assert decision.capability != Capability.IMAGE_GENERATION


# ── 3. Parameter Extraction Tests ─────────────────────────────────────────────

class TestParameterExtraction:
    """Test natural-language extraction of image generation parameters."""

    def test_dimensions_extraction_inline(self):
        text = "Generate an image of a refinery pump 768x512 steps: 25"
        params = extract_image_generation_params(text)
        assert params["width"] == 768
        assert params["height"] == 512
        assert params["steps"] == 25

    def test_seed_and_guidance_and_negative_prompt(self):
        text = "Create a picture of a boiler --seed 42 --guidance 8.5 --negative blurry, noisy"
        params = extract_image_generation_params(text)
        assert params["seed"] == 42
        assert params["guidance_scale"] == 8.5
        assert params["negative_prompt"] == "blurry, noisy"
        assert "boiler" in params["prompt"].lower()

    def test_save_as_filename_extraction(self):
        text = "Make an image showing heat exchanger E-101 save as heat_exchanger.png"
        params = extract_image_generation_params(text)
        assert params["filename"] == "heat_exchanger.png"
        assert "heat exchanger e-101" in params["prompt"].lower()

    def test_prompt_cleanup_strips_generative_triggers(self):
        text = "Please generate an image of an industrial centrifugal pump in refinery"
        params = extract_image_generation_params(text)
        assert params["prompt"].lower() == "industrial centrifugal pump in refinery"

    def test_explicit_kwargs_take_precedence(self):
        text = "Generate an image of a pump 512x512 --steps 20"
        params = extract_image_generation_params(
            text,
            width=1024,
            height=768,
            steps=50,
            seed=999,
            filename="custom_pump.png",
        )
        assert params["width"] == 1024
        assert params["height"] == 768
        assert params["steps"] == 50
        assert params["seed"] == 999
        assert params["filename"] == "custom_pump.png"


# ── 4. SovereignAgent End-to-End Execution Tests ──────────────────────────────

class TestSovereignAgentImageGenerationDispatch:
    """Test SovereignAgent handle() execution for image generation."""

    def test_agent_handle_generates_image_successfully(self, agent, mock_generator):
        query = "Generate an image of an industrial centrifugal pump 512x512 steps: 25 save as test_pump.png"
        response: AgentResponse = agent.handle(query)

        assert response.status == "completed"
        assert response.is_verified is True
        assert response.requires_approval is False
        assert response.error is None
        assert "image_generator" in response.execution_trace

        assert len(response.reasoning_steps) == 1
        step = response.reasoning_steps[0]
        assert step["name"] == "Local Stable Diffusion Image Synthesis"
        assert step["status"] == "success"
        assert step["is_verified"] is True

        assert isinstance(response.output, dict)
        assert response.output.get("filename") == "mock_pump_output.png"
        assert response.output.get("width") == 512
        assert response.output.get("height") == 512

        # Verify mock generator called with correct params
        mock_generator.generate.assert_called_once()
        call_kwargs = mock_generator.generate.call_args[1]
        assert call_kwargs["width"] == 512
        assert call_kwargs["height"] == 512
        assert call_kwargs["steps"] == 25
        assert call_kwargs["filename"] == "test_pump.png"

    def test_agent_handle_preserves_vision_fast_path_for_existing_images(self, agent, tmp_path):
        sample_img = tmp_path / "sample_inspection.jpg"
        sample_img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

        with patch.object(agent.tool_executor, "execute") as mock_exec:
            mock_exec.return_value = ToolResult(
                tool_name="vision_inspector",
                status="success",
                output={"text": "Inspection verified", "routing_decision": "inspection_report"},
                is_verified=True,
            )
            resp = agent.handle(f"Analyze this image {sample_img}")
            assert resp.status == "completed"
            assert "vision_inspector" in resp.execution_trace
            mock_exec.assert_called_once()
            assert mock_exec.call_args[0][0] == "vision_inspector"


# ── 5. RAGPipeline & CLI ask.py End-to-End Image Generation Tests ─────────────

class TestRAGPipelineAndCLIImageGeneration:
    """Test RAGPipeline.answer() and scripts/ask.py image generation routing."""

    def test_rag_pipeline_answer_routes_to_image_generator_without_document_retrieval(self, mock_generator):
        from rag_engine.generation.generation_config import GenerationConfig
        from rag_engine.pipeline.rag_pipeline import RAGPipeline

        mock_retrieval = MagicMock()
        config = GenerationConfig(default_model_name="deterministic_test", cache_enabled=False)
        pipeline = RAGPipeline(retrieval_pipeline=mock_retrieval, config=config)

        with patch("agent.tool_executor.ImageGeneratorTool.generator", new=mock_generator):
            query = "Generate an image of a large industrial heat exchanger inside a refinery, detailed realistic engineering visualization."
            response = pipeline.answer(query)

            # Assert document retrieval was NOT called
            mock_retrieval.retrieve.assert_not_called()

            # Assert artifact was generated
            assert response.artifact is not None
            assert response.artifact.artifact_type == "png"
            assert response.artifact.status == "success"
            assert response.is_grounded is True
            assert "mock_pump_output.png" in response.artifact.filename

            # Assert CLI clean output contains artifact details and no sources
            clean_out = response.format_clean_cli_output()
            assert "[Generated Artifact]" in clean_out
            assert "PNG" in clean_out

    def test_cli_ask_main_handles_image_generation_query(self, mock_generator, capsys):
        from scripts.ask import main

        with patch("agent.tool_executor.ImageGeneratorTool.generator", new=mock_generator):
            query = "Generate an image of a large industrial heat exchanger inside a refinery, detailed realistic engineering visualization."
            main(["--model", "deterministic_test", query])

            captured = capsys.readouterr().out
            assert "[Generated Artifact]" in captured or "mock_pump_output.png" in captured
            assert "PNG" in captured
