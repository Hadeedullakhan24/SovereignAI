"""Comprehensive unit, integration, and end-to-end tests for central intent classification.

Verifies:
1. Previously failing natural visual requests route correctly to Capability.IMAGE_GENERATION.
2. Diverse natural phrasings of generative visual requests (appearance, perspective, intrinsic verbs).
3. Negative preservation: Analytical visual inspection, OCR, defect detection route to Capability.VISION.
4. Negative preservation: Document questions, SOPs, safety standards route to Capability.RAG.
5. Negative preservation: Formulas and sizing route to Capability.CALCULATION.
6. End-to-end CLI ask execution path: Verifies scripts/ask.py executes ImageGeneratorTool with
   use_rag_context=False and zero Qdrant document citations.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from agent.intent import (
    ActionType,
    IntentDecision,
    OutputModality,
    TargetState,
    classify_intent,
)
from agent.router import Capability, TaskRouter, get_router
from rag_engine.vision.image_generator import ImageGenerationResult


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def router():
    """Default TaskRouter instance."""
    return TaskRouter()


@pytest.fixture
def mock_gen_result(tmp_path):
    """Realistic ImageGenerationResult for mock diffusion execution."""
    out_file = tmp_path / "mock_generated_refinery.png"
    out_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 300)
    return ImageGenerationResult(
        success=True,
        image_path=str(out_file),
        prompt="modern oil refinery from above",
        negative_prompt="blurry, low quality",
        width=512,
        height=512,
        steps=20,
        guidance_scale=7.5,
        seed=42,
        model_name="stable-diffusion-v1-5",
        device="cuda",
        generation_time_seconds=1.2,
        peak_vram_mb=2450.0,
        error=None,
        metadata={"file_size_bytes": out_file.stat().st_size},
    )


@pytest.fixture
def mock_generator(mock_gen_result):
    """Mock DiffusionImageGenerator."""
    gen = MagicMock()
    gen.is_available.return_value = True
    gen.generate.return_value = mock_gen_result
    gen.unload = MagicMock()
    return gen


# ── 1. Reported Failing Prompts Now Route to Image Generation ────────────────

class TestReportedProblemPrompts:
    """Explicitly verify the three user-reported prompts route to IMAGE_GENERATION."""

    @pytest.mark.parametrize(
        "query",
        [
            "Can you make a picture of a refinery control room?",
            "Show me what a modern oil refinery might look like from above.",
            "Can you visualize a distillation tower and the equipment around it?",
            "Generate a realistic industrial centrifugal pump inside an oil refinery.",
        ],
    )
    def test_reported_prompts_route_to_image_generation(self, router, query):
        decision = router.route(query)
        assert decision.capability == Capability.IMAGE_GENERATION, (
            f"Expected Capability.IMAGE_GENERATION for '{query}', got {decision.capability}. Reason: {decision.reason}"
        )
        assert decision.tool_name == "image_generator"
        assert decision.use_rag_context is False
        assert decision.is_actionable() is True


# ── 2. Generalization: Natural Phrasings for New Visual Synthesis ─────────────

class TestNaturalVisualGenerationGeneralization:
    """Verify generalization across diverse linguistic constructs without explicit 'image' keywords."""

    @pytest.mark.parametrize(
        "query",
        [
            # Appearance constructs: "show me what ... looks like / might look like / how ... looks"
            "Show me what an offshore petroleum platform looks like during a storm.",
            "Can you show me how a crude distillation unit looks from the inside?",
            "What would a catalytic cracking unit look like in 3D?",
            "How might a storage tank farm look at dusk?",
            # Perspective view constructs: aerial, bird's eye, isometric
            "Aerial view of crude oil storage tanks.",
            "Bird's eye view of the refinery pipeline manifold.",
            "Satellite view of an offshore petrochemical terminal.",
            "Isometric view of heat exchanger E-201.",
            # Intrinsic visual generation verbs: visualize, depict, render, illustrate, draw, sketch, picture
            "Can you visualize the reactor cooling loop in realistic detail?",
            "Depict an industrial compressor station inside a refinery plant.",
            "Render a distillation tower surrounded by control valves and pipe racks.",
            "Illustrate a flare stack burning excess gas at night.",
            "Sketch an emergency shutdown valve on a high-pressure line.",
            "Draw a concept of a spherical pressurized gas storage tank.",
            "Could you picture what the main distillation column looks like?",
            # Creative synthesis
            "Synthesize a 3D visual of a centrifugal pump skid.",
            "Create a realistic rendering of a refinery control room with SCADA monitors.",
        ],
    )
    def test_natural_generative_requests_route_to_image_generation(self, router, query):
        intent = classify_intent(query)
        assert intent.is_image_generation is True
        assert intent.action == ActionType.CREATE_NEW
        assert intent.output_modality == OutputModality.NEW_VISUAL
        assert intent.target_state == TargetState.NEW_SYNTHESIS

        decision = router.route(query)
        assert decision.capability == Capability.IMAGE_GENERATION
        assert decision.tool_name == "image_generator"
        assert decision.use_rag_context is False


# ── 3. Negative Preservation: Existing Visual Analysis & OCR ─────────────────

class TestNegativeVisualAnalysisPreservation:
    """Verify that analytical requests on existing visuals are NEVER routed to image generation."""

    @pytest.mark.parametrize(
        "query",
        [
            "Analyze this image.",
            "What does this P&ID show?",
            "Read the nameplate from this image.",
            "Describe the defects visible in this image.",
            "Inspect this engineering drawing.",
            "Extract the text from this scanned drawing.",
            "Find images similar to this.",
            "Inspect the scanned document for weld defects.",
            "What anomalies are shown in this radiograph scan?",
            "OCR extraction from datasets/engineering_drawings/PID/PID_002.jpg",
        ],
    )
    def test_analytical_visual_requests_route_to_vision(self, router, query):
        intent = classify_intent(query)
        assert intent.is_image_generation is False
        assert intent.capability_name == "vision"
        assert intent.action == ActionType.ANALYZE_EXISTING
        assert intent.output_modality == OutputModality.EXISTING_VISUAL_ANALYSIS
        assert intent.target_state == TargetState.EXISTING_ARTIFACT

        decision = router.route(query)
        assert decision.capability == Capability.VISION
        assert decision.tool_name == "vision_inspector"
        assert decision.capability != Capability.IMAGE_GENERATION


# ── 4. Negative Preservation: Document RAG & Calculations ─────────────────────

class TestNegativeRAGAndCalculationPreservation:
    """Verify that standard RAG queries and calculations are NEVER routed to image generation."""

    @pytest.mark.parametrize(
        "query, expected_capability, expected_tool",
        [
            ("What is the design operating pressure of centrifugal pump P-203?", Capability.RAG, "rag_pipeline"),
            ("Show me the safety procedure for hot work permits.", Capability.RAG, "rag_pipeline"),
            ("Explain the crude distillation process according to OISD standards.", Capability.RAG, "rag_pipeline"),
            ("What are the statutory requirements for confined space entry?", Capability.RAG, "rag_pipeline"),
            ("List the preventive maintenance steps for steam turbines.", Capability.RAG, "rag_pipeline"),
            ("Troubleshoot high vibration in compressor K-101.", Capability.RAG, "rag_pipeline"),
            ("Calculate the relief valve sizing for vessel V-305.", Capability.CALCULATION, "calculator"),
            ("Verify MAWP calculation for column C-101.", Capability.CALCULATION, "calculator"),
            ("Export inspection report as actual pdf artifact.", Capability.DOCUMENT_GENERATION, "pdf_generator"),
            ("Write a Python script to parse the inspection report.", Capability.CODING, "code_interpreter"),
        ],
    )
    def test_non_visual_queries_do_not_route_to_image_generation(self, router, query, expected_capability, expected_tool):
        decision = router.route(query)
        assert decision.capability == expected_capability
        assert decision.tool_name == expected_tool
        assert decision.capability != Capability.IMAGE_GENERATION


# ── 5. End-to-End CLI Pipeline Test (scripts/ask.py path) ─────────────────────

class TestCLIAskEndToEndExecution:
    """Test full execution path through scripts/ask.py."""

    def test_cli_ask_generates_image_for_reported_prompts(self, mock_generator, capsys):
        from scripts.ask import main

        test_queries = [
            "Show me what a modern oil refinery might look like from above.",
            "Can you visualize a distillation tower and the equipment around it?",
        ]

        for query in test_queries:
            with patch("agent.tool_executor.ImageGeneratorTool.generator", new=mock_generator):
                main(["--model", "deterministic_test", query])

                out = capsys.readouterr().out

                # 1. Image generation artifact must be displayed
                assert "[Generated Artifact]" in out, f"Output missing [Generated Artifact] for '{query}'"
                assert "PNG" in out, f"Output missing PNG artifact indicator for '{query}'"
                assert "mock_generated_refinery.png" in out

                # 2. RAG citations / sources must NOT be present
                assert "### Sources" not in out, f"RAG citations improperly printed for '{query}'"
                assert "[1] Oil Industry Safety Directorate" not in out
                assert "[1] V-310 Engineering E222" not in out
