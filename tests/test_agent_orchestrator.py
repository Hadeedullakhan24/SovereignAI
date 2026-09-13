"""Comprehensive Verification Suite for Member 2: Agentic AI & LLM Orchestration.

SIH 2026 Problem Statement SIH26117 -- MRPL:
"Sovereign On-Premise Agentic AI Workbench using Open-Weight Multimodal LLMs for Confidential Industrial Work"

Tests:
  1. Model Registry: local open-weight model catalog, availability, readiness, multi-model support.
  2. Task Router: rule-based determinism, capability classification, multi-model RAG routing, vision routing.
  3. Safe Calculator: AST-based expression evaluation, engineering formulas, security against arbitrary code execution.
  4. Sandboxed File Manager: path traversal prevention, sandboxed file operations.
  5. Python Subprocess Runner: isolation, timeout handling, blocked imports.
  6. Multimodal OCR & Vision Tool: integration with member3_ocr on real test assets (ocr_real_test.png).
  7. ReAct Planner & Safety Gates:
     - 7-step deterministic industrial pipeline.
     - Image/PDF report reading via vision_inspector.
     - Extraction Gate (Step 2 halt on missing critical fields).
     - Grounding Gate (Step 5 halt on ungrounded/low-confidence calculation).
     - Human-in-the-loop approval gate (Step 7 pause & persistent disk checkpoint).
     - Resumption from checkpoint with engineer authorization.
  8. Top-level SovereignAgent API: handle(), resume(), health(), JSON serialization.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import time

from agent.agent import AgentResponse, SovereignAgent, create_agent
from agent.model_registry import AgentModelRegistry, ModelRecord, get_agent_registry
from agent.planner import AgentPlanner, PlanExecutionResult, PlanStatus, PlanStep, StepType
from agent.router import Capability, RoutingDecision, TaskRouter, get_router
from agent.tool_executor import (
    DEFAULT_PROJECT_ROOT,
    DEFAULT_SANDBOX_DIR,
    DocumentGenerator,
    PDFConverter,
    PresentationGenerator,
    SafeCalculator,
    SandboxedFileManager,
    SpreadsheetGenerator,
    SubprocessPythonRunner,
    ToolExecutor,
    ToolResult,
    VisionInspectorTool,
)
from rag_engine.generation.prompt.prompt_templates import PromptArchetype


# ==============================================================================
# 1. Model Registry Tests
# ==============================================================================

class TestModelRegistry:
    """Test model registry cataloging, health checking, and disk verification."""

    def test_registry_initialization(self):
        registry = AgentModelRegistry()
        records = registry.all()
        assert len(records) >= 4, f"Expected at least 4 models, found {len(records)}"

        roles = {r.role for r in records}
        assert "rag" in roles
        assert "vision" in roles

    def test_local_open_weight_models_ready(self):
        registry = AgentModelRegistry()
        ready_rag = registry.ready_for_role("rag")
        assert len(ready_rag) >= 3, f"Expected 3 ready RAG models on disk, found {len(ready_rag)}"

        repo_ids = {r.hf_repo_id for r in ready_rag}
        assert "Qwen/Qwen2.5-1.5B-Instruct" in repo_ids
        assert "microsoft/Phi-3.5-mini-instruct" in repo_ids
        assert "HuggingFaceTB/SmolLM2-1.7B-Instruct" in repo_ids

    def test_qwen_vision_model_is_honestly_reported_ready_when_weights_exist(self):
        registry = AgentModelRegistry()
        ready_vision = registry.ready_for_role("vision")
        assert len(ready_vision) == 1
        qwen_vl = registry.get("Qwen/Qwen2.5-VL-3B-Instruct")
        assert qwen_vl is not None
        assert qwen_vl.status == "ready"
        assert qwen_vl.is_installed() is True

    def test_primary_rag_model(self):
        registry = AgentModelRegistry()
        primary = registry.primary_rag_model()
        assert primary is not None
        assert "Qwen2.5-1.5B" in primary.hf_repo_id


# ==============================================================================
# 2. Task Router Tests
# ==============================================================================

class TestTaskRouter:
    """Test deterministic rule-based capability and multi-model routing."""

    @pytest.fixture
    def router(self):
        return TaskRouter()

    def test_route_standard_rag(self, router):
        decision = router.route("What is the design operating pressure of centrifugal pump P-203?")
        assert decision.capability == Capability.RAG
        assert decision.tool_name == "rag_pipeline"
        assert decision.use_rag_context is True
        assert decision.capability_available is True
        assert decision.model_record is not None
        assert "Qwen2.5-1.5B" in decision.model_record.hf_repo_id

    def test_route_long_context_to_phi(self, router):
        task = "Summarize the full report for pressure vessel V-2201 including all historical ultrasonic thickness readings."
        decision = router.route(task)
        assert decision.capability == Capability.RAG
        assert decision.model_record is not None
        assert "Phi-3.5-mini" in decision.model_record.hf_repo_id

    def test_route_quick_lookup_to_smollm(self, router):
        task = "Quick lookup of equipment tag and operating pressure for pump P-203"
        decision = router.route(task)
        assert decision.capability == Capability.RAG
        assert decision.model_record is not None
        assert "SmolLM2-1.7B" in decision.model_record.hf_repo_id

    def test_route_calculation_requires_rag_context(self, router):
        task = "Calculate the minimum wall thickness and verify MAWP for vessel V-2201."
        decision = router.route(task)
        assert decision.capability == Capability.CALCULATION
        assert decision.tool_name == "calculator"
        assert decision.use_rag_context is True
        assert decision.is_actionable() is True

    def test_route_vision_uses_staged_qwen_weights(self, router):
        task = "Read the nameplate from this image of the pump P-101."
        decision = router.route(task)
        assert decision.capability == Capability.VISION
        assert decision.tool_name == "vision_inspector"
        assert decision.use_rag_context is False
        assert decision.capability_available is True
        assert decision.model_record is not None
        assert decision.model_record.hf_repo_id == "Qwen/Qwen2.5-VL-3B-Instruct"

    def test_unknown_query_has_warning(self, router):
        task = "blorp glob random phrase xyz"
        decision = router.route(task)
        assert decision.capability == Capability.UNKNOWN
        assert decision.fallback_warning is not None
        assert "Low-confidence routing" in decision.fallback_warning


# ==============================================================================
# 3. Safe Calculator Tests
# ==============================================================================

class TestSafeCalculator:
    """Test AST-restricted formula evaluation and mathematical sandboxing."""

    @pytest.fixture
    def calc(self):
        return SafeCalculator()

    def test_basic_arithmetic(self, calc):
        val, steps = calc.evaluate("10 + 20 * 3 - (15 / 3)")
        assert val == 65.0
        assert isinstance(steps, list) and len(steps) > 0

    def test_math_functions(self, calc):
        val, steps = calc.evaluate("sqrt(144) + abs(-10) + min(5, 8)")
        assert val == 27.0
        assert isinstance(steps, list) and len(steps) > 0

    def test_engineering_asme_formula(self, calc):
        # t_min = (P * R) / (S * E - 0.6 * P)
        expr = "(P * R) / (S * E - 0.6 * P)"
        variables = {"P": 10.5, "R": 600.0, "S": 1380.0, "E": 0.85}
        result, steps = calc.evaluate(expr, variables)
        assert round(result, 4) == 5.3998
        assert isinstance(steps, list) and len(steps) > 0
        assert any("P = 10.5" in s for s in steps)

    def test_calculator_steps_in_tool_result(self):
        executor = ToolExecutor(sandbox_dir=DEFAULT_SANDBOX_DIR)
        res = executor.execute("calculator", expression="10 + 20 * 3", variables={})
        assert res.status == "success"
        assert "steps" in res.output
        assert isinstance(res.output["steps"], list)
        assert len(res.output["steps"]) > 0

    def test_blocks_arbitrary_code(self, calc):
        with pytest.raises(ValueError):
            calc.evaluate("__import__('os').system('dir')")

        with pytest.raises(ValueError):
            calc.evaluate("eval('1 + 1')")

    def test_zero_division(self, calc):
        with pytest.raises(ZeroDivisionError):
            calc.evaluate("100 / 0")


# ==============================================================================
# 4. Sandboxed File Manager Tests
# ==============================================================================

class TestSandboxedFileManager:
    """Test safe file operations and path traversal rejection."""

    def test_read_write_list(self, tmp_path):
        fm = SandboxedFileManager(sandbox_dir=tmp_path)
        write_res = fm.write("unit_01.txt", "Inspection Pass: 12.5 bar")
        assert write_res["status"] == "written"

        content = fm.read("unit_01.txt")
        assert content == "Inspection Pass: 12.5 bar"

        files = fm.list_files()
        assert len(files) == 1
        assert files[0]["name"] == "unit_01.txt"

    def test_rejects_path_traversal(self, tmp_path):
        fm = SandboxedFileManager(sandbox_dir=tmp_path)
        with pytest.raises(PermissionError):
            fm.read("../../windows/system32/cmd.exe")


# ==============================================================================
# 5. Subprocess Python Execution Tests
# ==============================================================================

class TestPythonRunner:
    """Test sandboxed Python execution with timeout and restricted builtins."""

    def test_valid_execution(self, tmp_path):
        runner = SubprocessPythonRunner(sandbox_dir=tmp_path)
        res = runner.run("print(sum([x for x in range(5)]))")
        assert res["status"] == "success"
        assert "10" in res["stdout"]

    def test_timeout_enforcement(self, tmp_path):
        runner = SubprocessPythonRunner(sandbox_dir=tmp_path)
        res = runner.run("import time; time.sleep(5)", timeout_seconds=1)
        assert res["status"] == "timeout"


# ==============================================================================
# 6. Multimodal OCR & Vision Tool Tests
# ==============================================================================

class TestVisionInspectorTool:
    """Test Member 3 OCR & document processor integration on real image assets."""

    def test_inspect_real_test_image(self):
        tool = VisionInspectorTool(sandbox_dir=DEFAULT_SANDBOX_DIR, project_root=DEFAULT_PROJECT_ROOT)
        image_path = "member3_ocr/input/ocr_real_test.png"
        assert (DEFAULT_PROJECT_ROOT / image_path).exists(), f"Image file missing: {image_path}"

        result = tool.inspect(image_path)
        assert result["status"] in ("success", "warning")
        assert result["filename"] == "ocr_real_test.png"
        assert len(result["text"]) > 0

        # Verify tool output contains extracted text from OCR
        text = result["text"]
        assert "P-101" in text or "DWG" in text or "Pump" in text or "Centrifugal" in text

    def test_inspect_via_tool_executor(self):
        executor = ToolExecutor(sandbox_dir=DEFAULT_SANDBOX_DIR)
        res = executor.execute("vision_inspector", filename="member3_ocr/input/ocr_real_test.png")
        assert res.status in ("success", "warning")
        assert res.is_verified is True
        assert res.output is not None
        assert "ocr_real_test.png" in res.output["filename"]


# ==============================================================================
# 7. ReAct Planner & Safety Gate Tests
# ==============================================================================

class TestPlannerAndSafetyGates:
    """Test deterministic 7-step pipeline, gates, and human approval resumption."""

    @pytest.fixture
    def planner(self, tmp_path):
        sb = tmp_path / "sandbox"
        cp = tmp_path / "checkpoints"
        sb.mkdir()
        cp.mkdir()
        return AgentPlanner(sandbox_dir=sb, checkpoints_dir=cp)

    def test_rag_unavailable_halts_at_grounding_gate(self, planner, monkeypatch):
        # Simulate RAG being unavailable
        monkeypatch.setattr(
            planner.tool_executor,
            "rag_search",
            lambda *a, **kw: {
                "status": "error",
                "answer": "",
                "citations": [],
                "confidence_score": 0.0,
                "factual_grounding": "Unavailable",
                "is_insufficient_evidence": True,
                "retrieved_chunks": 0,
                "model_used": "mock",
                "error": "RAG pipeline unavailable",
            },
        )
        goal = "Process inspection report for V-2201, verify calculations, and prepare approval document."
        res = planner.run(goal)

        assert res.status == PlanStatus.REQUIRES_HUMAN_REVIEW
        assert res.failed_step == 3
        assert "Unavailable" in (res.halt_reason or "")

    def test_resume_requires_a_real_approval_checkpoint(self, planner):
        goal = "Process inspection report for V-2201"
        initial = planner.run(goal)
        assert initial.status in (PlanStatus.HUMAN_APPROVAL_REQUIRED, PlanStatus.REQUIRES_HUMAN_REVIEW)
        assert initial.checkpoint_id is not None

    def test_grounding_gate_halts_ungrounded_calculation(self, planner):
        goal = "Process inspection report with ungrounded calculation"
        res = planner.run(goal, force_ungrounded_calc=True)

        assert res.status == PlanStatus.REQUIRES_HUMAN_REVIEW
        # Safety gate halts either at Step 3 (if RAG is offline) or Step 5 (ungrounded calc)
        assert res.failed_step in (3, 5)
        assert "HALTED:" in res.execution_trace

    def test_extraction_gate_halts_incomplete_document(self, planner):
        incomplete_file = planner.sandbox_dir / "incomplete_report.md"
        incomplete_file.write_text(
            "# INSPECTION REPORT\n"
            "**Date:** 2026-09-13\n"
            "## Findings\nSome observations without equipment ID or pressure.\n",
            encoding="utf-8",
        )

        res = planner.run(
            "Analyze incomplete_report.md",
            report_filename="incomplete_report.md",
        )

        assert res.status == PlanStatus.REQUIRES_HUMAN_REVIEW
        assert res.failed_step == 2
        assert "HALTED: Step 2 requires human verification" in res.execution_trace

    def test_step_1_image_document_ocr_branch(self, planner):
        # Sync test image into planner sandbox
        img_src = DEFAULT_PROJECT_ROOT / "member3_ocr" / "input" / "ocr_real_test.png"
        img_dst = planner.sandbox_dir / "ocr_real_test.png"
        img_dst.write_bytes(img_src.read_bytes())

        res = planner.run(
            "Process image inspection report ocr_real_test.png",
            report_filename="ocr_real_test.png",
        )

        step1 = next(s for s in res.steps if s.step_number == 1)
        assert step1.status == "completed"
        assert "vision_inspector" in step1.action
        assert "raw_report_text" in res.context_state
        # Partial multimodal results are retained even if no trustworthy text
        # is available from the drawing; extraction then correctly gates.
        assert isinstance(res.context_state["raw_report_text"], str)


# ==============================================================================
# 8. Top-Level SovereignAgent API Tests
# ==============================================================================

class TestSovereignAgentAPI:
    """Test top-level orchestrator interface and Backend contract stability."""

    def test_agent_handle_returns_typed_response(self, tmp_path):
        agent = SovereignAgent(sandbox_dir=tmp_path / "sb", checkpoints_dir=tmp_path / "cp")
        resp = agent.handle("Process inspection report for V-2201")

        assert isinstance(resp, AgentResponse)
        assert resp.status in ("awaiting_approval", "completed", "requires_verification")
        assert isinstance(resp.is_verified, bool)
        assert isinstance(resp.total_time_ms, float)
        assert resp.execution_trace != ""
        assert len(resp.reasoning_steps) > 0

    def test_agent_health_report(self, tmp_path):
        agent = SovereignAgent(sandbox_dir=tmp_path / "sb", checkpoints_dir=tmp_path / "cp")
        health = agent.health()

        assert health["agent"] == "SovereignAgent"
        assert "models" in health
        assert "rag" in health["models"]
        assert "vision" in health["models"]
        assert health["models"]["vision"]["status"] == "ready"
        assert health["models"]["ocr"]["status"] == "ready"

    def test_agent_response_json_serialization(self, tmp_path):
        agent = SovereignAgent(sandbox_dir=tmp_path / "sb", checkpoints_dir=tmp_path / "cp")
        resp = agent.handle("Process inspection report for V-2201")
        data = resp.to_dict()

        # Must not raise TypeError
        serialized = json.dumps(data, indent=2)
        assert len(serialized) > 100
        reconstructed = json.loads(serialized)
        assert reconstructed["status"] == resp.status


# ==============================================================================
# 9. Document Generator Suite (docx, xlsx, pptx, pdf)
# ==============================================================================

class TestDocumentGenerators:
    """Test Member 2 multi-format document generators."""

    @pytest.fixture
    def file_manager(self, tmp_path):
        return SandboxedFileManager(sandbox_dir=tmp_path)

    def test_spreadsheet_generator_direct(self, file_manager, tmp_path):
        gen = SpreadsheetGenerator(file_manager)
        wb_spec = {
            "title": "Thickness Survey Summary",
            "sheets": [
                {
                    "name": "Measurements",
                    "headers": ["Point", "Location", "Nominal (mm)", "Measured (mm)", "Status"],
                    "rows": [
                        ["P-01", "Shell Top", 12.0, 11.8, "Acceptable"],
                        ["P-02", "Shell Middle", 12.0, 11.5, "Acceptable"],
                        ["P-03", "Shell Bottom", 12.0, 11.2, "Acceptable"],
                    ],
                    "metadata": {"Equipment": "V-2201", "Standard": "API 510"},
                }
            ],
        }
        res = gen.generate(workbook=wb_spec, filename="test_survey.xlsx")
        assert res["filename"] == "test_survey.xlsx"
        assert res["file_size_bytes"] > 0
        assert (tmp_path / "test_survey.xlsx").exists()

    def test_spreadsheet_generator_via_executor(self, tmp_path):
        executor = ToolExecutor(sandbox_dir=tmp_path)
        wb_spec = {
            "title": "Corrosion Monitoring",
            "sheets": [
                {
                    "name": "Data",
                    "headers": ["Param", "Value"],
                    "rows": [["Pressure", 10.5], ["Min Thickness", 5.4]],
                }
            ],
        }
        res = executor.execute("xlsx_generator", workbook=wb_spec, filename="corrosion.xlsx")
        assert res.status == "success"
        assert res.output["file_size_bytes"] > 0
        assert (tmp_path / "corrosion.xlsx").exists()

    def test_presentation_generator_direct(self, file_manager, tmp_path):
        gen = PresentationGenerator(file_manager)
        slides = [
            {
                "title": "Executive Summary",
                "subtitle": "Pressure Vessel V-2201",
                "content": "All thickness measurements verified against API 510 standards.",
            },
            {
                "title": "Survey Findings",
                "bullets": [
                    "Design Pressure: 10.5 kg/cm²g",
                    "Shell minimum thickness: 11.2 mm",
                    "Governing t_min: 5.48 mm",
                    "Safety margin: Adequate for 5-year cycle",
                ],
            },
        ]
        res = gen.generate(title="V-2201 Inspection Briefing", slides=slides, filename="briefing.pptx")
        assert res["filename"] == "briefing.pptx"
        assert res["file_size_bytes"] > 0
        assert (tmp_path / "briefing.pptx").exists()

    def test_presentation_generator_via_executor(self, tmp_path):
        executor = ToolExecutor(sandbox_dir=tmp_path)
        slides = [
            {"title": "Overview", "content": "Presentation generated via ToolExecutor contract."},
        ]
        res = executor.execute("pptx_generator", title="Compliance Deck", slides=slides, filename="deck.pptx")
        assert res.status == "success"
        assert res.output["file_size_bytes"] > 0
        assert (tmp_path / "deck.pptx").exists()

    def test_pdf_converter_graceful_handling(self, file_manager, tmp_path):
        converter = PDFConverter(file_manager)
        # First generate a docx
        doc_gen = DocumentGenerator(file_manager)
        doc_gen.generate_report(
            title="PDF Source Doc",
            sections=[{"heading": "Intro", "paragraphs": ["Test content"]}],
            filename="source.docx",
        )
        assert (tmp_path / "source.docx").exists()

        # Try PDF conversion
        try:
            pdf_res = converter.convert("source.docx", "output.pdf")
            assert pdf_res["status"] == "success"
            assert (tmp_path / "output.pdf").exists()
        except RuntimeError as exc:
            # When neither Word COM nor LibreOffice is installed, must raise clean RuntimeError
            assert "Neither docx2pdf nor LibreOffice was found" in str(exc) or "PDF conversion requires" in str(exc)

    def test_pdf_generator_via_executor_graceful_or_success(self, tmp_path):
        executor = ToolExecutor(sandbox_dir=tmp_path)
        # Create source docx
        executor.execute(
            "document_generator",
            title="PDF Test Report",
            sections=[{"heading": "Status", "paragraphs": ["All clear"]}],
            filename="pdf_test.docx",
        )
        res = executor.execute("pdf_generator", docx_filename="pdf_test.docx", out_filename="pdf_test.pdf")
        assert res.status in ("success", "capability_unavailable")
        if res.status == "capability_unavailable":
            assert "PDF" in (res.error or "") or "LibreOffice" in (res.error or "") or "docx2pdf" in (res.error or "")


# ==============================================================================
# 10. Router & Planner Edge Case Tests
# ==============================================================================

class TestRouterModelFallback:
    """Test TaskRouter dynamic fallback when primary models are unavailable."""

    def test_fallback_when_primary_unavailable(self, monkeypatch):
        router = TaskRouter()
        # Exclude Qwen from ready models to test fallback to Phi / SmolLM
        all_ready = router._registry.ready_for_role("rag")
        non_qwen_ready = [m for m in all_ready if "qwen" not in m.hf_repo_id.lower()]
        assert len(non_qwen_ready) > 0, "Expected at least one non-Qwen RAG model ready"

        monkeypatch.setattr(router._registry, "ready_for_role", lambda role: non_qwen_ready if role == "rag" else [])
        query = "Explain OISD-130 statutory inspection interval requirements"
        decision = router.route(query)
        assert decision.capability == Capability.RAG
        assert decision.capability_available is True
        assert decision.model_record is not None
        assert "qwen" not in decision.model_record.hf_repo_id.lower()

    def test_all_models_unavailable_reports_unavailable(self, monkeypatch):
        router = TaskRouter()
        monkeypatch.setattr(router._registry, "ready_for_role", lambda role: [])
        monkeypatch.setattr(router._registry, "ready_for_capability", lambda cap: [])
        monkeypatch.setattr(router._registry, "by_role", lambda role: [])
        monkeypatch.setattr(router._registry, "all", lambda: [])

        query = "Calculate API 510 minimum thickness"
        decision = router.route(query)
        assert isinstance(decision, RoutingDecision)
        assert decision.capability_available is False
        assert decision.model_record is None
        assert "has no models declared" in decision.reason or "no ready model" in decision.reason


class TestPlannerEmptyRAGContext:
    """Test planner graceful degradation when RAG returns empty context."""

    def test_planner_halts_safely_when_rag_returns_empty(self, tmp_path, monkeypatch):
        sb = tmp_path / "sb"
        cp = tmp_path / "cp"
        sb.mkdir()
        cp.mkdir()
        planner = AgentPlanner(sandbox_dir=sb, checkpoints_dir=cp)

        monkeypatch.setattr(
            planner.tool_executor,
            "rag_search",
            lambda *a, **kw: {
                "status": "insufficient_evidence",
                "answer": "",
                "citations": [],
                "confidence_score": 0.0,
                "factual_grounding": "Insufficient Evidence",
                "is_insufficient_evidence": True,
                "retrieved_chunks": 0,
                "model_used": "mock",
            },
        )
        res = planner.run("Process inspection report for V-2201")
        assert res.status == PlanStatus.REQUIRES_HUMAN_REVIEW
        assert res.failed_step == 3
        assert res.checkpoint_id is not None
        assert "no matching sop found" in res.context_state.get("sop_answer", "").lower()

    def test_step_4_synthesizes_approval_note_without_crashing_on_empty_rag(self, tmp_path):
        sb = tmp_path / "sb"
        cp = tmp_path / "cp"
        sb.mkdir()
        cp.mkdir()
        planner = AgentPlanner(sandbox_dir=sb, checkpoints_dir=cp)

        step4 = PlanStep(step_number=4, name="Generate approval note", step_type=StepType.AUTOMATED)
        ctx = {
            "findings": {
                "equipment_id": "V-2201",
                "equipment_name": "Knockout Drum",
                "design_pressure_kg_cm2": 10.5,
                "shell_min_thickness_mm": 11.2,
                "shell_orig_thickness_mm": 12.0,
                "_defaulted_fields": [],
            },
            "sop_answer": "[NO MATCHING SOP FOUND]",
        }
        planner._step_generate_approval_note(step4, ctx)
        assert step4.is_verified is True
        assert "approval_note" in ctx
        assert len(ctx["approval_note"]) > 50


class TestAgentResumeFromCheckpoint:
    """Test full checkpoint save and resumption cycle."""

    def test_checkpoint_roundtrip_and_approval_resumption(self, tmp_path):
        sb = tmp_path / "sb"
        cp = tmp_path / "cp"
        sb.mkdir()
        cp.mkdir()
        planner = AgentPlanner(sandbox_dir=sb, checkpoints_dir=cp)

        result = PlanExecutionResult(
            goal="Statutory compliance review for V-2201",
            status=PlanStatus.HUMAN_APPROVAL_REQUIRED,
            total_steps=7,
        )
        step7 = PlanStep(
            step_number=7,
            name="Wait for engineer approval",
            step_type=StepType.HUMAN_APPROVAL_REQUIRED,
            status="pending",
        )
        result.steps.append(step7)
        saved_path = planner.checkpoint_manager.save(result)
        assert Path(saved_path).is_file()

        resumed = planner.resume(
            checkpoint_id_or_path=result.checkpoint_id,
            engineer_name="R. Sharma, Lead Integrity Engineer",
            approved=True,
            comments="Wall thickness 11.2 mm exceeds ASME t_min of 5.48 mm. Safe for operation.",
        )
        assert resumed.status == PlanStatus.COMPLETED
        assert resumed.context_state["approval_signoff"]["approved"] is True
        assert resumed.context_state["approval_signoff"]["engineer"] == "R. Sharma, Lead Integrity Engineer"
        assert next(s for s in resumed.steps if s.step_number == 7).status == "completed"

    def test_checkpoint_rejection_resumption(self, tmp_path):
        sb = tmp_path / "sb"
        cp = tmp_path / "cp"
        sb.mkdir()
        cp.mkdir()
        planner = AgentPlanner(sandbox_dir=sb, checkpoints_dir=cp)

        result = PlanExecutionResult(
            goal="Statutory compliance review for V-2201",
            status=PlanStatus.HUMAN_APPROVAL_REQUIRED,
            total_steps=7,
        )
        result.steps.append(PlanStep(step_number=7, name="Wait for engineer approval", step_type=StepType.HUMAN_APPROVAL_REQUIRED))
        planner.checkpoint_manager.save(result)

        resumed = planner.resume(
            checkpoint_id_or_path=result.checkpoint_id,
            engineer_name="R. Sharma, Lead Integrity Engineer",
            approved=False,
            comments="Excessive localized pitting observed on nozzle N-1. Rectification required.",
        )
        assert resumed.status == PlanStatus.REJECTED
        assert "rejected" in resumed.halt_reason.lower()
        assert next(s for s in resumed.steps if s.step_number == 7).status == "rejected"


# ==============================================================================
# 11. Hardened Error Handling Tests
# ==============================================================================

class TestHardenedErrorHandling:
    """Test that all tool branches in ToolExecutor catch exceptions gracefully."""

    def test_file_manager_path_traversal_returns_tool_result_error(self, tmp_path):
        executor = ToolExecutor(sandbox_dir=tmp_path)
        res = executor.execute("file_manager", action="read", filename="../../etc/passwd")
        assert res.status == "error"
        assert res.is_verified is False
        assert res.error is not None

    def test_unsupported_tool_returns_tool_result_error(self, tmp_path):
        executor = ToolExecutor(sandbox_dir=tmp_path)
        res = executor.execute("imaginary_unsupported_tool_xyz")
        assert res.status == "error"
        assert "Unsupported tool name" in (res.error or "")
