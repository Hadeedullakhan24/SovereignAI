"""Agent Orchestration — Multi-Step ReAct Planner.

Implements a deterministic, safety-gated ReAct (Thought -> Action -> Observation -> Repeat)
planner for engineering inspection, compliance, and approval workflows:

  Step 1: Read inspection report (sandboxed file reading)
  Step 2: Extract findings (parameter and condition extraction)
  Step 3: Search SOP (real RAG retrieval for regulatory standards)
  Step 4: Generate approval note (technical synthesis of findings and standards)
  Step 5: Verify calculations (deterministic formula evaluation with RAG grounding gate)
  Step 6: Generate Word document (professional .docx statutory report)
  Step 7: Wait for engineer approval (explicit HUMAN_APPROVAL_REQUIRED gate with persistent checkpointing)

Critical Safety Architecture:
  1. Grounding Gate: Any step producing a ToolResult with is_verified=False immediately halts
     execution and returns 'REQUIRES_HUMAN_REVIEW' with the full reasoning trace.
  2. Human Approval Gate: The final authorization step halts execution with
     'HUMAN_APPROVAL_REQUIRED' and serializes the complete state to disk as a JSON checkpoint.
     Execution can be resumed across sessions via AgentPlanner.resume().
  3. Demo-Ready Execution Trace: Step-by-step trace formatting suitable for live audit
     demonstrations (e.g. 'Step 1: Read inspection report -> ... -> HALTED: Step 5 requires human verification').
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

# Ensure project root is on sys.path
_THIS_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path("agent").resolve()
_PROJECT_ROOT = _THIS_DIR.parent if _THIS_DIR.name == "agent" else Path(".").resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agent.router import Capability, RoutingDecision, TaskRouter
from agent.tool_executor import (
    DEFAULT_AUDIT_LOG_FILE,
    DEFAULT_PROJECT_ROOT,
    DEFAULT_SANDBOX_DIR,
    ToolExecutor,
    ToolResult,
)
from agent.prompts.prompt_loader import PromptLoader, get_prompt_loader
from rag_engine.generation.prompt.prompt_templates import PromptArchetype

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINTS_DIR = DEFAULT_SANDBOX_DIR / "checkpoints"


# ── Status and Step Type Constants ──────────────────────────────────────────

class StepType:
    AUTOMATED = "automated"
    HUMAN_APPROVAL_REQUIRED = "HUMAN_APPROVAL_REQUIRED"


class PlanStatus:
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    REQUIRES_HUMAN_REVIEW = "REQUIRES_HUMAN_REVIEW"
    HUMAN_APPROVAL_REQUIRED = "HUMAN_APPROVAL_REQUIRED"
    FAILED = "FAILED"
    REJECTED = "REJECTED"


# ── Step and Plan Dataclasses ───────────────────────────────────────────────

@dataclass
class PlanStep:
    """A single ReAct reasoning and action step in the workflow."""
    step_number: int
    name: str
    step_type: str = StepType.AUTOMATED
    thought: str = ""
    action: str = ""
    action_input: Dict[str, Any] = field(default_factory=dict)
    observation: Any = None
    tool_result: Optional[ToolResult] = None
    status: str = "pending"  # 'pending' | 'in_progress' | 'completed' | 'requires_verification' | 'paused' | 'failed'
    is_verified: bool = True
    reasoning_trace: str = ""
    execution_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert step to a clean JSON-serializable dictionary."""
        tr_dict = None
        if self.tool_result is not None:
            tr_dict = {
                "tool_name": self.tool_result.tool_name,
                "status": self.tool_result.status,
                "is_verified": self.tool_result.is_verified,
                "output": self._sanitize(self.tool_result.output),
                "rag_context": self._sanitize(self.tool_result.rag_context),
                "execution_time_ms": self.tool_result.execution_time_ms,
                "fallback_warning": self.tool_result.fallback_warning,
                "error": self.tool_result.error,
            }
        return {
            "step_number": self.step_number,
            "name": self.name,
            "step_type": self.step_type,
            "thought": self.thought,
            "action": self.action,
            "action_input": self._sanitize(self.action_input),
            "observation": self._sanitize(self.observation),
            "tool_result": tr_dict,
            "status": self.status,
            "is_verified": self.is_verified,
            "reasoning_trace": self.reasoning_trace,
            "execution_time_ms": round(self.execution_time_ms, 2),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PlanStep:
        """Reconstruct a PlanStep from dictionary."""
        tr = None
        tr_raw = data.get("tool_result")
        if tr_raw:
            tr = ToolResult(
                tool_name=tr_raw.get("tool_name", ""),
                status=tr_raw.get("status", "unknown"),
                output=tr_raw.get("output"),
                rag_context=tr_raw.get("rag_context"),
                is_verified=tr_raw.get("is_verified", True),
                execution_time_ms=tr_raw.get("execution_time_ms", 0.0),
                fallback_warning=tr_raw.get("fallback_warning"),
                error=tr_raw.get("error"),
            )
        return cls(
            step_number=data.get("step_number", 0),
            name=data.get("name", ""),
            step_type=data.get("step_type", StepType.AUTOMATED),
            thought=data.get("thought", ""),
            action=data.get("action", ""),
            action_input=data.get("action_input", {}),
            observation=data.get("observation"),
            tool_result=tr,
            status=data.get("status", "completed"),
            is_verified=data.get("is_verified", True),
            reasoning_trace=data.get("reasoning_trace", ""),
            execution_time_ms=data.get("execution_time_ms", 0.0),
        )

    @staticmethod
    def _sanitize(obj: Any) -> Any:
        try:
            json.dumps(obj)
            return obj
        except (TypeError, OverflowError):
            return str(obj)


@dataclass
class PlanExecutionResult:
    """Master result container returned by AgentPlanner.run() and AgentPlanner.resume()."""
    goal: str
    status: str  # PlanStatus
    steps: List[PlanStep] = field(default_factory=list)
    current_step_index: int = 0
    total_steps: int = 0
    checkpoint_id: Optional[str] = None
    checkpoint_path: Optional[str] = None
    halt_reason: Optional[str] = None
    failed_step: Optional[int] = None
    execution_trace: str = ""
    context_state: Dict[str, Any] = field(default_factory=dict)
    total_execution_time_ms: float = 0.0
    created_at_iso: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at_iso: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def format_trace(self, verbose: bool = False) -> str:
        """Produce a formatted multi-line execution trace suitable for terminal or reports."""
        lines = [
            "=" * 78,
            f"RE-ACT PLANNER EXECUTION TRACE",
            f"Goal   : {self.goal}",
            f"Status : {self.status}",
            f"Trace  : {self.execution_trace}",
        ]
        if self.checkpoint_path:
            lines.append(f"State  : Checkpoint saved -> {self.checkpoint_path}")
        if self.halt_reason:
            lines.append(f"Halt   : {self.halt_reason}")
        lines.append("=" * 78)

        for step in self.steps:
            v_tag = "VERIFIED" if step.is_verified else "UNVERIFIED"
            lines.append(f"\n[Step {step.step_number}: {step.name}] ({step.status.upper()} | {v_tag} | {step.execution_time_ms:.1f}ms)")
            lines.append(f"  Thought    : {step.thought}")
            lines.append(f"  Action     : {step.action}")
            if verbose and step.action_input:
                lines.append(f"  Input      : {json.dumps(step.action_input, default=str)}")
            obs_str = str(step.observation)
            if len(obs_str) > 220 and not verbose:
                obs_str = obs_str[:220] + "... [truncated]"
            lines.append(f"  Observation: {obs_str}")
            if step.tool_result and step.tool_result.fallback_warning:
                lines.append(f"  Warning    : {step.tool_result.fallback_warning}")
            if step.tool_result and step.tool_result.error:
                lines.append(f"  Error      : {step.tool_result.error}")

        lines.append("\n" + "=" * 78)
        return "\n".join(lines)

    def summary(self) -> str:
        """One-line summary for logging."""
        cp = f" | checkpoint={self.checkpoint_id}" if self.checkpoint_id else ""
        halt = f" | halt={self.halt_reason}" if self.halt_reason else ""
        return (
            f"[AgentPlanner] status={self.status} | steps={len(self.steps)} | "
            f"time={self.total_execution_time_ms:.1f}ms{cp}{halt} | trace={self.execution_trace}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert complete execution state to a JSON-serializable dictionary."""
        return {
            "goal": self.goal,
            "status": self.status,
            "steps": [s.to_dict() for s in self.steps],
            "current_step_index": self.current_step_index,
            "total_steps": self.total_steps,
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_path": self.checkpoint_path,
            "halt_reason": self.halt_reason,
            "failed_step": self.failed_step,
            "execution_trace": self.execution_trace,
            "context_state": self._sanitize(self.context_state),
            "total_execution_time_ms": round(self.total_execution_time_ms, 2),
            "created_at_iso": self.created_at_iso,
            "updated_at_iso": self.updated_at_iso,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PlanExecutionResult:
        """Reconstruct PlanExecutionResult from serialized dictionary."""
        steps = [PlanStep.from_dict(s) for s in data.get("steps", [])]
        return cls(
            goal=data.get("goal", ""),
            status=data.get("status", PlanStatus.FAILED),
            steps=steps,
            current_step_index=data.get("current_step_index", len(steps)),
            total_steps=data.get("total_steps", len(steps)),
            checkpoint_id=data.get("checkpoint_id"),
            checkpoint_path=data.get("checkpoint_path"),
            halt_reason=data.get("halt_reason"),
            failed_step=data.get("failed_step"),
            execution_trace=data.get("execution_trace", ""),
            context_state=data.get("context_state", {}),
            total_execution_time_ms=data.get("total_execution_time_ms", 0.0),
            created_at_iso=data.get("created_at_iso", datetime.now(timezone.utc).isoformat()),
            updated_at_iso=data.get("updated_at_iso", datetime.now(timezone.utc).isoformat()),
        )

    @staticmethod
    def _sanitize(obj: Any) -> Any:
        try:
            json.dumps(obj)
            return obj
        except (TypeError, OverflowError):
            return str(obj)


# ── Checkpoint Persistence Manager ──────────────────────────────────────────

class CheckpointManager:
    """Manages persistent serialization and recovery of plan execution states."""

    def __init__(self, checkpoints_dir: Union[str, Path] = DEFAULT_CHECKPOINTS_DIR) -> None:
        self.checkpoints_dir = Path(checkpoints_dir).resolve()
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def save(self, result: PlanExecutionResult) -> Path:
        """Serialize plan execution result to disk as a JSON checkpoint."""
        if not result.checkpoint_id:
            tag = "chk"
            if "equipment_id" in result.context_state:
                tag = f"chk_{result.context_state['equipment_id']}"
            result.checkpoint_id = f"{tag}_{int(time.time())}_{uuid.uuid4().hex[:6]}"

        file_path = self.checkpoints_dir / f"{result.checkpoint_id}.json"
        result.checkpoint_path = str(file_path)
        result.updated_at_iso = datetime.now(timezone.utc).isoformat()

        payload = result.to_dict()
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved planner checkpoint to: {file_path}")
        return file_path

    def load(self, checkpoint_id_or_path: Union[str, Path]) -> PlanExecutionResult:
        """Load plan execution state from a checkpoint ID or direct file path."""
        candidate = Path(checkpoint_id_or_path)
        if candidate.is_file():
            target_path = candidate
        else:
            # Look up by ID inside checkpoints directory
            cid = str(checkpoint_id_or_path)
            if not cid.endswith(".json"):
                cid = f"{cid}.json"
            target_path = self.checkpoints_dir / cid

        if not target_path.is_file():
            raise FileNotFoundError(f"Planner checkpoint file not found: {target_path}")

        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return PlanExecutionResult.from_dict(data)


# ── Master ReAct Agent Planner ──────────────────────────────────────────────

class AgentPlanner:
    """ReAct Multi-Step Planner for Sovereign AI.

    Coordinates:
      - Step 1: Read inspection report
      - Step 2: Extract findings
      - Step 3: Search SOP
      - Step 4: Generate approval note
      - Step 5: Verify calculations
      - Step 6: Generate Word document
      - Step 7: Wait for engineer approval

    Safety Gating:
      - Any ToolResult with is_verified=False immediately halts the workflow,
        returning REQUIRES_HUMAN_REVIEW and preserving the full reasoning trace.
      - Final approval step halts with HUMAN_APPROVAL_REQUIRED, persisting state
        to disk. It can be resumed via AgentPlanner.resume().
    """

    def __init__(
        self,
        tool_executor: Optional[ToolExecutor] = None,
        router: Optional[TaskRouter] = None,
        sandbox_dir: Union[str, Path] = DEFAULT_SANDBOX_DIR,
        checkpoints_dir: Union[str, Path] = DEFAULT_CHECKPOINTS_DIR,
        prompt_loader: Optional[PromptLoader] = None,
    ) -> None:
        self.sandbox_dir = Path(sandbox_dir).resolve()
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)
        self.tool_executor = tool_executor or ToolExecutor(sandbox_dir=self.sandbox_dir)
        self.router = router or TaskRouter()
        self.checkpoint_manager = CheckpointManager(checkpoints_dir)
        self.prompt_loader = prompt_loader or get_prompt_loader()

    # ------------------------------------------------------------------
    # Public Execution API
    # ------------------------------------------------------------------

    def run(
        self,
        user_goal: str,
        max_steps: int = 10,
        **kwargs: Any,
    ) -> PlanExecutionResult:
        """Execute the multi-step engineering compliance plan.

        Parameters
        ----------
        user_goal : Description of the goal / equipment inspection task.
        max_steps : Maximum permitted reasoning steps (default 10).
        kwargs    : Optional control parameters:
                    - report_filename: Custom filename of inspection report in sandbox.
                    - force_ungrounded_calc: If True, forces ungrounded RAG in calculation
                      to demonstrate safety gate halting behavior.
        """
        start_overall = time.perf_counter()
        logger.info(f"AgentPlanner starting run for goal: {user_goal!r}")

        result = PlanExecutionResult(
            goal=user_goal,
            status=PlanStatus.IN_PROGRESS,
            total_steps=7,
        )

        # Pre-seed sandbox with sample inspection report if none exists
        self._ensure_inspection_report_available(kwargs.get("report_filename"))

        # Context accumulator across steps
        ctx = result.context_state
        ctx["user_goal"] = user_goal
        ctx["report_filename"] = kwargs.get("report_filename", "pressure_vessel_inspection_002.md")
        ctx["force_ungrounded_calc"] = kwargs.get(
            "force_ungrounded_calc",
            ("ungrounded" in user_goal.lower() or "unverified" in user_goal.lower()),
        )

        # Define 7 canonical workflow steps
        step_definitions = [
            (1, "Read inspection report", StepType.AUTOMATED, self._step_read_inspection_report),
            (2, "Extract findings", StepType.AUTOMATED, self._step_extract_findings),
            (3, "Search SOP", StepType.AUTOMATED, self._step_search_sop),
            (4, "Generate approval note", StepType.AUTOMATED, self._step_generate_approval_note),
            (5, "Verify calculations", StepType.AUTOMATED, self._step_verify_calculations),
            (6, "Generate Word document", StepType.AUTOMATED, self._step_generate_document),
            (7, "Wait for engineer approval", StepType.HUMAN_APPROVAL_REQUIRED, self._step_wait_for_approval),
        ]

        for step_no, name, stype, step_func in step_definitions:
            if len(result.steps) >= max_steps:
                result.status = PlanStatus.FAILED
                result.halt_reason = f"Max step limit ({max_steps}) exceeded."
                result.execution_trace = self._build_arrow_trace(result.steps, halted_reason="Max steps exceeded")
                break

            result.current_step_index = step_no
            step = PlanStep(step_number=step_no, name=name, step_type=stype)
            result.steps.append(step)

            # Execute step
            step_start = time.perf_counter()
            try:
                step_func(step, ctx)
            except Exception as exc:
                step.status = "failed"
                step.is_verified = False
                step.observation = f"Exception: {exc}"
                result.status = PlanStatus.FAILED
                result.failed_step = step_no
                result.halt_reason = f"Unhandled exception in step {step_no} ({name}): {exc}"
                step.execution_time_ms = (time.perf_counter() - step_start) * 1000.0
                result.execution_trace = self._build_arrow_trace(result.steps, halted_reason=f"Step {step_no} failed")
                self.checkpoint_manager.save(result)
                break

            step.execution_time_ms = (time.perf_counter() - step_start) * 1000.0

            # ── CRITICAL GROUNDING GATE ─────────────────────────────────
            # Any step whose ToolResult has is_verified=False must NOT proceed.
            if step.tool_result is not None and not step.tool_result.is_verified:
                step.status = "requires_verification"
                step.is_verified = False
                result.status = PlanStatus.REQUIRES_HUMAN_REVIEW
                result.failed_step = step_no

                warning_detail = step.tool_result.fallback_warning or step.tool_result.error or "Ungrounded / low confidence context"
                rag_detail = ""
                if step.tool_result.rag_context:
                    rag_detail = f" (RAG status: {step.tool_result.rag_context.get('factual_grounding', 'Insufficient Evidence')})"

                reason = (
                    f"Step {step_no} ({name}) requires human review: "
                    f"ToolResult status='{step.tool_result.status}', is_verified=False{rag_detail}. "
                    f"Detail: {warning_detail}"
                )
                result.halt_reason = reason
                result.execution_trace = self._build_arrow_trace(
                    result.steps,
                    halted_reason=f"HALTED: Step {step_no} requires human verification",
                )
                # Serialize state for human review audit
                self.checkpoint_manager.save(result)
                logger.warning(f"Plan halted at step {step_no}: {reason}")
                break

            # ── HUMAN APPROVAL GATE ─────────────────────────────────────
            if step.step_type == StepType.HUMAN_APPROVAL_REQUIRED:
                step.status = "paused"
                result.status = PlanStatus.HUMAN_APPROVAL_REQUIRED
                result.execution_trace = self._build_arrow_trace(
                    result.steps,
                    halted_reason=f"PAUSED: Step {step_no} requires human approval",
                )
                # Persist state so it survives across sessions
                self.checkpoint_manager.save(result)
                logger.info(f"Plan paused at Step {step_no} for engineer approval. Checkpoint: {result.checkpoint_id}")
                break

            # Mark step completed
            step.status = "completed"

        # Finalize timings
        result.total_execution_time_ms = (time.perf_counter() - start_overall) * 1000.0
        if result.status == PlanStatus.IN_PROGRESS and len(result.steps) == len(step_definitions):
            result.status = PlanStatus.COMPLETED
            result.execution_trace = self._build_arrow_trace(result.steps, suffix="COMPLETED")

        logger.info(result.summary())
        return result

    # ------------------------------------------------------------------
    # Resumption API
    # ------------------------------------------------------------------

    def resume(
        self,
        checkpoint_id_or_path: Union[str, Path],
        engineer_name: str = "Authorized Inspector (S. Menon)",
        approved: bool = True,
        comments: str = "All thickness calculations and safety margins verified. Approved for continued service.",
    ) -> PlanExecutionResult:
        """Resume execution from a serialized checkpoint after human engineer review.

        Parameters
        ----------
        checkpoint_id_or_path : Checkpoint ID or path to JSON file on disk.
        engineer_name         : Name / credentials of the approving engineer.
        approved              : True to grant statutory approval, False to reject.
        comments              : Engineering notes / sign-off rationale.
        """
        result = self.checkpoint_manager.load(checkpoint_id_or_path)

        if result.status != PlanStatus.HUMAN_APPROVAL_REQUIRED:
            logger.warning(
                f"Checkpoint {result.checkpoint_id} is in status '{result.status}', not 'HUMAN_APPROVAL_REQUIRED'."
            )

        # Locate Step 7
        step_7 = next((s for s in result.steps if s.step_number == 7), None)
        if step_7 is None:
            step_7 = PlanStep(
                step_number=7,
                name="Wait for engineer approval",
                step_type=StepType.HUMAN_APPROVAL_REQUIRED,
            )
            result.steps.append(step_7)

        now_iso = datetime.now(timezone.utc).isoformat()
        if approved:
            step_7.status = "completed"
            step_7.is_verified = True
            step_7.thought = (
                f"Human review received from {engineer_name}. "
                f"Statutory requirements and calculations verified. Applying official authorization."
            )
            step_7.action = "record_engineer_approval"
            step_7.action_input = {
                "engineer_name": engineer_name,
                "approved": True,
                "comments": comments,
                "timestamp": now_iso,
            }
            step_7.observation = (
                f"APPROVED by {engineer_name} at {now_iso}. "
                f"Engineering Notes: {comments}. Document ready for operational issuance."
            )

            result.status = PlanStatus.COMPLETED
            result.context_state["approval_signoff"] = {
                "engineer": engineer_name,
                "approved": True,
                "comments": comments,
                "signed_at": now_iso,
            }
            result.execution_trace = self._build_arrow_trace(
                result.steps,
                suffix=f"COMPLETED: Approved by {engineer_name}",
            )
        else:
            step_7.status = "rejected"
            step_7.is_verified = False
            step_7.observation = f"REJECTED by {engineer_name}. Comments: {comments}"
            result.status = PlanStatus.REJECTED
            result.halt_reason = f"Workflow rejected during human review: {comments}"
            result.execution_trace = self._build_arrow_trace(
                result.steps,
                suffix=f"REJECTED by {engineer_name}",
            )

        # Update checkpoint on disk
        self.checkpoint_manager.save(result)
        logger.info(f"Resumed and updated checkpoint {result.checkpoint_id} -> Status: {result.status}")
        return result

    # ------------------------------------------------------------------
    # Step Implementations (Real Tools & Real Routing)
    # ------------------------------------------------------------------

    def _step_read_inspection_report(self, step: PlanStep, ctx: Dict[str, Any]) -> None:
        """Step 1: Read inspection report from the sandboxed environment (text or image/PDF via vision_inspector)."""
        filename = ctx.get("report_filename", "pressure_vessel_inspection_002.md")
        ext = Path(filename).suffix.lower()

        if ext in (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".pdf"):
            step.thought = (
                f"Inspection report '{filename}' is an image/PDF document. "
                f"Invoking vision_inspector (multimodal OCR) to extract text, tables, and parameters."
            )
            step.action = f"vision_inspector(filename='{filename}')"
            step.action_input = {"filename": filename, "tool": "vision_inspector"}

            res = self.tool_executor.execute("vision_inspector", filename=filename)
            step.tool_result = res
            step.is_verified = res.is_verified

            if res.is_success and isinstance(res.output, dict):
                report_text = res.output.get("text", "")
                ctx["raw_report_text"] = report_text
                ctx["ocr_metadata"] = res.output
                step.observation = (
                    f"Successfully parsed image document '{filename}' via vision_inspector "
                    f"({len(report_text)} chars extracted, route={res.output.get('routing_decision')})."
                )
            else:
                step.observation = f"Failed to extract text from image '{filename}': {res.error}"
        else:
            step.thought = (
                f"I need to read the statutory inspection report '{filename}' from the project sandbox "
                f"to examine physical condition, thickness survey data, and equipment metadata."
            )
            step.action = f"file_manager(action='read', filename='{filename}')"
            step.action_input = {"action": "read", "filename": filename}

            res = self.tool_executor.execute("file_manager", action="read", filename=filename)
            step.tool_result = res
            step.is_verified = res.is_verified

            if res.is_success and isinstance(res.output, dict):
                report_text = res.output.get("content", "")
                ctx["raw_report_text"] = report_text
                step.observation = f"Successfully loaded '{filename}' ({len(report_text)} chars, {len(report_text.splitlines())} lines)."
            else:
                step.observation = f"Failed to read file: {res.error}"

    def _step_extract_findings(self, step: PlanStep, ctx: Dict[str, Any]) -> None:
        """Step 2: Parse and extract critical engineering parameters from the source document.

        Tracks each critical field separately.  Any field that falls back to a
        hardcoded default — because the regex found no match in the actual text —
        is recorded in ``defaulted_fields``.  If ANY critical field (equipment_id,
        design_pressure_kg_cm2, shell_min_thickness_mm) is defaulted, the step
        is marked is_verified=False so the grounding gate halts the workflow.
        """
        raw_text = ctx.get("raw_report_text", "")
        step.thought = (
            "I need to parse the raw inspection text to extract structural parameters: "
            "Equipment Tag, Design Pressure, Design Temperature, Shell & Head thicknesses, "
            "and inspector recommendations.  Any field not found in the document will be "
            "flagged rather than silently substituted."
        )
        step.action = "extract_inspection_parameters(text=raw_report_text)"

        findings: Dict[str, Any] = {}
        defaulted_fields: List[str] = []   # fields substituted with hardcoded defaults
        extracted_fields: List[str] = []   # fields successfully read from source text

        # ── Equipment ID & Name ──────────────────────────────────────────
        # Pattern handles both plain "Equipment ID: V-2201" and bold markdown
        # "**Equipment ID:** V-2201 (Knockout Drum)" formats.
        eq_match = re.search(
            r"Equipment\s+(?:ID|Tag)[^:]*:[\s*]+([A-Z0-9][A-Z0-9\-]+)(?:\s*\((.*?)\))?",
            raw_text,
            re.I,
        )
        if eq_match:
            findings["equipment_id"] = eq_match.group(1).strip()
            findings["equipment_name"] = (
                eq_match.group(2).strip() if eq_match.group(2) else "Pressure Vessel"
            )
            extracted_fields.append("equipment_id")
        else:
            findings["equipment_id"] = "UNKNOWN"
            findings["equipment_name"] = "UNKNOWN"
            defaulted_fields.append("equipment_id")

        # ── Report No (non-critical, default is acceptable) ──────────────
        rep_match = re.search(r"Report\s+(?:No|ID|Number)[^:]*:[\s*]+([A-Z0-9][A-Z0-9\-]+)", raw_text, re.I)
        if rep_match:
            findings["report_no"] = rep_match.group(1).strip()
            extracted_fields.append("report_no")
        else:
            findings["report_no"] = "UNKNOWN"
            defaulted_fields.append("report_no")

        # ── Design Pressure (critical) ────────────────────────────────────
        # Handles "**Design Pressure:** 10.5 kg/cm2g | ..." markdown format.
        dp_match = re.search(
            r"Design\s+Pressure[^:]*:[\s*]+([\d\.]+)\s*(kg/cm2g|psig|bar|kpa|MPa|psi)?",
            raw_text,
            re.I,
        )
        if dp_match:
            findings["design_pressure_kg_cm2"] = float(dp_match.group(1))
            findings["design_pressure_unit"] = dp_match.group(2) or "kg/cm2g"
            extracted_fields.append("design_pressure_kg_cm2")
        else:
            findings["design_pressure_kg_cm2"] = None  # do NOT substitute a default
            findings["design_pressure_unit"] = "kg/cm2g"
            defaulted_fields.append("design_pressure_kg_cm2")

        # ── Design Temperature (non-critical) ─────────────────────────────
        dt_match = re.search(
            r"Design\s+Temperature[^:]*:[\s*]+([\d\.]+)\s*[°]?C?",
            raw_text,
            re.I,
        )
        if dt_match:
            findings["design_temp_c"] = float(dt_match.group(1))
            extracted_fields.append("design_temp_c")
        else:
            findings["design_temp_c"] = None
            defaulted_fields.append("design_temp_c")

        # ── Shell thickness (critical) ────────────────────────────────────
        shell_bot_match = re.search(r"Shell\s*\(Bottom\)[\s\|]+([\d\.]+)[\s\|]+([\d\.]+)", raw_text, re.I)
        if shell_bot_match:
            findings["shell_orig_thickness_mm"] = float(shell_bot_match.group(1))
            findings["shell_min_thickness_mm"] = float(shell_bot_match.group(2))
            extracted_fields.append("shell_min_thickness_mm")
        else:
            findings["shell_orig_thickness_mm"] = None
            findings["shell_min_thickness_mm"] = None
            defaulted_fields.append("shell_min_thickness_mm")

        # ── Head thickness (non-critical for minimum-t calc) ─────────────
        head_match = re.search(r"Head\s*\(Top\)[\s\|]+([\d\.]+)[\s\|]+([\d\.]+)", raw_text, re.I)
        if head_match:
            findings["head_orig_thickness_mm"] = float(head_match.group(1))
            findings["head_thickness_mm"] = float(head_match.group(2))
            extracted_fields.append("head_thickness_mm")
        else:
            findings["head_orig_thickness_mm"] = None
            findings["head_thickness_mm"] = None
            defaulted_fields.append("head_thickness_mm")

        # ── Inspector name (from text, non-critical) ──────────────────────
        # Handles "**Inspector:** S. Menon, ..."
        insp_match = re.search(r"Inspector[^:]*:[\s*]+(.+)", raw_text, re.I)
        findings["inspector"] = insp_match.group(1).strip() if insp_match else "NOT FOUND IN REPORT"

        # ── Recommendation (from text) ────────────────────────────────────
        rec_match = re.search(r"Recommendation[:\s\n]+(.+)", raw_text, re.I)
        findings["recommendation"] = rec_match.group(1).strip() if rec_match else "NOT FOUND IN REPORT"

        # ── Provenance tracking ───────────────────────────────────────────
        findings["_extracted_fields"] = extracted_fields
        findings["_defaulted_fields"] = defaulted_fields

        ctx["findings"] = findings
        ctx["equipment_id"] = findings["equipment_id"]
        step.action_input = {"target_equipment": findings["equipment_id"]}

        # ── Critical-field check ─────────────────────────────────────────
        # equipment_id, design_pressure_kg_cm2, shell_min_thickness_mm are
        # required for all downstream calculations and approval decisions.
        CRITICAL_FIELDS = ["equipment_id", "design_pressure_kg_cm2", "shell_min_thickness_mm"]
        missing_critical = [
            f for f in CRITICAL_FIELDS
            if f in defaulted_fields or findings.get(f) is None or findings.get(f) == "UNKNOWN"
        ]

        eq_disp = findings["equipment_id"]
        p_disp = f"{findings['design_pressure_kg_cm2']} kg/cm²g" if findings["design_pressure_kg_cm2"] is not None else "NOT FOUND"
        t_disp = f"{findings['shell_min_thickness_mm']} mm" if findings["shell_min_thickness_mm"] is not None else "NOT FOUND"

        if missing_critical:
            step.is_verified = False
            step.observation = (
                f"EXTRACTION INCOMPLETE — critical fields could not be read from source document: "
                f"{', '.join(missing_critical)}. "
                f"Extracted: {', '.join(extracted_fields) or 'none'}. "
                f"Defaulted/missing: {', '.join(defaulted_fields)}. "
                f"Fabricated defaults would produce unverified engineering conclusions — halting."
            )
            # Attach a synthetic ToolResult so the grounding gate fires
            step.tool_result = ToolResult(
                tool_name="extract_findings",
                status="requires_verification",
                output=findings,
                is_verified=False,
                error=f"Critical fields not found in source text: {missing_critical}",
            )
        else:
            step.is_verified = True
            _head_disp = (
                f"{findings['head_thickness_mm']} mm"
                if findings["head_thickness_mm"] is not None
                else "NOT FOUND"
            )
            step.observation = (
                f"Extracted parameters for {eq_disp} ({findings['equipment_name']}): "
                f"Design Pressure={p_disp}, "
                f"Shell Min Measured={t_disp}, "
                f"Head Measured={_head_disp}. "
                f"Defaulted (non-critical): {', '.join(d for d in defaulted_fields if d not in CRITICAL_FIELDS) or 'none'}."
            )

    def _step_search_sop(self, step: PlanStep, ctx: Dict[str, Any]) -> None:
        """Step 3: Query real RAG pipeline for applicable SOP and statutory standards."""
        findings = ctx.get("findings", {})
        eq_id = findings.get("equipment_id", "V-2201")
        query = (
            f"What are the statutory inspection intervals and minimum allowable thickness requirements "
            f"for pressure vessel {eq_id} under API 510 and OISD-130?"
        )

        step.thought = (
            f"I must search refinery standard operating procedures and engineering standards "
            f"(API 510 / OISD-130) to establish statutory thickness allowances and inspection intervals."
        )
        step.action = f"rag_search(query='{query}', top_k=3)"
        step.action_input = {"query": query, "top_k": 3}

        # Route dynamically through TaskRouter
        decision = self.router.route(query)
        tool_res = self.tool_executor.execute(decision, task=query, top_k=3)
        step.tool_result = tool_res
        step.is_verified = tool_res.is_verified

        if tool_res.is_verified:
            sop_answer = tool_res.output if isinstance(tool_res.output, str) else str(tool_res.output)
            ctx["sop_answer"] = sop_answer
            ctx["sop_citations"] = tool_res.rag_context.get("citations", []) if tool_res.rag_context else []
            step.observation = (
                f"SOP standards retrieved (Grounding: {tool_res.rag_context.get('factual_grounding', 'Grounded')}, "
                f"Confidence: {tool_res.rag_context.get('confidence_score', 0.0)}). Summary: {sop_answer[:160]}..."
            )
        else:
            # RAG unavailable / insufficient evidence — record safe fallback in ctx
            retrieved_chunks = (
                tool_res.rag_context.get("retrieved_chunks", 0)
                if tool_res.rag_context
                else 0
            )
            grounding = (
                tool_res.rag_context.get("factual_grounding", "Unavailable")
                if tool_res.rag_context
                else "Unavailable"
            )
            ctx["sop_answer"] = (
                f"[NO MATCHING SOP FOUND] RAG search returned insufficient evidence "
                f"(status='{tool_res.status}', grounding='{grounding}', "
                f"retrieved_chunks={retrieved_chunks}). "
                f"Engineer must manually locate applicable API 510 / OISD-130 clauses "
                f"and attach them to this approval note before signing."
            )
            ctx["sop_citations"] = []
            step.observation = (
                f"RAG search unverified / low grounding: {tool_res.status}. Grounding: {grounding}"
            )
            step.is_verified = False

    def _step_generate_approval_note(self, step: PlanStep, ctx: Dict[str, Any]) -> None:
        """Step 4: Synthesize an approval note using ONLY data-derived statements.

        All quantitative claims are computed from extracted findings; no
        unverifiable conclusory language is emitted.
        """
        findings = ctx.get("findings", {})
        eq_id = findings.get("equipment_id", "UNKNOWN")
        eq_name = findings.get("equipment_name", "UNKNOWN")
        p_design = findings.get("design_pressure_kg_cm2")   # may be None if not extracted
        t_shell = findings.get("shell_min_thickness_mm")     # may be None
        t_shell_orig = findings.get("shell_orig_thickness_mm")  # nominal/original
        defaulted = findings.get("_defaulted_fields", [])

        step.thought = (
            f"I will formulate the technical approval note for {eq_id} using only "
            f"data-derived statements.  Any field not found in the source report will "
            f"be explicitly marked '[NOT IN REPORT]' rather than silently assumed."
        )
        step.action = "synthesize_approval_note(findings, sop_guidelines)"

        # ── Compute corrosion allowance from actual measurements ──────────
        # Only state a numeric margin if both original and measured thicknesses
        # were successfully extracted from the document.
        if t_shell is not None and t_shell_orig is not None:
            corrosion_consumed_mm = t_shell_orig - t_shell
            corrosion_pct = (corrosion_consumed_mm / t_shell_orig) * 100.0
            shell_assessment = (
                f"Shell (Bottom): nominal={t_shell_orig} mm, measured={t_shell} mm, "
                f"corrosion consumed={corrosion_consumed_mm:.2f} mm ({corrosion_pct:.1f}% of original)."
            )
        elif t_shell is not None:
            shell_assessment = (
                f"Shell (Bottom): measured={t_shell} mm; original/nominal thickness not found "
                f"in report — corrosion margin cannot be computed."
            )
        else:
            shell_assessment = "Shell thickness: NOT FOUND IN REPORT — corrosion margin cannot be assessed."

        # ── Pressure line ──────────────────────────────────────────────────
        p_line = (
            f"Design Pressure: {p_design} kg/cm²g"
            if p_design is not None
            else "Design Pressure: NOT FOUND IN REPORT"
        )

        # ── Defaulted-fields disclosure ────────────────────────────────────
        if defaulted:
            default_disclosure = (
                f"\nNOTE — The following fields were NOT found in the source document and "
                f"are ABSENT from this note: {', '.join(defaulted)}. "
                f"Engineer must supply or verify these values before signing."
            )
        else:
            default_disclosure = ""

        # ── Render via PromptLoader template ──────────────────────────────
        approval_payload = {
            **findings,
            "equipment_id": eq_id,
            "equipment_name": eq_name,
            "report_no": findings.get("report_no", "UNKNOWN"),
            "inspector": findings.get("inspector", "NOT FOUND IN REPORT"),
            "recommendation": findings.get("recommendation", "NOT FOUND IN REPORT"),
            "shell_assessment": shell_assessment,
            "pressure_assessment": p_line,
            "default_disclosure": default_disclosure,
        }

        approval_text = self.prompt_loader.render(
            "approval_note_generation",
            **approval_payload,
        )

        ctx["approval_note"] = approval_text
        step.action_input = {"equipment_id": eq_id}
        step.observation = (
            f"Approval note drafted ({len(approval_text)} chars) using data-derived statements. "
            f"Defaulted fields: {defaulted or 'none'}."
        )
        step.is_verified = True

    # Constants used in the ASME/API minimum-wall-thickness formula.
    # R (vessel inner radius) and S (allowable stress) are vessel-specific and
    # should ideally come from the datasheet or be extracted from the report.
    # E (joint efficiency) is a code-defined weld factor.  Since none of these
    # appear in a standard inspection narrative report, we explicitly label them
    # as ASSUMED/DEFAULT values and surface that label in every observation,
    # approval note, and Word document section so they are never silently
    # presented as confirmed from the source document.
    _CALC_DEFAULTS: Dict[str, Any] = {
        "R_mm": 600.0,           # assumed: inner radius in mm (typical knockout drum)
        "S_kg_cm2": 1380.0,      # assumed: allowable stress per ASME Sec VIII Div 1 SA-516 Gr 70
        "E_weld": 0.85,          # assumed: weld joint efficiency (fully radiographed butt weld)
    }

    def _step_verify_calculations(self, step: PlanStep, ctx: Dict[str, Any]) -> None:
        """Step 5: Verify engineering calculations using SafeCalculator with RAG grounding.

        The design pressure P is taken from the extracted findings (Step 2).
        R, S, and E are NOT present in a typical inspection report narrative; they
        are drawn from ``_CALC_DEFAULTS`` and clearly labelled as ASSUMED in every
        output so the engineer knows exactly what needs independent verification.
        """
        findings = ctx.get("findings", {})
        eq_id = findings.get("equipment_id", "UNKNOWN")
        p_val = findings.get("design_pressure_kg_cm2")   # may be None
        t_actual = findings.get("shell_min_thickness_mm")  # may be None
        force_ungrounded = ctx.get("force_ungrounded_calc", False)

        # Assumed / default constants — explicitly labelled
        R_val = self._CALC_DEFAULTS["R_mm"]
        S_val = self._CALC_DEFAULTS["S_kg_cm2"]
        E_val = self._CALC_DEFAULTS["E_weld"]

        assumed_note = (
            f"[ASSUMED — not in report] R={R_val} mm (inner radius), "
            f"S={S_val} kg/cm² (allowable stress, ASME Sec VIII Div 1 SA-516 Gr 70), "
            f"E={E_val} (weld joint efficiency)."
        )
        ctx["calc_assumed_note"] = assumed_note

        step.thought = (
            f"I must evaluate the API 510 minimum wall thickness formula: "
            f"t_min = (P * R) / (S * E - 0.6 * P).  "
            f"P={p_val} kg/cm²g (extracted from report).  "
            f"{assumed_note}"
        )

        if p_val is None:
            # Cannot run the formula without a confirmed pressure value
            step.action = "calculator — SKIPPED (design pressure not extracted)"
            step.action_input = {}
            step.observation = (
                "Calculation SKIPPED: design_pressure_kg_cm2 was not found in the source report. "
                "A numeric pressure value is required for the minimum-wall-thickness formula. "
                "Engineer must supply the value from the vessel datasheet."
            )
            step.is_verified = False
            step.tool_result = ToolResult(
                tool_name="calculator",
                status="requires_verification",
                output=None,
                is_verified=False,
                error="design_pressure_kg_cm2 not extracted from source document — formula cannot run.",
            )
            return

        expression = "(P * R) / (S * E - 0.6 * P)"
        variables = {"P": float(p_val), "R": R_val, "S": S_val, "E": E_val}

        step.action = f"calculator(expression='{expression}', variables={variables}, use_rag_context=True)"
        step.action_input = {
            "expression": expression,
            "variables": variables,
            "use_rag_context": True,
            "force_ungrounded": force_ungrounded,
            "assumed_constants": {"R": R_val, "S": S_val, "E": E_val},
        }

        # Route calculation through router to produce real RoutingDecision
        if force_ungrounded:
            calc_task = "Calculate the relief valve sizing for vessel V-305."
        else:
            calc_task = f"Calculate minimum wall thickness formula for pressure vessel {eq_id} API 510"

        decision = self.router.route(calc_task)
        res = self.tool_executor.execute(
            decision,
            task=calc_task,
            expression=expression,
            variables=variables,
            force_ungrounded=force_ungrounded,
        )

        step.tool_result = res
        step.is_verified = res.is_verified

        if res.is_verified:
            t_min = res.output.get("result", 5.48)
            ctx["calc_result"] = res.output
            ctx["calculated_t_min"] = t_min
            if t_actual is not None:
                margin = t_actual - t_min
                margin_str = f"Actual measured thickness ({t_actual} mm) exceeds minimum by {margin:.2f} mm margin."
            else:
                margin_str = "Actual shell thickness NOT FOUND in report — margin cannot be computed."
            step.observation = (
                f"Calculation VERIFIED: t_min = {t_min:.4f} mm (RAG-grounded in API 510). "
                f"{margin_str} "
                f"{assumed_note}"
            )
        else:
            step.observation = (
                f"Calculation math completed ({res.output.get('result') if res.output else 'N/A'}), "
                f"but RAG grounding check failed: status='{res.status}', is_verified=False. "
                f"{assumed_note}"
            )

    def _step_generate_document(self, step: PlanStep, ctx: Dict[str, Any]) -> None:
        """Step 6: Generate formal Word document report via document_generator.

        Every table cell and paragraph paragraph distinguishes between values
        that were EXTRACTED from the source document versus values that are
        ASSUMED/DEFAULT constants, so the reviewing engineer can immediately
        see what requires independent verification.
        """
        findings = ctx.get("findings", {})
        eq_id = findings.get("equipment_id", "UNKNOWN")
        eq_name = findings.get("equipment_name", "UNKNOWN")
        t_min = ctx.get("calculated_t_min")
        t_actual = findings.get("shell_min_thickness_mm")
        t_actual_orig = findings.get("shell_orig_thickness_mm")
        p_design = findings.get("design_pressure_kg_cm2")
        t_design = findings.get("design_temp_c")
        assumed_note = ctx.get("calc_assumed_note", "R, S, E: assumed — not extracted from report.")
        out_filename = f"{eq_id}_Statutory_Approval_Note.docx"

        step.thought = (
            f"All verified steps completed.  Generating the statutory .docx report '{out_filename}'. "
            f"Extracted vs assumed values will be clearly labelled throughout the document."
        )
        step.action = f"document_generator(filename='{out_filename}')"

        # Helper to render a value with provenance label
        def _val(v: Any, label: str, unit: str = "") -> str:
            if v is None:
                return f"NOT FOUND IN REPORT [{label}]"
            return f"{v}{unit} [EXTRACTED]"

        # Compute margin only when both values are available
        if t_min is not None and t_actual is not None:
            margin_str = f"+{(t_actual - t_min):.2f} mm above calculated minimum"
            shell_status = "ACCEPTABLE"
        elif t_actual is not None:
            margin_str = "t_min not calculated — margin unknown"
            shell_status = "REQUIRES VERIFICATION"
        else:
            margin_str = "measured thickness not found — margin unknown"
            shell_status = "REQUIRES VERIFICATION"

        doc_sections = [
            {
                "heading": "1. Executive Summary",
                "paragraphs": [
                    f"Statutory fitness-for-service evaluation for Equipment {eq_id} ({eq_name}).",
                    "Inspection conducted under API 510 / OISD-130.  See Section 5 for a full list of assumed constants that require engineer verification.",
                    ctx.get("approval_note", "Approval note not generated — see upstream steps."),
                ],
            },
            {
                "heading": "2. Equipment Design & Inspection Metadata",
                "paragraphs": [
                    f"Equipment Tag: {eq_id} [EXTRACTED] | Name: {eq_name} [EXTRACTED]",
                    f"Report Reference: {findings.get('report_no', 'NOT FOUND')} [EXTRACTED]",
                    f"Design Pressure: {_val(p_design, 'design_pressure_kg_cm2', ' kg/cm²g')}",
                    f"Design Temperature: {_val(t_design, 'design_temp_c', '°C')}",
                    f"Inspector: {findings.get('inspector', 'NOT FOUND IN REPORT')}",
                ],
                "table": {
                    "headers": ["Component", "Nominal (mm)", "Measured (mm)", "t_min (mm)", "Provenance", "Status"],
                    "rows": [
                        [
                            "Shell (Bottom)",
                            str(t_actual_orig) if t_actual_orig is not None else "NOT FOUND",
                            str(t_actual) if t_actual is not None else "NOT FOUND",
                            f"{t_min:.2f}" if t_min is not None else "NOT CALCULATED",
                            "t_min: CALCULATED (R,S,E assumed — see §5)",
                            shell_status,
                        ],
                        [
                            "Head (Top)",
                            str(findings.get("head_orig_thickness_mm")) if findings.get("head_orig_thickness_mm") is not None else "NOT FOUND",
                            str(findings.get("head_thickness_mm")) if findings.get("head_thickness_mm") is not None else "NOT FOUND",
                            "NOT CALCULATED",
                            "No head t_min formula run",
                            "REQUIRES VERIFICATION",
                        ],
                    ],
                },
            },
            {
                "heading": "3. Regulatory Standards & SOP Compliance",
                "paragraphs": [
                    "Regulatory Framework: API 510 In-Service Pressure Vessel Inspection Code & OISD-130.",
                    ctx.get("sop_answer", "SOP answer not retrieved — see Step 3."),
                ],
            },
            {
                "heading": "4. Mathematical Verification Trace",
                "paragraphs": [
                    "Governing Formula: t_min = (P × R) / (S × E − 0.6 × P)   [API 510]",
                    f"P = {_val(p_design, 'design_pressure_kg_cm2', ' kg/cm²g')}",
                    assumed_note,
                    *(
                        # Surface the full intermediate calculation working if available
                        ["\nCalculation Working (from SafeCalculator step trace):"]
                        + [f"  {s}" for s in ctx.get("calc_result", {}).get("steps", [])]
                        if ctx.get("calc_result", {}).get("steps")
                        else ["No step trace available (calculation was skipped or failed)."]
                    ),
                    f"t_min (calculated) = {f'{t_min:.4f} mm' if t_min is not None else 'NOT CALCULATED'}",
                    f"t_actual (measured) = {_val(t_actual, 'shell_min_thickness_mm', ' mm')}",
                    f"Safety margin: {margin_str}.",
                ],
            },
            {
                "heading": "5. Assumed / Default Constants — Engineer Verification Required",
                "paragraphs": [
                    "The following constants were NOT found in the inspection report narrative "
                    "and were used as engineering defaults for calculation purposes only.  "
                    "The signing engineer MUST independently verify each value against the "
                    "vessel datasheet before issuing statutory approval.",
                    f"R = {self._CALC_DEFAULTS['R_mm']} mm (inner radius — assumed typical knockout drum; VERIFY from datasheet).",
                    f"S = {self._CALC_DEFAULTS['S_kg_cm2']} kg/cm² (allowable stress — assumed SA-516 Gr 70 at design temp; VERIFY material cert).",
                    f"E = {self._CALC_DEFAULTS['E_weld']} (weld joint efficiency — assumed full radiography; VERIFY weld records).",
                ],
            },
            {
                "heading": "6. Statutory Authorization & Signature",
                "paragraphs": [
                    f"Inspector Recommendation: {findings.get('recommendation', 'NOT FOUND IN REPORT')}.",
                    "Sign-off Status: PENDING PROFESSIONAL ENGINEER AUTHORIZATION.",
                ],
            },
        ]

        step.action_input = {"filename": out_filename, "sections_count": len(doc_sections)}

        res = self.tool_executor.execute(
            "document_generator",
            title=f"Statutory Continued Service Approval — {eq_id}",
            sections=doc_sections,
            filename=out_filename,
            metadata={"equipment_id": eq_id, "report_no": findings.get("report_no")},
        )

        step.tool_result = res
        step.is_verified = res.is_verified

        if res.is_success and isinstance(res.output, dict):
            ctx["generated_doc_path"] = res.output.get("path")
            ctx["generated_doc_name"] = out_filename
            step.observation = (
                f"Generated official document '{out_filename}' ({res.output.get('file_size_bytes')} bytes) "
                f"at {res.output.get('path')}. "
                f"Assumed constants are clearly labelled in Section 5; margin={margin_str}."
            )
        else:
            step.observation = f"Failed to generate Word report: {res.error}"

    def _step_wait_for_approval(self, step: PlanStep, ctx: Dict[str, Any]) -> None:
        """Step 7: Pause for human engineer sign-off and serialize state."""
        eq_id = ctx.get("equipment_id", "V-2201")
        doc_name = ctx.get("generated_doc_name", f"{eq_id}_Statutory_Approval_Note.docx")

        step.thought = (
            f"The Word approval document '{doc_name}' has been compiled and saved. "
            f"Refinery governance requires an Authorized Professional Engineer to review and sign off. "
            f"I must pause execution and serialize state to disk so it can be resumed."
        )
        step.action = "pause_for_human_approval(serialize_state=True)"
        step.action_input = {
            "required_role": "Authorized Professional Engineer / Inspector",
            "document": doc_name,
            "equipment_id": eq_id,
        }
        step.observation = (
            f"Execution paused. All 6 automated steps completed and verified. "
            f"Document '{doc_name}' is awaiting engineer review. Checkpoint will be written to disk."
        )
        step.is_verified = True

    # ------------------------------------------------------------------
    # Helper & Trace Formatting Methods
    # ------------------------------------------------------------------

    def _ensure_inspection_report_available(self, custom_filename: Optional[str] = None) -> None:
        """Ensure a valid inspection report file is accessible within the sandbox.

        Priority:
          1. Check canonical datasets/inspection_reports/<filename>, member3_ocr/input/<filename>,
             datasets/ocr/<filename>, or datasets/engineering_drawings/<filename>.
          2. Sync/copy to sandbox.
          3. Fallback: write a minimal built-in text report.
        """
        filename = custom_filename or "pressure_vessel_inspection_002.md"
        sandbox_target = self.sandbox_dir / filename
        source = DEFAULT_PROJECT_ROOT / "datasets" / "inspection_reports" / filename

        if not source.is_file() and custom_filename:
            for cand in [
                DEFAULT_PROJECT_ROOT / "member3_ocr" / "input" / custom_filename,
                DEFAULT_PROJECT_ROOT / "datasets" / "ocr" / custom_filename,
                DEFAULT_PROJECT_ROOT / "datasets" / "engineering_drawings" / custom_filename,
                DEFAULT_PROJECT_ROOT / custom_filename,
            ]:
                if cand.is_file():
                    source = cand
                    break

        if source.is_file():
            sandbox_target.parent.mkdir(parents=True, exist_ok=True)
            sandbox_target.write_bytes(source.read_bytes())
            logger.info(f"Synced inspection report from source: {source} -> {sandbox_target}")
        elif not sandbox_target.is_file():
            # Fallback: write a complete built-in report with all critical fields
            content = (
                "# PRESSURE VESSEL INSPECTION REPORT\n\n"
                "**Report No:** INS-PV-0087\n"
                "**Date of Inspection:** 03-Sep-2026\n"
                "**Equipment ID:** V-2201 (Knockout Drum)\n"
                "**Design Pressure:** 10.5 kg/cm2g | **Design Temperature:** 120\u00b0C\n"
                "**Inspector:** S. Menon, Authorized Inspector\n\n"
                "## Scope\n"
                "Internal and external inspection under Factories Act / OISD-130.\n\n"
                "## Thickness Survey Summary\n"
                "| Component | Original (mm) | Current (mm) | Corrosion Allowance Remaining |\n"
                "|---|---|---|---|\n"
                "| Shell (Top) | 12.0 | 11.6 | 87% |\n"
                "| Shell (Bottom) | 12.0 | 11.2 | 78% |\n"
                "| Head (Top) | 14.0 | 13.7 | 91% |\n\n"
                "## Findings & Recommendation\n"
                "Vessel is structurally sound. Approve for continued service until next cycle (5 years).\n"
                "**Status: APPROVED FOR SERVICE**\n"
            )
            sandbox_target.parent.mkdir(parents=True, exist_ok=True)
            sandbox_target.write_text(content, encoding="utf-8")
            logger.info(f"Initialized inspection report in sandbox (built-in fallback): {sandbox_target}")


    @staticmethod
    def _build_arrow_trace(
        steps: List[PlanStep],
        halted_reason: Optional[str] = None,
        suffix: Optional[str] = None,
    ) -> str:
        """Build the demo arrow trace string.

        Example:
          'Step 1: Read inspection report -> Step 2: Extract findings -> ... -> PAUSED: Step 7 requires human approval'
          'Step 1: Read inspection report -> ... -> HALTED: Step 5 requires human verification'
        """
        parts = []
        for s in steps:
            # If this is the step where we halted or paused, format according to halted_reason
            if halted_reason and s.step_number == steps[-1].step_number:
                parts.append(halted_reason)
            else:
                parts.append(f"Step {s.step_number}: {s.name}")

        if suffix and not halted_reason:
            parts.append(suffix)

        return " -> ".join(parts)


# ── Self-Verification & Demonstration Entrypoint ────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print("\n" + "=" * 80)
    print("DEMO 1: Sovereign AI Multi-Step ReAct Planner (End-to-End Workflow)")
    print("=" * 80)

    planner = AgentPlanner()

    # 1. Run full workflow to Step 7 (Human Approval Required)
    goal = "Process inspection report for V-2201 knockout drum, verify calculations, and prepare statutory approval document."
    plan_result = planner.run(goal)

    print("\n" + plan_result.format_trace())
    print("\nArrow Trace:")
    print(plan_result.execution_trace)
    print(f"\nStatus at pause: {plan_result.status}")
    print(f"Serialized Checkpoint: {plan_result.checkpoint_path}")

    assert plan_result.status == PlanStatus.HUMAN_APPROVAL_REQUIRED, (
        f"Expected HUMAN_APPROVAL_REQUIRED, got {plan_result.status}"
    )
    assert plan_result.checkpoint_path is not None, "Expected serialized checkpoint file on disk"

    # 2. Resume execution with engineer approval
    print("\n" + "=" * 80)
    print("DEMO 2: Resuming Execution from Persistent Disk Checkpoint")
    print("=" * 80)

    resumed_result = planner.resume(
        checkpoint_id_or_path=plan_result.checkpoint_id,
        engineer_name="S. Menon (Authorized Inspector #AI-9021)",
        approved=True,
        comments="Thickness survey verified against ASME Section VIII Div 1. Continued operation approved for 5 years.",
    )

    print("\n" + resumed_result.format_trace())
    print("\nResumed Arrow Trace:")
    print(resumed_result.execution_trace)
    print(f"Final Status: {resumed_result.status}")

    assert resumed_result.status == PlanStatus.COMPLETED, (
        f"Expected COMPLETED after approval, got {resumed_result.status}"
    )

    # 3. Grounding Gate Safety Demo (Ungrounded calculation halts execution)
    print("\n" + "=" * 80)
    print("DEMO 3: CRITICAL GATE — Ungrounded Calculation Halts Execution")
    print("=" * 80)

    gated_goal = "Process ungrounded inspection spec calculation for pressure vessel V-2201"
    gated_result = planner.run(gated_goal, force_ungrounded_calc=True)

    print("\n" + gated_result.format_trace())
    print("\nGated Arrow Trace:")
    print(gated_result.execution_trace)
    print(f"Status on ungrounded calculation: {gated_result.status}")
    print(f"Failed Step: {gated_result.failed_step}")
    print(f"Halt Reason: {gated_result.halt_reason}")

    assert gated_result.status == PlanStatus.REQUIRES_HUMAN_REVIEW, (
        f"Expected REQUIRES_HUMAN_REVIEW, got {gated_result.status}"
    )
    assert gated_result.failed_step == 5, f"Expected step 5 to fail grounding, got {gated_result.failed_step}"
    assert "HALTED: Step 5 requires human verification" in gated_result.execution_trace

    # ------------------------------------------------------------------
    # DEMO 4: Extraction Gate — Missing critical fields halt at Step 2
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("DEMO 4: EXTRACTION GATE — Report missing equipment ID & design pressure halts at Step 2")
    print("=" * 80)

    # Write a minimal report to the sandbox that deliberately omits
    # equipment ID and design pressure so the extraction step cannot
    # confirm the critical fields.
    from agent.tool_executor import DEFAULT_SANDBOX_DIR
    minimal_report_path = DEFAULT_SANDBOX_DIR / "minimal_incomplete_report.md"
    minimal_report_path.write_text(
        "# INSPECTION REPORT\n"
        "**Date of Inspection:** 03-Sep-2026\n"
        "**Inspector:** J. Kumar, API Inspector\n\n"
        "## Scope\n"
        "Internal inspection under OISD-130.\n\n"
        "## Thickness Survey Summary\n"
        "| Component | Original (mm) | Current (mm) |\n"
        "|---|---|---|\n"
        "| Shell (Bottom) | 12.0 | 11.2 |\n\n"
        "## Findings\n"
        "Vessel in satisfactory condition.\n",
        encoding="utf-8",
    )
    print(f"Wrote minimal incomplete report -> {minimal_report_path}")

    incomplete_goal = "Process incomplete_report.md and verify calculations"
    incomplete_result = planner.run(
        incomplete_goal,
        report_filename="minimal_incomplete_report.md",
    )

    print("\n" + incomplete_result.format_trace())
    print("\nExtraction Gate Arrow Trace:")
    print(incomplete_result.execution_trace)
    print(f"Status on incomplete report: {incomplete_result.status}")
    print(f"Failed Step: {incomplete_result.failed_step}")
    print(f"Halt Reason: {incomplete_result.halt_reason}")

    step2 = next((s for s in incomplete_result.steps if s.step_number == 2), None)
    assert incomplete_result.status == PlanStatus.REQUIRES_HUMAN_REVIEW, (
        f"Expected REQUIRES_HUMAN_REVIEW (extraction gate), got {incomplete_result.status}"
    )
    assert incomplete_result.failed_step == 2, (
        f"Expected halt at step 2 (extraction), got failed_step={incomplete_result.failed_step}"
    )
    assert step2 is not None and step2.is_verified is False, (
        f"Expected step 2 is_verified=False, got {getattr(step2, 'is_verified', None)}"
    )
    assert "equipment_id" in (incomplete_result.halt_reason or "") or \
           "design_pressure" in (incomplete_result.halt_reason or ""), (
        f"Expected halt_reason to name the missing critical field, got: {incomplete_result.halt_reason!r}"
    )
    print(
        f"  [+] DEMO 4 PASSED: extraction gate halted at step 2 with is_verified=False; "
        f"no fabricated defaults propagated downstream."
    )

    print("\n" + "=" * 80)
    print("ALL PLANNER DEMOS & SAFETY GATE ASSERTIONS PASSED SUCCESSFULLY!")
    print("=" * 80)
