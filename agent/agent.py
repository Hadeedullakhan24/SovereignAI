"""Agent Orchestration -- Top-Level Orchestrator (agent.py).

This is the ONLY interface the Backend team needs.  It wires together:

  ModelRegistry   -> catalogue of available models
  TaskRouter      -> deterministic rule-based routing
  ToolExecutor    -> sandboxed tool execution with audit log
  AgentPlanner    -> 7-step ReAct safety-gated workflow

Public API
----------
  agent = SovereignAgent()

  # Start a new workflow:
  response = agent.handle(user_request: str, **kwargs) -> AgentResponse

  # Resume from a human-approval checkpoint:
  response = agent.resume(
      checkpoint_id : str,
      approved      : bool,
      engineer_name : str,
      comments      : str = "",
  ) -> AgentResponse

AgentResponse fields (always present, even on error)
-----------------------------------------------------
  status            : str   -- "completed" | "awaiting_approval" |
                               "requires_verification" | "failed" | "rejected"
  output            : Any   -- primary output (step trace dict, etc.)
  requires_approval : bool  -- True iff a human engineer must approve next
  checkpoint_id     : str | None -- ID for resume() when paused
  checkpoint_path   : str | None -- filesystem path of serialised checkpoint
  execution_trace   : str   -- human-readable arrow-trace of steps executed
  reasoning_steps   : list  -- raw list of step dicts for detailed inspection
  halt_reason       : str | None -- why execution halted (if it did)
  failed_step       : int | None -- step number that triggered the halt
  is_verified       : bool  -- False if any unverified step was encountered
  total_time_ms     : float
  model_registry_status : dict -- snapshot of which models are ready
"""

from __future__ import annotations

import logging
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# -- Ensure project root is importable ----------------------------------------
_THIS_DIR = (
    Path(__file__).resolve().parent
    if "__file__" in globals()
    else Path("agent").resolve()
)
_PROJECT_ROOT = _THIS_DIR.parent if _THIS_DIR.name == "agent" else Path(".").resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agent.model_registry import AgentModelRegistry, get_agent_registry
from agent.router import Capability, RoutingDecision, TaskRouter
from agent.tool_executor import DEFAULT_SANDBOX_DIR, ToolExecutor
from agent.planner import (
    AgentPlanner,
    DEFAULT_CHECKPOINTS_DIR,
    PlanExecutionResult,
    PlanStatus,
)

logger = logging.getLogger(__name__)


# =============================================================================
# AgentResponse
# =============================================================================

@dataclass
class AgentResponse:
    """Stable, Backend-facing response object.

    All fields are always present so the caller never needs to handle missing
    attributes.  Key off ``status`` and ``requires_approval`` to decide the
    next action.
    """

    # -- Core status ----------------------------------------------------------
    status: str = "failed"
    requires_approval: bool = False
    is_verified: bool = True

    # -- Payload --------------------------------------------------------------
    output: Any = None                          # step trace dict, answer, etc.
    execution_trace: str = ""                   # "Step 1 -> ... -> COMPLETED"
    reasoning_steps: List[Dict[str, Any]] = field(default_factory=list)

    # -- Pause / resume support -----------------------------------------------
    checkpoint_id: Optional[str] = None
    checkpoint_path: Optional[str] = None

    # -- Diagnostics ----------------------------------------------------------
    halt_reason: Optional[str] = None
    failed_step: Optional[int] = None
    total_time_ms: float = 0.0
    error: Optional[str] = None

    # -- Observability --------------------------------------------------------
    model_registry_status: Dict[str, Any] = field(default_factory=dict)

    # -- Convenience helpers --------------------------------------------------

    def is_complete(self) -> bool:
        return self.status == "completed"

    def is_paused(self) -> bool:
        return self.requires_approval

    def is_failed(self) -> bool:
        return self.status == "failed"

    def to_dict(self) -> Dict[str, Any]:
        """Return a fully JSON-serialisable representation."""
        return {
            "status": self.status,
            "requires_approval": self.requires_approval,
            "is_verified": self.is_verified,
            "output": _safe_serialise(self.output),
            "execution_trace": self.execution_trace,
            "reasoning_steps": self.reasoning_steps,
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_path": self.checkpoint_path,
            "halt_reason": self.halt_reason,
            "failed_step": self.failed_step,
            "total_time_ms": round(self.total_time_ms, 2),
            "error": self.error,
            "model_registry_status": self.model_registry_status,
        }


# -- Internal helpers ---------------------------------------------------------

# Image and scanned-document extensions that should bypass the 7-step plan
# and route directly to VisionInspectorTool.
_IMAGE_EXTENSIONS: frozenset = frozenset({
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp",
    ".pdf",   # treat as scanned document
    ".svg",   # engineering drawing vector format
})

# Regex that matches any filesystem-style token (forward or back slashes,
# dot-extension) with a recognised image extension.  We accept paths that
# include directory separators so that e.g.
#   datasets/handwritten_notes/HN_0001.jpg
# is captured as a whole.
_IMAGE_PATH_RE = re.compile(
    r"(?:^|\s|['\"])"   # start of string, whitespace, or quote
    r"([\w.:/\\-]+"     # path characters (no spaces, includes Windows drive letter colon)
    r"(?:" + "|".join(re.escape(e) for e in sorted(_IMAGE_EXTENSIONS)) + r"))"
    r"(?:$|\s|['\"])",  # end of string, whitespace, or quote
    re.IGNORECASE,
)


def _extract_image_path(text: str) -> Optional[str]:
    """Return the first image/document file path found in *text*, or None.

    Searches for tokens that look like filesystem paths ending in a recognised
    image or scanned-document extension.  Accepts both forward-slash and
    back-slash separators.
    """
    m = _IMAGE_PATH_RE.search(text)
    return m.group(1) if m else None


def _safe_serialise(obj: Any) -> Any:
    """Return obj unchanged if JSON-serialisable, else str(obj)."""
    import json
    try:
        json.dumps(obj)
        return obj
    except (TypeError, OverflowError):
        return str(obj)


def _plan_to_response(
    result: PlanExecutionResult,
    elapsed_ms: float,
    registry_snapshot: Dict[str, Any],
) -> AgentResponse:
    """Convert an internal PlanExecutionResult to a public AgentResponse."""
    _STATUS_MAP: Dict[str, str] = {
        PlanStatus.COMPLETED:               "completed",
        PlanStatus.HUMAN_APPROVAL_REQUIRED: "awaiting_approval",
        PlanStatus.REQUIRES_HUMAN_REVIEW:   "requires_verification",
        PlanStatus.FAILED:                  "failed",
        PlanStatus.REJECTED:                "rejected",
        PlanStatus.IN_PROGRESS:             "in_progress",
    }
    status = _STATUS_MAP.get(result.status, result.status.lower())
    requires_approval = result.status in (
        PlanStatus.HUMAN_APPROVAL_REQUIRED,
        PlanStatus.REQUIRES_HUMAN_REVIEW,
    )
    # Aggregate: False if ANY step came back unverified
    is_verified = all(s.is_verified for s in result.steps)

    return AgentResponse(
        status=status,
        requires_approval=requires_approval,
        is_verified=is_verified,
        output=result.to_dict(),
        execution_trace=result.execution_trace,
        reasoning_steps=[s.to_dict() for s in result.steps],
        checkpoint_id=result.checkpoint_id,
        checkpoint_path=(
            str(result.checkpoint_path) if result.checkpoint_path else None
        ),
        halt_reason=result.halt_reason,
        failed_step=result.failed_step,
        total_time_ms=elapsed_ms,
        model_registry_status=registry_snapshot,
    )


# =============================================================================
# SovereignAgent
# =============================================================================

class SovereignAgent:
    """Top-level orchestrator for Sovereign AI.

    Instantiate once and re-use; all internal components are safe for
    sequential ``handle()`` / ``resume()`` calls.

    Parameters
    ----------
    registry        : AgentModelRegistry (default: auto-discovered from models.yaml).
    tool_executor   : ToolExecutor (default: sandboxed to workspace_sandbox/).
    router          : TaskRouter (default: rule-based deterministic router).
    planner         : AgentPlanner (default: constructed internally).
    sandbox_dir     : Override sandbox root for file / doc operations.
    checkpoints_dir : Override checkpoint persistence directory.
    """

    def __init__(
        self,
        registry: Optional[AgentModelRegistry] = None,
        tool_executor: Optional[ToolExecutor] = None,
        router: Optional[TaskRouter] = None,
        planner: Optional[AgentPlanner] = None,
        sandbox_dir: Union[str, Path] = DEFAULT_SANDBOX_DIR,
        checkpoints_dir: Union[str, Path] = DEFAULT_CHECKPOINTS_DIR,
    ) -> None:
        self.sandbox_dir = Path(sandbox_dir).resolve()
        self.checkpoints_dir = Path(checkpoints_dir).resolve()

        # Build the component graph; each layer accepts injected instances for
        # testing while defaulting to production singletons.
        self.registry: AgentModelRegistry = registry or get_agent_registry()
        self.tool_executor: ToolExecutor = tool_executor or ToolExecutor(
            sandbox_dir=self.sandbox_dir
        )
        self.router: TaskRouter = router or TaskRouter(registry=self.registry)
        self.planner: AgentPlanner = planner or AgentPlanner(
            tool_executor=self.tool_executor,
            router=self.router,
            sandbox_dir=self.sandbox_dir,
            checkpoints_dir=self.checkpoints_dir,
        )
        logger.info(
            "SovereignAgent initialised | sandbox=%s | checkpoints=%s",
            self.sandbox_dir,
            self.checkpoints_dir,
        )

    # -- Internal helpers -----------------------------------------------------

    def _registry_snapshot(self) -> Dict[str, Any]:
        """Lightweight dict of model readiness for observability."""
        try:
            records = self.registry.all()
            snap = {}
            for r in records:
                installed = r.is_installed() if callable(r.is_installed) else bool(r.is_installed)
                if r.enabled and installed:
                    status = "ready"
                elif r.enabled:
                    status = "not_installed"
                else:
                    status = "disabled"
                snap[r.role] = {"enabled": r.enabled, "installed": installed, "status": status}
            return snap
        except Exception as exc:
            logger.warning("Registry snapshot failed: %s", exc)
            return {"error": str(exc)}

    # =========================================================================
    # Public API
    # =========================================================================

    def handle(self, user_request: str, **kwargs: Any) -> AgentResponse:
        """Start a new Sovereign AI workflow.

        Parameters
        ----------
        user_request : Free-text goal from the user / Backend caller.
        **kwargs     : Forwarded to AgentPlanner.run():
                       - report_filename (str)         -- custom sandbox filename
                       - force_ungrounded_calc (bool)  -- force safety-gate demo
                       - max_steps (int)               -- cap on ReAct iterations
                       - vision_file_path (str)        -- explicit image/PDF path
                         (also auto-detected from user_request text).

        Returns
        -------
        AgentResponse
            Always returned -- never raises.  Check ``.status`` and
            ``.requires_approval`` to determine the next action.

        Safety Guarantees (inherited from AgentPlanner)
        -----------------------------------------------
        - Missing critical extraction fields (equipment_id, design_pressure,
          shell_min_thickness) halt at Step 2 with status="requires_verification"
          -- no fabricated defaults propagate downstream.
        - Any ToolResult with is_verified=False immediately halts execution.
        - The human-approval gate at Step 7 returns status="awaiting_approval"
          with a checkpoint_id the caller passes back to resume().

        Vision Fast-Path
        ----------------
        If user_request contains a file path with an image or scanned-document
        extension (.jpg, .jpeg, .png, .pdf, .tiff, .bmp, .webp, .svg) — or if
        ``vision_file_path`` is supplied explicitly — the request is short-
        circuited to ``ToolExecutor.execute('vision_inspector')`` directly,
        bypassing the 7-step inspection plan.  The response ``output`` dict
        contains::

            {
              "file_path"       : str   -- resolved path used,
              "routing_decision": str   -- e.g. 'plain_document' / 'engineering_drawing',
              "text"            : str   -- raw OCR / extracted text,
              "key_value_fields": list  -- structured key-value pairs,
              "equipment_list"  : list  -- equipment tags found,
              "tables"          : list,
              "summary"         : dict,
              "question"        : str | None,
              "execution_time_ms": float,
            }
        """
        t0 = time.perf_counter()
        registry_snap = self._registry_snapshot()
        # ── Router Dispatch ─────────────────────────────────────────────────
        # Determine capability, archetype, and designated execution tool.
        decision: RoutingDecision = self.router.route(user_request)

        # ── Image Generation Direct Route ───────────────────────────────────
        # Route generative text-to-image synthesis requests directly to
        # ToolExecutor.execute("image_generator", ...) bypassing the document planner.
        if decision.capability == Capability.IMAGE_GENERATION or decision.tool_name == "image_generator":
            return self._handle_image_generation(user_request, decision, t0, registry_snap, **kwargs)

        # ── Vision Fast-Path ────────────────────────────────────────────────
        # Detect image/PDF paths in the request and short-circuit to the
        # vision tool, bypassing the 7-step document-inspection planner which
        # has no mechanism to pick up image paths from free-text goals.
        vision_file_path: Optional[str] = (
            kwargs.pop("vision_file_path", None)
            or _extract_image_path(user_request)
        )
        if vision_file_path:
            # An email that names an image/P&ID must not terminate at the
            # inspection fast-path: send the extracted source through the
            # canonical RAG/evidence/email path instead.
            from rag_engine.generation.prompt.task_intent import OutputFormat, TaskClassifier
            email_intent = TaskClassifier.classify(user_request)
            if email_intent.output_format == OutputFormat.EMAIL:
                return self._handle_grounded_email(
                    user_request, vision_file_path, t0, registry_snap, **kwargs
                )
            return self._handle_vision(user_request, vision_file_path, t0, registry_snap, **kwargs)

        # ── Standard 7-step planner path ────────────────────────────────────
        try:
            max_steps = int(kwargs.pop("max_steps", 10))
            result: PlanExecutionResult = self.planner.run(
                user_goal=user_request,
                max_steps=max_steps,
                **kwargs,
            )
        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000.0
            logger.error(
                "SovereignAgent.handle unhandled exception: %s", exc, exc_info=True
            )
            return AgentResponse(
                status="failed",
                error=str(exc),
                total_time_ms=elapsed,
                model_registry_status=registry_snap,
            )

        elapsed = (time.perf_counter() - t0) * 1000.0
        response = _plan_to_response(result, elapsed, registry_snap)
        logger.info(
            "SovereignAgent.handle done | status=%s | time=%.1fms | checkpoint=%s",
            response.status,
            response.total_time_ms,
            response.checkpoint_id,
        )
        return response

    def _handle_grounded_email(
        self,
        user_request: str,
        source_path: str,
        t0: float,
        registry_snap: Dict[str, Any],
        **kwargs: Any,
    ) -> AgentResponse:
        """Run an email with an attached visual source through canonical RAG."""
        try:
            result = self.tool_executor.rag_search(
                query=user_request,
                top_k=kwargs.pop("top_k", 10),
                source_paths=[source_path],
            )
        except Exception as exc:
            return AgentResponse(status="failed", error=str(exc), total_time_ms=(time.perf_counter() - t0) * 1000.0, model_registry_status=registry_snap)
        elapsed = (time.perf_counter() - t0) * 1000.0
        verified = result.get("status") == "success"
        return AgentResponse(
            status="completed" if verified else "requires_verification",
            is_verified=verified,
            output={"email": result.get("answer", ""), "sources": result.get("citations", [])},
            execution_trace="Email -> source extraction -> RAG evidence gate -> grounded email",
            reasoning_steps=[{"step_number": 1, "name": "Grounded email drafting", "step_type": "automated", "action": "rag_search with supplied source", "status": result.get("status"), "is_verified": verified}],
            error=result.get("error"), total_time_ms=elapsed, model_registry_status=registry_snap,
        )

    def _handle_vision(
        self,
        user_request: str,
        file_path: str,
        t0: float,
        registry_snap: Dict[str, Any],
        **kwargs: Any,
    ) -> AgentResponse:
        """Direct vision fast-path: invoke VisionInspectorTool and return results.

        Bypasses the 7-step ReAct planner entirely.  Accepts an explicit
        ``question`` kwarg that is forwarded to the VLM layer if available.
        """
        logger.info(
            "SovereignAgent._handle_vision | file=%r | request=%r",
            file_path, user_request[:80],
        )
        question = kwargs.pop("question", user_request)  # use the full query as VLM prompt
        document_category = kwargs.pop("document_category", "inspection_report")
        use_vlm = kwargs.pop("use_vlm", False)

        try:
            tool_result = self.tool_executor.execute(
                "vision_inspector",
                file_path=file_path,
                question=question,
                document_category=document_category,
                use_vlm=use_vlm,
            )
        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000.0
            logger.error("SovereignAgent._handle_vision error: %s", exc, exc_info=True)
            return AgentResponse(
                status="failed",
                error=str(exc),
                total_time_ms=elapsed,
                model_registry_status=registry_snap,
            )

        elapsed = (time.perf_counter() - t0) * 1000.0

        if tool_result.status in ("success", "warning"):
            agent_status = "completed"
        elif tool_result.status == "capability_unavailable":
            agent_status = "failed"
        else:
            agent_status = "requires_verification"

        # Flatten the VisionInspectorTool output into a clean response payload
        raw_out: Dict[str, Any] = tool_result.output or {}
        output_payload: Dict[str, Any] = {
            "file_path"        : file_path,
            "routing_decision" : raw_out.get("routing_decision"),
            "text"             : raw_out.get("text", ""),
            "key_value_fields" : raw_out.get("key_value_fields", []),
            "equipment_list"   : raw_out.get("equipment_list", []),
            "tables"           : raw_out.get("tables", []),
            "sections"         : raw_out.get("sections", []),
            "summary"          : raw_out.get("summary", {}),
            "question"         : question,
            "execution_time_ms": raw_out.get("execution_time_ms", elapsed),
        }
        if "vlm" in raw_out:
            output_payload["vlm"] = raw_out["vlm"]

        response = AgentResponse(
            status=agent_status,
            requires_approval=False,
            is_verified=tool_result.is_verified,
            output=output_payload,
            execution_trace=(
                f"Vision fast-path: vision_inspector({file_path!r}) "
                f"-> {tool_result.status} "
                f"[{raw_out.get('routing_decision', 'unknown')}]"
            ),
            reasoning_steps=[{
                "step_number": 1,
                "name": "Vision OCR / Document Parse",
                "step_type": "automated",
                "action": f"vision_inspector(file_path={file_path!r})",
                "observation": (
                    f"Extracted {len(raw_out.get('text', ''))} chars | "
                    f"routing={raw_out.get('routing_decision')} | "
                    f"kv_fields={len(raw_out.get('key_value_fields', []))} | "
                    f"equipment={len(raw_out.get('equipment_list', []))}"
                ),
                "status": tool_result.status,
                "is_verified": tool_result.is_verified,
                "execution_time_ms": round(elapsed, 2),
            }],
            error=tool_result.error,
            total_time_ms=elapsed,
            model_registry_status=registry_snap,
        )
        logger.info(
            "SovereignAgent._handle_vision done | status=%s | chars=%d | time=%.1fms",
            agent_status,
            len(raw_out.get("text", "")),
            elapsed,
        )
        return response

    def _handle_image_generation(
        self,
        user_request: str,
        decision: RoutingDecision,
        t0: float,
        registry_snap: Dict[str, Any],
        **kwargs: Any,
    ) -> AgentResponse:
        """Direct image-generation path: invoke ImageGeneratorTool and return results.

        Bypasses the 7-step ReAct planner. Extracts generation parameters
        from natural language task or kwargs, runs local Stable Diffusion via
        ToolExecutor, and formats the response.
        """
        logger.info(
            "SovereignAgent._handle_image_generation | request=%r",
            user_request[:100],
        )
        from agent.router import extract_image_generation_params

        # Extract generation parameters
        params = extract_image_generation_params(user_request, **kwargs)

        try:
            tool_result = self.tool_executor.execute(
                decision,
                task=user_request,
                **params,
            )
        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000.0
            logger.error("SovereignAgent._handle_image_generation error: %s", exc, exc_info=True)
            return AgentResponse(
                status="failed",
                error=str(exc),
                total_time_ms=elapsed,
                model_registry_status=registry_snap,
            )

        elapsed = (time.perf_counter() - t0) * 1000.0

        if tool_result.status == "success":
            agent_status = "completed"
        elif tool_result.status == "capability_unavailable":
            agent_status = "failed"
        else:
            agent_status = "failed"

        output_payload: Dict[str, Any] = tool_result.output or {}
        if tool_result.metadata:
            output_payload.setdefault("metadata", tool_result.metadata)

        prompt_used = params.get("prompt") or user_request
        trace_summary = (
            f"Image generation: image_generator({prompt_used!r}) "
            f"-> {tool_result.status} "
            f"[{output_payload.get('filename', 'artifact')}]"
        )

        reasoning_step = {
            "step_number": 1,
            "name": "Local Stable Diffusion Image Synthesis",
            "step_type": "automated",
            "action": f"image_generator(prompt={prompt_used!r})",
            "observation": (
                f"Generated {output_payload.get('filename')} "
                f"({output_payload.get('width')}x{output_payload.get('height')}) in "
                f"{output_payload.get('generation_time_seconds', 0):.2f}s | "
                f"Peak VRAM: {output_payload.get('peak_vram_mb', 0):.1f} MB"
                if tool_result.status == "success"
                else f"Generation failed: {tool_result.error}"
            ),
            "status": tool_result.status,
            "is_verified": tool_result.is_verified,
            "execution_time_ms": round(elapsed, 2),
        }

        response = AgentResponse(
            status=agent_status,
            requires_approval=False,
            is_verified=tool_result.is_verified,
            output=output_payload,
            execution_trace=trace_summary,
            reasoning_steps=[reasoning_step],
            error=tool_result.error,
            total_time_ms=elapsed,
            model_registry_status=registry_snap,
        )
        logger.info(
            "SovereignAgent._handle_image_generation done | status=%s | file=%s | time=%.1fms",
            agent_status,
            output_payload.get("filename"),
            elapsed,
        )
        return response

    def resume(
        self,
        checkpoint_id: str,
        approved: bool,
        engineer_name: str,
        comments: str = "",
    ) -> AgentResponse:
        """Resume a paused workflow after human engineer review.

        Parameters
        ----------
        checkpoint_id : Checkpoint ID returned in a previous AgentResponse.
        approved      : True = grant approval; False = reject.
        engineer_name : Full name / credentials of the reviewing engineer.
        comments      : Engineering notes / sign-off rationale (full audit trail).

        Returns
        -------
        AgentResponse
            status = "completed" if approved, "rejected" if not.
        """
        t0 = time.perf_counter()
        registry_snap = self._registry_snapshot()
        logger.info(
            "SovereignAgent.resume | checkpoint=%s | approved=%s | engineer=%s",
            checkpoint_id,
            approved,
            engineer_name,
        )

        try:
            result: PlanExecutionResult = self.planner.resume(
                checkpoint_id_or_path=checkpoint_id,
                engineer_name=engineer_name,
                approved=approved,
                comments=comments,
            )
        except FileNotFoundError as exc:
            elapsed = (time.perf_counter() - t0) * 1000.0
            logger.error("SovereignAgent.resume: checkpoint not found: %s", exc)
            return AgentResponse(
                status="failed",
                error=f"Checkpoint not found: {checkpoint_id} -- {exc}",
                total_time_ms=elapsed,
                model_registry_status=registry_snap,
            )
        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000.0
            logger.error(
                "SovereignAgent.resume unhandled exception: %s", exc, exc_info=True
            )
            return AgentResponse(
                status="failed",
                error=str(exc),
                total_time_ms=elapsed,
                model_registry_status=registry_snap,
            )

        elapsed = (time.perf_counter() - t0) * 1000.0
        response = _plan_to_response(result, elapsed, registry_snap)
        logger.info(
            "SovereignAgent.resume done | status=%s | engineer=%s | time=%.1fms",
            response.status,
            engineer_name,
            response.total_time_ms,
        )
        return response

    def warmup(self, throwaway_query: str = "Warmup: verify equipment status and inspection standards.") -> float:
        """Pre-warm the SovereignAgent pipeline by executing an initial throwaway query.

        This absorbs the ~30-second cold-start latency (loading PyTorch LLM weights
        into memory, initializing SentenceTransformer embeddings, and opening Qdrant
        vector indices) before any real user-facing queries are handled.
        Subsequent queries execute rapidly (~150ms on cache hit, a few seconds when
        cache misses but models are warm).

        Parameters
        ----------
        throwaway_query : Throwaway goal to prime RAG, embeddings, and models.

        Returns
        -------
        float
            Warmup duration in milliseconds.
        """
        logger.info("Pre-warming SovereignAgent (absorbing cold-start model load cost)...")
        t0 = time.perf_counter()
        self.handle(throwaway_query)
        duration_ms = (time.perf_counter() - t0) * 1000.0
        logger.info("SovereignAgent warmup complete in %.2f ms", duration_ms)
        return duration_ms

    def health(self) -> Dict[str, Any]:
        """Return a health / status snapshot for monitoring endpoints."""
        return {
            "agent": "SovereignAgent",
            "sandbox_dir": str(self.sandbox_dir),
            "checkpoints_dir": str(self.checkpoints_dir),
            "models": self._registry_snapshot(),
        }


# -- Convenience factory function ---------------------------------------------

def create_agent(**kwargs: Any) -> SovereignAgent:
    """Factory -- Backend can do: from agent.agent import create_agent"""
    return SovereignAgent(**kwargs)


# =============================================================================
# Self-verification demos
# =============================================================================

if __name__ == "__main__":
    import json as _json

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    agent = SovereignAgent()

    # -------------------------------------------------------------------------
    # WARMUP: Absorb initial ~30s cold-start latency before user-facing queries
    # -------------------------------------------------------------------------
    # Calling agent.handle() once with a throwaway query immediately after
    # SovereignAgent() construction absorbs the ~30 second cold-start overhead
    # (PyTorch weights loading, SentenceTransformer embeddings, and Qdrant index).
    # Subsequent queries run fast (confirmed ~150ms when cache hits, few seconds
    # when cache misses but model is warm).
    print("\n" + "=" * 80)
    print("WARMUP: Pre-warming SovereignAgent (absorbing ~30s cold-start overhead)...")
    print("=" * 80)
    warmup_duration_ms = agent.warmup()
    print(f"  [+] Warmup complete in {warmup_duration_ms:.1f} ms. System is hot and ready for demo.")

    # -------------------------------------------------------------------------
    # DEMO 1: Normal end-to-end run (should pause at Step 7 for approval)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("DEMO 1: Normal end-to-end run via agent.handle()")
    print("=" * 80)

    r1 = agent.handle(
        "Process inspection report for V-2201 knockout drum, "
        "verify calculations, and prepare statutory approval document."
    )
    print(f"\nStatus           : {r1.status}")
    print(f"Requires Approval: {r1.requires_approval}")
    print(f"Is Verified      : {r1.is_verified}")
    print(f"Checkpoint ID    : {r1.checkpoint_id}")
    print(f"Execution Trace  : {r1.execution_trace}")
    print(f"Total Time       : {r1.total_time_ms:.1f} ms")
    print(f"Model Registry   : {_json.dumps(r1.model_registry_status, indent=2)}")

    assert r1.status == "awaiting_approval", (
        f"Expected 'awaiting_approval', got {r1.status!r}"
    )
    assert r1.requires_approval is True, "Expected requires_approval=True"
    assert r1.checkpoint_id is not None, "Expected a checkpoint_id"
    assert r1.is_verified is True, "Expected all steps verified in normal run"
    print("\n  [+] DEMO 1 PASSED: normal run correctly paused for engineer approval")

    # -------------------------------------------------------------------------
    # DEMO 2: Resume from checkpoint after engineer approval
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("DEMO 2: Resume after engineer approval via agent.resume()")
    print("=" * 80)

    r2 = agent.resume(
        checkpoint_id=r1.checkpoint_id,
        approved=True,
        engineer_name="S. Menon (Authorized Inspector #AI-9021)",
        comments=(
            "Thickness survey verified against ASME Section VIII Div 1. "
            "Continued operation approved for 5 years."
        ),
    )
    print(f"\nStatus           : {r2.status}")
    print(f"Requires Approval: {r2.requires_approval}")
    print(f"Execution Trace  : {r2.execution_trace}")
    print(f"Total Time       : {r2.total_time_ms:.1f} ms")

    assert r2.status == "completed", f"Expected 'completed', got {r2.status!r}"
    assert r2.requires_approval is False
    print("\n  [+] DEMO 2 PASSED: resumed workflow completed after approval")

    # -------------------------------------------------------------------------
    # DEMO 3: Ungrounded calculation halts at Step 5
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("DEMO 3: CRITICAL GATE -- Ungrounded calculation halts at Step 5")
    print("=" * 80)

    r3 = agent.handle(
        "Process ungrounded inspection spec calculation for pressure vessel V-2201",
        force_ungrounded_calc=True,
    )
    print(f"\nStatus           : {r3.status}")
    print(f"Requires Approval: {r3.requires_approval}")
    print(f"Is Verified      : {r3.is_verified}")
    print(f"Failed Step      : {r3.failed_step}")
    print(f"Halt Reason      : {r3.halt_reason}")
    print(f"Execution Trace  : {r3.execution_trace}")

    assert r3.status == "requires_verification", (
        f"Expected 'requires_verification', got {r3.status!r}"
    )
    assert r3.requires_approval is True
    assert r3.failed_step == 5, f"Expected step 5, got {r3.failed_step}"
    assert r3.is_verified is False, "Expected is_verified=False on ungrounded run"
    print(
        "\n  [+] DEMO 3 PASSED: ungrounded calculation halted at Step 5 "
        "with status='requires_verification'"
    )

    # -------------------------------------------------------------------------
    # DEMO 4: Incomplete report triggers extraction gate at Step 2
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("DEMO 4: EXTRACTION GATE -- Missing Equipment ID / Design Pressure")
    print("=" * 80)

    from agent.tool_executor import DEFAULT_SANDBOX_DIR as _SANDBOX
    _incomplete = _SANDBOX / "minimal_incomplete_report.md"
    _incomplete.write_text(
        "# INSPECTION REPORT\n"
        "**Date of Inspection:** 03-Sep-2026\n"
        "**Inspector:** J. Kumar, API Inspector\n\n"
        "## Thickness Survey Summary\n"
        "| Component | Original (mm) | Current (mm) |\n"
        "|---|---|---|\n"
        "| Shell (Bottom) | 12.0 | 11.2 |\n\n"
        "## Findings\n"
        "Vessel in satisfactory condition.\n",
        encoding="utf-8",
    )
    print(f"Wrote incomplete report -> {_incomplete}")

    r4 = agent.handle(
        "Process incomplete_report.md and verify calculations",
        report_filename="minimal_incomplete_report.md",
    )
    print(f"\nStatus           : {r4.status}")
    print(f"Requires Approval: {r4.requires_approval}")
    print(f"Is Verified      : {r4.is_verified}")
    print(f"Failed Step      : {r4.failed_step}")
    print(f"Halt Reason      : {r4.halt_reason}")
    print(f"Execution Trace  : {r4.execution_trace}")

    assert r4.status == "requires_verification", (
        f"Expected 'requires_verification' (extraction gate), got {r4.status!r}"
    )
    assert r4.failed_step == 2, f"Expected halt at step 2, got {r4.failed_step}"
    assert r4.is_verified is False
    assert (
        "equipment_id" in (r4.halt_reason or "")
        or "design_pressure" in (r4.halt_reason or "")
    ), f"Halt reason should name the missing field: {r4.halt_reason!r}"
    print(
        "\n  [+] DEMO 4 PASSED: extraction gate halted at Step 2 -- "
        "no fabricated defaults reached downstream steps."
    )

    # -------------------------------------------------------------------------
    # DEMO 5: agent.health() returns valid snapshot
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("DEMO 5: agent.health() snapshot")
    print("=" * 80)
    h = agent.health()
    print(_json.dumps(h, indent=2))
    assert "models" in h and "sandbox_dir" in h
    print("\n  [+] DEMO 5 PASSED: health snapshot returned")

    # -------------------------------------------------------------------------
    # DEMO 6: AgentResponse.to_dict() is fully JSON-serialisable
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("DEMO 6: AgentResponse.to_dict() JSON round-trip")
    print("=" * 80)
    for i, resp in enumerate([r1, r2, r3, r4], start=1):
        d = resp.to_dict()
        _json.dumps(d)   # raises TypeError if not serialisable
        print(f"  Demo {i} response to_dict() OK -- status={d['status']!r}")
    print("  [+] DEMO 6 PASSED: all AgentResponse objects serialise cleanly")

    print("\n" + "=" * 80)
    print("ALL AGENT DEMOS & SAFETY ASSERTIONS PASSED SUCCESSFULLY!")
    print("=" * 80)
