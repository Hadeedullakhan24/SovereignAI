from pathlib import Path

agent_file = Path(r"e:\SovereignAI\agent\agent.py")
content = agent_file.read_text(encoding="utf-8")

old_marker = """def _plan_to_response(
    result: PlanExecutionResult,
    elapsed_ms: float,
    registry_snapshot: Dict[str, Any],
) -> AgentResponse:"""

# We find where _plan_to_response begins
start_idx = content.find(old_marker)
if start_idx == -1:
    print("Could not find start marker")
    exit(1)

# And find where `def __init__(` begins
init_marker = "    def __init__("
init_idx = content.find(init_marker, start_idx)
if init_idx == -1:
    print("Could not find init marker")
    exit(1)

# Find the start of class SovereignAgent docstring before init
class_doc_marker = '    """\n\n    def __init__('
doc_end_idx = content.find(class_doc_marker, start_idx)

new_code = '''def _plan_to_response(
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
    is_verified = all(s.is_verified for s in result.steps)

    ctx = result.context_state or {}
    answer = ctx.get("approval_note") or ctx.get("sop_answer") or result.halt_reason or result.execution_trace
    artifact = None
    doc_out = ctx.get("document_result")
    if isinstance(doc_out, dict) and "filename" in doc_out:
        fname = doc_out["filename"]
        ext = Path(fname).suffix.lower().lstrip(".")
        artifact = {
            "artifact_type": ext.upper() if ext else "DOCX",
            "filename": fname,
            "file_path": str(doc_out.get("path") or ""),
            "file_size_bytes": int(doc_out.get("file_size_bytes") or 0),
            "status": "verified" if is_verified else "generated",
            "metadata": doc_out.get("metadata", {}),
        }

    return AgentResponse(
        status=status,
        requires_approval=requires_approval,
        is_verified=is_verified,
        output=result.to_dict(),
        answer=answer,
        artifact=artifact,
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
'''

if doc_end_idx != -1:
    replaced = content[:start_idx] + new_code + content[doc_end_idx:]
    agent_file.write_text(replaced, encoding="utf-8")
    print("Successfully replaced _plan_to_response and SovereignAgent header")
else:
    print("doc_end_idx not found, looking for class SovereignAgent")
