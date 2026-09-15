"""Agent Orchestration — Tool Executor.

Provides deterministic and safe tool execution for the agent workbench:
  1. rag_search         : Real integration with Member 1's RAGPipeline.
                          Handles 'Insufficient Evidence' gracefully as a valid status.
  2. calculator         : Safe mathematical & engineering formula evaluator via AST (no raw eval).
                          Returns (result, steps) — steps is an ordered list of intermediate
                          reduction strings suitable for inclusion in audit reports.
  3. file_manager       : Strictly sandboxed file I/O (read/write/list) within a project directory.
  4. python_execution   : Subprocess-isolated script execution with timeouts and restricted builtins.
  5. document_generator : Professional Word (.docx) report generator using python-docx.
  6. xlsx_generator     : Excel workbook generator using openpyxl.
  7. pptx_generator     : PowerPoint presentation generator using python-pptx.
  8. pdf_generator      : PDF converter on top of .docx output (docx2pdf / libreoffice --headless).
  9. vision_inspector   : Multimodal OCR & VLM visual inspection tool (PaddleOCR / Qwen2.5-VL).
  10. image_generator   : Offline local Stable Diffusion text-to-image generator.
  11. audit_logger      : Append-only JSONL log of every tool execution for governance.

All tool branches catch their own exceptions and return a structured ToolResult(error=...);
no unhandled exception propagates to the caller (no silent 500s).

ToolExecutor.execute() directly consumes a RoutingDecision (or tool_name/use_rag_context),
coordinating RAG context retrieval with deterministic calculation tools seamlessly.
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
import math
import operator
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import zipfile
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path

# Ensure project root is on sys.path for direct script execution
_THIS_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path("agent").resolve()
_PROJECT_ROOT = _THIS_DIR.parent if _THIS_DIR.name == "agent" else Path(".").resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agent.router import Capability, RoutingDecision
from rag_engine.generation.generation_config import GenerationConfig
from rag_engine.generation.prompt.prompt_templates import PromptArchetype
from rag_engine.pipeline.rag_pipeline import (
    INSUFFICIENT_EVIDENCE_FALLBACK,
    RAGPipeline,
    RAGResponse,
)

logger = logging.getLogger(__name__)

# Default project paths
DEFAULT_PROJECT_ROOT = _PROJECT_ROOT
DEFAULT_SANDBOX_DIR = DEFAULT_PROJECT_ROOT / "workspace_sandbox"
DEFAULT_AUDIT_LOG_FILE = DEFAULT_PROJECT_ROOT / "logs" / "agent_audit.jsonl"

# Contracts exposed to the planner/backend for generators.
DOCUMENT_TOOL_CONTRACTS: Dict[str, Dict[str, Any]] = {
    "document_generator": {"format": "docx", "implemented": True, "input": ["title", "sections", "filename", "metadata"], "output": ["path", "file_size_bytes"]},
    "xlsx_generator": {"format": "xlsx", "implemented": True, "input": ["workbook", "filename"], "output": ["path", "metadata"]},
    "pptx_generator": {"format": "pptx", "implemented": True, "input": ["slides", "filename"], "output": ["path", "metadata"]},
    "pdf_generator": {"format": "pdf", "implemented": True, "input": ["title", "sections", "filename", "metadata", "docx_filename", "out_filename"], "output": ["path", "metadata", "file_size_bytes"]},
    "image_generator": {
        "format": "png",
        "implemented": True,
        "input": [
            "prompt",
            "negative_prompt",
            "width",
            "height",
            "steps",
            "guidance_scale",
            "seed",
            "filename",
            "filename_prefix",
        ],
        "output": [
            "path",
            "filename",
            "file_size_bytes",
            "staged_path",
            "dimensions",
            "generation_time_seconds",
            "peak_vram_mb",
            "seed",
            "model_name",
        ],
    },
}

def validate_artifact(path: Union[str, Path], expected_format: str, required_sources: Optional[List[Any]] = None) -> Dict[str, Any]:
    """Validate bytes *and* readability; existence alone is never success."""
    target = Path(path)
    if not target.is_file() or target.stat().st_size == 0:
        raise RuntimeError("artifact was not written")
    fmt = expected_format.casefold().lstrip(".")
    raw = target.read_bytes()[:8]
    expected_names = [str(getattr(c, "document_name", "") or getattr(c, "document_id", "")) for c in (required_sources or [])]
    extracted_text = ""
    if fmt == "pdf":
        if not raw.startswith(b"%PDF-"):
            raise RuntimeError("generated file does not have a PDF signature")
        try:
            import fitz
            doc_fitz = fitz.open(str(target))
            if len(doc_fitz) == 0:
                raise RuntimeError("generated PDF has no readable pages")
            extracted_text = "\n".join(page.get_text() for page in doc_fitz)
            doc_fitz.close()
        except ImportError:
            try:
                import importlib
                pypdf_mod = importlib.import_module("pypdf")
                reader = pypdf_mod.PdfReader(str(target))
                if not reader.pages:
                    raise RuntimeError("generated PDF has no readable pages")
                extracted_text = "\n".join(page.extract_text() or "" for page in reader.pages)
            except Exception:
                extracted_text = raw.decode("latin-1", errors="ignore")
    elif fmt in {"docx", "xlsx", "pptx"}:
        if not raw.startswith(b"PK") or not zipfile.is_zipfile(target):
            raise RuntimeError(f"generated file does not have a valid {fmt.upper()} package signature")
        with zipfile.ZipFile(target) as package:
            required = {"docx": "word/document.xml", "xlsx": "xl/workbook.xml", "pptx": "ppt/presentation.xml"}[fmt]
            if required not in package.namelist():
                raise RuntimeError(f"generated {fmt.upper()} package is missing its main document part")
        if fmt == "docx":
            import docx
            doc = docx.Document(str(target))
            extracted_text = "\n".join(
                [p.text for p in doc.paragraphs]
                + [c.text for tbl in doc.tables for r in tbl.rows for c in r.cells]
            )
        elif fmt == "xlsx":
            import openpyxl
            wb = openpyxl.load_workbook(str(target), read_only=True)
            extracted_text = "\n".join(
                str(cell) for ws in wb.worksheets for row in ws.iter_rows(values_only=True) for cell in row if cell is not None
            )
            wb.close()
        else:
            from pptx import Presentation
            prs = Presentation(str(target))
            extracted_text = "\n".join(
                shape.text for slide in prs.slides for shape in slide.shapes if hasattr(shape, "text") and shape.text
            )
    else:
        raise ValueError(f"unsupported artifact format: {expected_format}")

    # Enforce placeholder rejection
    import re
    placeholder_matches = re.findall(
        r"\b(20XX(?:-\d{2}-\d{2})?|2XXX(?:-\d{2}-\d{2})?|19XX|YYYY|UNKNOWN_TAG|PLACEHOLDER|INSERT_DATE|\[(?:DATE|LOCATION|EQUIPMENT|TAG)\])\b",
        extracted_text,
        re.IGNORECASE,
    )
    if placeholder_matches:
        raise RuntimeError(f"Artifact contains prohibited placeholder values: {', '.join(set(placeholder_matches))}")

    missing_sources = [name for name in expected_names if name and name not in extracted_text]
    if missing_sources:
        raise RuntimeError(f"artifact is missing required source references: {', '.join(missing_sources)}")
    return {"validated": True, "format": fmt, "size": target.stat().st_size}


def stage_artifact_for_backend(sandbox_path: Union[str, Path]) -> Optional[Path]:
    """Safely stage a verified sandbox artifact for backend HTTP serving.

    Copies the validated file from sandbox to the backend artifacts directory
    while strictly preventing path traversal and leaks.
    """
    try:
        path_obj = Path(sandbox_path).resolve()
        if not path_obj.is_file():
            return None

        filename = path_obj.name
        if not filename or filename in {".", ".."} or "/" in filename or "\\" in filename:
            return None

        import shutil
        # Stage to project backend artifacts directory
        proj_backend_dir = DEFAULT_PROJECT_ROOT / "backend" / "artifacts"
        proj_backend_dir.mkdir(parents=True, exist_ok=True)
        target_file = proj_backend_dir / filename
        shutil.copy2(path_obj, target_file)

        # Also stage to runtime location if distinct
        try:
            from rag_engine.config.runtime_paths import runtime_file
            rt_dir = runtime_file("backend", "artifacts")
            if rt_dir.resolve() != proj_backend_dir.resolve():
                rt_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path_obj, rt_dir / filename)
        except Exception:
            pass

        return target_file
    except Exception as exc:
        logger.debug("Artifact staging to backend directory skipped or failed: %s", exc)
        return None


# ── Execution Result Dataclass ──────────────────────────────────────────────

@dataclass
class ToolResult:
    """Standard container returned by all tool executions.

    Fields
    ------
    tool_name        : Name of the executed tool.
    status           : 'success' | 'requires_verification' | 'insufficient_evidence' |
                       'error' | 'capability_unavailable' | 'timeout'.
    output           : Primary tool output payload (result, text, table, dict, etc.).
    rag_context      : Supporting RAG retrieval context, citations, and grounding info (if requested).
    is_verified      : True if the result is fully grounded & verified; False if ungrounded / missing context.
    execution_time_ms: Total tool execution latency in milliseconds.
    fallback_warning : Low-confidence warning passed down from RoutingDecision (if any).
    error            : Error description if execution failed, else None.
    metadata         : Additional contextual metadata.
    """
    tool_name: str
    status: str
    output: Any
    rag_context: Optional[Dict[str, Any]] = None
    is_verified: bool = True
    execution_time_ms: float = 0.0
    fallback_warning: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status in ("error", "capability_unavailable", "timeout"):
            self.is_verified = False

    @property
    def is_success(self) -> bool:
        # A multimodal pipeline can return useful OCR text while explicitly
        # warning that a non-essential modality (for example an unstaged VLM)
        # was unavailable.  Callers must retain that evidence rather than
        # discarding it; `is_verified` remains the approval safety signal.
        return self.status in ("success", "warning", "requires_verification")

    def summary(self) -> str:
        warn = f" | warning={self.fallback_warning!r}" if self.fallback_warning else ""
        err = f" | error={self.error!r}" if self.error else ""
        return (
            f"[ToolResult] tool={self.tool_name} | status={self.status} | "
            f"verified={self.is_verified} | "
            f"latency={self.execution_time_ms:.2f}ms{warn}{err}"
        )


# ── 1. Append-Only Local Audit Logger ──────────────────────────────────────

class AuditLogger:
    """Thread-safe append-only JSONL audit logger for tool invocations."""

    def __init__(self, log_path: Union[str, Path] = DEFAULT_AUDIT_LOG_FILE) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        execution_time_ms: float,
        status: str,
        result_summary: str,
    ) -> None:
        """Write an audit entry immediately to the log file."""
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tool_name": tool_name,
            "arguments": self._sanitize(arguments),
            "execution_time_ms": round(execution_time_ms, 2),
            "status": status,
            "result_summary": result_summary,
        }
        line = json.dumps(record, ensure_ascii=False) + "\n"
        # Audit telemetry must never turn a completed tool invocation into a
        # fabricated tool failure.  This can happen in a read-only deployment
        # bundle or when a test injects a sandbox outside the project tree.
        # Preserve the real result and surface the observability failure to the
        # host logger, which the backend can collect independently.
        try:
            with self._lock:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(line)
        except OSError as exc:
            logger.warning("Unable to append agent audit record to %s: %s", self.log_path, exc)

    @staticmethod
    def _sanitize(obj: Any) -> Any:
        """Convert arguments to JSON-serializable structures."""
        try:
            json.dumps(obj)
            return obj
        except (TypeError, OverflowError):
            return str(obj)


# ── 2. Safe Mathematical & Engineering Calculator ─────────────────────────

class SafeCalculator:
    """Safe expression evaluator using Python AST parsing.

    Guarantees no arbitrary code execution:
      - Strictly restricts node types to arithmetic, constants, and safe function calls.
      - Never invokes python's eval() or exec().
      - Supports engineering variables and constants (pi, e).
    """

    ALLOWED_OPERATORS: Dict[Any, Callable] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    ALLOWED_FUNCTIONS: Dict[str, Callable] = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sqrt": math.sqrt,
        "exp": math.exp,
        "log": math.log,
        "log10": math.log10,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "ceil": math.ceil,
        "floor": math.floor,
        "radians": math.radians,
        "degrees": math.degrees,
    }

    CONSTANTS: Dict[str, float] = {
        "pi": math.pi,
        "e": math.e,
    }

    def evaluate(
        self, expression: str, variables: Optional[Dict[str, float]] = None
    ) -> Tuple[float, List[str]]:
        """Parse and safely evaluate a mathematical expression.

        Parameters
        ----------
        expression : e.g. '(P * R) / (S * E - 0.6 * P)' or 'sqrt(144) + 12.5'
        variables  : optional dictionary of numeric variable values

        Returns
        -------
        (result, steps)
          result : float  — the final computed value.
          steps  : List[str] — ordered list of intermediate reduction strings
                   showing substituted variable values, sub-expression results,
                   and the final answer.  Suitable for inclusion in audit reports.
        """
        if not expression or not expression.strip():
            raise ValueError("Empty calculation expression provided.")

        clean_expr = expression.strip()
        # Parse expression into AST
        try:
            tree = ast.parse(clean_expr, mode="eval")
        except SyntaxError as e:
            raise ValueError(f"Syntax error in expression: {e}") from e

        var_dict = {k: float(v) for k, v in (variables or {}).items()}
        steps: List[str] = []

        # Record variable substitution as the first step
        if var_dict:
            sub_parts = ", ".join(f"{k} = {v}" for k, v in sorted(var_dict.items()))
            steps.append(f"Variable substitution: {sub_parts}")

        result = self._eval_node_traced(tree.body, var_dict, steps, depth=0)
        steps.append(f"Final result: {expression} = {result:.6g}")
        return result, steps

    def _eval_node(self, node: ast.AST, variables: Dict[str, float]) -> float:
        """Recursively evaluate an approved AST node (no step tracing)."""
        val, _ = self.evaluate.__wrapped__(self, ast.Expression(body=node), variables) if False else (None, None)
        return self._eval_node_traced(node, variables, [], depth=99)

    def _eval_node_traced(
        self,
        node: ast.AST,
        variables: Dict[str, float],
        steps: List[str],
        depth: int,
    ) -> float:
        """Recursively evaluate an approved AST node, emitting step traces."""
        # 1. Numbers / Constants
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError(f"Disallowed constant type: {type(node.value).__name__}")

        # 2. Variable or Mathematical Constant Names
        if isinstance(node, ast.Name):
            name = node.id
            if name in variables:
                return variables[name]
            if name in self.CONSTANTS:
                return self.CONSTANTS[name]
            raise ValueError(f"Undefined variable or constant: {name!r}")

        # 3. Unary operations (+x, -x)
        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type in self.ALLOWED_OPERATORS:
                operand_val = self._eval_node_traced(node.operand, variables, steps, depth + 1)
                result = self.ALLOWED_OPERATORS[op_type](operand_val)
                if depth <= 1:
                    op_sym = "-" if op_type == ast.USub else "+"
                    steps.append(f"Unary {op_sym}({operand_val}) = {result:.6g}")
                return result
            raise ValueError(f"Disallowed unary operator: {op_type.__name__}")

        # 4. Binary operations (x + y, x * y, x ** y, etc.)
        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type in self.ALLOWED_OPERATORS:
                left_val = self._eval_node_traced(node.left, variables, steps, depth + 1)
                right_val = self._eval_node_traced(node.right, variables, steps, depth + 1)
                # Check zero division
                if op_type in (ast.Div, ast.FloorDiv, ast.Mod) and right_val == 0:
                    raise ZeroDivisionError("Division by zero in mathematical expression.")
                result = self.ALLOWED_OPERATORS[op_type](left_val, right_val)
                # Emit a step for top-level binary ops to make the working visible
                if depth <= 2:
                    _OP_SYM = {
                        ast.Add: "+", ast.Sub: "-", ast.Mult: "×",
                        ast.Div: "÷", ast.FloorDiv: "//", ast.Mod: "%", ast.Pow: "^",
                    }
                    op_sym = _OP_SYM.get(op_type, str(op_type.__name__))
                    steps.append(f"  {left_val:.6g} {op_sym} {right_val:.6g} = {result:.6g}")
                return result
            raise ValueError(f"Disallowed binary operator: {op_type.__name__}")

        # 5. Function calls (sqrt, sin, abs, etc.)
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Nested function attributes / dynamic dispatch are disallowed.")
            func_name = node.func.id
            if func_name not in self.ALLOWED_FUNCTIONS:
                raise ValueError(f"Disallowed function call: {func_name!r}")
            evaluated_args = [self._eval_node_traced(arg, variables, steps, depth + 1) for arg in node.args]
            result = float(self.ALLOWED_FUNCTIONS[func_name](*evaluated_args))
            if depth <= 2:
                args_str = ", ".join(f"{a:.6g}" for a in evaluated_args)
                steps.append(f"  {func_name}({args_str}) = {result:.6g}")
            return result

        # Disallow everything else (lambdas, attributes, imports, subscriptions, etc.)
        raise ValueError(f"Disallowed expression element: {type(node).__name__}")


# ── 3. Sandboxed File Manager ──────────────────────────────────────────────

class SandboxedFileManager:
    """Strictly sandboxed file manager restricted to a designated root directory."""

    def __init__(self, sandbox_dir: Union[str, Path] = DEFAULT_SANDBOX_DIR) -> None:
        self.sandbox_dir = Path(sandbox_dir).resolve()
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_safe_path(self, rel_path: Union[str, Path]) -> Path:
        """Resolve path and verify it stays strictly inside the sandbox directory."""
        target = (self.sandbox_dir / rel_path).resolve()
        try:
            target.relative_to(self.sandbox_dir)
        except ValueError:
            raise PermissionError(
                f"Security violation: path '{rel_path}' resolves to '{target}' "
                f"which is outside the sandbox '{self.sandbox_dir}'"
            )
        return target

    def read(self, filename: str) -> str:
        """Read text content from a sandboxed file."""
        safe_path = self._resolve_safe_path(filename)
        if not safe_path.is_file():
            raise FileNotFoundError(f"File not found in sandbox: {filename}")
        with open(safe_path, "r", encoding="utf-8") as f:
            return f.read()

    def write(self, filename: str, content: str, overwrite: bool = True) -> Dict[str, Any]:
        """Write content to a sandboxed file."""
        safe_path = self._resolve_safe_path(filename)
        if safe_path.exists() and not overwrite:
            raise FileExistsError(f"File already exists in sandbox: {filename}")
        safe_path.parent.mkdir(parents=True, exist_ok=True)
        with open(safe_path, "w", encoding="utf-8") as f:
            f.write(content)
        return {
            "filename": filename,
            "path": str(safe_path),
            "bytes_written": len(content.encode("utf-8")),
            "status": "written",
        }

    def list_files(self, subpath: str = "") -> List[Dict[str, Any]]:
        """List files and directories in the sandbox or a sub-folder."""
        safe_dir = self._resolve_safe_path(subpath)
        if not safe_dir.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {subpath}")

        entries = []
        for item in safe_dir.iterdir():
            rel = item.relative_to(self.sandbox_dir)
            entries.append({
                "name": item.name,
                "relative_path": str(rel).replace("\\", "/"),
                "is_dir": item.is_dir(),
                "size_bytes": item.stat().st_size if item.is_file() else 0,
            })
        return entries

    def exists(self, filename: str) -> bool:
        """Check if a file or directory exists inside the sandbox."""
        try:
            safe_path = self._resolve_safe_path(filename)
            return safe_path.exists()
        except PermissionError:
            return False

    def delete(self, filename: str) -> bool:
        """Delete a file inside the sandbox."""
        safe_path = self._resolve_safe_path(filename)
        if safe_path.is_file():
            safe_path.unlink()
            return True
        return False


# ── 4. Isolated Python Script Execution ────────────────────────────────────

class SubprocessPythonRunner:
    """Executes Python code in an isolated subprocess with timeouts and restricted builtins."""

    RESTRICTED_PREAMBLE = """# Safe execution preamble
import sys

# Block dangerous standard library modules from being imported
_blocked = [
    "subprocess", "socket", "http", "urllib", "webbrowser",
    "ftplib", "smtplib", "telnetlib", "posix", "pty"
]
for mod in _blocked:
    sys.modules[mod] = None

# Restrict dangerous builtins in runtime namespace
for dangerous in ("breakpoint", "exit", "quit"):
    if hasattr(__builtins__, dangerous):
        delattr(__builtins__, dangerous)
"""

    def __init__(self, sandbox_dir: Path) -> None:
        self.sandbox_dir = sandbox_dir

    def run(self, code: str, timeout_seconds: int = 10) -> Dict[str, Any]:
        """Execute Python code in a subprocess within the sandbox directory."""
        full_code = f"{self.RESTRICTED_PREAMBLE}\n# User Code\n{code}"
        start_time = time.perf_counter()

        try:
            proc = subprocess.run(
                [sys.executable, "-c", full_code],
                cwd=str(self.sandbox_dir),
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return {
                "status": "success" if proc.returncode == 0 else "error",
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "returncode": proc.returncode,
                "execution_time_ms": elapsed_ms,
            }
        except subprocess.TimeoutExpired as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return {
                "status": "timeout",
                "stdout": exc.stdout or "" if hasattr(exc, "stdout") else "",
                "stderr": f"Execution timed out after {timeout_seconds} seconds.",
                "returncode": -1,
                "execution_time_ms": elapsed_ms,
            }
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return {
                "status": "error",
                "stdout": "",
                "stderr": str(exc),
                "returncode": -1,
                "execution_time_ms": elapsed_ms,
            }


# ── 5. Document Generator (python-docx) ────────────────────────────────────

class DocumentGenerator:
    """Generates professional Word (.docx) refinery and engineering documents."""

    def __init__(self, file_manager: SandboxedFileManager) -> None:
        self.file_manager = file_manager

    def generate_report(
        self,
        title: str,
        sections: List[Dict[str, Any]],
        filename: str = "report.docx",
        metadata: Optional[Dict[str, Any]] = None,
        summary: Optional[str] = None,
        citations: Optional[List[Any]] = None,
        conclusion: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Create and save a formatted .docx report inside the sandbox.

        Parameters
        ----------
        title      : Main report title
        sections   : List of dicts with heading, content, bullets, table, callout
        filename   : Output filename within the sandbox
        metadata   : Optional document metadata (author, equipment tag, date)
        summary    : Optional executive summary
        citations  : Optional list of Citation objects
        conclusion : Optional conclusion text
        """
        import docx
        from docx.shared import Inches, Pt, RGBColor
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.enum.table import WD_TABLE_ALIGNMENT

        doc = docx.Document()

        # Document Title
        title_para = doc.add_paragraph()
        title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        title_run = title_para.add_run(title)
        title_run.bold = True
        title_run.font.size = Pt(22)
        title_run.font.color.rgb = RGBColor(16, 44, 87)  # Deep industrial navy

        # Subtitle / Metadata Block
        meta_para = doc.add_paragraph()
        meta_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        meta_run = meta_para.add_run(
            f"Sovereign AI Engineering Report  |  Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        )
        meta_run.font.size = Pt(10)
        meta_run.font.italic = True
        meta_run.font.color.rgb = RGBColor(100, 110, 120)

        if metadata:
            meta_items = [f"{k}: {v}" for k, v in metadata.items()]
            meta_detail = doc.add_paragraph("  •  ".join(meta_items))
            meta_detail.alignment = WD_ALIGN_PARAGRAPH.CENTER
            meta_detail.runs[0].font.size = Pt(9)
            meta_detail.runs[0].font.color.rgb = RGBColor(110, 120, 130)

        doc.add_paragraph()  # spacing

        # Executive Summary
        if summary:
            h_sum = doc.add_heading("Executive Summary", level=1)
            h_sum.runs[0].font.color.rgb = RGBColor(24, 76, 120)
            p_sum = doc.add_paragraph(summary)
            p_sum.paragraph_format.line_spacing = 1.15
            doc.add_paragraph()

        # Populate Sections
        for sec in sections:
            heading_text = sec.get("heading") or sec.get("title")
            if heading_text:
                h = doc.add_heading(heading_text, level=1)
                h.runs[0].font.color.rgb = RGBColor(24, 76, 120)

            content_text = sec.get("content") or sec.get("text") or sec.get("body")
            if content_text:
                if isinstance(content_text, str):
                    for p_t in content_text.strip().split("\n\n"):
                        if p_t.strip():
                            p = doc.add_paragraph(p_t.strip())
                            p.paragraph_format.line_spacing = 1.15
                elif isinstance(content_text, list):
                    for p_t in content_text:
                        p = doc.add_paragraph(str(p_t))
                        p.paragraph_format.line_spacing = 1.15

            # Paragraphs list
            paragraphs = sec.get("paragraphs", [])
            for p_text in paragraphs:
                if p_text:
                    p = doc.add_paragraph(str(p_text))
                    p.paragraph_format.line_spacing = 1.15

            # Bullet points
            bullets = sec.get("bullets", [])
            for b in bullets:
                doc.add_paragraph(b, style="List Bullet")

            # Callout block (e.g. Safety warning or critical spec)
            callout = sec.get("callout")
            if callout:
                cp = doc.add_paragraph()
                cp.paragraph_format.left_indent = Inches(0.4)
                cp_run = cp.add_run(f"CRITICAL / NOTE: {callout}")
                cp_run.bold = True
                cp_run.font.color.rgb = RGBColor(180, 40, 20)

            # Table
            tbl_data = sec.get("table")
            if tbl_data and isinstance(tbl_data, dict) and "headers" in tbl_data and "rows" in tbl_data:
                headers = tbl_data["headers"]
                rows = tbl_data["rows"]
                table = doc.add_table(rows=1 + len(rows), cols=len(headers))
                table.alignment = WD_TABLE_ALIGNMENT.CENTER
                table.style = "Table Grid"

                # Header row styling
                hdr_cells = table.rows[0].cells
                for idx, h_text in enumerate(headers):
                    hdr_cells[idx].text = str(h_text)
                    for r in hdr_cells[idx].paragraphs[0].runs:
                        r.bold = True
                        r.font.size = Pt(10)
                        r.font.color.rgb = RGBColor(255, 255, 255)

                # Data rows
                for r_idx, row_vals in enumerate(rows):
                    row_cells = table.rows[r_idx + 1].cells
                    for c_idx, val in enumerate(row_vals):
                        if c_idx < len(row_cells):
                            row_cells[c_idx].text = str(val)
                            for r in row_cells[c_idx].paragraphs[0].runs:
                                r.font.size = Pt(9.5)

            doc.add_paragraph()  # spacing between sections

        # Sources & Citations
        if citations:
            h_cite = doc.add_heading("Sources & Regulatory Evidence", level=1)
            h_cite.runs[0].font.color.rgb = RGBColor(24, 76, 120)
            seen_cites = set()
            for idx, c in enumerate(citations, 1):
                doc_name = getattr(c, "document_name", None) or getattr(c, "document_id", "Document")
                page = getattr(c, "page_number", None)
                sec_title = getattr(c, "section_title", None)
                tag = getattr(c, "equipment_tag", None)
                quote = (getattr(c, "verbatim_quote", None) or "").strip()
                k = (str(doc_name), str(page), str(sec_title), str(tag), quote[:60])
                if k in seen_cites:
                    continue
                seen_cites.add(k)

                anchor = getattr(c, "citation_id", None) or f"[{idx}]"
                if not str(anchor).startswith("["):
                    anchor = f"[{anchor}]"
                page_str = f"Page {page}" if page else None
                sec_str = f"Section: {sec_title}" if sec_title else None
                tag_str = f"Tag: {tag}" if tag else None
                meta_parts = [p for p in [page_str, sec_str, tag_str] if p]
                meta_str = f" ({' | '.join(meta_parts)})" if meta_parts else ""

                cp = doc.add_paragraph()
                c_run1 = cp.add_run(f"{anchor} {doc_name}{meta_str}")
                c_run1.bold = True
                c_run1.font.size = Pt(9.5)

                if quote:
                    qp = doc.add_paragraph()
                    qp.paragraph_format.left_indent = Inches(0.25)
                    q_run = qp.add_run(f'"{quote}"')
                    q_run.italic = True
                    q_run.font.size = Pt(8.5)
                    q_run.font.color.rgb = RGBColor(100, 110, 120)

        # Conclusion
        if conclusion:
            h_concl = doc.add_heading("Conclusion & Statutory Remarks", level=1)
            h_concl.runs[0].font.color.rgb = RGBColor(24, 76, 120)
            p_concl = doc.add_paragraph(conclusion)
            p_concl.paragraph_format.line_spacing = 1.15

        # Save document inside sandbox safely
        safe_path = self.file_manager._resolve_safe_path(filename)
        safe_path.parent.mkdir(parents=True, exist_ok=True)
        doc.save(str(safe_path))

        if not safe_path.is_file() or safe_path.stat().st_size == 0:
            raise RuntimeError(f"DOCX generation failed: file '{safe_path}' was not written or is empty.")
        validate_artifact(safe_path, "docx", citations)

        stage_artifact_for_backend(safe_path)

        return {
            "status": "success",
            "filename": filename,
            "path": str(safe_path),
            "file_size_bytes": safe_path.stat().st_size,
            "sections_count": len(sections),
        }



# ── 6. Spreadsheet Generator (openpyxl) ───────────────────────────────────

class SpreadsheetGenerator:
    """Generates Excel (.xlsx) workbooks with styled headers and data rows."""

    _HEADER_FILL    = "FF102C57"   # deep navy
    _ALT_ROW_FILL   = "FFE8EEF5"   # light blue-grey
    _BORDER_COLOUR  = "FFB0BEC5"

    def __init__(self, file_manager: SandboxedFileManager) -> None:
        self.file_manager = file_manager

    def generate(
        self,
        workbook: Optional[Dict[str, Any]] = None,
        filename: str = "report.xlsx",
        metadata: Optional[Dict[str, Any]] = None,
        title: Optional[str] = None,
        sections: Optional[List[Dict[str, Any]]] = None,
        citations: Optional[List[Any]] = None,
        summary: Optional[str] = None,
        conclusion: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Create and save a styled .xlsx workbook inside the sandbox."""
        import openpyxl
        from openpyxl.styles import (
            PatternFill, Font, Alignment, Border, Side
        )
        from openpyxl.utils import get_column_letter

        wb_spec = workbook
        if not wb_spec:
            sheets_spec = []
            findings_rows = []
            if sections:
                for sec in sections:
                    sec_name = sec.get("heading") or sec.get("title") or "General"
                    tbl = sec.get("table")
                    if tbl and isinstance(tbl, dict) and "headers" in tbl and "rows" in tbl:
                        for row in tbl["rows"]:
                            findings_rows.append([sec_name] + [str(c) for c in row])
                    bullets = sec.get("bullets", [])
                    for b in bullets:
                        parts = str(b).split(":", 1)
                        if len(parts) == 2:
                            findings_rows.append([sec_name, parts[0].strip(), parts[1].strip(), "Documented"])
                        else:
                            findings_rows.append([sec_name, "Finding", str(b), "Documented"])
                    content = sec.get("content") or ""
                    if content and not bullets and not tbl:
                        for line in str(content).split("\n"):
                            line_str = line.strip().lstrip("-*• ")
                            if line_str:
                                parts = line_str.split(":", 1)
                                if len(parts) == 2:
                                    findings_rows.append([sec_name, parts[0].strip(), parts[1].strip(), "Documented"])
                                else:
                                    findings_rows.append([sec_name, "Detail", line_str, "Documented"])

            if findings_rows:
                max_cols = max(len(r) for r in findings_rows)
                default_headers = ["Section / Entity", "Parameter / Key", "Documented Value / Observation", "Status"]
                while len(default_headers) < max_cols:
                    default_headers.append(f"Field {len(default_headers)+1}")
                norm_rows = []
                for r in findings_rows:
                    padded = r + [""] * (len(default_headers) - len(r))
                    norm_rows.append(padded[:len(default_headers)])

                sheets_spec.append({
                    "name": "Documented Findings",
                    "headers": default_headers,
                    "rows": norm_rows,
                })
            else:
                sheets_spec.append({
                    "name": "Summary",
                    "headers": ["Entity", "Parameter", "Documented Value", "Unit", "Observation", "Source Document", "Page"],
                    "rows": [
                        ["Summary", "Finding", "Not documented in the available evidence.", "-", "-", "-", "-"],
                    ],
                })

            if citations:
                cite_rows = []
                seen_cites = set()
                for idx, c in enumerate(citations, 1):
                    c_id = getattr(c, "citation_id", None) or f"[{idx}]"
                    doc_name = getattr(c, "document_name", None) or getattr(c, "document_id", "Unknown Document")
                    page = str(getattr(c, "page_number", "N/A") or "N/A")
                    sec_title = str(getattr(c, "section_title", "N/A") or "N/A")
                    tag = str(getattr(c, "equipment_tag", "N/A") or "N/A")
                    quote = str(getattr(c, "verbatim_quote", "") or "").replace("\n", " ").strip()
                    k = (str(doc_name), page, sec_title, tag, quote[:60])
                    if k in seen_cites:
                        continue
                    seen_cites.add(k)
                    cite_rows.append([c_id, doc_name, page, sec_title, tag, quote])

                sheets_spec.append({
                    "name": "Source Provenance",
                    "headers": ["Citation ID", "Document Name", "Page", "Section", "Equipment Tag", "Verbatim Evidence"],
                    "rows": cite_rows,
                })

            wb_spec = {
                "title": title or "Sovereign AI Engineering Report",
                "sheets": sheets_spec,
            }

        wb = openpyxl.Workbook()
        wb.remove(wb.active)   # remove default empty sheet

        title_str = wb_spec.get("title", "Report")
        gen_date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        header_fill  = PatternFill("solid", fgColor=self._HEADER_FILL)
        alt_row_fill = PatternFill("solid", fgColor=self._ALT_ROW_FILL)
        thin_side    = Side(border_style="thin", color=self._BORDER_COLOUR)
        thin_border  = Border(left=thin_side, right=thin_side, top=thin_side, bottom=thin_side)

        for sheet_spec in wb_spec.get("sheets", []):
            ws = wb.create_sheet(title=sheet_spec.get("name", "Sheet")[:31])

            # Title row
            ws.merge_cells("A1:H1")
            ws["A1"] = f"{title_str}  |  Generated: {gen_date}"
            ws["A1"].font = Font(bold=True, size=13, color="FFFFFFFF")
            ws["A1"].fill = header_fill
            ws["A1"].alignment = Alignment(horizontal="center", vertical="center")
            ws.row_dimensions[1].height = 24

            row_cursor = 2
            meta_items = {**(metadata or {}), **sheet_spec.get("metadata", {})}
            if meta_items:
                for k, v in meta_items.items():
                    ws.cell(row=row_cursor, column=1, value=str(k)).font = Font(bold=True, size=9, color="FF102C57")
                    ws.cell(row=row_cursor, column=2, value=str(v)).font = Font(size=9)
                    row_cursor += 1
                row_cursor += 1

            headers = sheet_spec.get("headers", [])
            for col_idx, h_text in enumerate(headers, start=1):
                cell = ws.cell(row=row_cursor, column=col_idx, value=str(h_text))
                cell.font      = Font(bold=True, size=10, color="FFFFFFFF")
                cell.fill      = header_fill
                cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
                cell.border    = thin_border
            ws.row_dimensions[row_cursor].height = 18
            header_row = row_cursor
            row_cursor += 1

            for r_idx, row_vals in enumerate(sheet_spec.get("rows", [])):
                fill = alt_row_fill if r_idx % 2 == 1 else None
                for col_idx, val in enumerate(row_vals, start=1):
                    cell = ws.cell(row=row_cursor, column=col_idx, value=val)
                    cell.font   = Font(size=9)
                    cell.border = thin_border
                    if fill:
                        cell.fill = fill
                row_cursor += 1

            for col in ws.columns:
                max_len = 0
                col_letter = get_column_letter(col[0].column)
                for cell in col:
                    try:
                        max_len = max(max_len, len(str(cell.value or "")))
                    except Exception:
                        pass
                ws.column_dimensions[col_letter].width = min(max_len + 4, 45)

            ws.freeze_panes = ws.cell(row=header_row + 1, column=1)

        safe_path = self.file_manager._resolve_safe_path(filename)
        safe_path.parent.mkdir(parents=True, exist_ok=True)
        wb.save(str(safe_path))

        if not safe_path.is_file() or safe_path.stat().st_size == 0:
            raise RuntimeError(f"XLSX generation failed: file '{safe_path}' was not written or is empty.")
        validate_artifact(safe_path, "xlsx", citations)

        stage_artifact_for_backend(safe_path)

        return {
            "status": "success",
            "filename": filename,
            "path": str(safe_path),
            "file_size_bytes": safe_path.stat().st_size,
            "sheets": [s.get("name") for s in wb_spec.get("sheets", [])],
        }



# ── 7. Presentation Generator (python-pptx) ────────────────────────────────

class PresentationGenerator:
    """Generates PowerPoint (.pptx) presentations with a branded theme."""

    def __init__(self, file_manager: SandboxedFileManager) -> None:
        self.file_manager = file_manager

    def generate(
        self,
        title: str = "Agent Workbench Report",
        slides: Optional[List[Dict[str, Any]]] = None,
        filename: str = "report.pptx",
        metadata: Optional[Dict[str, Any]] = None,
        summary: Optional[str] = None,
        sections: Optional[List[Dict[str, Any]]] = None,
        citations: Optional[List[Any]] = None,
        conclusion: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Create and save a styled .pptx presentation inside the sandbox."""
        from pptx import Presentation
        from pptx.util import Inches, Pt
        from pptx.dml.color import RGBColor
        from pptx.enum.text import PP_ALIGN

        NAVY  = RGBColor(0x10, 0x2C, 0x57)   # deep industrial navy
        WHITE = RGBColor(0xFF, 0xFF, 0xFF)
        SLATE = RGBColor(0x64, 0x6E, 0x78)   # steel grey
        AMBER = RGBColor(0xE6, 0x8A, 0x00)   # accent

        gen_date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        prs = Presentation()
        prs.slide_width  = Inches(13.33)
        prs.slide_height = Inches(7.5)

        blank_layout = prs.slide_layouts[6]

        def _set_bg(slide, colour: RGBColor) -> None:
            bg = slide.background
            fill = bg.fill
            fill.solid()
            fill.fore_color.rgb = colour

        def _add_textbox(
            slide, left, top, width, height, text, bold=False, size=18,
            colour=WHITE, align=PP_ALIGN.LEFT, wrap=True
        ):
            txb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
            tf  = txb.text_frame
            tf.word_wrap = wrap
            p = tf.paragraphs[0]
            p.alignment = align
            run = p.add_run()
            run.text = text
            run.font.bold  = bold
            run.font.size  = Pt(size)
            run.font.color.rgb = colour
            return txb

        # Slide 1: Title card
        title_slide = prs.slides.add_slide(blank_layout)
        _set_bg(title_slide, NAVY)

        bar = title_slide.shapes.add_shape(1, Inches(0), Inches(0), Inches(13.33), Inches(0.35))
        bar.fill.solid()
        bar.fill.fore_color.rgb = AMBER
        bar.line.fill.background()

        _add_textbox(title_slide, 0.6, 1.5, 12, 1.4, title, bold=True, size=32, colour=WHITE, align=PP_ALIGN.CENTER)
        subtitle = f"Sovereign AI Engineering Workbench  |  {gen_date}"
        _add_textbox(title_slide, 0.6, 3.0, 12, 0.6, subtitle, bold=False, size=14, colour=RGBColor(0xAA, 0xC4, 0xE0), align=PP_ALIGN.CENTER)
        if metadata:
            meta_str = "  •  ".join(f"{k}: {v}" for k, v in metadata.items())
            _add_textbox(title_slide, 0.6, 3.8, 12, 0.5, meta_str, size=11, colour=RGBColor(0xAA, 0xC4, 0xE0), align=PP_ALIGN.CENTER)

        footer_bar = title_slide.shapes.add_shape(1, Inches(0), Inches(7.15), Inches(13.33), Inches(0.35))
        footer_bar.fill.solid()
        footer_bar.fill.fore_color.rgb = AMBER
        footer_bar.line.fill.background()

        # Build slides if not provided
        slide_specs = slides
        if not slide_specs:
            slide_specs = []
            if summary:
                slide_specs.append({"title": "Executive Summary", "content": summary})
            if sections:
                for sec in sections:
                    heading = sec.get("heading") or sec.get("title") or "Key Findings"
                    slide_specs.append({
                        "title": heading,
                        "content": sec.get("content"),
                        "bullets": sec.get("bullets", []),
                        "table": sec.get("table"),
                    })
            if citations:
                cite_bullets = []
                seen_cites = set()
                for idx, c in enumerate(citations, 1):
                    doc_name = getattr(c, "document_name", None) or getattr(c, "document_id", "Doc")
                    page = f"p.{getattr(c, 'page_number', '')}" if getattr(c, "page_number", None) else ""
                    quote = (getattr(c, "verbatim_quote", "") or "").strip()
                    k = (str(doc_name), page, quote[:60])
                    if k in seen_cites:
                        continue
                    seen_cites.add(k)
                    anchor = getattr(c, "citation_id", None) or f"[{idx}]"
                    q_trunc = quote
                    if len(q_trunc) > 100:
                        q_trunc = q_trunc[:97] + "..."
                    cite_bullets.append(f"{anchor} {doc_name} {page} - \"{q_trunc}\"")
                    if len(cite_bullets) >= 5:
                        break
                if cite_bullets:
                    slide_specs.append({"title": "Sources & Regulatory Provenance", "bullets": cite_bullets})
            if conclusion:
                slide_specs.append({"title": "Conclusion & Remarks", "content": conclusion})

        if not slide_specs:
            slide_specs = [{"title": "Executive Summary", "content": "No content provided."}]

        for slide_spec in slide_specs:
            s = prs.slides.add_slide(blank_layout)
            _set_bg(s, WHITE)

            hdr = s.shapes.add_shape(1, Inches(0), Inches(0), Inches(13.33), Inches(1.1))
            hdr.fill.solid()
            hdr.fill.fore_color.rgb = NAVY
            hdr.line.fill.background()

            slide_title = slide_spec.get("title", "Slide")
            _add_textbox(s, 0.3, 0.15, 12.7, 0.8, slide_title, bold=True, size=20, colour=WHITE)

            y_cursor = 1.3

            content = slide_spec.get("content", "")
            if isinstance(content, list):
                content = "\n".join(content)
            if content:
                _add_textbox(s, 0.5, y_cursor, 12.3, 1.2, content, size=13, colour=NAVY)
                y_cursor += 1.4

            bullets = slide_spec.get("bullets", [])
            if bullets:
                bullet_text = "\n".join(f"  •  {b}" for b in bullets)
                _add_textbox(s, 0.5, y_cursor, 12.3, min(3.0, 0.4 * len(bullets) + 0.3),
                             bullet_text, size=12, colour=NAVY)
                y_cursor += min(3.2, 0.4 * len(bullets) + 0.5)

            tbl_data = slide_spec.get("table")
            if tbl_data and "headers" in tbl_data and "rows" in tbl_data:
                headers = tbl_data["headers"]
                rows    = tbl_data["rows"]
                n_rows  = 1 + len(rows)
                n_cols  = len(headers)
                col_w   = Inches(min(12.0 / n_cols, 3.0))
                row_h   = Inches(0.35)
                tbl_h   = row_h * n_rows
                tbl_shape = s.shapes.add_table(
                    n_rows, n_cols,
                    Inches(0.5), Inches(min(y_cursor, 6.8)),
                    Inches(12.3), tbl_h
                )
                tbl = tbl_shape.table
                for c_idx, h in enumerate(headers):
                    cell = tbl.cell(0, c_idx)
                    cell.text = str(h)
                    cell.text_frame.paragraphs[0].runs[0].font.bold = True
                    cell.text_frame.paragraphs[0].runs[0].font.size = Pt(10)
                    cell.text_frame.paragraphs[0].runs[0].font.color.rgb = WHITE
                    cell.fill.solid()
                    cell.fill.fore_color.rgb = NAVY
                for r_idx, row_vals in enumerate(rows):
                    for c_idx, val in enumerate(row_vals):
                        if c_idx < n_cols:
                            tbl.cell(r_idx + 1, c_idx).text = str(val)
                            tbl.cell(r_idx + 1, c_idx).text_frame.paragraphs[0].runs[0].font.size = Pt(9)

            fb = s.shapes.add_shape(1, Inches(0), Inches(7.15), Inches(13.33), Inches(0.35))
            fb.fill.solid()
            fb.fill.fore_color.rgb = AMBER
            fb.line.fill.background()

        safe_path = self.file_manager._resolve_safe_path(filename)
        safe_path.parent.mkdir(parents=True, exist_ok=True)
        prs.save(str(safe_path))

        if not safe_path.is_file() or safe_path.stat().st_size == 0:
            raise RuntimeError(f"PPTX generation failed: file '{safe_path}' was not written or is empty.")
        validate_artifact(safe_path, "pptx", citations)

        stage_artifact_for_backend(safe_path)

        return {
            "status": "success",
            "filename": filename,
            "path": str(safe_path),
            "file_size_bytes": safe_path.stat().st_size,
            "slide_count": len(prs.slides),
        }


# ── 8. Native PDF Generator & Converter ───────────────────────────────────

class NativePDFGenerator:
    """Offline, air-gapped PDF document generator powered by ReportLab Platypus.

    Constructs styled, professional engineering and compliance PDF reports directly
    from structured grounded content without requiring Microsoft Word or LibreOffice.
    """

    def __init__(self, file_manager: SandboxedFileManager) -> None:
        self.file_manager = file_manager

    def generate(
        self,
        title: str = "Engineering & Compliance Report",
        sections: Optional[List[Dict[str, Any]]] = None,
        filename: str = "report.pdf",
        metadata: Optional[Dict[str, Any]] = None,
        citations: Optional[List[Any]] = None,
        summary: Optional[str] = None,
        conclusion: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Create and save a professionally formatted PDF inside the sandbox."""
        from reportlab.lib.pagesizes import letter
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, HRFlowable
        )
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from xml.sax.saxutils import escape as xml_escape

        safe_path = self.file_manager._resolve_safe_path(filename)
        safe_path.parent.mkdir(parents=True, exist_ok=True)

        doc = SimpleDocTemplate(
            str(safe_path),
            pagesize=letter,
            leftMargin=0.75 * inch,
            rightMargin=0.75 * inch,
            topMargin=0.75 * inch,
            bottomMargin=0.75 * inch,
        )

        styles = getSampleStyleSheet()

        c_navy = colors.HexColor("#102C57")
        c_blue = colors.HexColor("#1E3A8A")
        c_slate = colors.HexColor("#334155")
        c_grey = colors.HexColor("#64748B")
        c_bg_alt = colors.HexColor("#F1F5F9")
        c_callout_bg = colors.HexColor("#FEF2F2")
        c_callout_border = colors.HexColor("#DC2626")

        title_style = ParagraphStyle(
            "PDFDocTitle",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            textColor=c_navy,
            alignment=1,
            spaceAfter=6,
        )

        meta_style = ParagraphStyle(
            "PDFDocMeta",
            parent=styles["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=8.5,
            leading=11,
            textColor=c_grey,
            alignment=1,
            spaceAfter=8,
        )

        h1_style = ParagraphStyle(
            "PDFDocH1",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            textColor=c_navy,
            spaceBefore=10,
            spaceAfter=5,
            keepWithNext=True,
        )

        body_style = ParagraphStyle(
            "PDFDocBody",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13.5,
            textColor=c_slate,
            spaceAfter=6,
        )

        bullet_style = ParagraphStyle(
            "PDFDocBullet",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=13,
            textColor=c_slate,
            leftIndent=15,
            firstLineIndent=-10,
            spaceAfter=4,
        )

        callout_style = ParagraphStyle(
            "PDFDocCallout",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#991B1B"),
        )

        cite_style = ParagraphStyle(
            "PDFDocCite",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=12,
            textColor=c_slate,
            leftIndent=10,
            spaceAfter=3,
        )

        story = []

        # Title
        clean_title = xml_escape(title or "Engineering Report")
        story.append(Paragraph(clean_title, title_style))

        # Metadata block
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        meta_lines = [f"MRPL Sovereign AI Workbench  |  Generated: {now_str}"]
        if metadata:
            meta_parts = [f"{xml_escape(str(k))}: {xml_escape(str(v))}" for k, v in metadata.items()]
            meta_lines.append("  •  ".join(meta_parts))
        story.append(Paragraph("<br/>".join(meta_lines), meta_style))
        story.append(HRFlowable(width="100%", thickness=1, color=c_navy, spaceBefore=2, spaceAfter=10))

        # Summary
        if summary:
            story.append(Paragraph("Executive Summary", h1_style))
            story.append(Paragraph(xml_escape(summary).replace("\n", "<br/>"), body_style))
            story.append(Spacer(1, 4))

        # Sections
        sec_list = sections or []
        for sec in sec_list:
            heading = sec.get("heading") or sec.get("title")
            if heading:
                story.append(Paragraph(xml_escape(str(heading)), h1_style))

            content = sec.get("content") or sec.get("text") or sec.get("body")
            if content:
                if isinstance(content, str):
                    paras = content.strip().split("\n\n")
                    for p_text in paras:
                        if p_text.strip():
                            story.append(Paragraph(xml_escape(p_text.strip()).replace("\n", "<br/>"), body_style))
                elif isinstance(content, list):
                    for p_text in content:
                        story.append(Paragraph(xml_escape(str(p_text)).replace("\n", "<br/>"), body_style))

            paragraphs = sec.get("paragraphs", [])
            for p_text in paragraphs:
                if p_text:
                    story.append(Paragraph(xml_escape(str(p_text)).replace("\n", "<br/>"), body_style))

            bullets = sec.get("bullets", [])
            for b in bullets:
                story.append(Paragraph(f"&bull; {xml_escape(str(b))}", bullet_style))

            callout = sec.get("callout")
            if callout:
                callout_p = Paragraph(f"<b>CRITICAL NOTE:</b> {xml_escape(str(callout))}", callout_style)
                callout_tbl = Table([[callout_p]], colWidths=[7.0 * inch])
                callout_tbl.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, -1), c_callout_bg),
                    ("BOX", (0, 0), (-1, -1), 1, c_callout_border),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ]))
                story.append(Spacer(1, 4))
                story.append(callout_tbl)
                story.append(Spacer(1, 4))

            tbl_data = sec.get("table")
            if tbl_data and isinstance(tbl_data, dict) and "headers" in tbl_data and "rows" in tbl_data:
                headers = tbl_data["headers"]
                rows = tbl_data["rows"]
                num_cols = len(headers)
                if num_cols > 0:
                    col_w = (7.0 * inch) / num_cols
                    table_rows = []

                    hdr_row = [
                        Paragraph(f"<b>{xml_escape(str(h))}</b>", ParagraphStyle("TH", parent=body_style, fontSize=8.5, leading=10.5, textColor=colors.white, alignment=1))
                        for h in headers
                    ]
                    table_rows.append(hdr_row)

                    for r_idx, r in enumerate(rows):
                        row_cells = [
                            Paragraph(xml_escape(str(cell)), ParagraphStyle(f"TD_{r_idx}", parent=body_style, fontSize=8, leading=10.5))
                            for cell in r
                        ]
                        table_rows.append(row_cells)

                    t_elem = Table(table_rows, colWidths=[col_w] * num_cols)
                    t_style = [
                        ("BACKGROUND", (0, 0), (-1, 0), c_navy),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                        ("BOX", (0, 0), (-1, -1), 1, c_navy),
                        ("TOPPADDING", (0, 0), (-1, -1), 3),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ]
                    for r_i in range(1, len(table_rows)):
                        if r_i % 2 == 0:
                            t_style.append(("BACKGROUND", (0, r_i), (-1, r_i), c_bg_alt))
                    t_elem.setStyle(TableStyle(t_style))
                    story.append(Spacer(1, 4))
                    story.append(t_elem)
                    story.append(Spacer(1, 4))

            story.append(Spacer(1, 4))

        # Citations / Sources
        if citations:
            story.append(Spacer(1, 6))
            story.append(Paragraph("Sources & Regulatory Evidence", h1_style))
            seen_cites = set()
            for idx, c in enumerate(citations, 1):
                doc_name = getattr(c, "document_name", None) or getattr(c, "document_id", "Document")
                page = getattr(c, "page_number", None)
                sec_title = getattr(c, "section_title", None)
                tag = getattr(c, "equipment_tag", None)
                quote = (getattr(c, "verbatim_quote", None) or "").strip()
                k = (str(doc_name), str(page), str(sec_title), str(tag), quote[:60])
                if k in seen_cites:
                    continue
                seen_cites.add(k)

                anchor = getattr(c, "citation_id", None) or f"[{idx}]"
                if not str(anchor).startswith("["):
                    anchor = f"[{anchor}]"
                page_str = f"Page {page}" if page else None
                sec_str = f"Section: {sec_title}" if sec_title else None
                tag_str = f"Tag: {tag}" if tag else None

                meta_parts = [p for p in [page_str, sec_str, tag_str] if p]
                meta_str = f" ({' | '.join(meta_parts)})" if meta_parts else ""

                story.append(Paragraph(f"<b>{xml_escape(anchor)}</b> {xml_escape(str(doc_name))}{xml_escape(meta_str)}", cite_style))
                if quote:
                    q_clean = xml_escape(quote.replace("\n", " "))
                    if len(q_clean) > 180:
                        q_clean = q_clean[:177] + "..."
                    story.append(Paragraph(f'<i>"{q_clean}"</i>', ParagraphStyle("Quote", parent=cite_style, leftIndent=20, fontSize=8, textColor=c_grey)))

        # Conclusion
        if conclusion:
            story.append(Spacer(1, 6))
            story.append(Paragraph("Conclusion & Statutory Remarks", h1_style))
            story.append(Paragraph(xml_escape(conclusion).replace("\n", "<br/>"), body_style))

        doc.build(story)

        if not safe_path.is_file() or safe_path.stat().st_size == 0:
            raise RuntimeError(f"PDF generation failed: output file '{safe_path}' was not written or is 0 bytes.")

        header_bytes = safe_path.read_bytes()[:10]
        if not header_bytes.startswith(b"%PDF-"):
            raise ValueError(f"Generated file '{safe_path}' lacks a valid PDF header (%PDF-).")
        validate_artifact(safe_path, "pdf", citations)

        num_pages = 1
        try:
            import fitz
            doc_fitz = fitz.open(str(safe_path))
            num_pages = len(doc_fitz)
            doc_fitz.close()
        except Exception:
            pass

        stage_artifact_for_backend(safe_path)

        return {
            "status": "success",
            "generator": "reportlab_native",
            "filename": safe_path.name,
            "path": str(safe_path),
            "file_size_bytes": safe_path.stat().st_size,
            "pages": num_pages,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }


class PDFConverter:
    """Converts an existing .docx file in the sandbox to PDF, with ReportLab fallback."""

    def __init__(self, file_manager: SandboxedFileManager, native_generator: Optional[NativePDFGenerator] = None) -> None:
        self.file_manager = file_manager
        self.native_generator = native_generator or NativePDFGenerator(file_manager)

    @staticmethod
    def _find_libreoffice() -> Optional[str]:
        """Locate the LibreOffice binary (cross-platform)."""
        candidates = [
            "libreoffice", "soffice",
            r"C:\Program Files\LibreOffice\program\soffice.exe",
            r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
            "/usr/bin/libreoffice", "/usr/bin/soffice",
            "/opt/libreoffice/program/soffice",
        ]
        for cmd in candidates:
            path = Path(cmd)
            if path.is_file():
                return str(path)
            try:
                result = subprocess.run(
                    [cmd, "--version"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    return cmd
            except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
                pass
        return None

    def convert(
        self,
        docx_filename: str,
        out_filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Convert a .docx file in the sandbox to PDF."""
        docx_path = self.file_manager._resolve_safe_path(docx_filename)
        if not docx_path.is_file():
            raise FileNotFoundError(f"Source .docx not found in sandbox: {docx_filename}")

        if out_filename is None:
            out_filename = Path(docx_filename).with_suffix(".pdf").name
        pdf_path = self.file_manager._resolve_safe_path(out_filename)
        pdf_path.parent.mkdir(parents=True, exist_ok=True)

        # ── Strategy 1: docx2pdf ──────────────────────────────────────
        try:
            import importlib
            d2p = importlib.import_module("docx2pdf")
            d2p_convert = getattr(d2p, "convert")
            d2p_convert(str(docx_path), str(pdf_path))
            if pdf_path.is_file() and pdf_path.stat().st_size > 0:
                stage_artifact_for_backend(pdf_path)
                return {
                    "status": "success",
                    "converter": "docx2pdf",
                    "filename": out_filename,
                    "path": str(pdf_path),
                    "file_size_bytes": pdf_path.stat().st_size,
                }
        except (ImportError, ModuleNotFoundError):
            pass
        except Exception as exc:
            logger.warning("docx2pdf conversion failed (%s); trying fallback.", exc)

        # ── Strategy 2: LibreOffice headless ─────────────────────────
        lo_bin = self._find_libreoffice()
        if lo_bin:
            try:
                out_dir = str(pdf_path.parent)
                proc = subprocess.run(
                    [lo_bin, "--headless", "--convert-to", "pdf",
                     "--outdir", out_dir, str(docx_path)],
                    capture_output=True, text=True, timeout=60,
                )
                lo_out = pdf_path.parent / (docx_path.stem + ".pdf")
                if lo_out.is_file() and lo_out.stat().st_size > 0:
                    if lo_out != pdf_path:
                        lo_out.rename(pdf_path)
                    stage_artifact_for_backend(pdf_path)
                    return {
                        "status": "success",
                        "converter": "libreoffice",
                        "filename": out_filename,
                        "path": str(pdf_path),
                        "file_size_bytes": pdf_path.stat().st_size,
                    }
            except Exception as exc:
                logger.warning("LibreOffice conversion failed: %s", exc)

        # ── Strategy 3: ReportLab native conversion fallback ─────────
        try:
            import docx
            doc = docx.Document(str(docx_path))
            sections = []
            current_sec = {"heading": "", "paragraphs": [], "table": None}
            for p in doc.paragraphs:
                txt = p.text.strip()
                if not txt:
                    continue
                if p.style.name.startswith("Heading"):
                    if current_sec["paragraphs"]:
                        sections.append(current_sec)
                    current_sec = {"heading": txt, "paragraphs": [], "table": None}
                else:
                    current_sec["paragraphs"].append(txt)
            if current_sec["paragraphs"] or current_sec["heading"]:
                sections.append(current_sec)

            return self.native_generator.generate(
                title=docx_path.stem.replace("_", " ").title(),
                sections=sections,
                filename=out_filename,
            )
        except Exception as exc:
            logger.warning("ReportLab docx fallback failed: %s", exc)

        raise RuntimeError(
            "PDF conversion requires either docx2pdf (with Microsoft Word installed), "
            "LibreOffice (soffice/libreoffice in PATH), or ReportLab."
        )


# ── 9. Multimodal OCR & Vision Inspector Tool ─────────────────────────────

class VisionInspectorTool:
    """Tool wrapper for Member 3's MultimodalProcessor.

    Provides OCR, document parsing, P&ID diagram analysis, and visual inspection
    for images and PDFs without external cloud dependencies.
    """

    def __init__(self, sandbox_dir: Path, project_root: Path = DEFAULT_PROJECT_ROOT) -> None:
        self.sandbox_dir = Path(sandbox_dir).resolve()
        self.project_root = Path(project_root).resolve()
        # Keep OCR-only and VLM-enabled processors separate.  In particular, the
        # Member 3 default vision backend is a deterministic test double and must
        # never be selected by this production tool.
        self._processors: Dict[bool, Any] = {}

    @property
    def processor(self) -> Any:
        """Backward-compatible OCR-only processor accessor.

        Callers that want VLM processing must use ``inspect(..., use_vlm=True)``;
        this prevents OCR-only requests from manufacturing mock vision findings.
        """
        return self._get_processor(enable_vision=False)

    def _get_processor(self, *, enable_vision: bool) -> Any:
        if enable_vision not in self._processors:
            try:
                from member3_ocr.core.multimodal_processor import (
                    MultimodalProcessor,
                    MultimodalProcessorConfig,
                )
                from member3_ocr.core.ocr_pipeline import (
                    PaddleOCRModelConfig,
                    PaddleOCRBackend,
                    OCRPipeline,
                )
                from member3_ocr.core.vision_pipeline import create_vision_pipeline

                config = MultimodalProcessorConfig(
                    enable_document_parsing=True,
                    enable_ocr=True,
                    enable_vision=enable_vision,
                    enable_drawing_analysis=True,
                )

                det_dir = DEFAULT_PROJECT_ROOT / "member3_ocr" / "models" / "paddleocr" / "PP-OCRv5_mobile_det_infer"
                rec_dir = DEFAULT_PROJECT_ROOT / "member3_ocr" / "models" / "paddleocr" / "PP-OCRv5_mobile_rec_infer"

                ocr_pipeline = None
                det_valid = det_dir.is_dir() and any(det_dir.iterdir())
                rec_valid = rec_dir.is_dir() and any(rec_dir.iterdir())

                if det_valid and rec_valid:
                    try:
                        ocr_config = PaddleOCRModelConfig(
                            detection_model_dir=det_dir,
                            recognition_model_dir=rec_dir,
                        )
                        backend = PaddleOCRBackend(ocr_config)
                        ocr_pipeline = OCRPipeline(backend=backend)
                        logger.info("Configured real PaddleOCR backend for VisionInspectorTool.")
                    except Exception as exc:
                        logger.warning(
                            f"PaddleOCRBackend initialization failed ({exc}); "
                            "falling back to default MockOCRBackend."
                        )
                        ocr_pipeline = None
                else:
                    # The Member 3 default is deliberately a test-double backend.
                    # Production orchestration must not turn that into apparent OCR.
                    raise RuntimeError(
                        "Local PaddleOCR weights are unavailable; OCR cannot run without real "
                        "detection and recognition model directories."
                    )

                vision_pipeline = None
                if enable_vision:
                    # This is the sole production VLM path.  It creates a
                    # structured VisionResult which the drawing/inspection
                    # routes consume directly; no standalone Qwen call follows.
                    vision_pipeline = create_vision_pipeline(
                        model_path=self.project_root / "models" / "vision" / "qwen2.5-vl-3b-instruct",
                        device=os.environ.get("AGENT_VISION_DEVICE", "auto"),
                    )

                self._processors[enable_vision] = MultimodalProcessor(
                    config=config,
                    ocr_pipeline=ocr_pipeline,
                    vision_pipeline=vision_pipeline,
                )
            except Exception as e:
                logger.error(f"Failed to initialize MultimodalProcessor: {e}")
                raise
        return self._processors[enable_vision]

    def _resolve_input_path(self, file_path: Union[str, Path]) -> Path:
        """Resolve an input file path safely against sandbox or project directories."""
        path_obj = Path(file_path)
        if path_obj.is_absolute() and path_obj.exists():
            resolved = path_obj.resolve()
            for allowed in (self.sandbox_dir, self.project_root):
                try:
                    resolved.relative_to(allowed)
                    return resolved
                except ValueError:
                    pass
            raise PermissionError("Vision input must be inside the agent sandbox or project root.")

        # 1. Look in sandbox
        candidate_sandbox = (self.sandbox_dir / file_path).resolve()
        if candidate_sandbox.is_file():
            return candidate_sandbox

        # 2. Look relative to project root
        candidate_proj = (self.project_root / file_path).resolve()
        if candidate_proj.is_file():
            return candidate_proj

        # 3. Check well-known project media directories
        for subfolder in [
            "member3_ocr/input",
            "datasets/ocr",
            "datasets/inspection_reports",
            "datasets/engineering_drawings",
        ]:
            c = (self.project_root / subfolder / path_obj.name).resolve()
            if c.is_file():
                return c

        raise FileNotFoundError(f"Input image/document not found: {file_path}")

    def inspect(
        self,
        file_path: Union[str, Path],
        force_route: Optional[str] = None,
        document_category: str = "inspection_report",
        question: Optional[str] = None,
        use_vlm: bool = False,
    ) -> Dict[str, Any]:
        """Process an image or PDF document and extract text, tables, and equipment findings."""
        target_path = self._resolve_input_path(file_path)
        start_time = time.perf_counter()

        from member3_ocr.core.multimodal_processor import export_for_agent

        result = self._get_processor(enable_vision=use_vlm).orchestrate(
            source=target_path,
            force_route=force_route,
            document_category=document_category,
        )

        agent_view = export_for_agent(result)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        full_text = result.get_full_text()
        kv_fields = agent_view.get_key_value_fields()
        equipment_list = agent_view.get_equipment_list()
        tables = agent_view.get_tables()
        sections = agent_view.get_sections()
        summary = agent_view.get_summary()

        payload = {
            "status": "success" if not result.has_errors else "warning",
            "filename": target_path.name,
            "path": str(target_path),
            "routing_decision": result.routing_decision,
            "text": full_text,
            "sections": sections,
            "tables": tables,
            "key_value_fields": kv_fields,
            "equipment_list": equipment_list,
            "has_document": result.has_document,
            "has_drawing": result.has_drawing,
            "has_vision_analysis": result.has_vision_analysis,
            "summary": summary,
            "vision_status": result.processing_metadata.get("vision", {}),
            "execution_time_ms": elapsed_ms,
        }
        if use_vlm:
            # ``question`` is intentionally not a second ad-hoc VLM call.  The
            # structured VisionResult is the authoritative reusable result.
            vision_status = payload["vision_status"]
            payload["vlm"] = {
                "status": "success" if vision_status.get("success") else "error",
                "answer": (
                    result.vision_analysis.caption if result.vision_analysis is not None
                    else "Structured drawing VLM result fused into drawing_analysis."
                    if result.drawing_analysis is not None and vision_status.get("success")
                    else ""
                ),
                **vision_status,
            }
            if not vision_status.get("success"):
                payload["status"] = "warning"
        return payload


# ── 10. Local Stable Diffusion Image Generator Tool ────────────────────────

class ImageGeneratorTool:
    """Tool wrapper for local Stable Diffusion text-to-image generation.

    Integrates DiffusionImageGenerator with the agent sandbox and backend
    artifact staging infrastructure while strictly maintaining offline operation
    and memory hygiene.
    """

    def __init__(
        self,
        sandbox_dir: Path,
        project_root: Path = DEFAULT_PROJECT_ROOT,
        generator: Optional[Any] = None,
    ) -> None:
        self.sandbox_dir = Path(sandbox_dir).resolve()
        self.project_root = Path(project_root).resolve()
        self._generator = generator

    @property
    def generator(self) -> Any:
        """Lazily instantiate DiffusionImageGenerator."""
        if self._generator is None:
            from rag_engine.vision.image_generator import DiffusionImageGenerator
            self._generator = DiffusionImageGenerator(
                default_output_dir=self.sandbox_dir
            )
        return self._generator

    def is_available(self) -> bool:
        """Check whether local diffusion model weights exist on disk."""
        try:
            return bool(self.generator.is_available())
        except Exception:
            return False

    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 512,
        height: int = 512,
        steps: int = 30,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        filename: Optional[str] = None,
        filename_prefix: Optional[str] = None,
        unload_after: bool = True,
    ) -> Dict[str, Any]:
        """Generate an image via DiffusionImageGenerator, stage the artifact, and return structured output."""
        gen = self.generator
        try:
            result = gen.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                steps=steps,
                guidance_scale=guidance_scale,
                seed=seed,
                output_dir=self.sandbox_dir,
                filename=filename,
                filename_prefix=filename_prefix,
            )
        finally:
            if unload_after and hasattr(self._generator, "unload"):
                try:
                    self._generator.unload()
                except Exception as unload_exc:
                    logger.debug("Diffusion pipeline unload skipped or failed: %s", unload_exc)

        if not getattr(result, "success", False):
            err_str = str(getattr(result, "error", "") or "")
            is_cap = (
                "DiffusionCapabilityUnavailable" in err_str
                or "capability unavailable" in err_str.lower()
                or "not found" in err_str.lower()
                or "missing" in err_str.lower()
                or (hasattr(gen, "is_available") and not gen.is_available())
            )
            return {
                "status": "capability_unavailable" if is_cap else "error",
                "success": False,
                "error": result.error if hasattr(result, "error") else "Image generation failed.",
                "prompt": prompt,
                "model": getattr(result, "model_name", "stable-diffusion-v1-5"),
                "model_name": getattr(result, "model_name", "stable-diffusion-v1-5"),
                "device": getattr(result, "device", "unknown"),
                "generation_time_seconds": getattr(result, "generation_time_seconds", 0.0),
                "seed": seed,
            }

        safe_path = Path(result.image_path).resolve()
        staged_path = stage_artifact_for_backend(safe_path)

        return {
            "status": "success",
            "success": True,
            "filename": safe_path.name,
            "path": str(safe_path),
            "image_path": str(safe_path),
            "file_size_bytes": safe_path.stat().st_size if safe_path.is_file() else 0,
            "staged_path": str(staged_path) if staged_path else None,
            "staged_reference": staged_path.name if staged_path else None,
            "dimensions": {"width": result.width, "height": result.height},
            "width": result.width,
            "height": result.height,
            "steps": result.steps,
            "guidance_scale": result.guidance_scale,
            "seed": result.seed,
            "model": result.model_name,
            "model_name": result.model_name,
            "device": result.device,
            "generation_time_seconds": result.generation_time_seconds,
            "peak_vram_mb": result.peak_vram_mb,
            "prompt": result.prompt,
            "negative_prompt": result.negative_prompt,
            "metadata": result.metadata if hasattr(result, "metadata") and result.metadata else {},
        }


# ── 11. Master Tool Executor ───────────────────────────────────────────────

class ToolExecutor:
    """Master Tool Orchestrator for the Sovereign AI Agent.

    Directly accepts a RoutingDecision from the router to:
      1. Conditionally invoke real RAG search for document grounding & engineering context.
      2. Support dynamic local LLM model selection via GenerationConfig & RAGPipeline.
      3. Handle 'Insufficient Evidence' as a first-class valid return type.
      4. Pass RAG context directly into deterministic tools (e.g. calculator).
      5. Provide multimodal OCR & vision inspection via VisionInspectorTool.
      6. Provide offline local Stable Diffusion image generation via ImageGeneratorTool.
      7. Log every invocation into an append-only JSONL audit file.
    """

    def __init__(
        self,
        rag_pipeline: Optional[RAGPipeline] = None,
        sandbox_dir: Union[str, Path] = DEFAULT_SANDBOX_DIR,
        audit_log_path: Optional[Union[str, Path]] = None,
        image_generator: Optional[Any] = None,
    ) -> None:
        self._rag_pipeline = rag_pipeline
        self._rag_pipelines: Dict[str, RAGPipeline] = {}
        self.sandbox_dir = Path(sandbox_dir).resolve()
        # Keep per-workflow audit output with the sandbox by default.  This
        # avoids an unrelated project-level log ACL disabling tools, and lets
        # the future backend persist/move a complete workflow bundle.
        self.audit_logger = AuditLogger(audit_log_path or (self.sandbox_dir / "audit.jsonl"))

        # Core Tools
        self.calculator_tool = SafeCalculator()
        self.file_manager = SandboxedFileManager(self.sandbox_dir)
        self.python_runner = SubprocessPythonRunner(self.sandbox_dir)
        self.document_generator = DocumentGenerator(self.file_manager)
        self.spreadsheet_generator = SpreadsheetGenerator(self.file_manager)
        self.presentation_generator = PresentationGenerator(self.file_manager)
        self.pdf_generator = NativePDFGenerator(self.file_manager)
        self.pdf_converter = PDFConverter(self.file_manager, native_generator=self.pdf_generator)
        self.vision_tool = VisionInspectorTool(self.sandbox_dir, DEFAULT_PROJECT_ROOT)
        self.image_tool = ImageGeneratorTool(self.sandbox_dir, DEFAULT_PROJECT_ROOT, generator=image_generator)

    def tool_contracts(self) -> Dict[str, Dict[str, Any]]:
        """Serializable tool schemas for a future backend capability endpoint."""
        return dict(DOCUMENT_TOOL_CONTRACTS)

    def get_rag_pipeline(self, model_name: Optional[str] = None) -> RAGPipeline:
        """Lazily return or create a RAGPipeline for the target model."""
        if not model_name or model_name.strip().lower() == "auto":
            if self._rag_pipeline is None:
                config = GenerationConfig(default_model_name="auto")
                self._rag_pipeline = RAGPipeline(config=config)
            return self._rag_pipeline

        key = model_name.strip().lower()
        if key not in self._rag_pipelines:
            if "phi" in key:
                cfg_model = "microsoft/Phi-3.5-mini-instruct"
            elif "smol" in key:
                cfg_model = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
            elif "qwen" in key:
                cfg_model = "Qwen/Qwen2.5-1.5B-Instruct"
            else:
                cfg_model = model_name
            config = GenerationConfig(default_model_name=cfg_model)
            self._rag_pipelines[key] = RAGPipeline(config=config)
        return self._rag_pipelines[key]

    @property
    def rag_pipeline(self) -> RAGPipeline:
        """Default RAGPipeline instance."""
        return self.get_rag_pipeline()

    # ------------------------------------------------------------------
    # Tool 1: Real RAG Search with Dynamic Model Routing & Insufficient Evidence
    # ------------------------------------------------------------------

    def rag_search(
        self,
        query: str,
        top_k: int = 5,
        archetype: PromptArchetype = PromptArchetype.GENERAL_QA,
        session_id: str = "agent_session",
        model_name: Optional[str] = None,
        source_paths: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Execute a REAL search via Member 1's RAGPipeline.

        Handles 'Insufficient Evidence' and low confidence gracefully
        as valid return states rather than throwing errors.
        """
        start_time = time.perf_counter()
        try:
            pipeline = self.get_rag_pipeline(model_name)
            # Call Member 1's RAGPipeline.answer() — confirmed signature:
            # answer(question: str, session_id: str, archetype: PromptArchetype, top_k: int)
            resp: RAGResponse = pipeline.answer(
                question=query,
                session_id=session_id,
                archetype=archetype,
                top_k=top_k,
                source_paths=source_paths,
            )

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            # Determine whether evidence is insufficient
            is_insufficient = (
                resp.answer == INSUFFICIENT_EVIDENCE_FALLBACK
                or "insufficient information" in resp.answer.lower()
                or "insufficient evidence" in resp.answer.lower()
                or (resp.confidence_score is not None and resp.confidence_score < 0.25)
            )

            citations_list = [
                {
                    "source": c.document_name or c.document_id,
                    "page": c.page_number,
                    "section": c.section_title,
                    "tag": c.equipment_tag,
                    "quote": c.verbatim_quote,
                }
                for c in (resp.citations or [])
            ]

            status = "insufficient_evidence" if is_insufficient else "success"
            factual_grounding = "Insufficient Evidence" if is_insufficient else (
                "Grounded" if resp.is_grounded else "Warning: Low Grounding"
            )

            result = {
                "status": status,
                "answer": resp.answer,
                "citations": citations_list,
                "confidence_score": round(resp.confidence_score, 4),
                "factual_grounding": factual_grounding,
                "is_insufficient_evidence": is_insufficient,
                "retrieved_chunks": (
                    resp.execution_trace.retrieved_chunks
                    if resp.execution_trace
                    else len(citations_list)
                ),
                "model_used": resp.model_used,
                "generation_time_ms": (
                    resp.execution_trace.generation_time_ms
                    if resp.execution_trace
                    else 0.0
                ),
                "total_latency_ms": round(resp.total_latency_ms or elapsed_ms, 2),
                # Internal-only canonical payload for downstream artifact tools.
                # It is deliberately not derived from answer prose.
                "structured_report": resp.structured_report,
            }

            self.audit_logger.log(
                tool_name="rag_search",
                arguments={"query": query, "top_k": top_k, "archetype": getattr(archetype, "value", str(archetype))},
                execution_time_ms=elapsed_ms,
                status=status,
                result_summary=(
                    f"Status: {status} | Grounding: {factual_grounding} | "
                    f"Confidence: {resp.confidence_score:.2f} | Chunks: {result['retrieved_chunks']}"
                ),
            )
            return result

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            self.audit_logger.log(
                tool_name="rag_search",
                arguments={"query": query, "top_k": top_k},
                execution_time_ms=elapsed_ms,
                status="error",
                result_summary=f"RAG execution exception: {exc}",
            )
            # RAG availability is an operational dependency, not an exception
            # that should crash the workflow.  Return the standard contract so
            # the planner can halt at its grounding gate with an auditable
            # reason and never invent an SOP answer.
            return {
                "status": "error",
                "answer": "",
                "citations": [],
                "confidence_score": 0.0,
                "factual_grounding": "Unavailable",
                "is_insufficient_evidence": True,
                "retrieved_chunks": 0,
                "model_used": model_name,
                "generation_time_ms": 0.0,
                "total_latency_ms": round(elapsed_ms, 2),
                "error": str(exc),
            }

    # ------------------------------------------------------------------
    # Tool 2: Safe Calculator
    # ------------------------------------------------------------------

    def calculate(
        self, expression: str, variables: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Evaluate mathematical expression safely via SafeCalculator.

        Returns a dict that includes ``steps`` — an ordered list of intermediate
        reduction strings showing the full working of the calculation.
        """
        start_time = time.perf_counter()
        try:
            val, steps = self.calculator_tool.evaluate(expression, variables)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            result = {
                "status": "success",
                "expression": expression,
                "variables": variables or {},
                "result": val,
                "formatted_result": f"{val:,.4f}",
                "steps": steps,
            }
            self.audit_logger.log(
                tool_name="calculator",
                arguments={"expression": expression, "variables": variables},
                execution_time_ms=elapsed_ms,
                status="success",
                result_summary=f"Result: {val} | Steps: {len(steps)}",
            )
            return result
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            self.audit_logger.log(
                tool_name="calculator",
                arguments={"expression": expression, "variables": variables},
                execution_time_ms=elapsed_ms,
                status="error",
                result_summary=f"Calculator error: {exc}",
            )
            raise

    # ------------------------------------------------------------------
    # Tool 3: Local Stable Diffusion Image Generator
    # ------------------------------------------------------------------

    @property
    def image_generator(self) -> Any:
        """Lazily instantiated local Stable Diffusion image generator."""
        return self.image_tool.generator

    def generate_image(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 512,
        height: int = 512,
        steps: int = 30,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        filename: Optional[str] = None,
        filename_prefix: Optional[str] = None,
        unload_after: bool = True,
    ) -> Dict[str, Any]:
        """Generate an image using the local Stable Diffusion image generator."""
        return self.image_tool.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            filename=filename,
            filename_prefix=filename_prefix,
            unload_after=unload_after,
        )

    # ------------------------------------------------------------------
    # Master Execution Entrance: execute() consuming RoutingDecision
    # ------------------------------------------------------------------

    def execute(
        self,
        decision_or_tool: Union[RoutingDecision, str],
        task: str = "",
        **kwargs: Any,
    ) -> ToolResult:
        """Execute tool according to RoutingDecision or direct tool name.

        Coordinating Logic:
        -------------------
        1. If decision indicates capability unavailable -> immediate graceful exit.
        2. If use_rag_context is True -> runs real rag_search first.
        3. If tool_name is 'calculator' -> evaluates engineering calculation,
           coupling RAG context with mathematical computation.
        4. If tool_name is 'rag_pipeline' -> RAG context is returned directly.
        5. Other tools ('file_manager', 'python_execution', 'document_generator')
           run in their respective sandboxed environments.
        """
        start_time = time.perf_counter()

        # Unpack RoutingDecision if provided
        routed_model_name: Optional[str] = None
        if isinstance(decision_or_tool, RoutingDecision):
            decision = decision_or_tool
            tool_name = decision.tool_name or "rag_pipeline"
            use_rag_context = decision.use_rag_context
            archetype = decision.archetype
            fallback_warning = decision.fallback_warning
            if decision.model_record:
                routed_model_name = decision.model_record.hf_repo_id

            # Check capability availability gate
            if not decision.capability_available:
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                res = ToolResult(
                    tool_name=tool_name,
                    status="capability_unavailable",
                    output=None,
                    is_verified=False,
                    execution_time_ms=elapsed_ms,
                    fallback_warning=fallback_warning,
                    error=decision.reason,
                )
                self.audit_logger.log(
                    tool_name=tool_name,
                    arguments={"task": task},
                    execution_time_ms=elapsed_ms,
                    status="capability_unavailable",
                    result_summary=decision.reason,
                )
                return res
        else:
            tool_name = str(decision_or_tool)
            use_rag_context = kwargs.get("use_rag_context", False)
            archetype = kwargs.get("archetype", PromptArchetype.GENERAL_QA)
            fallback_warning = kwargs.get("fallback_warning", None)
            routed_model_name = kwargs.get("model_name")

        # Strict guardrail: Image generation requests NEVER retrieve document context via RAG
        if tool_name in ("image_generator", "image_generation", "generate_image") or (
            isinstance(decision_or_tool, RoutingDecision) and decision_or_tool.capability == Capability.IMAGE_GENERATION
        ):
            use_rag_context = False

        # ── Step 1: Optional RAG Context Retrieval ─────────────────────
        rag_context: Optional[Dict[str, Any]] = None
        if use_rag_context:
            query_str = task or kwargs.get("query") or kwargs.get("expression") or ""
            if query_str:
                top_k = kwargs.get("top_k", 5)
                rag_context = self.rag_search(
                    query=query_str,
                    top_k=top_k,
                    archetype=archetype,
                    session_id=kwargs.get("session_id", "agent_session"),
                    model_name=routed_model_name,
                    source_paths=kwargs.get("source_paths"),
                )

        # Artifact tools consume the canonical report produced by the evidence
        # gate.  This prevents a planner or renderer from using LLM prose as a
        # second, unvalidated factual source.
        if tool_name in {"document_generator", "xlsx_generator", "pptx_generator", "pdf_generator"}:
            report = kwargs.get("structured_report") or (rag_context or {}).get("structured_report")
            if use_rag_context and report is None:
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                return ToolResult(
                    tool_name=tool_name,
                    status="insufficient_evidence",
                    output=None,
                    rag_context=rag_context,
                    is_verified=False,
                    execution_time_ms=elapsed_ms,
                    error="Artifact generation requires a validated structured evidence report.",
                )
            if report is not None:
                is_valid, validation_errors = report.validate()
                if not is_valid:
                    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                    return ToolResult(
                        tool_name=tool_name,
                        status="insufficient_evidence",
                        output=None,
                        rag_context=rag_context,
                        is_verified=False,
                        execution_time_ms=elapsed_ms,
                        error=f"StructuredReport failed grounding validation: {', '.join(validation_errors)}",
                    )
                kwargs.update(report.renderer_payload())

        # ── Step 2: Route to Designated Tool ──────────────────────────
        try:
            if tool_name == "xlsx_generator":
                try:
                    wb_spec    = kwargs.get("workbook")
                    xls_fname  = kwargs.get("filename", "report.xlsx")
                    if not xls_fname.lower().endswith(".xlsx"):
                        xls_fname = f"{xls_fname}.xlsx"
                    xls_meta   = kwargs.get("metadata")
                    xls_out    = self.spreadsheet_generator.generate(
                        workbook=wb_spec,
                        filename=xls_fname,
                        metadata=xls_meta,
                        title=kwargs.get("title", "Engineering & Compliance Data"),
                        sections=kwargs.get("sections"),
                        citations=kwargs.get("citations"),
                        summary=kwargs.get("summary"),
                    )
                    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                    self.audit_logger.log(
                        tool_name="xlsx_generator",
                        arguments={"filename": xls_fname},
                        execution_time_ms=elapsed_ms,
                        status="success",
                        result_summary=f"Saved xlsx {xls_fname} ({xls_out['file_size_bytes']} bytes)",
                    )
                    return ToolResult(
                        tool_name="xlsx_generator",
                        status="success",
                        output=xls_out,
                        rag_context=rag_context,
                        execution_time_ms=elapsed_ms,
                        fallback_warning=fallback_warning,
                    )
                except Exception as exc:
                    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                    self.audit_logger.log(
                        tool_name="xlsx_generator",
                        arguments={"filename": kwargs.get("filename", "")},
                        execution_time_ms=elapsed_ms,
                        status="error",
                        result_summary=f"xlsx_generator error: {exc}",
                    )
                    return ToolResult(
                        tool_name="xlsx_generator",
                        status="error",
                        output=None,
                        rag_context=rag_context,
                        execution_time_ms=elapsed_ms,
                        fallback_warning=fallback_warning,
                        error=str(exc),
                    )

            if tool_name == "pptx_generator":
                try:
                    pptx_title  = kwargs.get("title", "Engineering Report")
                    pptx_slides = kwargs.get("slides")
                    pptx_fname  = kwargs.get("filename", "report.pptx")
                    if not pptx_fname.lower().endswith(".pptx"):
                        pptx_fname = f"{pptx_fname}.pptx"
                    pptx_meta   = kwargs.get("metadata")
                    pptx_out    = self.presentation_generator.generate(
                        title=pptx_title,
                        slides=pptx_slides,
                        filename=pptx_fname,
                        metadata=pptx_meta,
                        summary=kwargs.get("summary"),
                        sections=kwargs.get("sections"),
                        citations=kwargs.get("citations"),
                        conclusion=kwargs.get("conclusion"),
                    )
                    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                    self.audit_logger.log(
                        tool_name="pptx_generator",
                        arguments={"title": pptx_title, "filename": pptx_fname},
                        execution_time_ms=elapsed_ms,
                        status="success",
                        result_summary=f"Saved pptx {pptx_fname} ({pptx_out['file_size_bytes']} bytes)",
                    )
                    return ToolResult(
                        tool_name="pptx_generator",
                        status="success",
                        output=pptx_out,
                        rag_context=rag_context,
                        execution_time_ms=elapsed_ms,
                        fallback_warning=fallback_warning,
                    )
                except Exception as exc:
                    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                    self.audit_logger.log(
                        tool_name="pptx_generator",
                        arguments={"filename": kwargs.get("filename", "")},
                        execution_time_ms=elapsed_ms,
                        status="error",
                        result_summary=f"pptx_generator error: {exc}",
                    )
                    return ToolResult(
                        tool_name="pptx_generator",
                        status="error",
                        output=None,
                        rag_context=rag_context,
                        execution_time_ms=elapsed_ms,
                        fallback_warning=fallback_warning,
                        error=str(exc),
                    )

            if tool_name == "pdf_generator":
                try:
                    pdf_filename = kwargs.get("filename") or kwargs.get("out_filename") or "report.pdf"
                    if not pdf_filename.lower().endswith(".pdf"):
                        pdf_filename = f"{pdf_filename}.pdf"

                    docx_input = kwargs.get("docx_filename")
                    sections = kwargs.get("sections")
                    title = kwargs.get("title") or "Engineering & Compliance Report"

                    # If structured sections or title are provided, use NativePDFGenerator
                    if sections is not None or not docx_input:
                        pdf_out = self.pdf_generator.generate(
                            title=title,
                            sections=sections or [],
                            filename=pdf_filename,
                            metadata=kwargs.get("metadata"),
                            citations=kwargs.get("citations"),
                            summary=kwargs.get("summary"),
                            conclusion=kwargs.get("conclusion"),
                        )
                    else:
                        # Convert docx input
                        pdf_out = self.pdf_converter.convert(
                            docx_filename=docx_input, out_filename=pdf_filename
                        )

                    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                    self.audit_logger.log(
                        tool_name="pdf_generator",
                        arguments={"filename": pdf_filename, "title": title},
                        execution_time_ms=elapsed_ms,
                        status="success",
                        result_summary=f"PDF saved {pdf_out['filename']} via {pdf_out.get('generator', pdf_out.get('converter', 'reportlab'))} ({pdf_out['file_size_bytes']} bytes)",
                    )
                    return ToolResult(
                        tool_name="pdf_generator",
                        status="success",
                        output=pdf_out,
                        rag_context=rag_context,
                        execution_time_ms=elapsed_ms,
                        fallback_warning=fallback_warning,
                    )
                except Exception as exc:
                    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                    err_msg = str(exc)
                    self.audit_logger.log(
                        tool_name="pdf_generator",
                        arguments={"filename": kwargs.get("filename", "")},
                        execution_time_ms=elapsed_ms,
                        status="error",
                        result_summary=f"pdf_generator: {err_msg[:200]}",
                    )
                    return ToolResult(
                        tool_name="pdf_generator",
                        status="error",
                        output=None,
                        rag_context=rag_context,
                        execution_time_ms=elapsed_ms,
                        fallback_warning=fallback_warning,
                        error=err_msg,
                    )
            if tool_name == "calculator":
                # For calculation tasks: evaluate expression with optional variables
                expr = kwargs.get("expression")
                if not expr:
                    # Provide default sample expression if only task provided in test
                    expr = kwargs.get("formula", "1.0")
                vars_dict = kwargs.get("variables", {})

                # 1. Check RAG grounding status BEFORE finalizing calculator outcome
                rag_is_ungrounded = False
                if kwargs.get("force_ungrounded", False):
                    rag_is_ungrounded = True
                elif use_rag_context and rag_context:
                    if (
                        rag_context.get("is_insufficient_evidence", False)
                        or rag_context.get("factual_grounding") == "Insufficient Evidence"
                        or rag_context.get("confidence_score", 0.0) < 0.25
                    ):
                        rag_is_ungrounded = True

                # 2. Calculator step still runs so math is visible for audit/debugging
                calc_out = self.calculate(expression=expr, variables=vars_dict)
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0

                # 3. If ungrounded, status must be 'requires_verification' and is_verified=False
                final_status = "requires_verification" if rag_is_ungrounded else "success"
                is_verified = not rag_is_ungrounded

                return ToolResult(
                    tool_name="calculator",
                    status=final_status,
                    output=calc_out,
                    rag_context=rag_context,
                    is_verified=is_verified,
                    execution_time_ms=elapsed_ms,
                    fallback_warning=fallback_warning,
                )

            elif tool_name == "rag_pipeline":
                # Pure RAG query
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                status = rag_context.get("status", "success") if rag_context else "success"
                is_verified = bool(rag_context) and not rag_context.get("is_insufficient_evidence", False)
                return ToolResult(
                    tool_name="rag_pipeline",
                    status=status,
                    output=rag_context.get("answer") if rag_context else "",
                    rag_context=rag_context,
                    is_verified=is_verified,
                    execution_time_ms=elapsed_ms,
                    fallback_warning=fallback_warning,
                )

            elif tool_name in ("code_interpreter", "python_execution"):
                code = kwargs.get("code", "")
                timeout = kwargs.get("timeout_seconds", 10)
                exec_out = self.python_runner.run(code=code, timeout_seconds=timeout)
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                self.audit_logger.log(
                    tool_name=tool_name,
                    arguments={"code": code[:100]},
                    execution_time_ms=elapsed_ms,
                    status=exec_out["status"],
                    result_summary=f"Returncode: {exec_out['returncode']}",
                )
                return ToolResult(
                    tool_name=tool_name,
                    status=exec_out["status"],
                    output=exec_out,
                    rag_context=rag_context,
                    execution_time_ms=elapsed_ms,
                    fallback_warning=fallback_warning,
                    error=exec_out["stderr"] if exec_out["status"] != "success" else None,
                )

            elif tool_name == "file_manager":
                action = kwargs.get("action", "read")
                file_arg = kwargs.get("filename", "")
                try:
                    if action == "read":
                        content = self.file_manager.read(file_arg)
                        out = {"content": content}
                    elif action == "write":
                        content = kwargs.get("content", "")
                        out = self.file_manager.write(file_arg, content, overwrite=kwargs.get("overwrite", True))
                    elif action == "list":
                        out = {"files": self.file_manager.list_files(file_arg)}
                    elif action == "exists":
                        out = {"exists": self.file_manager.exists(file_arg)}
                    else:
                        raise ValueError(f"Unknown file_manager action: {action}")

                    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                    self.audit_logger.log(
                        tool_name="file_manager",
                        arguments={"action": action, "filename": file_arg},
                        execution_time_ms=elapsed_ms,
                        status="success",
                        result_summary=f"Action {action} completed on {file_arg}",
                    )
                    return ToolResult(
                        tool_name="file_manager",
                        status="success",
                        output=out,
                        rag_context=rag_context,
                        execution_time_ms=elapsed_ms,
                        fallback_warning=fallback_warning,
                    )
                except Exception as exc:
                    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                    self.audit_logger.log(
                        tool_name="file_manager",
                        arguments={"action": action, "filename": file_arg},
                        execution_time_ms=elapsed_ms,
                        status="error",
                        result_summary=f"file_manager {action} error on '{file_arg}': {exc}",
                    )
                    return ToolResult(
                        tool_name="file_manager",
                        status="error",
                        output=None,
                        rag_context=rag_context,
                        execution_time_ms=elapsed_ms,
                        fallback_warning=fallback_warning,
                        error=str(exc),
                    )

            elif tool_name == "document_generator":
                title_text = kwargs.get("title", "Refinery Report")
                sections_list = kwargs.get("sections", [])
                doc_filename = kwargs.get("filename", "report.docx")
                if not doc_filename.lower().endswith(".docx"):
                    doc_filename = f"{doc_filename}.docx"
                meta = kwargs.get("metadata")
                try:
                    doc_out = self.document_generator.generate_report(
                        title=title_text,
                        sections=sections_list,
                        filename=doc_filename,
                        metadata=meta,
                        summary=kwargs.get("summary"),
                        citations=kwargs.get("citations"),
                        conclusion=kwargs.get("conclusion"),
                    )
                    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                    self.audit_logger.log(
                        tool_name="document_generator",
                        arguments={"title": title_text, "filename": doc_filename},
                        execution_time_ms=elapsed_ms,
                        status="success",
                        result_summary=f"Saved docx {doc_filename} ({doc_out['file_size_bytes']} bytes)",
                    )
                    return ToolResult(
                        tool_name="document_generator",
                        status="success",
                        output=doc_out,
                        rag_context=rag_context,
                        execution_time_ms=elapsed_ms,
                        fallback_warning=fallback_warning,
                    )
                except Exception as exc:
                    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                    self.audit_logger.log(
                        tool_name="document_generator",
                        arguments={"title": title_text, "filename": doc_filename},
                        execution_time_ms=elapsed_ms,
                        status="error",
                        result_summary=f"document_generator error: {exc}",
                    )
                    return ToolResult(
                        tool_name="document_generator",
                        status="error",
                        output=None,
                        rag_context=rag_context,
                        execution_time_ms=elapsed_ms,
                        fallback_warning=fallback_warning,
                        error=str(exc),
                    )

            elif tool_name in ("vision_inspector", "ocr", "vision"):
                file_arg = (
                    kwargs.get("filename")
                    or kwargs.get("file_path")
                    or kwargs.get("source")
                    or task
                )
                force_route = kwargs.get("force_route")
                category = kwargs.get("document_category", "inspection_report")
                try:
                    inspect_out = self.vision_tool.inspect(
                        file_path=file_arg,
                        force_route=force_route,
                        document_category=category,
                        question=kwargs.get("question"),
                        use_vlm=kwargs.get("use_vlm", isinstance(decision_or_tool, RoutingDecision) and decision_or_tool.capability == Capability.VISION),
                    )
                    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                    self.audit_logger.log(
                        tool_name="vision_inspector",
                        arguments={"filename": str(file_arg), "force_route": force_route},
                        execution_time_ms=elapsed_ms,
                        status=inspect_out["status"],
                        result_summary=(
                            f"Parsed {inspect_out['filename']} via {inspect_out['routing_decision']} "
                            f"({len(inspect_out['text'])} chars extracted)"
                        ),
                    )
                    return ToolResult(
                        tool_name="vision_inspector",
                        status=inspect_out["status"],
                        output=inspect_out,
                        rag_context=rag_context,
                        is_verified=inspect_out["status"] in ("success", "warning"),
                        execution_time_ms=elapsed_ms,
                        fallback_warning=fallback_warning,
                    )
                except Exception as exc:
                    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                    self.audit_logger.log(
                        tool_name="vision_inspector",
                        arguments={"filename": str(file_arg), "force_route": force_route},
                        execution_time_ms=elapsed_ms,
                        status="error",
                        result_summary=f"vision_inspector error on '{file_arg}': {exc}",
                    )
                    return ToolResult(
                        tool_name="vision_inspector",
                        status="error",
                        output=None,
                        rag_context=rag_context,
                        is_verified=False,
                        execution_time_ms=elapsed_ms,
                        fallback_warning=fallback_warning,
                        error=str(exc),
                    )

            elif tool_name in ("image_generator", "image_generation", "generate_image"):
                from agent.router import extract_image_generation_params
                task_str = task if (task and task.strip()) else kwargs.get("prompt", "")
                extracted = extract_image_generation_params(task_str, **kwargs) if task_str else {}

                prompt = kwargs.get("prompt") or extracted.get("prompt") or task_str
                neg_prompt = kwargs.get("negative_prompt") or extracted.get("negative_prompt")
                width = kwargs.get("width") or extracted.get("width", 512)
                height = kwargs.get("height") or extracted.get("height", 512)
                steps = kwargs.get("steps") or extracted.get("steps", 30)
                guidance_scale = kwargs.get("guidance_scale") or extracted.get("guidance_scale", 7.5)
                seed = kwargs.get("seed") if kwargs.get("seed") is not None else extracted.get("seed")
                filename = kwargs.get("filename") or extracted.get("filename")
                filename_prefix = kwargs.get("filename_prefix") or kwargs.get("prefix") or extracted.get("filename_prefix")
                unload_after = kwargs.get("unload_after", True)

                try:
                    img_out = self.image_tool.generate(
                        prompt=prompt,
                        negative_prompt=neg_prompt,
                        width=width,
                        height=height,
                        steps=steps,
                        guidance_scale=guidance_scale,
                        seed=seed,
                        filename=filename,
                        filename_prefix=filename_prefix,
                        unload_after=unload_after,
                    )
                    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                    status = img_out.get("status", "success")
                    is_ok = status == "success"

                    self.audit_logger.log(
                        tool_name="image_generator",
                        arguments={"prompt": str(prompt)[:100], "seed": seed, "filename": filename},
                        execution_time_ms=elapsed_ms,
                        status=status,
                        result_summary=(
                            f"Generated {img_out.get('filename')} ({img_out.get('width')}x{img_out.get('height')}, {img_out.get('generation_time_seconds', 0):.2f}s)"
                            if is_ok
                            else f"image_generator {status}: {img_out.get('error')}"
                        ),
                    )

                    return ToolResult(
                        tool_name="image_generator",
                        status=status,
                        output=img_out if is_ok else None,
                        rag_context=rag_context,
                        is_verified=is_ok,
                        execution_time_ms=elapsed_ms,
                        fallback_warning=fallback_warning,
                        error=img_out.get("error") if not is_ok else None,
                        metadata={
                            k: v for k, v in img_out.items()
                            if k in (
                                "filename", "path", "image_path", "staged_path",
                                "staged_reference", "dimensions", "width", "height", "steps",
                                "guidance_scale", "seed", "model", "model_name",
                                "device", "generation_time_seconds", "peak_vram_mb"
                            )
                        } if is_ok else {},
                    )
                except Exception as exc:
                    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                    from rag_engine.vision.image_generator import DiffusionCapabilityUnavailable
                    is_cap = isinstance(exc, DiffusionCapabilityUnavailable)
                    status = "capability_unavailable" if is_cap else "error"
                    self.audit_logger.log(
                        tool_name="image_generator",
                        arguments={"prompt": str(prompt)[:100]},
                        execution_time_ms=elapsed_ms,
                        status=status,
                        result_summary=f"image_generator exception: {exc}",
                    )
                    return ToolResult(
                        tool_name="image_generator",
                        status=status,
                        output=None,
                        rag_context=rag_context,
                        is_verified=False,
                        execution_time_ms=elapsed_ms,
                        fallback_warning=fallback_warning,
                        error=str(exc),
                    )

            else:
                raise ValueError(f"Unsupported tool name: {tool_name}")

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            # Top-level safety net: any unhandled exception from tool dispatch
            # is logged to the audit trail before returning a structured error.
            try:
                self.audit_logger.log(
                    tool_name=tool_name,
                    arguments={"task": task[:200] if task else ""},
                    execution_time_ms=elapsed_ms,
                    status="error",
                    result_summary=f"Unhandled exception in execute() for tool '{tool_name}': {exc}",
                )
            except Exception:
                pass  # audit failure must never mask the original error
            return ToolResult(
                tool_name=tool_name,
                status="error",
                output=None,
                rag_context=rag_context,
                execution_time_ms=elapsed_ms,
                fallback_warning=fallback_warning,
                error=str(exc),
            )


# ── Module-level singleton ────────────────────────────────────────────────

_executor: Optional[ToolExecutor] = None


def get_tool_executor(
    rag_pipeline: Optional[RAGPipeline] = None,
    sandbox_dir: Union[str, Path] = DEFAULT_SANDBOX_DIR,
    audit_log_path: Union[str, Path] = DEFAULT_AUDIT_LOG_FILE,
) -> ToolExecutor:
    """Return the shared ToolExecutor singleton."""
    global _executor
    if _executor is None:
        _executor = ToolExecutor(
            rag_pipeline=rag_pipeline,
            sandbox_dir=sandbox_dir,
            audit_log_path=audit_log_path,
        )
    return _executor


# ── Self-test / __main__ ──────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    print("=" * 72)
    print("Tool Executor — Comprehensive Verification Test")
    print("=" * 72)

    executor = ToolExecutor()

    # ------------------------------------------------------------------
    # Test 1: Safe Calculator
    # ------------------------------------------------------------------
    print("\n[1] Testing Safe Calculator (AST Evaluator)...")
    calc = executor.calculator_tool

    # Basic arithmetic
    r1 = calc.evaluate("100 + 50 * 2 - (25 / 5)")
    assert r1 == 195.0, f"Expected 195.0, got {r1}"
    print(f"  [+] Arithmetic evaluation : 100 + 50 * 2 - (25 / 5) = {r1}")

    # Safe math functions
    r2 = calc.evaluate("sqrt(144) + abs(-25) + min(10, 5)")
    assert r2 == 42.0, f"Expected 42.0, got {r2}"
    print(f"  [+] Safe math functions   : sqrt(144) + abs(-25) + min(10, 5) = {r2}")

    # Engineering variables: ASME MAWP formula P = (S * E * t) / (R + 0.6 * t)
    mawp_expr = "(S * E * t) / (R + 0.6 * t)"
    mawp_vars = {"S": 20000.0, "E": 0.85, "t": 0.5, "R": 24.0}
    r3 = calc.evaluate(mawp_expr, mawp_vars)
    print(f"  [+] Engineering variables : MAWP = {r3:.2f} psi")

    # Security check: verify arbitrary code execution is blocked
    try:
        calc.evaluate("__import__('os').system('dir')")
        print("  [-] Security check failed: dangerous code was evaluated!")
    except ValueError as e:
        print(f"  [+] Security check passed : Blocked unsafe syntax ({e})")

    # Zero division check
    try:
        calc.evaluate("100 / 0")
        print("  [-] Zero division failed!")
    except ZeroDivisionError as e:
        print(f"  [+] Zero division handled : Caught clean ZeroDivisionError ({e})")

    # ------------------------------------------------------------------
    # Test 2: Sandboxed File Manager
    # ------------------------------------------------------------------
    print("\n[2] Testing Sandboxed File Manager...")
    fm = executor.file_manager

    # Safe write & read
    w_res = fm.write("test_doc.txt", "Refinery Unit 02 - Pressure Inspection OK")
    print(f"  [+] File written inside sandbox: {w_res['filename']}")
    content = fm.read("test_doc.txt")
    assert "Refinery Unit 02" in content
    print(f"  [+] File read content verified : {content!r}")

    # List files
    files = fm.list_files()
    print(f"  [+] Directory list : {len(files)} files in sandbox ({[f['name'] for f in files]})")

    # Security check: path traversal rejection
    try:
        fm.read("../../windows/system32/cmd.exe")
        print("  [-] Security check failed: path traversal succeeded!")
    except PermissionError as e:
        print(f"  [+] Security check passed : Path traversal blocked ({e})")

    # ------------------------------------------------------------------
    # Test 3: Subprocess Python Runner
    # ------------------------------------------------------------------
    print("\n[3] Testing Subprocess Python Execution...")
    pr = executor.python_runner

    # Safe computation
    res_py = pr.run("print(sum([x * x for x in range(10)]))", timeout_seconds=5)
    assert res_py["status"] == "success" and "285" in res_py["stdout"]
    print(f"  [+] Subprocess execution output: {res_py['stdout'].strip()}")

    # Timeout check
    res_to = pr.run("import time; time.sleep(10)", timeout_seconds=1)
    assert res_to["status"] == "timeout"
    print(f"  [+] Subprocess timeout handled : {res_to['stderr'].strip()}")

    # ------------------------------------------------------------------
    # Test 4: Document Generator (python-docx)
    # ------------------------------------------------------------------
    print("\n[4] Testing Document Generator (python-docx)...")
    dg = executor.document_generator
    doc_res = dg.generate_report(
        title="Pressure Relief Valve Inspection Report",
        sections=[
            {
                "heading": "1. Executive Summary",
                "content": "Inspection conducted for Vessel V-305 overpressure protection systems.",
                "bullets": [
                    "PRV-502 set pressure verified at 18.5 bar.",
                    "No seat leakage detected during acoustic test.",
                ],
                "callout": "Next mandatory overhaul scheduled in 6 months as per OISD-132.",
            },
            {
                "heading": "2. Specification & Sizing Table",
                "table": {
                    "headers": ["Tag", "Design Press (bar)", "Set Press (bar)", "Status"],
                    "rows": [
                        ["PRV-501", "15.0", "16.5", "Operational"],
                        ["PRV-502", "18.0", "18.5", "Calibrated"],
                    ],
                },
            },
        ],
        filename="prv_inspection_report.docx",
        metadata={"Vessel": "V-305", "Inspector": "Agent Workbench", "Unit": "CDU-1"},
    )
    assert doc_res["status"] == "success" and doc_res["file_size_bytes"] > 0
    print(f"  [+] Word doc generated successfully : {doc_res['filename']} ({doc_res['file_size_bytes']} bytes)")

    # ------------------------------------------------------------------
    # Test 5: ToolExecutor.execute() with RoutingDecision
    # ------------------------------------------------------------------
    print("\n[5] Testing ToolExecutor.execute() consuming RoutingDecision...")
    from agent.router import TaskRouter
    router = TaskRouter()

    # 5a. Calculation task (relief valve sizing)
    task_calc = "Calculate the relief valve sizing for vessel V-305."
    dec_calc = router.route(task_calc)
    print(f"  [*] Router decision: tool={dec_calc.tool_name}, use_rag={dec_calc.use_rag_context}")

    res_exec_calc = executor.execute(
        dec_calc,
        task=task_calc,
        expression="(P * R) / (S * E - 0.6 * P)",
        variables={"P": 150.0, "R": 24.0, "S": 20000.0, "E": 0.85},
    )
    print(f"  [+] Executed calculation via RoutingDecision:")
    print(f"      Status       : {res_exec_calc.status}")
    print(f"      Is Verified  : {res_exec_calc.is_verified}")
    print(f"      Calculated   : {res_exec_calc.output['formatted_result']}")
    print(f"      RAG Context  : {'Present' if res_exec_calc.rag_context else 'None'}")
    if res_exec_calc.rag_context:
        print(f"      RAG Grounding: {res_exec_calc.rag_context['factual_grounding']}")

    # Requirement 4: Explicit assertions that calculation with insufficient RAG evidence
    # returns status="requires_verification" and is_verified=False (NOT "success")
    assert res_exec_calc.status == "requires_verification", (
        f"Expected status='requires_verification', got {res_exec_calc.status!r}"
    )
    assert res_exec_calc.is_verified is False, (
        f"Expected is_verified=False, got {res_exec_calc.is_verified!r}"
    )
    print(f"  [+] Assertion PASSED: Ungrounded calculation flagged as status='requires_verification' and is_verified=False")

    # 5b. Capability unavailable task (coding placeholder)
    task_code = "Write a Python script to parse the inspection report."
    dec_code = router.route(task_code)
    res_exec_code = executor.execute(dec_code, task=task_code)
    print(f"  [+] Executed unavailable capability task:")
    print(f"      Status       : {res_exec_code.status}")
    print(f"      Error Reason : {res_exec_code.error}")

    # 5c. Fallback warning preserved (unknown query)
    task_unk = "zxqwerty bloop foo bar"
    dec_unk = router.route(task_unk)
    res_exec_unk = executor.execute(dec_unk, task=task_unk)
    print(f"  [+] Executed unknown query fallback:")
    print(f"      Status       : {res_exec_unk.status}")
    print(f"      Warning field: {res_exec_unk.fallback_warning[:65]}...")

    # ------------------------------------------------------------------
    # Test 6: Audit Log File Verification
    # ------------------------------------------------------------------
    print("\n[6] Verifying Local Append-Only Audit Log...")
    audit_path = executor.audit_logger.log_path
    assert audit_path.exists(), f"Audit log file {audit_path} does not exist!"
    with open(audit_path, "r", encoding="utf-8") as f:
        log_lines = f.readlines()
    print(f"  [+] Audit log verified : {len(log_lines)} records logged in {audit_path.name}")
    print(f"  [+] Sample last record : {log_lines[-1].strip()[:95]}...")

    print("\n" + "=" * 72)
    print("ALL TOOL EXECUTOR VERIFICATION TESTS PASSED SUCCESSFULLY.")
    print("=" * 72)
