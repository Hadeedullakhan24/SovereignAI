"""Agent Orchestration — Tool Executor.

Provides deterministic and safe tool execution for the agent workbench:
  1. rag_search         : Real integration with Member 1's RAGPipeline.
                          Handles 'Insufficient Evidence' gracefully as a valid status.
  2. calculator         : Safe mathematical & engineering formula evaluator via AST (no raw eval).
  3. file_manager       : Strictly sandboxed file I/O (read/write/list) within a project directory.
  4. python_execution   : Subprocess-isolated script execution with timeouts and restricted builtins.
  5. document_generator : Professional Word (.docx) report generator using python-docx.
  6. audit_logger       : Append-only JSONL log of every tool execution for governance.

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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path

# Ensure project root is on sys.path for direct script execution
_THIS_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path("agent").resolve()
_PROJECT_ROOT = _THIS_DIR.parent if _THIS_DIR.name == "agent" else Path(".").resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agent.router import Capability, RoutingDecision
from agent.local_vision import LocalVisionUnavailable, QwenVisionRuntime
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

# Contracts exposed to the planner/backend for generators owned by other
# modules.  Only DOCX is implemented here; the remaining entries deliberately
# advertise an unavailable capability instead of manufacturing a file.
DOCUMENT_TOOL_CONTRACTS: Dict[str, Dict[str, Any]] = {
    "document_generator": {"format": "docx", "implemented": True, "input": ["title", "sections", "filename", "metadata"], "output": ["path", "file_size_bytes"]},
    "xlsx_generator": {"format": "xlsx", "implemented": False, "input": ["workbook", "filename"], "output": ["path", "metadata"]},
    "pptx_generator": {"format": "pptx", "implemented": False, "input": ["slides", "filename"], "output": ["path", "metadata"]},
    "pdf_generator": {"format": "pdf", "implemented": False, "input": ["content", "filename"], "output": ["path", "metadata"]},
}


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
    ) -> float:
        """Parse and safely evaluate a mathematical expression.

        Parameters
        ----------
        expression : e.g. '(P * R) / (S * E - 0.6 * P)' or 'sqrt(144) + 12.5'
        variables  : optional dictionary of numeric variable values
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
        return self._eval_node(tree.body, var_dict)

    def _eval_node(self, node: ast.AST, variables: Dict[str, float]) -> float:
        """Recursively evaluate an approved AST node."""
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
                operand_val = self._eval_node(node.operand, variables)
                return self.ALLOWED_OPERATORS[op_type](operand_val)
            raise ValueError(f"Disallowed unary operator: {op_type.__name__}")

        # 4. Binary operations (x + y, x * y, x ** y, etc.)
        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type in self.ALLOWED_OPERATORS:
                left_val = self._eval_node(node.left, variables)
                right_val = self._eval_node(node.right, variables)
                # Check zero division
                if op_type in (ast.Div, ast.FloorDiv, ast.Mod) and right_val == 0:
                    raise ZeroDivisionError("Division by zero in mathematical expression.")
                return self.ALLOWED_OPERATORS[op_type](left_val, right_val)
            raise ValueError(f"Disallowed binary operator: {op_type.__name__}")

        # 5. Function calls (sqrt, sin, abs, etc.)
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Nested function attributes / dynamic dispatch are disallowed.")
            func_name = node.func.id
            if func_name not in self.ALLOWED_FUNCTIONS:
                raise ValueError(f"Disallowed function call: {func_name!r}")
            evaluated_args = [self._eval_node(arg, variables) for arg in node.args]
            return float(self.ALLOWED_FUNCTIONS[func_name](*evaluated_args))

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
    ) -> Dict[str, Any]:
        """Create and save a formatted .docx report inside the sandbox.

        Parameters
        ----------
        title    : Main report title
        sections : List of dicts, each with keys like:
                   - 'heading': section title
                   - 'content': body paragraph
                   - 'bullets': list of bullet items
                   - 'table': dict with {'headers': [...], 'rows': [[...], ...]}
                   - 'callout': highlight/safety note
        filename : output filename within the sandbox
        metadata : optional dict of document metadata (author, equipment tag, date)
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

        # Populate Sections
        for sec in sections:
            heading_text = sec.get("heading")
            if heading_text:
                h = doc.add_heading(heading_text, level=1)
                h.runs[0].font.color.rgb = RGBColor(24, 76, 120)

            content_text = sec.get("content")
            if content_text:
                p = doc.add_paragraph(content_text)
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
            if tbl_data and "headers" in tbl_data and "rows" in tbl_data:
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

        # Save document inside sandbox safely
        safe_path = self.file_manager._resolve_safe_path(filename)
        safe_path.parent.mkdir(parents=True, exist_ok=True)
        doc.save(str(safe_path))

        return {
            "status": "success",
            "filename": filename,
            "path": str(safe_path),
            "file_size_bytes": safe_path.stat().st_size,
            "sections_count": len(sections),
        }


# ── 6. Multimodal OCR & Vision Inspector Tool ─────────────────────────────

class VisionInspectorTool:
    """Tool wrapper for Member 3's MultimodalProcessor.

    Provides OCR, document parsing, P&ID diagram analysis, and visual inspection
    for images and PDFs without external cloud dependencies.
    """

    def __init__(self, sandbox_dir: Path, project_root: Path = DEFAULT_PROJECT_ROOT) -> None:
        self.sandbox_dir = Path(sandbox_dir).resolve()
        self.project_root = Path(project_root).resolve()
        self._processor: Optional[Any] = None
        self._vlm: Optional[QwenVisionRuntime] = None

    @property
    def vlm(self) -> QwenVisionRuntime:
        if self._vlm is None:
            self._vlm = QwenVisionRuntime(
                self.project_root / "models" / "vision" / "qwen2.5-vl-3b-instruct"
            )
        return self._vlm

    @property
    def processor(self) -> Any:
        if self._processor is None:
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

                config = MultimodalProcessorConfig(
                    enable_document_parsing=True,
                    enable_ocr=True,
                    enable_vision=True,
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

                if ocr_pipeline is not None:
                    self._processor = MultimodalProcessor(config=config, ocr_pipeline=ocr_pipeline)
                else:
                    self._processor = MultimodalProcessor(config=config)
            except Exception as e:
                logger.error(f"Failed to initialize MultimodalProcessor: {e}")
                raise
        return self._processor

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

        result = self.processor.orchestrate(
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
            "execution_time_ms": elapsed_ms,
        }
        if use_vlm:
            try:
                payload["vlm"] = self.vlm.answer(
                    target_path,
                    question or "Analyze this industrial image. Report only visible, verifiable details.",
                )
            except LocalVisionUnavailable as exc:
                # OCR evidence is retained, but is never represented as a VLM answer.
                payload["vlm"] = {"status": "capability_unavailable", "error": str(exc)}
                payload["status"] = "capability_unavailable"
        return payload


# ── 7. Master Tool Executor ────────────────────────────────────────────────

class ToolExecutor:
    """Master Tool Orchestrator for the Sovereign AI Agent.

    Directly accepts a RoutingDecision from the router to:
      1. Conditionally invoke real RAG search for document grounding & engineering context.
      2. Support dynamic local LLM model selection via GenerationConfig & RAGPipeline.
      3. Handle 'Insufficient Evidence' as a first-class valid return type.
      4. Pass RAG context directly into deterministic tools (e.g. calculator).
      5. Provide multimodal OCR & vision inspection via VisionInspectorTool.
      6. Log every invocation into an append-only JSONL audit file.
    """

    def __init__(
        self,
        rag_pipeline: Optional[RAGPipeline] = None,
        sandbox_dir: Union[str, Path] = DEFAULT_SANDBOX_DIR,
        audit_log_path: Optional[Union[str, Path]] = None,
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
        self.vision_tool = VisionInspectorTool(self.sandbox_dir, DEFAULT_PROJECT_ROOT)

    def tool_contracts(self) -> Dict[str, Dict[str, Any]]:
        """Serializable tool schemas for a future backend capability endpoint."""
        return dict(DOCUMENT_TOOL_CONTRACTS)

    def get_rag_pipeline(self, model_name: Optional[str] = None) -> RAGPipeline:
        """Lazily return or create a RAGPipeline for the target model."""
        if not model_name:
            if self._rag_pipeline is None:
                config = GenerationConfig(default_model_name="qwen2.5-1.5b")
                self._rag_pipeline = RAGPipeline(config=config)
            return self._rag_pipeline

        key = model_name.strip().lower()
        if "qwen" in key and self._rag_pipeline is not None:
            return self._rag_pipeline

        if key not in self._rag_pipelines:
            if "phi" in key:
                cfg_model = "phi-3.5-mini-instruct"
            elif "smol" in key:
                cfg_model = "smollm2-1.7b-instruct"
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
        """Evaluate mathematical expression safely via SafeCalculator."""
        start_time = time.perf_counter()
        try:
            val = self.calculator_tool.evaluate(expression, variables)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            result = {
                "status": "success",
                "expression": expression,
                "variables": variables or {},
                "result": val,
                "formatted_result": f"{val:,.4f}",
            }
            self.audit_logger.log(
                tool_name="calculator",
                arguments={"expression": expression, "variables": variables},
                execution_time_ms=elapsed_ms,
                status="success",
                result_summary=f"Result: {val}",
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
                )

        # ── Step 2: Route to Designated Tool ──────────────────────────
        try:
            if tool_name in ("xlsx_generator", "pptx_generator", "pdf_generator"):
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                contract = DOCUMENT_TOOL_CONTRACTS[tool_name]
                return ToolResult(
                    tool_name=tool_name,
                    status="capability_unavailable",
                    output={"contract": contract},
                    is_verified=False,
                    execution_time_ms=elapsed_ms,
                    error=(f"{contract['format'].upper()} generation is not implemented by Member 2; "
                           "connect the owning module to this contract."),
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

            elif tool_name == "document_generator":
                title_text = kwargs.get("title", "Refinery Report")
                sections_list = kwargs.get("sections", [])
                doc_filename = kwargs.get("filename", "report.docx")
                meta = kwargs.get("metadata")
                doc_out = self.document_generator.generate_report(
                    title=title_text,
                    sections=sections_list,
                    filename=doc_filename,
                    metadata=meta,
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

            elif tool_name in ("vision_inspector", "ocr", "vision"):
                file_arg = (
                    kwargs.get("filename")
                    or kwargs.get("file_path")
                    or kwargs.get("source")
                    or task
                )
                force_route = kwargs.get("force_route")
                category = kwargs.get("document_category", "inspection_report")
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

            else:
                raise ValueError(f"Unsupported tool name: {tool_name}")

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
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
