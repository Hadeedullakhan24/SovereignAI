"""Agent Orchestration — Task Router.

Given a free-text task description, determines:
  - Which *capability* is needed  (rag | calculation | coding | vision)
  - Whether that capability has a *ready model* in the AgentModelRegistry
  - Which *PromptArchetype* best matches the query (for RAG tasks)
  - A plain-English *reason* string for the audit trail

Design principles:
  - Completely deterministic rule-based matching (no LLM calls, no ML).
  - Returns a typed RoutingDecision — never raises, never silently falls back.
  - "capability not available" is a first-class result: the caller (planner)
    must handle it explicitly rather than silently using the RAG model.
  - All keyword lists are centralized in ROUTING_RULES so they can be
    maintained without touching logic.
"""

from __future__ import annotations

import logging
from pathlib import Path
import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

_THIS_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path("agent").resolve()
_PROJECT_ROOT = _THIS_DIR.parent if _THIS_DIR.name == "agent" else Path(".").resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agent.model_registry import AgentModelRegistry, ModelRecord, get_agent_registry

# We import PromptArchetype for typing; it lives in Member 1's rag_engine.
from rag_engine.generation.prompt.prompt_templates import PromptArchetype

logger = logging.getLogger(__name__)


# ── Capability enum ────────────────────────────────────────────────────────

class Capability(str, Enum):
    """Logical capability the router maps a task to."""
    RAG                 = "rag"                  # document-grounded Q&A via RAGPipeline
    DOCUMENT_GENERATION = "document_generation"  # grounded report/file artifact generation
    CALCULATION         = "calculation"          # numerical / engineering verification
    CODING              = "coding"               # code / script generation
    VISION              = "vision"               # image / OCR / diagram understanding
    IMAGE_GENERATION    = "image_generation"     # local text-to-image synthesis
    UNKNOWN             = "unknown"              # no rule matched; falls back to general qa


# ── TaskType enum ──────────────────────────────────────────────────────────

class TaskType(str, Enum):
    """Specific task classification for Member 2 routing."""
    GENERAL_QA                   = "general_qa"
    DOCUMENT_QA                  = "document_qa"
    EXTRACTION                   = "extraction"
    COMPARISON                   = "comparison"
    REASONING                    = "reasoning"
    DOCUMENT_GENERATION          = "document_generation"
    SUMMARIZATION                = "summarization"
    VISION_OCR                   = "vision_ocr"
    ENGINEERING_DRAWING_ANALYSIS = "engineering_drawing_analysis"
    CALCULATION                  = "calculation"
    CODING                       = "coding"
    IMAGE_GENERATION             = "image_generation"


# ── Routing outcome ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RoutingDecision:
    """Immutable result of a routing call.

    Fields
    ------
    capability           : The capability the router selected (rag | calculation | coding | vision | image_generation | unknown).
    archetype            : Recommended PromptArchetype (RAG tasks only).
    model_record         : The ModelRecord that will serve this task, or None
                           if capability_available is False.
    capability_available : True if a ready model exists for the capability.
    tool_name            : Specific tool to execute (e.g. 'calculator', 'rag_pipeline',
                           'code_interpreter', 'vision_inspector', 'image_generator'), or None.
    use_rag_context      : True if the task requires RAG context retrieval (e.g. looking
                           up parameters or document grounding).
    fallback_warning     : Structured warning string for low-confidence/unknown fallbacks,
                           or None for confident routing matches.
    reason               : Plain-English explanation for the audit trail.
    matched_keywords     : The keyword(s) that triggered this routing decision.
    task_type            : Specific classified task type (e.g. general_qa, document_qa, comparison, etc.).
    """
    capability: Capability
    archetype: PromptArchetype
    model_record: Optional[ModelRecord]
    capability_available: bool
    tool_name: Optional[str] = None
    use_rag_context: bool = True
    fallback_warning: Optional[str] = None
    reason: str = ""
    matched_keywords: List[str] = field(default_factory=list)
    task_type: TaskType = TaskType.DOCUMENT_QA

    def is_actionable(self) -> bool:
        """True if the planner can proceed (capable model exists)."""
        return self.capability_available and self.model_record is not None

    def summary(self) -> str:
        """One-line summary for logs and CLI display."""
        status = "AVAILABLE" if self.capability_available else "NOT AVAILABLE"
        model = self.model_record.hf_repo_id if self.model_record else "none"
        tool = self.tool_name or "none"
        warn = f" | warning={self.fallback_warning!r}" if self.fallback_warning else ""
        return (
            f"[Router] task_type={self.task_type.value} | "
            f"capability={self.capability.value} | "
            f"archetype={self.archetype.value} | "
            f"model={model} | "
            f"tool={tool} | "
            f"use_rag={self.use_rag_context} | "
            f"status={status}{warn} | "
            f"reason={self.reason!r}"
        )


# ── Rule tables ────────────────────────────────────────────────────────────
# Each rule is a tuple: (TaskType, Capability, PromptArchetype, [trigger_keywords]).
# Rules are checked in order; first match wins.
# Keywords are matched case-insensitively against the full task string.
# A keyword may be a plain word OR a regex pattern (prefixed with "re:").

_ROUTING_RULES: List[Tuple[TaskType, Capability, PromptArchetype, List[str]]] = [

    # ── Document / Report / Artifact generation ───────────────────────────
    (
        TaskType.DOCUMENT_GENERATION,
        Capability.DOCUMENT_GENERATION,
        PromptArchetype.REPORT,
        [
            r"re:.*\b(?:create|generate|export|prepare|save|output|produce|build|draft|write|download|make)\b.*\b(?:actual\s+)?(?:pdf|docx|word\s+document|excel|spreadsheet|xlsx|powerpoint|pptx)\b",
            r"re:.*\b(?:pdf|docx|xlsx|pptx)\s+(?:file|document|report|artifact|export)\b",
            r"re:.*\bexport\s+(?:as|to)\s*(?:pdf|docx|word|excel|xlsx|pptx|powerpoint)\b",
            r"re:.*\bas\s+(?:an?\s+)?(?:actual\s+)?(?:pdf|docx|xlsx|pptx)\s+artifact\b",
        ],
    ),

    # ── Image Generation / Visual Synthesis ──────────────────────────────
    (
        TaskType.IMAGE_GENERATION,
        Capability.IMAGE_GENERATION,
        PromptArchetype.GENERAL_QA,
        [
            r"re:.*\b(?:generate|create|make|render|draw|produce|synthesize|output|build|paint|sketch|illustrate)\b.*\b(?:an?\s+)?(?:image|picture|photo|photograph|illustration|diagram|rendering|graphic|visual\s+representation|visual|schematic\s+image)\b",
            r"re:.*\b(?:text-to-image|txt2img|stable\s*diffusion|diffusion\s*image|diffusion\s*model)\b",
            r"re:.*\b(?:visual\s+representation\s+of)\b",
        ],
    ),

    # ── Code / script generation ──────────────────────────────────────────
    (
        TaskType.CODING,
        Capability.CODING,
        PromptArchetype.GENERAL_QA,
        [
            "write code", "generate code", "write a script", "write script",
            "write python", "write sql", "generate sql", "generate python",
            "create a function", "implement a", "code snippet", "write function",
            "script to", "automate", "generate report script",
        ],
    ),

    # ── Engineering Drawing / P&ID Analysis (Qwen2.5-VL via Member 3) ────
    (
        TaskType.ENGINEERING_DRAWING_ANALYSIS,
        Capability.VISION,
        PromptArchetype.GENERAL_QA,
        [
            "p&id", "pid diagram", "piping and instrumentation", "engineering drawing",
            "schematic diagram", "pfd", "process flow diagram", "isometric drawing",
            "blue print", "blueprint", "ga drawing", "wiring diagram", "loop diagram",
            "logic diagram", "drawing analysis", "p&id drawing", "read drawing",
            "analyze drawing", "inspect drawing", "schematic",
        ],
    ),

    # ── Vision / Scanned Document OCR (Member 3 processor) ────────────────
    (
        TaskType.VISION_OCR,
        Capability.VISION,
        PromptArchetype.GENERAL_QA,
        [
            "ocr", "scan", "scanned document", "handwritten", "nameplate",
            "plate label", "read text from image", "extract text from image",
            "image text", "ocr table", "invoice scan", "inspection report scan",
            "document image", "photo of nameplate", "photograph",
        ],
    ),

    # ── Numerical / engineering calculation ───────────────────────────────
    (
        TaskType.CALCULATION,
        Capability.CALCULATION,
        PromptArchetype.EQUIPMENT_LOOKUP,
        [
            "calculate", "compute", "verify calculation", "check calculation",
            "engineering calculation", "stress calculation", "pressure calculation",
            "flow rate calculation", "heat duty", "pipe sizing", "relief valve sizing",
            "thickness calculation", "corrosion allowance", "design margin",
            "factor of safety", "burst pressure", "mawp", "maximum allowable",
            "hydrostatic test", "nozzle load",
        ],
    ),

    # ── Comparison (Phi-3.5-mini) ─────────────────────────────────────────
    (
        TaskType.COMPARISON,
        Capability.RAG,
        PromptArchetype.COMPARISON,
        [
            "compare", "comparison", "versus", " vs ", "difference between",
            "better than", "which is better", "trade-off", "tradeoff",
            "alternative", "option a vs", "option b",
        ],
    ),

    # ── Reasoning / Troubleshooting (Phi-3.5-mini) ────────────────────────
    (
        TaskType.REASONING,
        Capability.RAG,
        PromptArchetype.TROUBLESHOOTING,
        [
            "troubleshoot", "fault", "failure", "root cause", "rca",
            "why is", "why did", "abnormal", "alarm", "trip", "high temperature",
            "high pressure", "low flow", "cavitation", "surge", "vibrating",
            "leak", "leaking", "chatter", "erratic", "unstable",
            "not working", "failed", "broken", "investigate cause",
            "failure analysis", "engineering reasoning", "determine why",
        ],
    ),

    # ── Summarization (Phi-3.5-mini for full report / SmolLM2 for brief) ──
    (
        TaskType.SUMMARIZATION,
        Capability.RAG,
        PromptArchetype.GENERAL_QA,
        [
            "summarize", "summary", "brief overview", "executive summary",
            "overview of", "synopsis", "recap", "summarize report",
            "summarize manual", "summarize document",
        ],
    ),

    # ── Extraction (Qwen2.5-1.5B) ─────────────────────────────────────────
    (
        TaskType.EXTRACTION,
        Capability.RAG,
        PromptArchetype.EQUIPMENT_LOOKUP,
        [
            "extract", "extraction", "list all tags", "extract parameters",
            "extract specifications", "extract values", "find tag",
            "pull out data", "extract table", "get all tags", "parse tags",
            "retrieve tags", "parameter extraction",
        ],
    ),

    # ── Document QA: Safety & compliance (Qwen2.5-1.5B) ───────────────────
    (
        TaskType.DOCUMENT_QA,
        Capability.RAG,
        PromptArchetype.SAFETY_COMPLIANCE,
        [
            "safety", "hazard", "oisd", "api 510", "api 570", "api 650",
            "asme", "pngrb", "permit to work", "ptw", "loto", "lockout",
            "hot work", "confined space", "emergency", "evacuation", "msds",
            "sds", "explosive", "flammable", "toxic", "h2s", "hydrogen sulphide",
            "fire & gas", "fire and gas", "ppe", "personal protective",
            "regulatory", "compliance", "statutory",
        ],
    ),

    # ── Document QA: SOP / procedure (Qwen2.5-1.5B) ───────────────────────
    (
        TaskType.DOCUMENT_QA,
        Capability.RAG,
        PromptArchetype.SOP_RETRIEVAL,
        [
            "sop", "procedure", "step by step", "step-by-step", "how to",
            "startup", "shutdown", "commissioning", "decommissioning",
            "isolation", "depressurization", "purging", "flushing",
            "handover", "shift handover", "operational sequence",
            "operating manual", "operation manual",
        ],
    ),

    # ── Document QA: Maintenance (Qwen2.5-1.5B) ───────────────────────────
    (
        TaskType.DOCUMENT_QA,
        Capability.RAG,
        PromptArchetype.MAINTENANCE,
        [
            "maintenance", "inspection", "overhaul", "turnaround", "ta ",
            "preventive", "predictive", "corrective", "lubrication",
            "alignment", "vibration", "bearing", "seal replacement",
            "gasket", "torque", "ndt", "non-destructive", "thickness measurement",
            "corrosion monitoring", "inspection report",
        ],
    ),

    # ── Document QA: Equipment lookup (Qwen2.5-1.5B) ──────────────────────
    (
        TaskType.DOCUMENT_QA,
        Capability.RAG,
        PromptArchetype.EQUIPMENT_LOOKUP,
        [
            "design pressure", "design temperature", "operating pressure",
            "operating temperature", "flow rate", "capacity", "rating",
            "metallurgy", "material of construction", "moc",
            "equipment tag", "tag number", "p-", "e-", "v-", "t-", "k-",
            "pump", "compressor", "vessel", "heat exchanger", "column",
            "reactor", "separator", "drum", "tank", "valve",
            "instrument", "transmitter", "controller", "datasheet",
            "data sheet", "specification", "refinery", "mrpl", "cdu", "vdu", "fccu",
        ],
    ),

    # ── General QA: Standard definitions / general questions (SmolLM2) ─────
    (
        TaskType.GENERAL_QA,
        Capability.RAG,
        PromptArchetype.GENERAL_QA,
        [
            "what is", "what are", "explain", "describe", "define",
            "tell me", "who is", "how does", "hello", "hi", "help me understand",
        ],
    ),
]


# ── Router ─────────────────────────────────────────────────────────────────

class TaskRouter:
    """Deterministic rule-based task router.

    Usage
    -----
        router = TaskRouter()
        decision = router.route("What is the operating pressure of pump P-203?")
        if decision.is_actionable():
            result = pipeline.answer(query, archetype=decision.archetype)
        else:
            print("Cannot handle:", decision.reason)
    """

    def __init__(self, registry: Optional[AgentModelRegistry] = None) -> None:
        self._registry = registry or get_agent_registry()

    @property
    def model_registry(self) -> AgentModelRegistry:
        """The underlying AgentModelRegistry instance."""
        return self._registry

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(self, task: str) -> RoutingDecision:
        """Classify *task* and return a RoutingDecision.

        Steps
        -----
        1. Normalize the task string.
        2. Walk ROUTING_RULES in order; collect the first match (TaskType, Capability, Archetype).
        3. Determine execution tool and whether RAG context is required.
        4. Check AgentModelRegistry for a ready model for that capability/task.
        5. If no ready model exists → capability_available=False.
        6. Return RoutingDecision with task_type, model, tool, context flag, warning, and reason.
        """
        normalized = self._normalize(task)

        task_type, capability, archetype, matched = self._match_rules(normalized)

        # Resolve execution tool and RAG context requirement
        tool_name, use_rag_context = self._resolve_tool(task_type, capability, task=normalized)

        # Resolve model, audit reason, and structured fallback warning
        model_record, available, reason, fallback_warning = self._resolve_model(
            task_type, capability, archetype, matched, task
        )

        decision = RoutingDecision(
            capability=capability,
            archetype=archetype,
            model_record=model_record,
            capability_available=available,
            tool_name=tool_name,
            use_rag_context=use_rag_context,
            fallback_warning=fallback_warning,
            reason=reason,
            matched_keywords=matched,
            task_type=task_type,
        )

        logger.info(decision.summary())
        return decision

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(text: str) -> str:
        """Lowercase, collapse whitespace, keep punctuation."""
        return re.sub(r"\s+", " ", text.strip().lower())

    def _match_rules(
        self, normalized: str
    ) -> Tuple[TaskType, Capability, PromptArchetype, List[str]]:
        """Return (task_type, capability, archetype, matched_keywords) for first matching rule."""
        from agent.intent import (
            ActionType,
            OutputModality,
            classify_intent,
        )

        intent = classify_intent(normalized)

        # 1. Image Generation Intent: Highest precedence over visual keywords
        if intent.is_image_generation:
            return (
                TaskType.IMAGE_GENERATION,
                Capability.IMAGE_GENERATION,
                PromptArchetype.GENERAL_QA,
                intent.matched_cues or ["image_generation"],
            )

        # 2. Existing Visual Analysis / Inspection Intent
        if intent.is_existing_visual_analysis:
            drawing_cues = [
                "p&id", "pid", "piping and instrumentation", "engineering drawing",
                "schematic diagram", "pfd", "process flow diagram", "isometric drawing",
                "blue print", "blueprint", "ga drawing", "wiring diagram", "loop diagram",
                "logic diagram", "drawing analysis", "p&id drawing", "read drawing",
                "analyze drawing", "inspect drawing", "schematic",
            ]
            is_eng_drawing = any(
                (cue in normalized if " " in cue else bool(re.search(rf"\b{re.escape(cue)}\b", normalized)))
                for cue in drawing_cues
            )
            if is_eng_drawing:
                return (
                    TaskType.ENGINEERING_DRAWING_ANALYSIS,
                    Capability.VISION,
                    PromptArchetype.GENERAL_QA,
                    intent.matched_cues or ["engineering_drawing"],
                )
            return (
                TaskType.VISION_OCR,
                Capability.VISION,
                PromptArchetype.GENERAL_QA,
                intent.matched_cues or ["vision_ocr"],
            )

        # 3. Document File Generation (PDF, DOCX, XLSX, PPTX)
        if intent.output_modality == OutputModality.DOCUMENT_FILE or (
            intent.action == ActionType.CREATE_NEW and intent.output_modality == OutputModality.DOCUMENT_FILE
        ):
            return (
                TaskType.DOCUMENT_GENERATION,
                Capability.DOCUMENT_GENERATION,
                PromptArchetype.REPORT,
                intent.matched_cues or ["document_generation"],
            )

        # 4. Numerical Calculation
        if intent.action == ActionType.CALCULATE or intent.output_modality == OutputModality.NUMERICAL_RESULT:
            return (
                TaskType.CALCULATION,
                Capability.CALCULATION,
                PromptArchetype.EQUIPMENT_LOOKUP,
                intent.matched_cues or ["calculation"],
            )

        # 5. Code Generation / Scripting
        if intent.action == ActionType.WRITE_CODE or intent.output_modality == OutputModality.CODE:
            return (
                TaskType.CODING,
                Capability.CODING,
                PromptArchetype.GENERAL_QA,
                intent.matched_cues or ["coding"],
            )

        # 6. Walk domain rules for text QA / specialized RAG archetypes
        for task_type, capability, archetype, keywords in _ROUTING_RULES:
            hits = []
            for kw in keywords:
                if kw.startswith("re:"):
                    pattern = kw[3:]
                    if re.search(pattern, normalized):
                        hits.append(kw)
                else:
                    if kw.lower() in normalized:
                        hits.append(kw)
            if hits:
                return task_type, capability, archetype, hits

        # No rule matched → treat as GENERAL_QA
        return TaskType.GENERAL_QA, Capability.UNKNOWN, PromptArchetype.GENERAL_QA, []

    @staticmethod
    def _resolve_tool(task_type: TaskType, capability: Capability, task: str = "") -> Tuple[Optional[str], bool]:
        """Determine which execution tool to run and whether RAG context is required.

        Returns
        -------
        (tool_name, use_rag_context)
          - tool_name: e.g. 'calculator' for calculation, 'rag_pipeline' for RAG,
                       'code_interpreter' for coding, 'vision_inspector' for vision,
                       'pdf_generator' / 'document_generator' / 'xlsx_generator' for artifacts.
          - use_rag_context: True if tool_executor should retrieve document context/specs
                             first before executing the tool.
        """
        if task_type == TaskType.CALCULATION or capability == Capability.CALCULATION:
            return "calculator", True
        elif task_type == TaskType.DOCUMENT_GENERATION or capability == Capability.DOCUMENT_GENERATION:
            t_lower = task.lower()
            if any(k in t_lower for k in ["xlsx", "excel", "spreadsheet"]):
                return "xlsx_generator", True
            elif any(k in t_lower for k in ["pptx", "powerpoint", "slide"]):
                return "pptx_generator", True
            elif any(k in t_lower for k in ["docx", "word"]):
                return "document_generator", True
            return "pdf_generator", True
        elif task_type == TaskType.IMAGE_GENERATION or capability == Capability.IMAGE_GENERATION:
            return "image_generator", False
        elif task_type in (TaskType.VISION_OCR, TaskType.ENGINEERING_DRAWING_ANALYSIS) or capability == Capability.VISION:
            return "vision_inspector", False
        elif task_type == TaskType.CODING or capability == Capability.CODING:
            return "code_interpreter", False
        elif task_type == TaskType.GENERAL_QA or capability == Capability.UNKNOWN:
            return "rag_pipeline", False
        elif task_type in (TaskType.DOCUMENT_QA, TaskType.EXTRACTION, TaskType.COMPARISON, TaskType.REASONING, TaskType.SUMMARIZATION) or capability == Capability.RAG:
            return "rag_pipeline", True
        return None, False

    LONG_CONTEXT_KEYWORDS: List[str] = [
        "long document", "full report", "entire report", "complete report",
        "entire document", "complete document", "long context", "full manual",
        "entire manual", "comprehensive report", "detailed manual", "comprehensive analysis",
    ]

    QUICK_LOOKUP_KEYWORDS: List[str] = [
        "quick lookup", "fast lookup", "quick check", "fast check",
        "tag lookup", "quick", "fast", "brief", "brief summary",
    ]

    LONG_CONTEXT_THRESHOLD_CHARS: int = 250

    def _select_model_for_task(
        self,
        task_type: TaskType,
        capability: Capability,
        ready_models: List[ModelRecord],
        task: str,
    ) -> Tuple[ModelRecord, str]:
        """Select the authoritative specialist model for the classified TaskType."""
        task_lower = task.lower()

        # Index ready models by family / repo id
        phi_model = next((m for m in ready_models if "phi" in m.family.lower() or "phi-3.5-mini" in m.hf_repo_id.lower()), None)
        smollm_model = next((m for m in ready_models if "smollm" in m.family.lower() or "smollm" in m.hf_repo_id.lower()), None)
        qwen_model = next((m for m in ready_models if "qwen" in m.family.lower() or "qwen2.5" in m.hf_repo_id.lower()), None)
        qwen_vl_model = next((m for m in ready_models if "qwen2.5-vl" in m.hf_repo_id.lower() or "qwen_vl" in m.family.lower()), None)
        ocr_model = next((m for m in ready_models if "member3" in m.hf_repo_id.lower() or m.role == "ocr"), None)

        # 1. GENERAL_QA -> SmolLM2-1.7B
        if task_type == TaskType.GENERAL_QA:
            if smollm_model is not None:
                return smollm_model, f"Selected {smollm_model.hf_repo_id}: specialist model for GENERAL_QA."
            if qwen_model is not None:
                return qwen_model, f"Selected {qwen_model.hf_repo_id}: fallback model for GENERAL_QA."
            return ready_models[0], f"Selected {ready_models[0].hf_repo_id}: available ready model for GENERAL_QA."

        # 2. DOCUMENT_QA -> Qwen2.5-1.5B (or SmolLM2 for quick lookup)
        if task_type == TaskType.DOCUMENT_QA:
            matched_quick = [
                kw for kw in self.QUICK_LOOKUP_KEYWORDS
                if (kw in task_lower if " " in kw else bool(re.search(rf"\b{re.escape(kw)}\b", task_lower)))
            ]
            if matched_quick and smollm_model is not None:
                return smollm_model, f"Selected {smollm_model.hf_repo_id}: specialist model for speed-optimized quick lookup."
            if qwen_model is not None:
                return qwen_model, f"Selected {qwen_model.hf_repo_id}: specialist model for DOCUMENT_QA (structured document-grounded engineering QA)."
            if phi_model is not None:
                return phi_model, f"Selected {phi_model.hf_repo_id}: fallback model for DOCUMENT_QA."
            return ready_models[0], f"Selected {ready_models[0].hf_repo_id}: available ready model for DOCUMENT_QA."

        # 3. EXTRACTION -> Qwen2.5-1.5B
        if task_type == TaskType.EXTRACTION:
            if qwen_model is not None:
                return qwen_model, f"Selected {qwen_model.hf_repo_id}: specialist model for EXTRACTION (high-precision structured parameter/tag extraction)."
            if phi_model is not None:
                return phi_model, f"Selected {phi_model.hf_repo_id}: fallback model for EXTRACTION."
            return ready_models[0], f"Selected {ready_models[0].hf_repo_id}: available ready model for EXTRACTION."

        # 4. COMPARISON -> Phi-3.5-mini
        if task_type == TaskType.COMPARISON:
            if phi_model is not None:
                return phi_model, f"Selected {phi_model.hf_repo_id}: specialist model for COMPARISON (multi-parameter synthesis and comparative evaluation)."
            if qwen_model is not None:
                return qwen_model, f"Selected {qwen_model.hf_repo_id}: fallback model for COMPARISON."
            return ready_models[0], f"Selected {ready_models[0].hf_repo_id}: available ready model for COMPARISON."

        # 5. REASONING -> Phi-3.5-mini
        if task_type == TaskType.REASONING:
            if phi_model is not None:
                return phi_model, f"Selected {phi_model.hf_repo_id}: specialist model for REASONING (deep engineering analysis, root cause, and troubleshooting)."
            if qwen_model is not None:
                return qwen_model, f"Selected {qwen_model.hf_repo_id}: fallback model for REASONING."
            return ready_models[0], f"Selected {ready_models[0].hf_repo_id}: available ready model for REASONING."

        # 6. DOCUMENT_GENERATION -> Phi-3.5-mini
        if task_type == TaskType.DOCUMENT_GENERATION:
            if phi_model is not None:
                return phi_model, f"Selected {phi_model.hf_repo_id}: specialist model for DOCUMENT_GENERATION (structured report narrative and document synthesis)."
            if qwen_model is not None:
                return qwen_model, f"Selected {qwen_model.hf_repo_id}: fallback model for DOCUMENT_GENERATION."
            return ready_models[0], f"Selected {ready_models[0].hf_repo_id}: available ready model for DOCUMENT_GENERATION."

        # 7. SUMMARIZATION -> Phi-3.5-mini (for long document) or SmolLM2-1.7B (for brief summary)
        if task_type == TaskType.SUMMARIZATION:
            matched_quick = [
                kw for kw in self.QUICK_LOOKUP_KEYWORDS
                if (kw in task_lower if " " in kw else bool(re.search(rf"\b{re.escape(kw)}\b", task_lower)))
            ]
            if matched_quick and smollm_model is not None:
                return smollm_model, f"Selected {smollm_model.hf_repo_id}: specialist model for speed-optimized SUMMARIZATION / quick overview."
            if phi_model is not None:
                return phi_model, f"Selected {phi_model.hf_repo_id}: specialist model for comprehensive SUMMARIZATION / full report synthesis."
            if smollm_model is not None:
                return smollm_model, f"Selected {smollm_model.hf_repo_id}: specialist model for SUMMARIZATION."
            if qwen_model is not None:
                return qwen_model, f"Selected {qwen_model.hf_repo_id}: fallback model for SUMMARIZATION."
            return ready_models[0], f"Selected {ready_models[0].hf_repo_id}: available ready model for SUMMARIZATION."

        # 8. ENGINEERING_DRAWING_ANALYSIS -> Qwen2.5-VL-3B-Instruct
        if task_type == TaskType.ENGINEERING_DRAWING_ANALYSIS:
            if qwen_vl_model is not None:
                return qwen_vl_model, f"Selected {qwen_vl_model.hf_repo_id}: specialist model for ENGINEERING_DRAWING_ANALYSIS (P&ID, schematic, and diagram visual reasoning)."
            if ocr_model is not None:
                return ocr_model, f"Selected {ocr_model.hf_repo_id}: fallback processor for ENGINEERING_DRAWING_ANALYSIS."
            return ready_models[0], f"Selected {ready_models[0].hf_repo_id}: available model for ENGINEERING_DRAWING_ANALYSIS."

        # 9. VISION_OCR -> Member 3 processor / Qwen2.5-VL
        if task_type == TaskType.VISION_OCR:
            if any(k in task_lower for k in ("image", "photo", "photograph", "visual", "drawing", "diagram")) and qwen_vl_model is not None:
                return qwen_vl_model, f"Selected {qwen_vl_model.hf_repo_id}: multimodal vision specialist for visual image reasoning."
            if ocr_model is not None:
                return ocr_model, f"Selected {ocr_model.hf_repo_id}: specialist processor for VISION_OCR (scanned documents, tables, and nameplate OCR)."
            if qwen_vl_model is not None:
                return qwen_vl_model, f"Selected {qwen_vl_model.hf_repo_id}: multimodal vision fallback for VISION_OCR."
            return ready_models[0], f"Selected {ready_models[0].hf_repo_id}: available processor for VISION_OCR."

        # 10. CALCULATION -> Qwen2.5-1.5B (or Phi-3.5-mini for long context)
        if task_type == TaskType.CALCULATION:
            if qwen_model is not None:
                return qwen_model, f"Selected {qwen_model.hf_repo_id}: context lookup model for CALCULATION."
            if phi_model is not None:
                return phi_model, f"Selected {phi_model.hf_repo_id}: context lookup model for CALCULATION."
            return ready_models[0], f"Selected {ready_models[0].hf_repo_id}: available model for CALCULATION."

        # 11. IMAGE_GENERATION -> local Stable Diffusion model
        if task_type == TaskType.IMAGE_GENERATION:
            diff_model = next((m for m in ready_models if "diffusion" in m.family.lower() or "stable-diffusion" in m.hf_repo_id.lower() or m.role == "image_generation"), None)
            if diff_model is not None:
                return diff_model, f"Selected {diff_model.hf_repo_id}: local diffusion model for IMAGE_GENERATION."
            return ready_models[0], f"Selected {ready_models[0].hf_repo_id}: available model for IMAGE_GENERATION."

        # Fallback for general RAG / unknown:
        matched_long = [kw for kw in self.LONG_CONTEXT_KEYWORDS if kw in task_lower]
        is_long_query = len(task) > self.LONG_CONTEXT_THRESHOLD_CHARS
        if (matched_long or is_long_query) and phi_model is not None:
            return phi_model, f"Selected {phi_model.hf_repo_id}: long-context reasoning model."
        if smollm_model is not None:
            return smollm_model, f"Selected {smollm_model.hf_repo_id}: default lightweight model for general query."
        if qwen_model is not None:
            return qwen_model, f"Selected {qwen_model.hf_repo_id}: ready model."
        return ready_models[0], f"Selected {ready_models[0].hf_repo_id}: available ready model."

    def _resolve_model(
        self,
        task_type: TaskType,
        capability: Capability,
        archetype: PromptArchetype,
        matched: List[str],
        original_task: str,
    ) -> Tuple[Optional[ModelRecord], bool, str, Optional[str]]:
        """Determine model availability, compose the audit-trail reason, and set fallback warning."""

        effective_role = capability.value
        if capability in (Capability.CALCULATION, Capability.DOCUMENT_GENERATION, Capability.UNKNOWN):
            effective_role = "rag"

        capability_key = {
            Capability.VISION: "vision",
            Capability.CODING: "code_generation",
            Capability.IMAGE_GENERATION: "image_generation",
        }.get(capability)

        if task_type in (TaskType.VISION_OCR, TaskType.ENGINEERING_DRAWING_ANALYSIS):
            ready_models = self._registry.ready_for_role("vision") + self._registry.ready_for_role("ocr")
        elif capability_key:
            ready_models = self._registry.ready_for_capability(capability_key) or self._registry.ready_for_role(effective_role)
        else:
            ready_models = self._registry.ready_for_role(effective_role)

        # ── No ready model ─────────────────────────────────────────────
        if not ready_models:
            all_for_role = (
                [r for r in self._registry.all() if capability_key in r.capabilities]
                if capability_key
                else self._registry.by_role(effective_role)
            )
            if not all_for_role:
                reason = (
                    f"Capability '{effective_role}' has no models declared in models.yaml. "
                    f"Add an entry with role: {effective_role} to enable this capability."
                )
            else:
                disabled = [r.hf_repo_id for r in all_for_role if not r.enabled]
                missing = [r.hf_repo_id for r in all_for_role if r.enabled and not r.is_installed()]
                if disabled:
                    reason = (
                        f"Capability '{effective_role}' requires a model that is currently disabled "
                        f"in models.yaml (set enabled: true for one of: "
                        f"{', '.join(disabled)}). "
                        f"Matched task keywords: {matched or ['<none — no keyword match>']!r}."
                    )
                elif missing:
                    reason = (
                        f"Capability '{effective_role}' has models declared and enabled but not "
                        f"installed on disk: {', '.join(missing)}. "
                        f"Run scripts/download_llm_models.py to download them."
                    )
                else:
                    reason = (
                        f"Capability '{effective_role}' has no ready model "
                        f"(unknown reason — check models.yaml)."
                    )
            return None, False, reason, None

        # ── Ready model found ──────────────────────────────────────────
        model, model_selection_reason = self._select_model_for_task(task_type, capability, ready_models, original_task)

        fallback_warning: Optional[str] = None

        if capability == Capability.UNKNOWN:
            fallback_warning = (
                f"Low-confidence routing: no specific domain keywords matched the task. "
                f"Defaulting to general QA via {model.hf_repo_id} without document retrieval. "
                f"Verify answer or refine the query if document search is needed."
            )
            reason = (
                f"No specific keyword match found for task. {model_selection_reason} "
                f"Defaulting to general QA via {model.hf_repo_id}."
            )
        elif task_type == TaskType.CALCULATION:
            reason = (
                f"Task contains engineering calculation keywords {matched!r}. "
                f"Routing to calculation tool ('calculator') with RAG context from "
                f"{model.hf_repo_id} (archetype: {archetype.value}). {model_selection_reason}"
            )
        elif task_type == TaskType.IMAGE_GENERATION:
            reason = (
                f"Task contains image generation keywords {matched!r}. "
                f"Routing to local diffusion tool ('image_generator') via {model.hf_repo_id}."
            )
        else:
            kw_preview = matched[:3]
            extra = f" (+{len(matched)-3} more)" if len(matched) > 3 else ""
            reason = (
                f"Matched task type '{task_type.value}' ({capability.value}) via keywords "
                f"{kw_preview!r}{extra}. "
                f"Using {model.hf_repo_id} with archetype '{archetype.value}'. "
                f"{model_selection_reason}"
            )

        return model, True, reason, fallback_warning

    def extract_generation_params(self, task: str, **kwargs: Any) -> Dict[str, Any]:
        """Extract inline and explicit parameters for image generation requests."""
        return extract_image_generation_params(task, **kwargs)


# ── Natural Language Parameter Extractor for Image Generation ──────────────

def extract_image_generation_params(text: str, **kwargs: Any) -> Dict[str, Any]:
    """Extract natural-language and structured parameters for image generation.

    Handles explicit keyword arguments as highest priority, then parses inline
    dimensions (e.g. 512x512), inference steps, random seeds, CFG guidance scale,
    negative prompts, target filenames, and cleans the prompt string.
    """
    params: Dict[str, Any] = {}
    consumed_spans: List[Tuple[int, int]] = []

    # 1. Start with explicit keyword arguments if provided
    for k in ("negative_prompt", "width", "height", "steps", "guidance_scale", "seed", "filename", "filename_prefix"):
        if k in kwargs and kwargs[k] is not None:
            params[k] = kwargs[k]

    # 2. Extract dimensions: e.g. "512x512", "768x512", "width=512, height=512", "--width 512"
    if "width" not in params or "height" not in params:
        dim_match = re.search(r"\b(\d{2,4})\s*[xX*]\s*(\d{2,4})\b", text)
        if dim_match:
            params.setdefault("width", int(dim_match.group(1)))
            params.setdefault("height", int(dim_match.group(2)))
            consumed_spans.append(dim_match.span())

    if "width" not in params:
        w_match = re.search(r"(?:--width|-w|width\s*[:=])\s*(\d+)", text, re.IGNORECASE)
        if w_match:
            params["width"] = int(w_match.group(1))
            consumed_spans.append(w_match.span())

    if "height" not in params:
        h_match = re.search(r"(?:--height|-h|height\s*[:=])\s*(\d+)", text, re.IGNORECASE)
        if h_match:
            params["height"] = int(h_match.group(1))
            consumed_spans.append(h_match.span())

    # 3. Extract steps: e.g. "steps: 25", "--steps 25", "steps=25", "25 steps"
    if "steps" not in params:
        steps_match = re.search(r"(?:--steps|steps\s*[:=])\s*(\d+)", text, re.IGNORECASE) or re.search(r"\b(\d+)\s+steps\b", text, re.IGNORECASE)
        if steps_match:
            params["steps"] = int(steps_match.group(1))
            consumed_spans.append(steps_match.span())

    # 4. Extract guidance scale / CFG: e.g. "guidance: 7.5", "--guidance 7.5", "cfg: 7.5", "scale: 7.5"
    if "guidance_scale" not in params:
        cfg_match = re.search(r"(?:--guidance(?:_scale)?|--cfg|guidance\s*[:=]|cfg\s*[:=]|scale\s*[:=])\s*([0-9]+(?:\.[0-9]+)?)", text, re.IGNORECASE)
        if cfg_match:
            params["guidance_scale"] = float(cfg_match.group(1))
            consumed_spans.append(cfg_match.span())

    # 5. Extract seed: e.g. "seed: 42", "--seed 42", "seed=42", "seed 42"
    if "seed" not in params:
        seed_match = re.search(r"(?:--seed|seed\s*[:=]|\bseed\s+)(\d+)", text, re.IGNORECASE)
        if seed_match:
            params["seed"] = int(seed_match.group(1))
            consumed_spans.append(seed_match.span())

    # 6. Extract filename: e.g. "save as pump.png", "filename: pump.png", "to pump.png", "--filename pump.png"
    if "filename" not in params:
        fn_match = re.search(r"(?:save\s+(?:as|to)|filename\s*[:=]|--filename|-f)\s*([a-zA-Z0-9_\-]+\.png)", text, re.IGNORECASE)
        if fn_match:
            params["filename"] = fn_match.group(1).strip()
            consumed_spans.append(fn_match.span())

    # 7. Extract negative prompt: e.g. "--negative blurry, noisy", "negative: blurry, distorted"
    if "negative_prompt" not in params:
        neg_match = re.search(r"(?:--negative(?:_prompt)?|negative\s*prompt\s*[:=]|negative\s*[:=])\s*([^-\n;]+)", text, re.IGNORECASE)
        if neg_match:
            neg_val = neg_match.group(1).strip()
            neg_val = re.sub(r"(?:save\s+(?:as|to)|filename\s*[:=]|--\w+).*", "", neg_val, flags=re.IGNORECASE).strip()
            if neg_val:
                params["negative_prompt"] = neg_val.strip(", ")
                consumed_spans.append(neg_match.span())

    # 8. Prompt extraction
    if "prompt" in kwargs and kwargs["prompt"]:
        params["prompt"] = kwargs["prompt"]
    else:
        cleaned_chars = list(text)
        for start, end in sorted(consumed_spans, reverse=True):
            cleaned_chars[start:end] = " "
        cleaned = "".join(cleaned_chars)

        cleaned = re.sub(r"--\w+\b", " ", cleaned)
        cleaned = re.sub(r"[()]", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        cleaned_prompt = re.sub(
            r"^\s*(?:please\s+)?(?:generate|create|make|render|draw|produce|synthesize|output|build|paint|sketch)\s+(?:an?\s+)?(?:image|picture|photo|photograph|illustration|diagram|rendering|graphic|visual\s+representation|visual)\s+(?:of|showing|depicting|illustrating|for|with)?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()

        if cleaned_prompt.lower().startswith("a ") and len(cleaned_prompt) > 2:
            cleaned_prompt = cleaned_prompt[2:].strip()
        elif cleaned_prompt.lower().startswith("an ") and len(cleaned_prompt) > 3:
            cleaned_prompt = cleaned_prompt[3:].strip()
        elif cleaned_prompt.lower().startswith("the ") and len(cleaned_prompt) > 4:
            cleaned_prompt = cleaned_prompt[4:].strip()

        params["prompt"] = cleaned_prompt if len(cleaned_prompt) >= 2 else text.strip()

    return params


# ── Module-level singleton ────────────────────────────────────────────────

_router: Optional[TaskRouter] = None


def get_router(registry: Optional[AgentModelRegistry] = None) -> TaskRouter:
    """Return the shared TaskRouter singleton."""
    global _router
    if _router is None:
        _router = TaskRouter(registry)
    return _router


# ── Self-test / __main__ ──────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    TEST_TASKS = [
        # General QA -> SmolLM2-1.7B (no RAG)
        ("What is thermodynamics?", TaskType.GENERAL_QA, "HuggingFaceTB/SmolLM2-1.7B-Instruct", False),
        ("Who is Isaac Newton?", TaskType.GENERAL_QA, "HuggingFaceTB/SmolLM2-1.7B-Instruct", False),
        # Document QA -> Qwen2.5-1.5B (with RAG)
        ("What is the design operating pressure of centrifugal pump P-203?", TaskType.DOCUMENT_QA, "Qwen/Qwen2.5-1.5B-Instruct", True),
        ("Explain the startup procedure for the crude distillation unit.", TaskType.DOCUMENT_QA, "Qwen/Qwen2.5-1.5B-Instruct", True),
        ("What are the OISD safety requirements for hot work permits?", TaskType.DOCUMENT_QA, "Qwen/Qwen2.5-1.5B-Instruct", True),
        # Extraction -> Qwen2.5-1.5B (with RAG)
        ("Extract the design temperature and metallurgy of vessel V-2201", TaskType.EXTRACTION, "Qwen/Qwen2.5-1.5B-Instruct", True),
        # Comparison -> Phi-3.5-mini (with RAG)
        ("Compare the design temperatures of heat exchangers E-101 and E-202.", TaskType.COMPARISON, "microsoft/Phi-3.5-mini-instruct", True),
        # Reasoning / Troubleshooting -> Phi-3.5-mini (with RAG)
        ("Troubleshoot high vibration on compressor K-101.", TaskType.REASONING, "microsoft/Phi-3.5-mini-instruct", True),
        # Document Generation -> Phi-3.5-mini
        ("Generate a statutory inspection report PDF for vessel V-2201", TaskType.DOCUMENT_GENERATION, "microsoft/Phi-3.5-mini-instruct", True),
        # Summarization -> Phi-3.5-mini (long) / SmolLM2 (quick)
        ("Summarize the full report for pressure vessel V-2201 including all historical ultrasonic thickness readings.", TaskType.SUMMARIZATION, "microsoft/Phi-3.5-mini-instruct", True),
        ("Quick lookup of equipment tag and operating pressure for pump P-203", TaskType.SUMMARIZATION, "HuggingFaceTB/SmolLM2-1.7B-Instruct", True),
        # Vision OCR -> Member 3 processor
        ("Read the nameplate from this scanned document of pump P-101.", TaskType.VISION_OCR, "member3_ocr/multimodal_processor", False),
        # Engineering Drawing -> Qwen2.5-VL-3B-Instruct
        ("Analyze this P&ID diagram and inspect piping connections.", TaskType.ENGINEERING_DRAWING_ANALYSIS, "Qwen/Qwen2.5-VL-3B-Instruct", False),
    ]

    router = TaskRouter()
    print("=" * 72)
    print("Task Router — Multi-Model Specialist Routing Verification")
    print("=" * 72)
    for task, expected_task_type, expected_model, expected_rag in TEST_TASKS:
        d = router.route(task)
        model = d.model_record.hf_repo_id if d.model_record else "none"
        print(f"\nTask       : {task[:65]!r}")
        print(f"  TaskType   : {d.task_type.value} (expected: {expected_task_type.value})")
        print(f"  Capability : {d.capability.value}")
        print(f"  Model      : {model} (expected: {expected_model})")
        print(f"  Use RAG    : {d.use_rag_context} (expected: {expected_rag})")
        print(f"  Reason     : {d.reason}")

        assert d.task_type == expected_task_type, f"TaskType mismatch for {task}: got {d.task_type}, expected {expected_task_type}"
        assert d.model_record is not None, f"Model is None for {task}"
        assert d.model_record.hf_repo_id == expected_model, f"Model mismatch for {task}: got {d.model_record.hf_repo_id}, expected {expected_model}"
        assert d.use_rag_context == expected_rag, f"use_rag_context mismatch for {task}: got {d.use_rag_context}, expected {expected_rag}"
        print("  --> [PASSED]")

    print("\nALL ROUTER MULTI-MODEL SELECTION ASSERTIONS PASSED SUCCESSFULLY!")

