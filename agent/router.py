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
    UNKNOWN             = "unknown"              # no rule matched; falls back to RAG with warning


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
            f"[Router] capability={self.capability.value} | "
            f"archetype={self.archetype.value} | "
            f"model={model} | "
            f"tool={tool} | "
            f"use_rag={self.use_rag_context} | "
            f"status={status}{warn} | "
            f"reason={self.reason!r}"
        )


# ── Rule tables ────────────────────────────────────────────────────────────
# Each rule is a tuple: (Capability, PromptArchetype, [trigger_keywords]).
# Rules are checked in order; first match wins.
# Keywords are matched case-insensitively against the full task string.
# A keyword may be a plain word OR a regex pattern (prefixed with "re:").

_ROUTING_RULES: List[Tuple[Capability, PromptArchetype, List[str]]] = [

    # ── Document / Report / Artifact generation ───────────────────────────
    (
        Capability.DOCUMENT_GENERATION,
        PromptArchetype.REPORT,
        [
            r"re:.*\b(?:create|generate|export|prepare|save|output|produce|build|draft|write|download|make)\b.*\b(?:actual\s+)?(?:pdf|docx|word\s+document|excel|spreadsheet|xlsx|powerpoint|pptx)\b",
            r"re:.*\b(?:pdf|docx|xlsx|pptx)\s+(?:file|document|report|artifact|export)\b",
            r"re:.*\bexport\s+(?:as|to)\s*(?:pdf|docx|word|excel|xlsx|pptx|powerpoint)\b",
            r"re:.*\bas\s+(?:an?\s+)?(?:actual\s+)?(?:pdf|docx|xlsx|pptx)\s+artifact\b",
        ],
    ),

    # ── Code / script generation ──────────────────────────────────────────
    (
        Capability.CODING,
        PromptArchetype.GENERAL_QA,   # not used if capability unavailable
        [
            "write code", "generate code", "write a script", "write script",
            "write python", "write sql", "generate sql", "generate python",
            "create a function", "implement a", "code snippet", "write function",
            "script to", "automate", "generate report script",
        ],
    ),

    # ── Image Generation / Visual Synthesis ──────────────────────────────
    (
        Capability.IMAGE_GENERATION,
        PromptArchetype.GENERAL_QA,
        [
            r"re:.*\b(?:generate|create|make|render|draw|produce|synthesize|output|build|paint|sketch|illustrate)\b.*\b(?:an?\s+)?(?:image|picture|photo|photograph|illustration|diagram|rendering|graphic|visual\s+representation|visual|schematic\s+image)\b",
            r"re:.*\b(?:text-to-image|txt2img|stable\s*diffusion|diffusion\s*image|diffusion\s*model)\b",
            r"re:.*\b(?:visual\s+representation\s+of)\b",
        ],
    ),

    # ── Vision / image / OCR ─────────────────────────────────────────────
    (
        Capability.VISION,
        PromptArchetype.GENERAL_QA,
        [
            "image", "photo", "picture", "diagram", "p&id", "pid diagram",
            "drawing", "scan", "ocr", "handwritten", "sketch", "schematic",
            "visual", "plate label", "nameplate", "photograph",
        ],
    ),

    # ── Numerical / engineering calculation ───────────────────────────────
    # (handled by calc tool; still uses RAG model for context if available)
    (
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

    # ── RAG: Safety & compliance ──────────────────────────────────────────
    (
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

    # ── RAG: SOP / procedure ─────────────────────────────────────────────
    (
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

    # ── RAG: Maintenance ─────────────────────────────────────────────────
    (
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

    # ── RAG: Troubleshooting ─────────────────────────────────────────────
    (
        Capability.RAG,
        PromptArchetype.TROUBLESHOOTING,
        [
            "troubleshoot", "fault", "failure", "root cause", "rca",
            "why is", "why did", "abnormal", "alarm", "trip", "high temperature",
            "high pressure", "low flow", "cavitation", "surge", "vibrating",
            "leak", "leaking", "chatter", "erratic", "unstable",
            "not working", "failed", "broken",
        ],
    ),

    # ── RAG: Equipment lookup ─────────────────────────────────────────────
    (
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
            "data sheet", "specification",
        ],
    ),

    # ── RAG: Comparison ──────────────────────────────────────────────────
    (
        Capability.RAG,
        PromptArchetype.COMPARISON,
        [
            "compare", "comparison", "versus", " vs ", "difference between",
            "better than", "which is better", "trade-off", "tradeoff",
            "alternative", "option a vs", "option b",
        ],
    ),

    # ── RAG: General inspection / report questions ────────────────────────
    (
        Capability.RAG,
        PromptArchetype.GENERAL_QA,
        [
            "what is", "what are", "explain", "describe", "define",
            "tell me", "give me", "show me", "list", "summarize",
            "pressure vessel", "piping", "instrument", "refinery",
            "process", "unit", "plant",
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
        2. Walk ROUTING_RULES in order; collect the first match.
        3. Determine execution tool and whether RAG context is required.
        4. Check AgentModelRegistry for a ready model for that capability.
        5. If no ready model exists → capability_available=False.
        6. Return RoutingDecision with model, tool, context flag, warning, and reason.
        """
        normalized = self._normalize(task)

        capability, archetype, matched = self._match_rules(normalized)

        # Resolve execution tool and RAG context requirement
        tool_name, use_rag_context = self._resolve_tool(capability, task=normalized)

        # Resolve model, audit reason, and structured fallback warning
        model_record, available, reason, fallback_warning = self._resolve_model(
            capability, archetype, matched, task
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
    ) -> Tuple[Capability, PromptArchetype, List[str]]:
        """Return (capability, archetype, matched_keywords) for first matching rule."""
        for capability, archetype, keywords in _ROUTING_RULES:
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
                return capability, archetype, hits

        # No rule matched → treat as general RAG
        return Capability.UNKNOWN, PromptArchetype.GENERAL_QA, []

    @staticmethod
    def _resolve_tool(capability: Capability, task: str = "") -> Tuple[Optional[str], bool]:
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
        if capability == Capability.CALCULATION:
            # Calculation requires deterministic tool computation, but uses RAG
            # pipeline to look up vessel specs / design parameters from docs first.
            return "calculator", True
        elif capability == Capability.DOCUMENT_GENERATION:
            t_lower = task.lower()
            if any(k in t_lower for k in ["xlsx", "excel", "spreadsheet"]):
                return "xlsx_generator", True
            elif any(k in t_lower for k in ["pptx", "powerpoint", "slide"]):
                return "pptx_generator", True
            elif any(k in t_lower for k in ["docx", "word"]):
                return "document_generator", True
            return "pdf_generator", True
        elif capability == Capability.IMAGE_GENERATION:
            return "image_generator", False
        elif capability in (Capability.RAG, Capability.UNKNOWN):
            return "rag_pipeline", True
        elif capability == Capability.CODING:
            return "code_interpreter", False
        elif capability == Capability.VISION:
            return "vision_inspector", False
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

    def _select_rag_model(
        self,
        ready_models: List[ModelRecord],
        task: str,
    ) -> Tuple[ModelRecord, str]:
        """Select between multiple ready RAG models based on task characteristics."""
        task_lower = task.lower()

        # Index ready models by family / repo id
        phi_model = next((m for m in ready_models if "phi" in m.family.lower() or "phi-3.5-mini" in m.hf_repo_id.lower()), None)
        smollm_model = next((m for m in ready_models if "smollm" in m.family.lower() or "smollm" in m.hf_repo_id.lower()), None)
        qwen_model = next((m for m in ready_models if "qwen" in m.family.lower() or "qwen2.5" in m.hf_repo_id.lower()), None)

        # 1. Long context / full document -> Phi-3.5-mini-instruct (context_window: 131,072)
        matched_long = [kw for kw in self.LONG_CONTEXT_KEYWORDS if kw in task_lower]
        is_long_query = len(task) > self.LONG_CONTEXT_THRESHOLD_CHARS

        if (matched_long or is_long_query) and phi_model is not None:
            trigger_detail = (
                f"matched keywords: {matched_long}"
                if matched_long
                else f"query length {len(task)} chars > {self.LONG_CONTEXT_THRESHOLD_CHARS} threshold"
            )
            reason = (
                f"Selected {phi_model.hf_repo_id}: task requires long-context reasoning "
                f"({trigger_detail}; context window: {phi_model.context_window:,} tokens)."
            )
            return phi_model, reason

        # 2. Speed-optimized / quick lookups -> SmolLM2-1.7B-Instruct (context_window: 8,192)
        matched_quick = [
            kw for kw in self.QUICK_LOOKUP_KEYWORDS
            if (kw in task_lower if " " in kw else bool(re.search(rf"\b{re.escape(kw)}\b", task_lower)))
        ]
        if matched_quick and smollm_model is not None:
            reason = (
                f"Selected {smollm_model.hf_repo_id}: task requests speed-optimized quick lookup "
                f"(matched keywords: {matched_quick}; context window: {smollm_model.context_window:,} tokens)."
            )
            return smollm_model, reason

        # 3. Default standard RAG queries -> Qwen2.5-1.5B-Instruct
        if qwen_model is not None:
            reason = (
                f"Selected {qwen_model.hf_repo_id}: default primary RAG model for standard "
                f"engineering queries (context window: {qwen_model.context_window:,} tokens)."
            )
            return qwen_model, reason

        # Fallback to first available ready model
        fallback = ready_models[0]
        reason = f"Selected {fallback.hf_repo_id}: default ready model for role 'rag'."
        return fallback, reason

    def _resolve_model(
        self,
        capability: Capability,
        archetype: PromptArchetype,
        matched: List[str],
        original_task: str,
    ) -> Tuple[Optional[ModelRecord], bool, str, Optional[str]]:
        """Determine model availability, compose the audit-trail reason, and set fallback warning."""

        # CALCULATION / DOCUMENT_GENERATION: uses RAG pipeline for context + tool
        # Map it to the RAG role for model resolution
        effective_role = capability.value
        if capability in (Capability.CALCULATION, Capability.DOCUMENT_GENERATION, Capability.UNKNOWN):
            effective_role = "rag"

        # Model choice is capability-driven.  Roles remain as a compatibility
        # grouping for existing text/RAG entries, while specialist routing is
        # resolved from declarative capabilities in models.yaml.
        capability_key = {
            Capability.VISION: "vision",
            Capability.CODING: "code_generation",
            Capability.IMAGE_GENERATION: "image_generation",
        }.get(capability)
        ready_models = (
            self._registry.ready_for_capability(capability_key)
            if capability_key
            else self._registry.ready_for_role(effective_role)
        )

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
        if effective_role == "rag":
            model, model_selection_reason = self._select_rag_model(ready_models, original_task)
        else:
            model = ready_models[0]
            model_selection_reason = f"Selected {model.hf_repo_id} for role '{effective_role}'."

        fallback_warning: Optional[str] = None

        if capability == Capability.UNKNOWN:
            fallback_warning = (
                f"Low-confidence routing: no specific domain keywords matched the task. "
                f"Defaulting to general RAG (general_qa) via {model.hf_repo_id}. "
                f"Verify answer grounding or refine the query."
            )
            reason = (
                f"No specific keyword match found for task. {model_selection_reason} "
                f"Defaulting to RAG / general_qa via {model.hf_repo_id}. "
                f"Consider refining the task description."
            )
        elif capability == Capability.CALCULATION:
            reason = (
                f"Task contains engineering calculation keywords {matched!r}. "
                f"Routing to calculation tool ('calculator') with RAG context from "
                f"{model.hf_repo_id} (archetype: {archetype.value}). {model_selection_reason}"
            )
        elif capability == Capability.IMAGE_GENERATION:
            reason = (
                f"Task contains image generation keywords {matched!r}. "
                f"Routing to local diffusion tool ('image_generator') via {model.hf_repo_id}."
            )
        else:
            kw_preview = matched[:3]
            extra = f" (+{len(matched)-3} more)" if len(matched) > 3 else ""
            reason = (
                f"Matched {capability.value} capability via keywords "
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
            # If "save as ..." was trailing in neg_val, trim it
            neg_val = re.sub(r"(?:save\s+(?:as|to)|filename\s*[:=]|--\w+).*", "", neg_val, flags=re.IGNORECASE).strip()
            if neg_val:
                params["negative_prompt"] = neg_val.strip(", ")
                consumed_spans.append(neg_match.span())

    # 8. Prompt extraction
    if "prompt" in kwargs and kwargs["prompt"]:
        params["prompt"] = kwargs["prompt"]
    else:
        # Rebuild text removing consumed spans
        cleaned_chars = list(text)
        for start, end in sorted(consumed_spans, reverse=True):
            cleaned_chars[start:end] = " "
        cleaned = "".join(cleaned_chars)

        # Remove extra punctuation/flags remnants
        cleaned = re.sub(r"--\w+\b", " ", cleaned)
        cleaned = re.sub(r"[()]", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        # Strip leading trigger phrases like "generate an image of", "create a picture of", etc.
        cleaned_prompt = re.sub(
            r"^\s*(?:please\s+)?(?:generate|create|make|render|draw|produce|synthesize|output|build|paint|sketch)\s+(?:an?\s+)?(?:image|picture|photo|photograph|illustration|diagram|rendering|graphic|visual\s+representation|visual)\s+(?:of|showing|depicting|illustrating|for|with)?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()

        # Strip leading articles like "a boiler" -> "boiler" or keep
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
        # RAG — standard (Qwen2.5-1.5B-Instruct)
        "What is the design operating pressure of centrifugal pump P-203?",
        "What is a pressure vessel?",
        "Explain the startup procedure for the crude distillation unit.",
        "What are the OISD safety requirements for hot work permits?",
        "Troubleshoot high vibration on compressor K-101.",
        "Compare the design temperatures of heat exchangers E-101 and E-202.",
        "Show me the maintenance schedule for pump P-101 bearings.",
        # RAG — Long Context / Full Report (Phi-3.5-mini-instruct)
        "Summarize the full report for pressure vessel V-2201 including all historical ultrasonic thickness readings.",
        "Perform a comprehensive multi-point structural integrity evaluation for column C-101 using all available statutory inspection logs, ultrasonic thickness surveys, corrosion coupon analysis reports, and metallurgical examination datasheets to determine remaining life under API 510.",
        # RAG — Speed-optimized Quick Lookup (SmolLM2-1.7B-Instruct)
        "Quick lookup of equipment tag and operating pressure for pump P-203",
        "Fast check of relief valve setting for V-305",
        # Calculation
        "Calculate the relief valve sizing for vessel V-305.",
        "Verify the MAWP calculation for this pressure vessel.",
        # Coding (disabled model)
        "Write a Python script to parse the inspection report.",
        # Vision (disabled model)
        "Read the nameplate from this image of the pump.",
        # Unknown
        "Help me understand something about the refinery.",
        # Exact no-match edge case
        "zxqwerty bloop foo bar",
    ]

    router = TaskRouter()
    print("=" * 72)
    print("Task Router — Self-Test")
    print("=" * 72)
    for task in TEST_TASKS:
        d = router.route(task)
        status = "ACTIONABLE" if d.is_actionable() else "NOT AVAILABLE"
        model = d.model_record.hf_repo_id if d.model_record else "none"
        print(f"\nTask       : {task[:65]!r}")
        print(f"  Capability : {d.capability.value}")
        print(f"  Archetype  : {d.archetype.value}")
        print(f"  Model      : {model}")
        print(f"  Tool       : {d.tool_name}")
        print(f"  Use RAG    : {d.use_rag_context}")
        print(f"  Warning    : {d.fallback_warning}")
        print(f"  Status     : {status}")
        print(f"  Keywords   : {d.matched_keywords[:4]}")
        print(f"  Reason     : {d.reason}")

    # Explicit multi-model routing verification
    print("\n" + "=" * 72)
    print("Multi-Model RAG Routing Verification Assertions")
    print("=" * 72)

    # 1. Standard RAG query -> Qwen2.5-1.5B-Instruct
    d_qwen = router.route("What is the design operating pressure of centrifugal pump P-203?")
    assert d_qwen.model_record is not None
    assert d_qwen.model_record.hf_repo_id == "Qwen/Qwen2.5-1.5B-Instruct"
    assert "default primary RAG model" in d_qwen.reason
    print("  [+] Standard query routes to Qwen2.5-1.5B-Instruct (PASSED)")

    # 2. Long Document / Full Report -> Phi-3.5-mini-instruct
    d_phi = router.route("Summarize the full report for pressure vessel V-2201 with all thickness surveys.")
    assert d_phi.model_record is not None
    assert d_phi.model_record.hf_repo_id == "microsoft/Phi-3.5-mini-instruct"
    assert "long-context reasoning" in d_phi.reason
    print("  [+] Long-document query routes to Phi-3.5-mini-instruct (PASSED)")

    # 3. Long Query (>250 chars) -> Phi-3.5-mini-instruct
    long_task = "Analyze the comprehensive multi-point statutory inspection dossier for crude column C-101 covering all tray inspection logs, ultrasonic thickness survey tables, corrosion coupon analyses, nozzle structural assessments, and safety re-certification under API 510."
    d_phi_long = router.route(long_task)
    assert d_phi_long.model_record is not None
    assert d_phi_long.model_record.hf_repo_id == "microsoft/Phi-3.5-mini-instruct"
    assert "long-context reasoning" in d_phi_long.reason
    print("  [+] Long-query (>250 chars) routes to Phi-3.5-mini-instruct (PASSED)")

    # 4. Quick Lookup -> SmolLM2-1.7B-Instruct
    d_smollm = router.route("Quick lookup of equipment tag and operating pressure for pump P-203")
    assert d_smollm.model_record is not None
    assert d_smollm.model_record.hf_repo_id == "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    assert "speed-optimized quick lookup" in d_smollm.reason
    print("  [+] Quick lookup query routes to SmolLM2-1.7B-Instruct (PASSED)")

    print("\nALL ROUTER MULTI-MODEL SELECTION ASSERTIONS PASSED SUCCESSFULLY!")
