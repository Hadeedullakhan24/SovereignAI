"""Representative Evaluation Harness for Member 3 Vision Pipeline.

Evaluates the offline local vision model (Qwen2.5-VL-3B-Instruct) across
curated representative samples of MRPL engineering drawings, industrial
inspection / defect images, and handwritten notes.

Datasets evaluated (strictly read-only):
  - C:\\SovereignAI\\datasets\\engineering_drawings (P&ID, PFD, Equipment, Instrumentation, Pump, Electrical)
  - C:\\SovereignAI\\datasets\\Vision (corrosion defects, industrial defects, infrared solar modules)
  - C:\\SovereignAI\\datasets\\handwritten_notes (industrial inspection logs)

Outputs written strictly to:
  C:\\SovereignAI\\member3_ocr\\output\\evaluation\\vision
  - vision_evaluation.json
  - vision_evaluation.csv
  - README.md

Confidence Score Derivation Note:
  The ``confidence`` field in EvaluationRecord is extracted from the ``understand`` task
  response. It is populated from ``_parse_understanding_response`` in vision_pipeline.py,
  which reads JSON embedded in the model's text output under the "confidence" key.
  This is a HEURISTIC self-report, NOT a calibrated probability (not log-softmax/logprob).
  The field ``confidence_method: "self_reported_heuristic"`` is stored in every record and
  in the JSON ``confidence_metadata`` block for unambiguous interpretation.
"""
from __future__ import annotations


import argparse
import csv
import json
import logging
import os
import re
import statistics
import subprocess
import signal
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

# Windows Torch-Paddle import ordering safeguard
try:
    import torch  # noqa: F401
except ImportError:
    pass

# Ensure project root is in sys.path
from member3_ocr.evaluation.paths import get_project_root, get_output_dir
PROJECT_ROOT = get_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image
import numpy as np

from member3_ocr.core.common_validators import (
    DRAWING_SYNONYMS,
    RE_DIMENSION,
    RE_GENERIC_TAG_PATTERN,
    RE_PRESSURE,
    RE_PROXIMITY_CONNECTION,
    RE_SAFETY_CLAIM,
    RE_TEMPERATURE,
    check_drawing_type_match,
    check_hallucinations,
    inspect_pid_outputs,
)
from member3_ocr.core.vision_pipeline import (
    DEFAULT_VISION_MODEL_PATH,
    LocalQwenVisionBackend,
    MockVisionBackend,
    VisionBackend,
    VisionModelConfig,
    VisionPipeline,
    VisionResult,
)
from member3_ocr.core.pdf_rendering import render_pdf_pages

LOGGER = logging.getLogger("vision_evaluator")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEFAULT_OUTPUT_DIR = get_output_dir() / "evaluation" / "vision"

# Confidence derivation method — single source of truth stored in every EvaluationRecord
CONFIDENCE_METHOD = "self_reported_heuristic"
"""
Confidence is extracted from the vision model's own JSON output under the 'confidence'
key in the 'understand' task response. It is NOT computed from model logprobs or softmax
probabilities. It is a heuristic self-assessment the model generates as plain text.
Use only for qualitative ordering; do not treat as a calibrated probability.
"""


# ──────────────────────────────────────────────────────────────────────────────
# Sample Manifest Definition
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SampleManifestItem:
    """Specification of a representative sample for vision evaluation."""

    sample_id: str
    source_path: str  # Relative to PROJECT_ROOT
    dataset: str
    category: str
    tasks: tuple[str, ...]
    expected_drawing_type: str | None = None
    expected_equipment: tuple[str, ...] = ()
    # expected_scene_keywords: keywords expected to appear in correct scene/caption outputs
    expected_scene_keywords: tuple[str, ...] = ()
    description: str = ""


REPRESENTATIVE_SAMPLES: tuple[SampleManifestItem, ...] = (
    # --- Engineering Drawings: P&ID (2 samples) ---
    SampleManifestItem(
        sample_id="PID_01",
        source_path="datasets/engineering_drawings/PID/PID_002_Process_Example.jpg",
        dataset="engineering_drawings",
        category="PID",
        tasks=("understand", "caption", "drawing", "equipment"),
        expected_drawing_type="pid",
        expected_equipment=("tank", "pump", "valve", "heat exchanger"),
        expected_scene_keywords=("p&id", "pid", "piping", "instrumentation", "process"),
        description="Process piping and instrumentation diagram with vessels, control valves, and pumps",
    ),
    SampleManifestItem(
        sample_id="PID_02",
        source_path="datasets/engineering_drawings/PID/PID_003_Heat_Exchanger_Instrumentation.jpg",
        dataset="engineering_drawings",
        category="PID",
        tasks=("understand", "caption", "drawing", "equipment"),
        expected_drawing_type="pid",
        expected_equipment=("heat exchanger", "valve", "instrument"),
        expected_scene_keywords=("p&id", "pid", "heat exchanger", "instrumentation", "loop"),
        description="Heat exchanger instrumentation loop with temperature/pressure sensors and bypass valves",
    ),
    # --- Engineering Drawings: PFD (2 samples) ---
    SampleManifestItem(
        sample_id="PFD_01",
        source_path="datasets/engineering_drawings/PFD/PFD_002_Refinery_Process_Flow.png",
        dataset="engineering_drawings",
        category="PFD",
        tasks=("understand", "caption", "drawing", "equipment"),
        expected_drawing_type="pfd",
        expected_equipment=("column", "reactor", "condenser", "pump"),
        expected_scene_keywords=("pfd", "process flow", "refinery", "flow diagram"),
        description="Refinery process flow diagram showing primary distillation flow and major streams",
    ),
    SampleManifestItem(
        sample_id="PFD_02",
        source_path="datasets/engineering_drawings/PFD/PFD_003_Refinery_Flow_Complex.png",
        dataset="engineering_drawings",
        category="PFD",
        tasks=("understand", "caption", "drawing", "equipment"),
        expected_drawing_type="pfd",
        expected_equipment=("separator", "vessel", "pump", "heat exchanger"),
        expected_scene_keywords=("pfd", "process flow", "refinery", "complex"),
        description="Complex multi-unit refinery flow diagram with recycle streams and flash drums",
    ),
    # --- Engineering Drawings: Equipment Drawings (2 samples) ---
    SampleManifestItem(
        sample_id="EQUIP_01",
        source_path="datasets/engineering_drawings/Equipment_Drawings/EQUIP_001_AlfaLaval_Plate_Heat_Exchanger_Drawing.pdf",
        dataset="engineering_drawings",
        category="Equipment_Drawings",
        tasks=("understand", "caption", "drawing", "equipment"),
        expected_drawing_type="equipment_drawing",
        expected_equipment=("heat exchanger", "plate heat exchanger"),
        expected_scene_keywords=("heat exchanger", "plate", "mechanical", "sectional"),
        description="Alfa Laval plate heat exchanger dimensional and sectional mechanical drawing",
    ),
    SampleManifestItem(
        sample_id="EQUIP_02",
        source_path="datasets/engineering_drawings/Equipment_Drawings/EQUIP_008_1Process_Equipment_Design_Drawings.pdf",
        dataset="engineering_drawings",
        category="Equipment_Drawings",
        tasks=("understand", "caption", "drawing", "equipment"),
        expected_drawing_type="equipment_drawing",
        expected_equipment=("pressure vessel", "tank", "reactor"),
        expected_scene_keywords=("equipment", "mechanical", "vessel", "design"),
        description="Process equipment mechanical design detail drawings with nozzle schedules",
    ),
    # --- Engineering Drawings: Instrumentation (1 sample) ---
    SampleManifestItem(
        sample_id="INST_01",
        source_path="datasets/engineering_drawings/Instrumentation/INST_001_Industrial_Control_Loop.jpg",
        dataset="engineering_drawings",
        category="Instrumentation",
        tasks=("understand", "caption", "drawing", "equipment"),
        expected_drawing_type="instrumentation",
        expected_equipment=("transmitter", "controller", "control valve"),
        expected_scene_keywords=("control loop", "instrumentation", "control", "loop"),
        description="Industrial feedback control loop diagram with sensor transmitter and valve actuator",
    ),
    # --- Engineering Drawings: Pump Diagrams (1 sample) ---
    SampleManifestItem(
        sample_id="PUMP_01",
        source_path="datasets/engineering_drawings/Pump_Diagrams/PUMP_001_Grundfos_CR_Sectional_Drawings.pdf",
        dataset="engineering_drawings",
        category="Pump_Diagrams",
        tasks=("understand", "caption", "drawing", "equipment"),
        expected_drawing_type="pump_diagram",
        expected_equipment=("pump", "centrifugal pump", "impeller", "motor"),
        expected_scene_keywords=("pump", "centrifugal", "sectional", "mechanical"),
        description="Grundfos CR multistage vertical centrifugal pump sectional cutaway drawing",
    ),
    # --- Engineering Drawings: Electrical (1 sample) ---
    SampleManifestItem(
        sample_id="ELEC_01",
        source_path="datasets/engineering_drawings/Electrical/ELEC_001_Industrial_Single_Line_Diagram.pdf",
        dataset="engineering_drawings",
        category="Electrical",
        tasks=("understand", "caption", "drawing"),
        expected_drawing_type="electrical",
        expected_equipment=("transformer", "switchgear", "circuit breaker", "busbar"),
        expected_scene_keywords=("electrical", "single line", "single-line", "sld", "power"),
        description="Industrial high-voltage electrical single-line power distribution diagram",
    ),
    # --- Vision: Corrosion / Surface Defects (2 samples) ---
    SampleManifestItem(
        sample_id="CORR_01",
        source_path="datasets/Vision/corrosion_defects/NEU_Surface_Defects/IMAGES/crazing_1.jpg",
        dataset="Vision",
        category="corrosion_defects",
        tasks=("understand", "caption", "observation"),
        expected_scene_keywords=("defect", "surface", "crack", "crazing", "steel"),
        description="NEU surface defect: metallic surface micro-fissuring and crazing pattern",
    ),
    SampleManifestItem(
        sample_id="CORR_02",
        source_path="datasets/Vision/corrosion_defects/NEU_Surface_Defects/IMAGES/pitted_surface_1.jpg",
        dataset="Vision",
        category="corrosion_defects",
        tasks=("understand", "caption", "observation"),
        expected_scene_keywords=("defect", "pitting", "surface", "corrosion", "steel"),
        description="NEU surface defect: localized pitting corrosion on hot-rolled steel strip",
    ),
    # --- Vision: General Industrial / Inspection Defect (2 samples) ---
    SampleManifestItem(
        sample_id="INSP_01",
        source_path="datasets/Vision/defect_detection/images/Missing_hole/01_missing_hole_01.jpg",
        dataset="Vision",
        category="defect_detection",
        tasks=("understand", "caption", "observation"),
        expected_scene_keywords=("pcb", "circuit", "board", "defect", "hole"),
        description="Industrial PCB automated optical inspection: missing drilled through-hole defect",
    ),
    SampleManifestItem(
        sample_id="INSP_02",
        source_path="datasets/Vision/defect_detection/images/Open_circuit/01_open_circuit_01.jpg",
        dataset="Vision",
        category="defect_detection",
        tasks=("understand", "caption", "observation"),
        expected_scene_keywords=("pcb", "circuit", "board", "open circuit", "defect"),
        description="Industrial PCB automated optical inspection: conductor open circuit trace disconnection",
    ),
    # --- Vision: Infrared Inspection (1 sample) ---
    SampleManifestItem(
        sample_id="IR_01",
        source_path="datasets/Vision/infrared/InfraredSolarModules/images/0.jpg",
        dataset="Vision",
        category="infrared",
        tasks=("understand", "caption", "observation"),
        expected_scene_keywords=("infrared", "thermal", "solar", "photovoltaic", "module"),
        description="Infrared thermographic inspection image of photovoltaic solar module showing thermal gradient",
    ),
    # --- Handwritten Industrial Notes (2 samples) ---
    SampleManifestItem(
        sample_id="HW_01",
        source_path="datasets/handwritten_notes/HN_0001_narrative_Pipeline_CT106B.jpg",
        dataset="handwritten_notes",
        category="handwritten_notes",
        tasks=("understand", "caption"),
        expected_scene_keywords=("handwritten", "notes", "log", "pipeline", "maintenance"),
        description="Handwritten plant shift narrative log regarding Pipeline CT-106B inspection and pressure drop",
    ),
    SampleManifestItem(
        sample_id="HW_02",
        source_path="datasets/handwritten_notes/HN_0005_structured_Pump_P203.jpg",
        dataset="handwritten_notes",
        category="handwritten_notes",
        tasks=("understand", "caption"),
        expected_scene_keywords=("handwritten", "notes", "pump", "maintenance", "log"),
        description="Handwritten semi-structured maintenance log for centrifugal Pump P-203 bearing replacement",
    ),
)

# Complete set of categories in the manifest — used for coverage gap detection
MANIFEST_CATEGORIES: frozenset[str] = frozenset(s.category for s in REPRESENTATIVE_SAMPLES)


# ──────────────────────────────────────────────────────────────────────────────
# Low-Resource / Chunked Execution Constants
# ──────────────────────────────────────────────────────────────────────────────

# Conservative fallback latency used when no matching run-mode history exists.
# Based on observed real-model run: ~573,000 ms/task on CPU at
#   --max-image-size 256, --max-new-tokens 48 (Qwen2.5-VL-3B-Instruct, 4-thread CPU).
COLD_START_LATENCY_MS_PER_TASK: float = 573_000.0

# Estimated latency speedup factor for --quick mode (192px, 24 tokens).
# Conservative estimate: image area ∝ (192/256)² = 0.5625; token halving ≈ 0.5;
# constant overhead tempers both → ~0.45 overall.  Updated once real quick-mode
# data is available.
QUICK_MODE_LATENCY_REDUCTION_FACTOR: float = 0.45


# ── CATEGORY_DISPLAY_NAMES ────────────────────────────────────────────────────
# User-facing canonical display names for report output, coverage warnings, and hints.
CATEGORY_DISPLAY_NAMES: dict[str, str] = {
    "PID": "PID",
    "PFD": "PFD",
    "Equipment_Drawings": "Equipment_Drawings",
    "Instrumentation": "Instrumentation",
    "Pump_Diagrams": "Pump_Diagrams",
    "Electrical": "Electrical",
    "corrosion_defects": "Corrosion_Defects",
    "defect_detection": "Defect_Detection",
    "infrared": "Infrared",
    "handwritten_notes": "Handwritten_Notes",
}

# ── CATEGORY_ALIASES ──────────────────────────────────────────────────────────
# Maps each canonical manifest category name to all CLI aliases accepted by
# --category.  Keys are SINGLE SOURCE OF TRUTH and must exactly match
# MANIFEST_CATEGORIES.  Verified by the startup assertion immediately below.
CATEGORY_ALIASES: dict[str, tuple[str, ...]] = {
    "PID": (
        "PID", "P&ID", "pid", "p&id",
        "piping and instrumentation", "piping instrumentation",
    ),
    "PFD": (
        "PFD", "pfd", "process flow", "Process Flow Diagram",
    ),
    "Equipment_Drawings": (
        "Equipment_Drawings", "equipment", "Equipment Drawings",
        "Equipment Drawing", "equip",
    ),
    "Instrumentation": (
        "Instrumentation", "instrumentation", "Inst",
        "control loop", "instrument",
    ),
    "Pump_Diagrams": (
        "Pump_Diagrams", "pump", "Pump Diagrams", "pump diagram",
    ),
    "Electrical": (
        "Electrical", "electrical", "Electrical SLD",
        "sld", "single line",
    ),
    "corrosion_defects": (
        "corrosion_defects", "Corrosion_Defects", "corrosion", "NEU Surface",
        "neu", "surface defect", "Corrosion Defects",
    ),
    "defect_detection": (
        "defect_detection", "Defect_Detection", "defect", "PCB",
        "pcb", "inspection defect", "Defect Detection",
    ),
    "infrared": (
        "infrared", "Infrared", "Infrared PV", "infrared pv",
        "thermal", "solar", "Infrared Photovoltaic",
    ),
    "handwritten_notes": (
        "handwritten_notes", "Handwritten_Notes", "handwritten", "Handwritten Notes",
        "notes", "hw",
    ),
}

# ── CHUNK_PLAN_SESSIONS ───────────────────────────────────────────────────────
# Planned 5-session groupings from CHUNKED_RUN_PLAN.md
CHUNK_PLAN_SESSIONS: tuple[tuple[str, ...], ...] = (
    ("PID", "PFD"),
    ("Equipment_Drawings", "Instrumentation"),
    ("Pump_Diagrams", "Electrical"),
    ("corrosion_defects", "defect_detection"),
    ("infrared", "handwritten_notes"),
)


# ── PRIMARY_TASK_MAP ──────────────────────────────────────────────────────────
# Maps each canonical category to its single most diagnostic task.
# Used by --quick / --tasks-per-sample 1 to select highest-value task per sample.
# Constraints (verified at startup): (a) keys == MANIFEST_CATEGORIES,
# (b) value ∈ sample.tasks for EVERY sample of that category.
PRIMARY_TASK_MAP: dict[str, str] = {
    "PID":                "drawing",      # P&ID type classification (all PID samples have 'drawing')
    "PFD":                "drawing",      # Process flow type classification
    "Equipment_Drawings": "drawing",      # Mechanical drawing recognition
    "Instrumentation":    "drawing",      # Control loop drawing identification
    "Pump_Diagrams":      "drawing",      # Pump sectional classification
    "Electrical":         "drawing",      # SLD identification (no 'equipment' on ELEC_01)
    "corrosion_defects":  "observation",  # Surface anomaly description
    "defect_detection":   "observation",  # PCB defect observation
    "infrared":           "observation",  # Thermal gradient observation
    "handwritten_notes":  "caption",      # Maintenance note summary (no 'observation' on HW samples)
}


# ── Startup Consistency Assertion (Fix 1 / Bug 2) ─────────────────────────────
# Runs once at import time.  Fails loudly with descriptive messages if
# CATEGORY_ALIASES, PRIMARY_TASK_MAP, or CATEGORY_DISPLAY_NAMES fall out of sync
# with MANIFEST_CATEGORIES, or if any display name cannot round-trip through
# filter_samples_by_category().
_manifest_cats: frozenset[str] = MANIFEST_CATEGORIES
_alias_cats: frozenset[str] = frozenset(CATEGORY_ALIASES.keys())
_primary_cats: frozenset[str] = frozenset(PRIMARY_TASK_MAP.keys())
_display_cats: frozenset[str] = frozenset(CATEGORY_DISPLAY_NAMES.keys())

assert _alias_cats == _manifest_cats, (
    f"CATEGORY_ALIASES keys don't match MANIFEST_CATEGORIES.\n"
    f"  Missing from aliases: {_manifest_cats - _alias_cats}\n"
    f"  Extra in aliases:     {_alias_cats - _manifest_cats}"
)
assert _primary_cats == _manifest_cats, (
    f"PRIMARY_TASK_MAP keys don't match MANIFEST_CATEGORIES.\n"
    f"  Missing from primary map: {_manifest_cats - _primary_cats}\n"
    f"  Extra in primary map:     {_primary_cats - _manifest_cats}"
)
assert _display_cats == _manifest_cats, (
    f"CATEGORY_DISPLAY_NAMES keys don't match MANIFEST_CATEGORIES.\n"
    f"  Missing from display map: {_manifest_cats - _display_cats}\n"
    f"  Extra in display map:     {_display_cats - _manifest_cats}"
)
for _s in REPRESENTATIVE_SAMPLES:
    _pt = PRIMARY_TASK_MAP[_s.category]
    assert _pt in _s.tasks, (
        f"PRIMARY_TASK_MAP[{_s.category!r}] = {_pt!r} is NOT in sample "
        f"{_s.sample_id!r} tasks={_s.tasks}. Update PRIMARY_TASK_MAP."
    )
del _manifest_cats, _alias_cats, _primary_cats, _display_cats, _s, _pt


# ──────────────────────────────────────────────────────────────────────────────
# Category Filtering
# ──────────────────────────────────────────────────────────────────────────────

def filter_samples_by_category(
    samples: Sequence[SampleManifestItem],
    category_names: list[str],
) -> list[SampleManifestItem]:
    """Filter samples to those matching any of the supplied category names (alias-aware).

    Args:
        samples: The manifest (or subset) to filter.
        category_names: Category names or aliases accepted by ``--category``.
            Case-insensitive.  Comma-splitting is NOT done here — caller must
            split on ``","`` if needed.

    Returns:
        Ordered list of matching SampleManifestItems (manifest order preserved).

    Raises:
        ValueError: If any supplied name cannot be resolved to a manifest category.
    """
    # Build case-insensitive reverse alias lookup: alias_lower -> canonical
    alias_to_canonical: dict[str, str] = {}
    for canonical, aliases in CATEGORY_ALIASES.items():
        for alias in aliases:
            alias_to_canonical[alias.lower()] = canonical

    canonical_requested: set[str] = set()
    unrecognised: list[str] = []
    for name in category_names:
        resolved = alias_to_canonical.get(name.strip().lower())
        if resolved is None:
            unrecognised.append(name)
        else:
            canonical_requested.add(resolved)

    if unrecognised:
        valid = sorted(CATEGORY_ALIASES.keys())
        raise ValueError(
            f"Unknown category name(s): {unrecognised!r}.\n"
            f"Valid canonical names: {valid}.\n"
            f"All accepted aliases are defined in CATEGORY_ALIASES."
        )

    return [s for s in samples if s.category in canonical_requested]


# Verify all canonical display names round-trip through filter_samples_by_category (Bug 2 Fix)
for _disp in CATEGORY_DISPLAY_NAMES.values():
    _matched = filter_samples_by_category(REPRESENTATIVE_SAMPLES, [_disp])
    assert len(_matched) > 0, f"CATEGORY_DISPLAY_NAMES value {_disp!r} cannot be resolved by filter_samples_by_category"
del _disp, _matched


def get_next_suggested_category_chunk(uncovered_categories: Sequence[str]) -> list[str]:
    """Return the next 2-3 categories to run based on CHUNKED_RUN_PLAN.md sessions."""
    # Build case-insensitive reverse alias lookup: alias_lower -> canonical
    alias_to_canonical = {}
    for canonical, aliases in CATEGORY_ALIASES.items():
        for alias in aliases:
            alias_to_canonical[alias.lower()] = canonical

    canonical_uncovered: set[str] = set()
    for cat in uncovered_categories:
        c = alias_to_canonical.get(cat.strip().lower(), cat.strip())
        canonical_uncovered.add(c)

    for session in CHUNK_PLAN_SESSIONS:
        uncovered_in_session = [c for c in session if c in canonical_uncovered]
        if uncovered_in_session:
            return [CATEGORY_DISPLAY_NAMES.get(c, c) for c in uncovered_in_session]

    return [CATEGORY_DISPLAY_NAMES.get(c, c) for c in sorted(canonical_uncovered)[:2]]


def format_next_run_hint(uncovered_categories: Sequence[str], run_mode: str = "quick") -> str:
    """Format a fully runnable, copy-pasteable next-step CLI command matching CHUNKED_RUN_PLAN.md."""
    if not uncovered_categories:
        return "python evaluate_vision.py --resume"
    next_chunk = get_next_suggested_category_chunk(uncovered_categories)
    cat_arg = ",".join(next_chunk)
    mode_flag = " --quick" if run_mode == "quick" else ""
    return f'python evaluate_vision.py --category "{cat_arg}"{mode_flag} --resume --device auto --max-run-minutes 15 --threads 4'



# ──────────────────────────────────────────────────────────────────────────────
# Drawing Category Matching Logic
# ──────────────────────────────────────────────────────────────────────────────

DRAWING_SYNONYMS: dict[str, set[str]] = {
    "PID": {
        "pid", "p&id", "p & id", "piping and instrumentation", "piping & instrumentation",
        "piping and instrumentation diagram", "process and instrumentation diagram",
        "process & instrumentation diagram",
    },
    "PFD": {
        "pfd", "process flow", "process flow diagram", "process flow sheet",
        "flowsheet", "flow sheet",
    },
    "Equipment_Drawings": {
        "equipment", "equipment drawing", "equipment_drawing", "mechanical drawing",
        "assembly drawing", "fabrication drawing", "sectional drawing", "vessel drawing",
        "heat exchanger drawing", "cutaway", "sectional",
    },
    "Instrumentation": {
        "instrumentation", "control loop", "instrument loop", "control schematic",
        "loop diagram", "instrument drawing",
    },
    "Pump_Diagrams": {
        "pump", "pump diagram", "pump sectional", "centrifugal pump", "pump assembly",
    },
    "Electrical": {
        "electrical", "single line", "single-line", "sld", "one-line diagram",
        "power distribution", "electrical schematic", "wiring diagram",
    },
}


def check_drawing_type_match(predicted_drawing_type: str, expected_category: str) -> bool | None:
    """Check if predicted drawing type is consistent with dataset category."""
    if not predicted_drawing_type or not expected_category:
        return None
    cat_synonyms = DRAWING_SYNONYMS.get(expected_category)
    if not cat_synonyms:
        return None
    pred_norm = predicted_drawing_type.strip().lower()
    for syn in cat_synonyms:
        if syn in pred_norm:
            return True
    return False


def check_scene_keyword_match(
    predicted_text: str,
    expected_keywords: tuple[str, ...],
) -> dict[str, Any]:
    """Check if predicted scene/caption text contains expected keywords.

    Returns a dict with:
      matched_keywords: list of expected keywords found in predicted_text
      total_expected: number of expected keywords
      keyword_hit_rate: fraction of expected keywords that appeared (None if no keywords)
      scored: True if any expected keywords exist, False if no ground truth
    """
    if not expected_keywords:
        return {
            "matched_keywords": [],
            "total_expected": 0,
            "keyword_hit_rate": None,
            "scored": False,
            "note": "unscored — no reference keywords available for this sample",
        }
    pred_lower = (predicted_text or "").lower()
    matched = [kw for kw in expected_keywords if kw.lower() in pred_lower]
    return {
        "matched_keywords": matched,
        "total_expected": len(expected_keywords),
        "keyword_hit_rate": round(len(matched) / len(expected_keywords), 4),
        "scored": True,
    }


def check_equipment_type_match(
    detected_types: list[str],
    expected_equipment: tuple[str, ...],
) -> dict[str, Any]:
    """Check overlap between detected equipment types and expected equipment keywords."""
    if not expected_equipment:
        return {
            "matched_equipment": [],
            "total_expected": 0,
            "equipment_hit_rate": None,
            "scored": False,
            "note": "unscored — no reference equipment available for this sample",
        }
    pred_lower = " ".join(detected_types).lower()
    matched = [eq for eq in expected_equipment if eq.lower() in pred_lower]
    return {
        "matched_equipment": matched,
        "total_expected": len(expected_equipment),
        "equipment_hit_rate": round(len(matched) / len(expected_equipment), 4),
        "scored": True,
    }


def detect_available_device() -> tuple[str, str]:
    """Detect available compute device and return (device_str, reason_message).

    Returns:
        Tuple of (device, reason) where device is 'cuda' or 'cpu'.
    """
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "CUDA"
            return "cuda", f"CUDA available: {device_name} ({torch.cuda.device_count()} device(s))"
        else:
            return "cpu", "CUDA not available (no CUDA-capable GPU detected or driver not installed)"
    except ImportError:
        return "cpu", "torch not importable; defaulting to cpu"


def resolve_device_for_harness(requested: str) -> tuple[str, str]:
    """Resolve the actual device to use given the requested device string.

    Returns:
        Tuple of (actual_device, resolution_note).
    """
    if requested == "cpu":
        return "cpu", "cpu explicitly requested"
    if requested == "cuda":
        available, reason = detect_available_device()
        if available == "cuda":
            return "cuda", f"cuda explicitly requested and available: {reason}"
        else:
            return "cpu", f"cuda requested but unavailable ({reason}), falling back to cpu"
    # "auto"
    available, reason = detect_available_device()
    return available, f"auto-selected {available}: {reason}"


def estimate_full_run_duration(
    samples: Sequence[SampleManifestItem],
    avg_latency_ms_per_task: float,
    tasks_per_sample: int | None = None,
) -> dict[str, Any]:
    """Estimate total wall-clock time for a full manifest run based on observed avg latency."""
    if tasks_per_sample == 1:
        total_tasks = len(samples)
    elif tasks_per_sample is not None and tasks_per_sample > 0:
        total_tasks = sum(min(tasks_per_sample, len(s.tasks)) for s in samples)
    else:
        total_tasks = sum(len(s.tasks) for s in samples)

    est_ms = total_tasks * avg_latency_ms_per_task
    est_min = est_ms / 60_000
    est_hours = est_min / 60
    return {
        "total_manifest_samples": len(samples),
        "total_manifest_tasks": total_tasks,
        "avg_latency_ms_per_task_basis": round(avg_latency_ms_per_task, 1),
        "estimated_total_ms": round(est_ms, 0),
        "estimated_total_minutes": round(est_min, 1),
        "estimated_total_hours": round(est_hours, 2),
        "note": (
            f"Projection based on observed avg latency of {avg_latency_ms_per_task:.0f} ms/task "
            f"x {total_tasks} tasks across {len(samples)} samples. "
            "Actual time varies by image resolution and model state."
        ),
    }


def estimate_run_duration(
    samples: Sequence[SampleManifestItem],
    tasks_per_sample: int | None,
    prior_json_path: Path | None = None,
    is_quick: bool = False,
) -> dict[str, Any]:
    """Estimate wall-clock time for a specific filtered evaluation run.

    Extends ``estimate_full_run_duration`` but scoped to the active
    ``--category`` / ``--limit`` / ``--quick`` filters for this invocation.

    Latency source priority:

    1. Prior JSON that matches the current ``run_mode`` (quick→quick, full→full).
    2. Prior JSON for a different mode, scaled by
       ``QUICK_MODE_LATENCY_REDUCTION_FACTOR`` if switching full→quick.
    3. ``COLD_START_LATENCY_MS_PER_TASK`` fallback (Fix 2: always well-defined).

    ``is_fallback_estimate`` is ``True`` for cases 2 and 3 so the caller can
    display the appropriate warning label.
    """
    # Count tasks this run will execute
    if tasks_per_sample is None or tasks_per_sample <= 0:
        total_tasks = sum(len(s.tasks) for s in samples)
    else:
        total_tasks = sum(min(tasks_per_sample, len(s.tasks)) for s in samples)

    avg_latency_ms: float | None = None
    latency_source: str = ""
    is_fallback: bool = True

    if prior_json_path is not None and prior_json_path.exists():
        try:
            data = json.loads(prior_json_path.read_text(encoding="utf-8"))
            prior_mode = (data.get("evaluation_metadata") or {}).get("run_mode", "full")
            prior_avg = (data.get("latency_summary") or {}).get("average_latency_ms", 0.0) or 0.0
            if prior_avg > 0:
                if (is_quick and prior_mode == "quick") or (not is_quick and prior_mode == "full"):
                    avg_latency_ms = prior_avg
                    latency_source = f"prior_{prior_mode}_run ({prior_avg:.0f} ms/task)"
                    is_fallback = False
                elif is_quick and prior_mode == "full":
                    # Scale full-mode observed latency down by the reduction factor
                    avg_latency_ms = prior_avg * QUICK_MODE_LATENCY_REDUCTION_FACTOR
                    latency_source = (
                        f"adjusted_from_full_run ({prior_avg:.0f} ms \u00d7 "
                        f"{QUICK_MODE_LATENCY_REDUCTION_FACTOR} quick-mode factor "
                        f"= {avg_latency_ms:.0f} ms/task)"
                    )
                    is_fallback = True  # still an estimate
                else:
                    avg_latency_ms = prior_avg
                    latency_source = f"prior_{prior_mode}_run_different_mode ({prior_avg:.0f} ms/task)"
                    is_fallback = True
        except Exception:
            pass

    if avg_latency_ms is None or avg_latency_ms <= 0:
        if is_quick:
            avg_latency_ms = COLD_START_LATENCY_MS_PER_TASK * QUICK_MODE_LATENCY_REDUCTION_FACTOR
            latency_source = (
                f"cold_start_fallback_quick ({COLD_START_LATENCY_MS_PER_TASK:.0f} ms "
                f"\u00d7 {QUICK_MODE_LATENCY_REDUCTION_FACTOR} = {avg_latency_ms:.0f} ms/task)"
            )
        else:
            avg_latency_ms = COLD_START_LATENCY_MS_PER_TASK
            latency_source = (
                f"cold_start_fallback ({avg_latency_ms:.0f} ms/task — "
                "from prior real-model observation on this hardware)"
            )
        is_fallback = True

    est_ms = total_tasks * avg_latency_ms
    est_min = est_ms / 60_000

    if is_fallback:
        note = (
            f"No prior data for these settings — showing conservative estimate: "
            f"~{est_min:.0f} min. Actual may differ."
        )
    else:
        note = (
            f"Estimated from {latency_source}: "
            f"~{est_min:.0f} min for {total_tasks} tasks across {len(samples)} samples."
        )

    return {
        "sample_count": len(samples),
        "tasks_per_sample": tasks_per_sample if tasks_per_sample else "all",
        "total_tasks_estimated": total_tasks,
        "avg_latency_ms_basis": round(avg_latency_ms, 1),
        "latency_source": latency_source,
        "is_fallback_estimate": is_fallback,
        "estimated_total_ms": round(est_ms, 0),
        "estimated_total_minutes": round(est_min, 1),
        "note": note,
    }


def _get_git_commit_hash() -> str:
    """Return current HEAD short git commit hash, or 'unavailable' if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=str(PROJECT_ROOT),
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unavailable"


# ──────────────────────────────────────────────────────────────────────────────
# Hallucination Check Engine
# ──────────────────────────────────────────────────────────────────────────────


# Unsupported specifications regexes
RE_PRESSURE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:bar|psi|kpa|mpa|atm)\b", re.IGNORECASE)
RE_TEMPERATURE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:°C|°F|deg\s*C|deg\s*F|Kelvin|K\b)", re.IGNORECASE)
RE_DIMENSION = re.compile(r"\b\d+(?:\.\d+)?\s*(?:mm|cm|inches|inch|meters|m\b)", re.IGNORECASE)
RE_SAFETY_CLAIM = re.compile(
    r"\b(?:equipment is safe|safe to operate|fully compliant|risk-free|no safety hazard|"
    r"operable condition|satisfactory operating condition|safe condition|"
    r"operating within safe limits|danger-free|guaranteed safe)\b",
    re.IGNORECASE,
)
RE_PROXIMITY_CONNECTION = re.compile(
    r"\b(?:connected to .* based on proximity|connected because .* adjacent|"
    r"assumed connected|inferred connection|connects .* based on proximity|"
    r"connected due to proximity)\b",
    re.IGNORECASE,
)
RE_GENERIC_TAG_PATTERN = re.compile(r"\b[A-Z]{1,4}[-_ ]?\d{2,5}[A-Z]?\b")


def check_hallucinations(
    text: str,
    equipment_items: Sequence[dict[str, Any]],
    expected_category: str,
    known_tags: Sequence[str] = (),
) -> list[str]:
    """Inspect model outputs and flag ungrounded or speculative assertions.

    Flags:
      - invented_equipment_tag
      - unsupported_numerical_specification
      - unsupported_pressure_value
      - unsupported_temperature_value
      - unsupported_dimension
      - unsupported_safety_conclusion
      - unsupported_connected_to_relationship
    """
    flags: list[str] = []

    # 1. Unsupported pressure values
    if RE_PRESSURE.search(text):
        flags.append("unsupported_pressure_value")

    # 2. Unsupported temperature values
    if RE_TEMPERATURE.search(text):
        flags.append("unsupported_temperature_value")

    # 3. Unsupported physical dimensions in non-dimensional drawings
    if expected_category not in ("Equipment_Drawings", "Pump_Diagrams") and RE_DIMENSION.search(text):
        flags.append("unsupported_dimension")

    # 4. Unsupported safety conclusions
    if RE_SAFETY_CLAIM.search(text):
        flags.append("unsupported_safety_conclusion")

    # 5. Unsupported proximity-based connectivity claims
    if RE_PROXIMITY_CONNECTION.search(text):
        flags.append("unsupported_connected_to_relationship")

    # 6. Invented equipment tags check
    # Check if equipment items claim specific tags when no tag is visually supported
    for item in equipment_items:
        tag = item.get("name_or_tag")
        if tag:
            tag_clean = str(tag).strip()
            # If tag looks like fabricated tag and not in known tags or filename
            if known_tags and not any(k.lower() in tag_clean.lower() for k in known_tags):
                # Only flag if evidence is missing or clearly fabricated
                evidence = str(item.get("evidence", "")).strip()
                if not evidence or "assumed" in evidence.lower() or "inferred" in evidence.lower():
                    flags.append(f"invented_equipment_tag:{tag_clean}")

    return sorted(list(set(flags)))


# ──────────────────────────────────────────────────────────────────────────────
# P&ID Inspection Analysis
# ──────────────────────────────────────────────────────────────────────────────

def inspect_pid_outputs(task_res: dict[str, Any]) -> dict[str, Any]:
    """Specialized evaluation for P&ID drawings examining labels, equipment, instruments."""
    equipment = task_res.get("equipment", [])
    visible_labels = task_res.get("visible_labels", []) or task_res.get("visible_text", [])
    raw = task_res.get("raw_output", "")

    equipment_names = [e.get("equipment_type", "") for e in equipment]
    instrument_keywords = ["transmitter", "controller", "valve", "sensor", "gauge", "indicator", "meter"]
    instruments_found = [
        name for name in equipment_names
        if any(kw in name.lower() for kw in instrument_keywords)
    ]

    has_unsupported_connectivity = bool(
        RE_PROXIMITY_CONNECTION.search(raw) or
        ("connected" in raw.lower() and "proximity" in raw.lower())
    )

    return {
        "labels_observed_count": len(visible_labels),
        "equipment_identified_count": len(equipment_names),
        "instruments_identified": instruments_found,
        "visual_structure_detected": any(w in raw.lower() for w in ("pipe", "line", "stream", "flow", "loop")),
        "unsupported_connectivity_claims": has_unsupported_connectivity,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation Record & Runner
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class EvaluationRecord:
    """Record of a single task inference on a sample."""

    sample_id: str
    source_path: str
    dataset: str
    category: str
    task: str
    success: bool
    scene_type: str = "unknown"
    caption: str = ""
    equipment_count: int = 0
    equipment_types: list[str] = field(default_factory=list)
    equipment_tags: list[str] = field(default_factory=list)
    observations_count: int = 0
    confidence: float = 0.0
    # confidence_method documents the derivation of confidence (see module docstring)
    confidence_method: str = CONFIDENCE_METHOD
    device_requested: str = "cpu"
    device_used: str = "cpu"
    # processing_time_ms: wall-clock time of pipeline.run_task() ONLY — model load excluded
    processing_time_ms: float = 0.0
    issues: list[str] = field(default_factory=list)
    model_name: str = "qwen2.5-vl-3b-instruct"
    model_path: str = ""
    predicted_drawing_type: str = ""
    expected_category: str = ""
    drawing_type_match: bool | None = None
    hallucination_flags: list[str] = field(default_factory=list)
    raw_output: str = ""
    pid_inspection: dict[str, Any] = field(default_factory=dict)
    # Ground-truth keyword scoring
    scene_keyword_score: dict[str, Any] = field(default_factory=dict)
    equipment_keyword_score: dict[str, Any] = field(default_factory=dict)
    # Repeat run tracking (0=first/only run, 1,2...=additional repeats)
    repeat_index: int = 0
    # Settings signature fields (Bug 1 Fix)
    run_mode: str = "full"
    max_image_size: int | None = None
    max_new_tokens: int | None = None



class MockVisionEvaluatorBackend(MockVisionBackend):
    """Mock backend supporting run_task for fast unit testing."""

    def run_task(self, image: np.ndarray, task: str) -> dict[str, Any]:
        if task == "understand":
            return {
                "task": task,
                "scene_type": "piping and instrumentation diagram (P&ID)",
                "confidence": 0.85,
                "equipment": [
                    {"equipment_type": "Centrifugal Pump", "name_or_tag": None, "confidence": 0.9, "evidence": "standard pump symbol"},
                    {"equipment_type": "Control Valve", "name_or_tag": None, "confidence": 0.88, "evidence": "valve symbol with actuator"},
                ],
                "observations": [
                    {"description": "Main process piping connects vessel to pump inlet", "confidence": 0.8},
                ],
                "visible_text": [{"text": "P-101", "confidence": 0.8}],
                "raw_output": "P&ID diagram showing process piping with pump and control valve.",
            }
        if task == "caption":
            return {
                "task": task,
                "caption": "Engineering P&ID diagram showing an industrial pumping circuit and piping lines.",
                "raw_output": "Engineering P&ID diagram showing an industrial pumping circuit and piping lines.",
            }
        if task == "equipment":
            return {
                "task": task,
                "equipment": [
                    {"equipment_type": "Centrifugal Pump", "name_or_tag": None, "confidence": 0.9, "evidence": "visible pump casing"},
                    {"equipment_type": "Gate Valve", "name_or_tag": None, "confidence": 0.85, "evidence": "two touching triangles"},
                ],
                "raw_output": "Detected centrifugal pump and gate valve.",
            }
        if task == "drawing":
            return {
                "task": task,
                "drawing_type": "P&ID",
                "visible_labels": [{"text": "FIC-101", "confidence": 0.85}],
                "note": "Spatial relationships NOT inferred from proximity.",
                "raw_output": "Drawing type: Piping and Instrumentation Diagram (P&ID).",
            }
        if task == "observation":
            return {
                "task": task,
                "observations": [
                    {"description": "Surface shows discoloration and pitting marks consistent with corrosion", "confidence": 0.82, "uncertainty": "requires metallurgical confirmation"},
                ],
                "raw_output": "Observed surface pitting marks with localized discoloration.",
            }
        return {"task": task, "raw_output": f"Mock output for task {task}"}


def load_image_input(sample_path: Path) -> np.ndarray:
    """Load image from disk or render page 1 if PDF. Strictly read-only."""
    if not sample_path.exists():
        raise FileNotFoundError(f"Sample path does not exist: {sample_path}")

    suffix = sample_path.suffix.lower()
    if suffix == ".pdf":
        pages = render_pdf_pages(sample_path, dpi=150)
        if not pages or pages[0].image is None:
            err = pages[0].error if pages else "No pages found"
            raise RuntimeError(f"Failed to render PDF page 1: {err}")
        # Convert OpenCV BGR to RGB
        import cv2
        return cv2.cvtColor(pages[0].image, cv2.COLOR_BGR2RGB)
    else:
        with Image.open(sample_path) as pil:
            return np.array(pil.convert("RGB"))


class VisionEvaluationHarness:
    """Evaluation harness coordinating samples, pipeline execution, hallucination detection, and reporting."""

    def __init__(
        self,
        output_dir: Path | str = DEFAULT_OUTPUT_DIR,
        model_path: str | None = None,
        device: str = "cpu",
        is_mock: bool = False,
        max_image_size: int | None = 512,
        max_new_tokens: int = 128,
        run_mode: str = "full",
        quick_mode_settings: dict[str, Any] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = model_path or DEFAULT_VISION_MODEL_PATH
        self.is_mock = is_mock
        self.max_image_size = max_image_size
        self.max_new_tokens = max_new_tokens
        self.run_mode = run_mode
        self.quick_mode_settings = quick_mode_settings or {}

        # Resolve and log actual device to be used before model load
        if is_mock:
            self.device_requested = "mock"
            self.device_used = "mock"
            self.device_resolution_note = "mock backend — no device selection"
        else:
            actual_device, resolution_note = resolve_device_for_harness(device)
            self.device_requested = device
            self.device_used = actual_device
            self.device_resolution_note = resolution_note
            LOGGER.info("Device resolution: %s (requested: %s)", actual_device, device)
            LOGGER.info("Device note: %s", resolution_note)

        self.model_load_time_ms: float = 0.0
        self.pipeline: VisionPipeline | None = None
        self._init_pipeline()

    def _matches_settings_signature(self, record: EvaluationRecord) -> bool:
        """Check if record matches the current run's settings signature.

        Signature comparison covers (run_mode, max_image_size, max_new_tokens).
        """
        if record.run_mode != self.run_mode:
            return False
        if self.max_image_size is not None and record.max_image_size is not None:
            if record.max_image_size != self.max_image_size:
                return False
        if self.max_new_tokens is not None and record.max_new_tokens is not None:
            if record.max_new_tokens != self.max_new_tokens:
                return False
        return True

    def _init_pipeline(self) -> None:
        """Initialize pipeline with chosen backend."""
        if self.is_mock:
            backend = MockVisionEvaluatorBackend()
            self.pipeline = VisionPipeline(backend=backend)
            self.model_load_time_ms = 1.0
            LOGGER.info("Initialized VisionEvaluationHarness with MockVisionEvaluatorBackend.")
        else:
            config = VisionModelConfig(
                model_name="qwen2.5-vl-3b-instruct",
                model_path=self.model_path,
                device=self.device_used,
                max_image_size=self.max_image_size,
                max_new_tokens=self.max_new_tokens,
            )
            backend = LocalQwenVisionBackend(config=config)
            self.pipeline = VisionPipeline(backend=backend)

            t0 = time.perf_counter()
            try:
                backend.initialize()
                # model_load_time_ms covers ONLY model init — never combined with inference latency
                self.model_load_time_ms = (time.perf_counter() - t0) * 1000
                LOGGER.info(
                    "Model initialized in %.1f ms on device %s",
                    self.model_load_time_ms,
                    backend.backend_info.device_used,
                )
            except Exception as exc:
                LOGGER.error("Failed to initialize Qwen backend: %s", exc)

    def run_inference_on_sample_task(
        self,
        sample: SampleManifestItem,
        task: str,
        image_array: np.ndarray,
        repeat_index: int = 0,
    ) -> EvaluationRecord:
        """Execute a single task on a preloaded sample image array.

        Latency measurement:
          - processing_time_ms measures ONLY pipeline.run_task() wall-clock time.
          - Model load time (one-time init) is tracked separately in self.model_load_time_ms.
          - This ensures reported latencies are purely per-inference and comparable across runs.
        """
        assert self.pipeline is not None

        t0 = time.perf_counter()
        try:
            task_res = self.pipeline.run_task(image_array, task)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            success = True
            issues: list[str] = []
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            success = False
            task_res = {}
            issues = [f"{type(exc).__name__}: {exc}"]
            LOGGER.warning("Inference failed for %s task=%s: %s", sample.sample_id, task, exc)

        bi = self.pipeline.backend.backend_info
        raw_output = str(task_res.get("raw_output", ""))

        # Extract fields based on task
        scene_type = str(task_res.get("scene_type", ""))
        caption = str(task_res.get("caption", ""))
        confidence = float(task_res.get("confidence", 0.0))

        equip_list = task_res.get("equipment", []) or []
        equip_types = [str(e.get("equipment_type", "")) for e in equip_list if e.get("equipment_type")]
        equip_tags = [str(e.get("name_or_tag", "")) for e in equip_list if e.get("name_or_tag")]

        obs_list = task_res.get("observations", []) or []
        obs_count = len(obs_list)

        predicted_drawing_type = str(task_res.get("drawing_type", ""))
        drawing_match = None
        if task == "drawing" and sample.expected_drawing_type:
            drawing_match = check_drawing_type_match(predicted_drawing_type, sample.category)

        # Hallucination detection
        combined_text = f"{raw_output} {caption}"
        hallucination_flags = check_hallucinations(
            text=combined_text,
            equipment_items=equip_list,
            expected_category=sample.category,
            known_tags=sample.expected_equipment,
        )

        # Ground-truth keyword scoring
        scene_text_to_score = f"{scene_type} {raw_output} {caption}"
        scene_kw_score = check_scene_keyword_match(scene_text_to_score, sample.expected_scene_keywords)
        equip_kw_score = check_equipment_type_match(equip_types, sample.expected_equipment)

        # P&ID specialized inspection
        pid_inspection: dict[str, Any] = {}
        if sample.category == "PID" and task in ("drawing", "understand", "equipment"):
            pid_inspection = inspect_pid_outputs(task_res)

        return EvaluationRecord(
            sample_id=sample.sample_id,
            source_path=sample.source_path,
            dataset=sample.dataset,
            category=sample.category,
            task=task,
            success=success,
            scene_type=scene_type,
            caption=caption,
            equipment_count=len(equip_types),
            equipment_types=equip_types,
            equipment_tags=equip_tags,
            observations_count=obs_count,
            confidence=round(confidence, 4),
            confidence_method=CONFIDENCE_METHOD,
            device_requested=bi.device_requested,
            device_used=bi.device_used,
            processing_time_ms=round(elapsed_ms, 2),
            issues=issues,
            model_name=bi.name,
            model_path=bi.model_path or "",
            predicted_drawing_type=predicted_drawing_type,
            expected_category=sample.category,
            drawing_type_match=drawing_match,
            hallucination_flags=hallucination_flags,
            raw_output=raw_output,
            pid_inspection=pid_inspection,
            scene_keyword_score=scene_kw_score,
            equipment_keyword_score=equip_kw_score,
            repeat_index=repeat_index,
            run_mode=self.run_mode,
            max_image_size=self.max_image_size,
            max_new_tokens=self.max_new_tokens,
        )

    def _flush_incremental(
        self,
        records: list[EvaluationRecord],
        target_samples: list[SampleManifestItem],
        stopped_early: bool = False,
    ) -> None:
        """Atomically flush minimal resume-compatible JSON after each inference."""
        json_file = self.output_dir / "vision_evaluation.json"
        device_used = self.pipeline.backend.backend_info.device_used if self.pipeline else self.device_used
        summary = {
            "evaluation_metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model_name": "qwen2.5-vl-3b-instruct" if not self.is_mock else "mock_vision_backend",
                "model_path": str(self.model_path),
                "is_mock": self.is_mock,
                "device_requested": self.device_requested,
                "device_used": device_used,
                "device_resolution_note": self.device_resolution_note,
                "model_load_time_ms": round(self.model_load_time_ms, 2),
                "total_inferences": len(records),
                "successful_inferences": sum(1 for r in records if r.success),
                "failed_inferences": sum(1 for r in records if not r.success),
                "run_mode": self.run_mode,
                "quick_mode_settings": self.quick_mode_settings,
                "stopped_early": stopped_early,
                "is_incremental_flush": True,
            },
            "per_sample_inferences": [asdict(r) for r in records],
        }
        tmp_file = json_file.with_suffix(".tmp")
        try:
            tmp_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            tmp_file.replace(json_file)
        except Exception as exc:
            LOGGER.warning("Failed incremental flush: %s", exc)

    def evaluate(
        self,
        samples: Sequence[SampleManifestItem] = REPRESENTATIVE_SAMPLES,
        limit: int | None = None,
        resume: bool = False,
        repeat: int = 1,
        tasks_per_sample: int | None = None,
        max_run_minutes: float | None = None,
    ) -> dict[str, Any]:
        """Run the evaluation across manifest samples subject to limit, resume, and repeat flags.

        Args:
            samples: sequence of SampleManifestItem to evaluate.
            limit: max number of samples (not tasks) to process; None means all.
            resume: if True, reload existing JSON and skip completed (sample_id, task, repeat_index) triples.
            repeat: number of times to run each (sample, task) pair (default 1).
                    When >1, variance statistics are reported per (sample_id, task).
            tasks_per_sample: limit number of tasks per sample (1 selects PRIMARY_TASK_MAP task).
            max_run_minutes: maximum wall-clock minutes; stops early gracefully and flushes if exceeded.
        """
        completed_keys: set[tuple[str, str, int]] = set()  # (sample_id, task, repeat_index)
        existing_records: list[EvaluationRecord] = []

        # Load existing records if resume is enabled
        json_path = self.output_dir / "vision_evaluation.json"
        if resume and json_path.exists():
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
                for r in data.get("per_sample_inferences", []):
                    # Backfill settings signature for legacy records if missing
                    if "run_mode" not in r or r["run_mode"] is None:
                        if r.get("sample_id") in ("PID_02", "PFD_01", "PFD_02") or (0 < r.get("processing_time_ms", 0) < 350_000):
                            r["run_mode"] = "quick"
                            r["max_image_size"] = 192
                            r["max_new_tokens"] = 24
                        else:
                            r["run_mode"] = "full"
                            r["max_image_size"] = 256
                            r["max_new_tokens"] = 48
                    rec = EvaluationRecord(**{
                        k: v for k, v in r.items()
                        if k in EvaluationRecord.__dataclass_fields__
                    })
                    existing_records.append(rec)
                    completed_keys.add((rec.sample_id, rec.task, rec.repeat_index))
                LOGGER.info("Resuming evaluation. Found %d completed task inferences.", len(existing_records))
            except Exception as exc:
                LOGGER.warning("Could not parse existing evaluation for resume: %s", exc)

        records: list[EvaluationRecord] = list(existing_records)

        # Determine which samples to process
        target_samples = list(samples)
        if limit is not None and limit > 0:
            target_samples = target_samples[:limit]

        def _get_tasks_for_sample(s: SampleManifestItem) -> list[str]:
            if tasks_per_sample == 1:
                return [PRIMARY_TASK_MAP.get(s.category, s.tasks[0])]
            elif tasks_per_sample is not None and tasks_per_sample > 0:
                return list(s.tasks[:tasks_per_sample])
            return list(s.tasks)

        total_task_calls = sum(len(_get_tasks_for_sample(s)) for s in target_samples) * repeat
        LOGGER.info(
            "Evaluating %d samples x %d repeat(s) = %d total task calls (limit=%s, resume=%s, tasks_per_sample=%s)...",
            len(target_samples), repeat, total_task_calls, limit, resume, tasks_per_sample,
        )

        start_time = time.time()
        max_run_seconds = (max_run_minutes * 60.0) if max_run_minutes is not None else None
        stopped_early = False

        try:
            for idx, sample in enumerate(target_samples, 1):
                if max_run_seconds is not None and (time.time() - start_time) >= max_run_seconds:
                    stopped_early = True
                    LOGGER.warning("Time budget reached (%.2f min). Stopping gracefully.", max_run_minutes)
                    break

                sample_tasks = _get_tasks_for_sample(sample)
                sample_full_path = PROJECT_ROOT / sample.source_path
                LOGGER.info("[%d/%d] Sample %s (%s): %s", idx, len(target_samples), sample.sample_id, sample.category, sample.source_path)

                try:
                    image_arr = load_image_input(sample_full_path)
                except Exception as exc:
                    LOGGER.error("Failed to load sample image %s: %s", sample.source_path, exc)
                    for t in sample_tasks:
                        for ri in range(repeat):
                            if (sample.sample_id, t, ri) in completed_keys:
                                continue
                            rec = EvaluationRecord(
                                sample_id=sample.sample_id,
                                source_path=sample.source_path,
                                dataset=sample.dataset,
                                category=sample.category,
                                task=t,
                                success=False,
                                issues=[f"ImageLoadError: {exc}"],
                                model_name=self.pipeline.backend.backend_info.name if self.pipeline else "unknown",
                                model_path=self.pipeline.backend.backend_info.model_path or "" if self.pipeline else "",
                                repeat_index=ri,
                                run_mode=self.run_mode,
                                max_image_size=self.max_image_size,
                                max_new_tokens=self.max_new_tokens,
                            )
                            records.append(rec)
                            self._flush_incremental(records, target_samples)
                    continue

                for t in sample_tasks:
                    if max_run_seconds is not None and (time.time() - start_time) >= max_run_seconds:
                        stopped_early = True
                        LOGGER.warning("Time budget reached (%.2f min). Stopping gracefully.", max_run_minutes)
                        break

                    for ri in range(repeat):
                        if (sample.sample_id, t, ri) in completed_keys:
                            LOGGER.info("  Skipping task=%s repeat=%d (already in resume file)", t, ri)
                            continue

                        LOGGER.info("  Running task: %s (repeat %d/%d)", t, ri + 1, repeat)
                        rec = self.run_inference_on_sample_task(sample, t, image_arr, repeat_index=ri)
                        records.append(rec)
                        completed_keys.add((sample.sample_id, t, ri))
                        self._flush_incremental(records, target_samples)

                        if max_run_seconds is not None and (time.time() - start_time) >= max_run_seconds:
                            stopped_early = True
                            LOGGER.warning("Time budget reached (%.2f min). Stopping gracefully.", max_run_minutes)
                            break
                    if stopped_early:
                        break
        except KeyboardInterrupt:
            LOGGER.warning("KeyboardInterrupt caught — flushing %d records before exiting...", len(records))
            self._flush_incremental(records, target_samples, stopped_early=True)
            raise

        total_elapsed = time.time() - start_time

        # Save artifacts
        summary = self._generate_and_save_artifacts(records, total_elapsed, target_samples, stopped_early=stopped_early)
        return summary

    def _generate_and_save_artifacts(
        self,
        records: list[EvaluationRecord],
        total_time_s: float,
        target_samples: list[SampleManifestItem],
        stopped_early: bool = False,
    ) -> dict[str, Any]:
        """Aggregate metrics and export vision_evaluation.json, vision_evaluation.csv, README.md."""
        total_inferences = len(records)
        successful_inferences = sum(1 for r in records if r.success)
        failed_inferences = total_inferences - successful_inferences

        # Bug 1 Fix: Scope latencies to records matching the current run's settings signature
        matching_records = [
            r for r in records
            if r.success and r.processing_time_ms > 0 and self._matches_settings_signature(r)
        ]
        matching_count = len(matching_records)
        confidence_caveat = None
        if matching_count < 2:
            confidence_caveat = f"(low confidence — based on {matching_count} sample(s) at these settings)"

        latencies = [r.processing_time_ms for r in matching_records]
        avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else 0.0
        min_latency = round(min(latencies), 2) if latencies else 0.0
        max_latency = round(max(latencies), 2) if latencies else 0.0

        # Latency by task (also scoped to matching signature records)
        per_task_latency: dict[str, dict[str, float]] = {}
        for t in ("understand", "caption", "equipment", "drawing", "observation"):
            task_lats = [r.processing_time_ms for r in matching_records if r.task == t]
            if task_lats:
                per_task_latency[t] = {
                    "count": len(task_lats),
                    "average_ms": round(sum(task_lats) / len(task_lats), 2),
                    "min_ms": round(min(task_lats), 2),
                    "max_ms": round(max(task_lats), 2),
                }

        # Scene types distribution
        scene_types: dict[str, int] = {}
        for r in records:
            if r.scene_type and r.scene_type != "unknown":
                scene_types[r.scene_type] = scene_types.get(r.scene_type, 0) + 1

        # Drawing matches
        drawing_records = [r for r in records if r.task == "drawing" and r.drawing_type_match is not None]
        drawing_matches = sum(1 for r in drawing_records if r.drawing_type_match is True)
        drawing_match_rate = round(drawing_matches / len(drawing_records), 4) if drawing_records else 0.0

        # Equipment recognized summary
        equipment_counts = [r.equipment_count for r in records if r.task in ("understand", "equipment")]
        total_equipment = sum(equipment_counts)

        # Hallucinations summary
        all_flags: list[str] = []
        for r in records:
            all_flags.extend(r.hallucination_flags)
        flag_breakdown: dict[str, int] = {}
        for f in all_flags:
            flag_breakdown[f] = flag_breakdown.get(f, 0) + 1

        device_used = self.pipeline.backend.backend_info.device_used if self.pipeline else self.device_used
        device_requested = self.device_requested

        # Per-category breakdown (keyed by canonical display name)
        category_breakdown = self._build_category_breakdown(records)

        # Repeat variance report
        repeat_variance = self._compute_repeat_variance(records)

        # Bug 2 Fix: Coverage gap analysis using canonical display names exclusively
        categories_evaluated: set[str] = set()
        for r in records:
            if r.success:
                categories_evaluated.add(r.category)
        all_manifest_categories = sorted(set(s.category for s in REPRESENTATIVE_SAMPLES))
        evaluated_categories = sorted(set(CATEGORY_DISPLAY_NAMES.get(c, c) for c in categories_evaluated))
        all_display_categories = sorted(set(CATEGORY_DISPLAY_NAMES.get(c, c) for c in all_manifest_categories))
        categories_without_coverage = sorted(set(all_display_categories) - set(evaluated_categories))

        # Bug 3 Fix: Next run hint
        next_hint = format_next_run_hint(categories_without_coverage, run_mode=self.run_mode)

        # Full-run latency projection
        full_run_projection: dict[str, Any] = {}
        if avg_latency > 0:
            tasks_per_sample_val = (
                self.quick_mode_settings.get("tasks_per_sample")
                if self.run_mode == "quick" else None
            )
            full_run_projection = estimate_full_run_duration(
                REPRESENTATIVE_SAMPLES,
                avg_latency,
                tasks_per_sample=tasks_per_sample_val,
            )
            if confidence_caveat:
                full_run_projection["confidence_caveat"] = confidence_caveat

        summary: dict[str, Any] = {
            "evaluation_metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model_name": "qwen2.5-vl-3b-instruct" if not self.is_mock else "mock_vision_backend",
                "model_path": str(self.model_path),
                "is_mock": self.is_mock,
                "device_requested": device_requested,
                "device_used": device_used,
                "device_resolution_note": self.device_resolution_note,
                # model_load_time_ms is SEPARATE from per-inference processing_time_ms
                "model_load_time_ms": round(self.model_load_time_ms, 2),
                "total_inferences": total_inferences,
                "successful_inferences": successful_inferences,
                "failed_inferences": failed_inferences,
                "total_execution_time_s": round(total_time_s, 2),
                "run_mode": self.run_mode,
                "quick_mode_settings": self.quick_mode_settings,
                "stopped_early": stopped_early,
            },
            "confidence_metadata": {
                "confidence_method": CONFIDENCE_METHOD,
                "confidence_derivation": (
                    "Confidence is a self-reported heuristic from the model's own JSON output. "
                    "It is NOT computed from model logprobs or softmax probabilities. "
                    "Use only for qualitative ordering; do not treat as a calibrated probability."
                ),
            },
            "latency_summary": {
                "note": (
                    "processing_time_ms measures ONLY pipeline.run_task() wall-clock time. "
                    "Model load time is tracked separately in model_load_time_ms. "
                    "Latency averages and projections are strictly scoped to records matching the current settings signature."
                ),
                "settings_signature": {
                    "run_mode": self.run_mode,
                    "max_image_size": self.max_image_size,
                    "max_new_tokens": self.max_new_tokens,
                },
                "matching_signature_inferences_count": matching_count,
                "confidence_caveat": confidence_caveat,
                "average_latency_ms": avg_latency,
                "minimum_latency_ms": min_latency,
                "maximum_latency_ms": max_latency,
                "per_task_latency": per_task_latency,
                "full_run_projection": full_run_projection,
            },
            "coverage_summary": {
                "manifest_total_samples": len(REPRESENTATIVE_SAMPLES),
                "manifest_total_categories": len(all_display_categories),
                "samples_evaluated": len(set(r.sample_id for r in records)),
                "categories_with_real_model_results": evaluated_categories,
                "categories_with_zero_real_model_coverage": categories_without_coverage,
                "coverage_gap_warning": (
                    f"WARNING: {len(categories_without_coverage)} categories have zero "
                    f"coverage: {categories_without_coverage}"
                ) if categories_without_coverage else "All manifest categories have results.",
                "next_run_hint": next_hint,
                "most_recent_run_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "git_commit_hash": _get_git_commit_hash(),
            },
            "scene_classification_summary": {
                "detected_scene_types": scene_types,
            },
            "equipment_recognition_summary": {
                "total_equipment_detected": total_equipment,
                "average_equipment_per_image": round(total_equipment / len(equipment_counts), 2) if equipment_counts else 0.0,
            },
            "drawing_analysis_summary": {
                "drawing_inferences_evaluated": len(drawing_records),
                "drawing_type_matches": drawing_matches,
                "drawing_type_match_rate": drawing_match_rate,
            },
            "hallucination_summary": {
                "total_hallucination_flags_raised": len(all_flags),
                "breakdown": flag_breakdown,
            },
            "category_breakdown": category_breakdown,
            "repeat_variance_report": repeat_variance,
            "per_sample_inferences": [asdict(r) for r in records],
        }

        # 1. Write vision_evaluation.json
        json_file = self.output_dir / "vision_evaluation.json"
        json_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        LOGGER.info("Saved JSON evaluation to %s", json_file)

        # 2. Write vision_evaluation.csv (extended headers)
        csv_file = self.output_dir / "vision_evaluation.csv"
        csv_headers = [
            "sample_id", "source_path", "dataset", "category", "task", "success",
            "scene_type", "caption", "equipment_count", "equipment_types",
            "equipment_tags", "observations_count", "confidence", "confidence_method",
            "device_requested", "device_used", "processing_time_ms", "issues",
            "model_name", "model_path", "predicted_drawing_type", "expected_category",
            "drawing_type_match", "hallucination_flags", "repeat_index",
            "scene_keyword_hit_rate", "equipment_keyword_hit_rate",
            "run_mode", "max_image_size", "max_new_tokens",
        ]
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(csv_headers)
            for r in records:
                scene_kw_rate = r.scene_keyword_score.get("keyword_hit_rate", "")
                equip_kw_rate = r.equipment_keyword_score.get("equipment_hit_rate", "")
                writer.writerow([
                    r.sample_id, r.source_path, r.dataset, r.category, r.task, r.success,
                    r.scene_type, r.caption.replace("\n", " "), r.equipment_count,
                    ";".join(r.equipment_types), ";".join(r.equipment_tags),
                    r.observations_count, r.confidence, r.confidence_method,
                    r.device_requested, r.device_used, r.processing_time_ms,
                    ";".join(r.issues), r.model_name, r.model_path,
                    r.predicted_drawing_type, r.expected_category,
                    "" if r.drawing_type_match is None else str(r.drawing_type_match),
                    ";".join(r.hallucination_flags), r.repeat_index,
                    "" if scene_kw_rate is None else scene_kw_rate,
                    "" if equip_kw_rate is None else equip_kw_rate,
                    r.run_mode,
                    "" if r.max_image_size is None else str(r.max_image_size),
                    "" if r.max_new_tokens is None else str(r.max_new_tokens),
                ])
        LOGGER.info("Saved CSV evaluation to %s", csv_file)

        # 3. Write README.md
        self._write_readme(summary, records)

        return summary

    def _build_category_breakdown(self, records: list[EvaluationRecord]) -> dict[str, Any]:
        """Compute per-category stats from evaluation records using canonical display names."""
        cat_map: dict[str, list[EvaluationRecord]] = defaultdict(list)
        for r in records:
            cat_display = CATEGORY_DISPLAY_NAMES.get(r.category, r.category)
            cat_map[cat_display].append(r)

        breakdown: dict[str, Any] = {}
        for cat, recs in sorted(cat_map.items()):
            success_count = sum(1 for r in recs if r.success)
            total_count = len(recs)
            latencies = [r.processing_time_ms for r in recs if r.success and r.processing_time_ms > 0]
            confs = [r.confidence for r in recs if r.success and r.confidence > 0 and r.task == "understand"]
            hall_flags: list[str] = []
            for r in recs:
                hall_flags.extend(r.hallucination_flags)
            kw_scores = [
                r.scene_keyword_score.get("keyword_hit_rate")
                for r in recs
                if r.scene_keyword_score.get("scored") and r.scene_keyword_score.get("keyword_hit_rate") is not None
            ]
            breakdown[cat] = {
                "sample_ids": sorted(set(r.sample_id for r in recs)),
                "sample_count": len(set(r.sample_id for r in recs)),
                "total_task_inferences": total_count,
                "successful_inferences": success_count,
                "task_success_rate": round(success_count / total_count, 4) if total_count else 0.0,
                "average_confidence": round(sum(confs) / len(confs), 4) if confs else None,
                "average_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else None,
                "hallucination_flag_count": len(hall_flags),
                "hallucination_flag_types": sorted(set(hall_flags)),
                "scene_keyword_hit_rate": round(sum(kw_scores) / len(kw_scores), 4) if kw_scores else None,
            }
        return breakdown

    def _compute_repeat_variance(self, records: list[EvaluationRecord]) -> dict[str, Any]:
        """Compute variance across repeated runs per (sample_id, task)."""
        grouped: dict[tuple[str, str], list[EvaluationRecord]] = defaultdict(list)
        for r in records:
            grouped[(r.sample_id, r.task)].append(r)

        variance_report: dict[str, Any] = {}
        for (sid, task), recs in sorted(grouped.items()):
            if len(recs) < 2:
                continue  # Only report variance when >1 repeat
            confs = [r.confidence for r in recs]
            successes = [r.success for r in recs]
            hall_flag_counts = [len(r.hallucination_flags) for r in recs]
            key = f"{sid}::{task}"
            variance_report[key] = {
                "sample_id": sid,
                "task": task,
                "repeat_count": len(recs),
                "success_rate": round(sum(successes) / len(successes), 4),
                "confidence_values": confs,
                "confidence_mean": round(statistics.mean(confs), 4) if confs else None,
                "confidence_stdev": round(statistics.stdev(confs), 4) if len(confs) > 1 else 0.0,
                "hallucination_flag_counts": hall_flag_counts,
                "hallucination_flag_stdev": round(statistics.stdev(hall_flag_counts), 4) if len(hall_flag_counts) > 1 else 0.0,
            }
        return variance_report

    def _write_readme(self, summary: dict[str, Any], records: list[EvaluationRecord]) -> None:
        """Write structured README.md separating mock tests, real results, and coverage gaps."""
        meta = summary["evaluation_metadata"]
        lat = summary["latency_summary"]
        draw = summary["drawing_analysis_summary"]
        hall = summary["hallucination_summary"]
        cov = summary["coverage_summary"]
        conf_meta = summary["confidence_metadata"]
        cat_breakdown = summary.get("category_breakdown", {})
        repeat_var = summary.get("repeat_variance_report", {})
        proj = lat.get("full_run_projection", {})

        gap_warning = ""
        if cov["categories_with_zero_real_model_coverage"]:
            gap_hint = cov.get("next_run_hint") or format_next_run_hint(
                cov["categories_with_zero_real_model_coverage"],
                meta.get("run_mode", "quick"),
            )
            gap_warning = (
                "\n> [!WARNING]\n"
                f"> **Coverage Gap:** {len(cov['categories_with_zero_real_model_coverage'])} categories "
                f"have **zero real-model results**: "
                f"`{'`, `'.join(cov['categories_with_zero_real_model_coverage'])}`\n"
                f"> **Next Recommended Run:** `{gap_hint}`\n"
            )

        projection_text = ""
        if proj:
            conf_caveat_note = ""
            if proj.get("confidence_caveat"):
                conf_caveat_note = f"\n> [!NOTE]\n> Estimate is {proj['confidence_caveat']}.\n"
            mode_name = meta.get("run_mode", "full")
            projection_text = (
                f"\n**Full-Manifest Run Estimate** (based on {proj['avg_latency_ms_per_task_basis']:.0f} ms/task avg "
                f"at `{mode_name}` settings):\n"
                f"- {proj['total_manifest_tasks']} tasks x {proj['avg_latency_ms_per_task_basis']:.0f} ms "
                f"≈ **{proj['estimated_total_minutes']:.0f} min** (~{proj['estimated_total_hours']:.1f} h)\n"
                f"{conf_caveat_note}"
            )

        cat_table_text = "\n## 3. Per-Category Breakdown\n\n"
        cat_table_text += "| Category | Samples | Inferences | Success | Avg Conf | Avg Latency (ms) | Hall. Flags | KW Hit Rate |\n"
        cat_table_text += "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
        for cat, stats in sorted(cat_breakdown.items()):
            conf_str = f"{stats['average_confidence']:.2f}" if stats["average_confidence"] is not None else "\u2014"
            lat_str = f"{stats['average_latency_ms']:.0f}" if stats["average_latency_ms"] is not None else "\u2014"
            kw_str = f"{stats['scene_keyword_hit_rate']:.0%}" if stats["scene_keyword_hit_rate"] is not None else "unscored"
            cat_table_text += (
                f"| `{cat}` | {stats['sample_count']} | {stats['total_task_inferences']} | "
                f"{stats['task_success_rate'] * 100:.0f}% | {conf_str} | {lat_str} | "
                f"{stats['hallucination_flag_count']} | {kw_str} |\n"
            )

        repeat_variance_text = ""
        if repeat_var:
            repeat_variance_text = "\n## 5. Repeat-Run Variance\n\n"
            repeat_variance_text += "| Key | Repeats | Success | Conf Mean | Conf Stdev | Hall. Stdev |\n"
            repeat_variance_text += "| :--- | :---: | :---: | :---: | :---: | :---: |\n"
            for key, v in repeat_var.items():
                repeat_variance_text += (
                    f"| `{key}` | {v['repeat_count']} | {v['success_rate'] * 100:.0f}% | "
                    f"{v['confidence_mean'] or 'N/A'} | {v['confidence_stdev']} | {v['hallucination_flag_stdev']} |\n"
                )

        matching_n = lat.get("matching_signature_inferences_count", meta["successful_inferences"])
        caveat_badge = f" {lat['confidence_caveat']}" if lat.get("confidence_caveat") else ""

        readme_text = f"""# Member 3 Vision Pipeline Evaluation Report

**Evaluation Date:** {meta['timestamp']}
**Evaluated Model:** `{meta['model_name']}`
**Model Path:** `{meta['model_path']}`
**Device Requested:** `{meta['device_requested']}` | **Device Used:** `{meta['device_used']}`
**Device Resolution:** {meta['device_resolution_note']}
**Model Load Time:** `{meta['model_load_time_ms']:.1f} ms` (one-time init — excluded from per-inference latency)
**Mock Execution:** `{meta['is_mock']}`
**Git Commit:** `{cov['git_commit_hash']}`

> [!NOTE]
> Results are from {meta['total_inferences']} inferences across {cov['samples_evaluated']} samples
> (full manifest: {cov['manifest_total_samples']} samples, {cov['manifest_total_categories']} categories).
> No metric generalizes beyond what was actually tested.
{gap_warning}

---

## 1. Report Sections

This report separates three evidence tiers:

1. **Harness Self-Tests (Section 4):** Mock-backend tests validating harness logic — no model quality claims.
2. **Real-Model Quantitative Results (Sections 2–3):** Based on samples run against real Qwen2.5-VL-3B-Instruct.
3. **Coverage Gaps (Section 1.1):** Categories and samples with zero real-model results.

> [!IMPORTANT]
> **Strict Ground Rules:**
> - No "high accuracy" claims without ground truth annotations.
> - Confidence scores use method `{conf_meta['confidence_method']}` — NOT calibrated probabilities (see Section 2.3).
> - All quantitative claims are traceable to `coverage_summary`, `category_breakdown`, or `latency_summary` in `vision_evaluation.json`.

### 1.1 Coverage Summary

- **Manifest:** {cov['manifest_total_samples']} samples, {cov['manifest_total_categories']} categories
- **Evaluated This Run:** {cov['samples_evaluated']} samples
- **Categories With Real-Model Results:** `{'`, `'.join(cov['categories_with_real_model_results']) or 'none'}`
- **Zero-Coverage Categories:** `{'`, `'.join(cov['categories_with_zero_real_model_coverage']) or 'none'}`

---

## 2. Quantitative Measurements (Real-Model Results Only)

### 2.1 Inference Performance & Latency
| Metric | Value | Description |
| :--- | :--- | :--- |
| **Total Inferences** | **{meta['total_inferences']}** | Task queries run across all sessions |
| **Successful** | **{meta['successful_inferences']}** ({meta['successful_inferences'] / max(1, meta['total_inferences']) * 100:.1f}%) | No pipeline exception |
| **Failed** | **{meta['failed_inferences']}** | Errors or image load failures |
| **Avg Inference Latency** | **{lat['average_latency_ms']:.1f} ms**{caveat_badge} | `pipeline.run_task()` only (n={matching_n} matching `{meta.get('run_mode', 'full')}` settings) |
| **Min / Max Latency** | **{lat['minimum_latency_ms']:.1f} / {lat['maximum_latency_ms']:.1f} ms** | Scoped to matching settings signature |
| **Model Load Time** | **{meta['model_load_time_ms']:.1f} ms** | One-time init, NOT in avg latency |
| **Total Wall-Clock** | **{meta['total_execution_time_s']:.1f} s** | End-to-end harness runtime |
{projection_text}
### 2.2 Per-Task Latency

| Task | Inferences | Avg (ms) | Min (ms) | Max (ms) |
| :--- | :---: | :---: | :---: | :---: |
"""
        for t, stat in lat.get("per_task_latency", {}).items():
            readme_text += f"| `{t}` | {stat['count']} | {stat['average_ms']:.1f} | {stat['min_ms']:.1f} | {stat['max_ms']:.1f} |\n"

        readme_text += f"""
### 2.3 Confidence Score Methodology

**Method:** `{conf_meta['confidence_method']}`

{conf_meta['confidence_derivation']}

Each record in `vision_evaluation.json` includes `confidence_method: "{conf_meta['confidence_method']}"` for unambiguous interpretation.

### 2.4 Drawing Category Match Rate

| Metric | Result |
| :--- | :---: |
| **Drawing Inferences Evaluated** | **{draw['drawing_inferences_evaluated']}** |
| **Category Matches (ground-truth-backed)** | **{draw['drawing_type_matches']}** |
| **Match Rate** | **{draw['drawing_type_match_rate'] * 100:.1f}%** |

---
{cat_table_text}
> Scene KW Hit Rate = fraction of expected scene keywords found in combined model output.
> "unscored" = no reference keywords provided for this sample.
> Avg Confidence reported only for `understand` task.

---

## 4. Hallucination & Safety Audit

| Category | Flags |
| :--- | :---: |
| Invented Equipment Tags | {sum(1 for f in hall['breakdown'] if 'invented_equipment_tag' in f)} |
| Unsupported Pressure | {hall['breakdown'].get('unsupported_pressure_value', 0)} |
| Unsupported Temperature | {hall['breakdown'].get('unsupported_temperature_value', 0)} |
| Unsupported Dimensions | {hall['breakdown'].get('unsupported_dimension', 0)} |
| Unsupported Safety Claims | {hall['breakdown'].get('unsupported_safety_conclusion', 0)} |
| Proximity-Based Connectivity | {hall['breakdown'].get('unsupported_connected_to_relationship', 0)} |

---
{repeat_variance_text}

## 6. Harness Self-Test Notes

> Results below are from **mock-backend unit tests only** — no model quality claims.
> Run: `./.venv/Scripts/python.exe -m pytest member3_ocr/tests/test_vision_evaluator.py -v`

---

## 7. Qualitative Observations (Non-Scored)

### 7.1 Engineering Drawings
- Qwen2.5-VL reliably classifies drawing macro-category (P&ID, PFD, mechanical assembly).
- Fine instrument bubble text (`FT-101`, `PT-202`) is NOT reliable at low resolutions.
- Proximity-based topology inference is suppressed; deferred to `drawing_analyzer.py`.

### 7.2 Industrial Inspection & Defect
- Surface anomaly descriptions generated with appropriate uncertainty qualifiers.
- Definitive safety diagnoses suppressed by system prompt.

### 7.3 Handwritten Notes
- Global maintenance context captured; character-level transcription requires PaddleOCR.

---

## 8. Production Recommendations

1. **CPU Bottleneck:** At ~{lat['average_latency_ms']:.0f} ms/task on CPU, CUDA GPU or INT4 quantization is needed for production.
2. **Speed Tradeoff:** `--max-image-size 256 --max-new-tokens 48` significantly reduces latency with acceptable quality for categorization.
3. **Hybrid Architecture:** Qwen2.5-VL for macro understanding; PaddleOCR + `drawing_analyzer.py` for fine text and topology.
"""
        readme_file = self.output_dir / "README.md"
        readme_file.write_text(readme_text, encoding="utf-8")
        LOGGER.info("Saved README report to %s", readme_file)




# ──────────────────────────────────────────────────────────────────────────────
# Command Line Interface
# ──────────────────────────────────────────────────────────────────────────────

def generate_chunked_run_plan(output_dir: Path | str = DEFAULT_OUTPUT_DIR) -> str:
    """Generate or update member3_ocr/CHUNKED_RUN_PLAN.md.

    Uses observed quick-mode latency from output_dir / 'vision_evaluation.json' if present,
    or falls back to provisional estimates with an explicit disclaimer.
    """
    out_path = Path(output_dir)
    json_path = out_path / "vision_evaluation.json"

    sessions = [
        ("Core Process Drawings", ["PID", "PFD"]),
        ("Equipment & Instrumentation", ["Equipment_Drawings", "Instrumentation"]),
        ("Rotating & Electrical", ["Pump_Diagrams", "Electrical"]),
        ("Defect Inspection", ["Corrosion_Defects", "Defect_Detection"]),
        ("Photovoltaic & Notes", ["Infrared", "Handwritten_Notes"]),
    ]

    session_rows = []
    total_samples = 0
    total_tasks = 0
    total_est_minutes = 0.0
    has_real_quick_data = False
    measured_ms = 0.0

    if json_path.exists():
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            meta = data.get("evaluation_metadata", {})
            lat = data.get("latency_summary", {})
            if meta.get("run_mode") == "quick" and not meta.get("is_mock", False):
                avg_lat = lat.get("average_latency_ms", 0.0)
                if avg_lat > 0:
                    has_real_quick_data = True
                    measured_ms = avg_lat
        except Exception:
            pass

    for s_idx, (s_name, s_cats) in enumerate(sessions, 1):
        cat_samples = filter_samples_by_category(REPRESENTATIVE_SAMPLES, s_cats)
        est = estimate_run_duration(cat_samples, tasks_per_sample=1, prior_json_path=json_path, is_quick=True)
        cat_arg = ",".join(s_cats)
        cmd = f"python evaluate_vision.py --category {cat_arg} --quick --resume --device auto --max-run-minutes 15 --threads 4"
        session_rows.append({
            "session_num": s_idx,
            "title": s_name,
            "categories": cat_arg,
            "samples": len(cat_samples),
            "tasks": est["total_tasks_estimated"],
            "minutes": est["estimated_total_minutes"],
            "command": cmd,
        })
        total_samples += len(cat_samples)
        total_tasks += est["total_tasks_estimated"]
        total_est_minutes += est["estimated_total_minutes"]

    if has_real_quick_data:
        header_notice = (
            f"> **Notice:** Time estimates below are **refined from measured quick-mode data** "
            f"({measured_ms / 1000:.1f} s/task measured on Qwen2.5-VL-3B-Instruct).\n"
        )
    else:
        header_notice = (
            "> **Notice:** Estimates below are provisional (based on non-quick settings) — "
            "will be refined after first real --quick run.\n"
        )

    lines = [
        "# Chunked Execution Run Plan for Vision Pipeline Evaluation",
        "",
        header_notice.strip(),
        "",
        "## Overview",
        "",
        "This execution plan divides the 16 representative samples across all 10 manifest categories into 5 independent 10–15 minute sessions using `--category`, `--quick`, and `--resume`.",
        "",
        "| Session | Focus Area | Categories | Samples | Tasks | Est. Time |",
        "| :---: | :--- | :--- | :---: | :---: | :---: |",
    ]

    for r in session_rows:
        lines.append(f"| {r['session_num']} | {r['title']} | `{r['categories']}` | {r['samples']} | {r['tasks']} | ~{r['minutes']:.0f} min |")

    lines.append(f"| **Total** | **All 10 Categories** | — | **{total_samples}** | **{total_tasks}** | **~{total_est_minutes:.0f} min** |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Session Commands")
    lines.append("")

    for r in session_rows:
        lines.append(f"### Session {r['session_num']}: {r['title']}")
        lines.append(f"```bash\n{r['command']}\n```")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Step 6: Consolidation & Full Report")
    lines.append("")
    lines.append("After completing all sessions, run the consolidation command without category filter:")
    lines.append("```bash\npython evaluate_vision.py --resume\n```")
    lines.append("")
    lines.append("This loads all accumulated inferences and generates final unified `vision_evaluation.json`, `vision_evaluation.csv`, and `README.md` artifacts.")
    lines.append("")

    content = "\n".join(lines)
    plan_file = PROJECT_ROOT / "member3_ocr" / "CHUNKED_RUN_PLAN.md"
    plan_file.write_text(content, encoding="utf-8")
    LOGGER.info("Wrote run plan to %s", plan_file)
    return content


# ──────────────────────────────────────────────────────────────────────────────
# Command Line Interface
# ──────────────────────────────────────────────────────────────────────────────

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for evaluation runner."""
    parser = argparse.ArgumentParser(
        description="Member 3 Vision Pipeline Representative Evaluation Harness"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=2,
        help="Number of representative samples to evaluate (default: 2 for CPU safety; 0 = all samples)",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Filter samples to specific category or comma-separated categories/aliases (e.g. 'PID', 'PFD', 'PID,PFD')",
    )
    parser.add_argument(
        "--list-categories",
        action="store_true",
        help="List all canonical manifest categories and accepted aliases, then exit",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run in low-resource quick mode (192px max size, 24 tokens, 1 primary task per sample)",
    )
    parser.add_argument(
        "--tasks-per-sample",
        type=int,
        default=None,
        help="Limit number of tasks to run per sample (1 selects PRIMARY_TASK_MAP task)",
    )
    parser.add_argument(
        "--max-run-minutes",
        type=float,
        default=None,
        help="Maximum run time budget in minutes. Evaluation stops gracefully and flushes to disk when reached.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Number of PyTorch CPU threads to use (calls torch.set_num_threads)",
    )
    parser.add_argument(
        "--quantize",
        type=str,
        default=None,
        help="Model quantization (e.g. '4bit', '8bit'). Note: not available in CPU-only environment.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume evaluation skipping already completed (sample_id, task, repeat_index) triples",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run harness with MockVisionEvaluatorBackend for instantaneous testing",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_VISION_MODEL_PATH,
        help=f"Path to local Qwen2.5-VL model (default: {DEFAULT_VISION_MODEL_PATH})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "auto"],
        help="Inference device: 'auto' detects CUDA and falls back to cpu. Device resolution is logged at startup.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory to write evaluation artifacts (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--max-image-size",
        type=int,
        default=None,
        help="Max image dimension in pixels (default: 512 in full mode, 192 in --quick mode)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Max generation tokens per task (default: 128 in full mode, 24 in --quick mode)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help=(
            "Number of times to repeat each (sample, task) pair (default: 1). "
            "Use >1 to measure output variance across runs. Variance report added to JSON/README."
        ),
    )
    parser.add_argument(
        "--regenerate-plan",
        action="store_true",
        help="Regenerate member3_ocr/CHUNKED_RUN_PLAN.md with current latency data and exit",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Main execution function."""
    args = parse_args(argv)

    if args.list_categories:
        print("\nAvailable Manifest Categories and Accepted Aliases:")
        print("-" * 75)
        for cat in sorted(CATEGORY_ALIASES.keys()):
            display_name = CATEGORY_DISPLAY_NAMES.get(cat, cat)
            aliases = ", ".join(f"'{a}'" for a in CATEGORY_ALIASES[cat])
            primary = PRIMARY_TASK_MAP.get(cat, "unknown")
            print(f"  {display_name:<22} (primary task: {primary:<11}) -> aliases: {aliases}")
        print("-" * 75)
        return 0

    if args.regenerate_plan:
        generate_chunked_run_plan(output_dir=args.output_dir)
        print("member3_ocr/CHUNKED_RUN_PLAN.md regenerated successfully.")
        return 0

    # Cheap speedup: configure PyTorch thread count if requested
    if args.threads is not None and args.threads > 0:
        try:
            import torch
            torch.set_num_threads(args.threads)
            LOGGER.info("Configured PyTorch CPU thread count to %d", args.threads)
        except Exception as exc:
            LOGGER.warning("Could not set torch thread count: %s", exc)

    if args.quantize is not None:
        LOGGER.warning(
            "Quantization flag '--quantize %s' passed, but bitsandbytes/optimum is not "
            "installed in this environment. Proceeding with standard precision.",
            args.quantize,
        )

    # Resolve quick-mode defaults
    if args.quick:
        max_image_size = args.max_image_size if args.max_image_size is not None else 192
        max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else 24
        tasks_per_sample = args.tasks_per_sample if args.tasks_per_sample is not None else 1
        run_mode = "quick"
        quick_mode_settings = {
            "max_image_size": max_image_size,
            "max_new_tokens": max_new_tokens,
            "tasks_per_sample": tasks_per_sample,
        }
    else:
        max_image_size = args.max_image_size if args.max_image_size is not None else 512
        max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else 128
        tasks_per_sample = args.tasks_per_sample
        run_mode = "full"
        quick_mode_settings = {}

    # Category filtering
    if args.category:
        category_names = [c.strip() for c in args.category.split(",") if c.strip()]
        try:
            target_samples = filter_samples_by_category(REPRESENTATIVE_SAMPLES, category_names)
        except ValueError as exc:
            LOGGER.error("Category resolution error: %s", exc)
            print(f"Error: {exc}")
            return 1
    else:
        target_samples = list(REPRESENTATIVE_SAMPLES)

    limit = args.limit if args.limit > 0 else None

    # Pre-run duration estimate (Fix 2)
    prior_json = Path(args.output_dir) / "vision_evaluation.json"
    est_samples = target_samples[:limit] if limit is not None else target_samples
    eta_info = estimate_run_duration(
        est_samples,
        tasks_per_sample=tasks_per_sample,
        prior_json_path=prior_json,
        is_quick=args.quick,
    )
    print(f"\n[Pre-Run Estimate] {eta_info['note']}")

    harness = VisionEvaluationHarness(
        output_dir=args.output_dir,
        model_path=args.model_path,
        device=args.device,
        is_mock=args.mock,
        max_image_size=max_image_size,
        max_new_tokens=max_new_tokens,
        run_mode=run_mode,
        quick_mode_settings=quick_mode_settings,
    )
    summary = harness.evaluate(
        samples=target_samples,
        limit=limit,
        resume=args.resume,
        repeat=args.repeat,
        tasks_per_sample=tasks_per_sample,
        max_run_minutes=args.max_run_minutes,
    )

    meta = summary["evaluation_metadata"]
    lat = summary["latency_summary"]
    cov = summary["coverage_summary"]
    proj = lat.get("full_run_projection", {})

    print("\n" + "=" * 70)
    print("VISION PIPELINE EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Run Mode         : {meta.get('run_mode', 'full').upper()}")
    if meta.get("quick_mode_settings"):
        print(f"Quick Settings   : {meta['quick_mode_settings']}")
    if meta.get("stopped_early"):
        print("Early Stop       : TRUE (stopped due to --max-run-minutes budget)")
    print(f"Total Inferences : {meta['total_inferences']}")
    print(f"Successful       : {meta['successful_inferences']}")
    print(f"Failed           : {meta['failed_inferences']}")
    print(f"Device Requested : {meta['device_requested']}")
    print(f"Device Used      : {meta['device_used']}")
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    print(f"  |-- {meta['device_resolution_note']}")
    print(f"Model Load Time  : {meta['model_load_time_ms']:.1f} ms (excluded from avg latency)")
    caveat_str = f" {lat['confidence_caveat']}" if lat.get("confidence_caveat") else ""
    matching_n = lat.get("matching_signature_inferences_count", meta.get("successful_inferences", 0))
    print(f"Avg Latency      : {lat['average_latency_ms']:.1f} ms (inference only, n={matching_n} matching settings){caveat_str}")
    if proj:
        proj_caveat = f" {proj['confidence_caveat']}" if proj.get("confidence_caveat") else ""
        print(f"Full-Run Est.    : ~{proj['estimated_total_minutes']:.0f} min for all {proj['total_manifest_tasks']} manifest tasks{proj_caveat}")
    if cov["categories_with_zero_real_model_coverage"]:
        print(f"\n[!] COVERAGE GAPS: {cov['categories_with_zero_real_model_coverage']}")
        print(f"   {cov['coverage_gap_warning']}")
        next_hint = cov.get("next_run_hint") or format_next_run_hint(cov["categories_with_zero_real_model_coverage"], meta.get("run_mode", "quick"))
        print(f"   Run: {next_hint}")
    else:
        print("\n[OK] FULL COVERAGE: All 10 manifest categories have evaluation results.")
    print(f"Output Directory : {harness.output_dir}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())

