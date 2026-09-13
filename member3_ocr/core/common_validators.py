"""Common validators and regex specifications shared across drawing and vision analysis.

Extracted from evaluate_vision.py to decouple core production code (drawing_analyzer.py)
from evaluation harnesses. Single source of truth for drawing synonyms, hallucination checks,
and P&ID inspection analysis.
"""
from __future__ import annotations


import re
from typing import Any, Sequence

# ──────────────────────────────────────────────────────────────────────────────
# Drawing Synonyms & Type Matcher
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


# ──────────────────────────────────────────────────────────────────────────────
# Hallucination Check Engine Regexes & Validators
# ──────────────────────────────────────────────────────────────────────────────

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
    for item in equipment_items:
        tag = item.get("name_or_tag")
        if tag:
            tag_clean = str(tag).strip()
            if known_tags and not any(k.lower() in tag_clean.lower() for k in known_tags):
                evidence = str(item.get("evidence", "")).strip()
                if not evidence or "assumed" in evidence.lower() or "inferred" in evidence.lower():
                    flags.append(f"invented_equipment_tag:{tag_clean}")

    return sorted(list(set(flags)))


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
