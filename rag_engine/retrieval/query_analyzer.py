"""Query Analyzer & Intent Classifier.

Performs deep semantic analysis and entity extraction on refinery engineering queries
to classify query intent and extract domain entities (equipment, units, standards, specs).
"""

from __future__ import annotations

from enum import StrEnum
import re
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from rag_engine.retrieval.retrieval_utils import (
    EQUIPMENT_TAG_REGEX,
    LINE_NUMBER_REGEX,
    PAGE_REF_REGEX,
    PLANT_UNIT_REGEX,
    PRESSURE_REGEX,
    REVISION_REGEX,
    SECTION_REF_REGEX,
    STANDARD_REGEX,
    TEMPERATURE_REGEX,
    normalize_whitespace,
)


class IntentType(StrEnum):
    """Refinery query intent classification."""

    INFORMATIONAL = "informational"
    EQUIPMENT_LOOKUP = "equipment_lookup"
    SAFETY_LOOKUP = "safety_lookup"
    INSPECTION_LOOKUP = "inspection_lookup"
    MAINTENANCE_LOOKUP = "maintenance_lookup"
    PROCEDURE_LOOKUP = "procedure_lookup"
    SPECIFICATION_LOOKUP = "specification_lookup"
    TABLE_LOOKUP = "table_lookup"
    NUMERICAL_LOOKUP = "numerical_lookup"
    TEMPLATE_LOOKUP = "template_lookup"
    COMPARISON = "comparison"
    TROUBLESHOOTING = "troubleshooting"


class ExtractedEntities(BaseModel):
    """Structured refinery domain entities extracted from a query."""

    model_config = ConfigDict(frozen=True)

    equipment_tags: list[str] = Field(default_factory=list)
    plant_units: list[str] = Field(default_factory=list)
    standards: list[str] = Field(default_factory=list)
    pressure_values: list[str] = Field(default_factory=list)
    temperature_values: list[str] = Field(default_factory=list)
    line_numbers: list[str] = Field(default_factory=list)
    dates: list[str] = Field(default_factory=list)
    engineer_names: list[str] = Field(default_factory=list)
    revision_numbers: list[str] = Field(default_factory=list)
    section_references: list[str] = Field(default_factory=list)
    page_references: list[str] = Field(default_factory=list)


class QueryIntent(BaseModel):
    """Complete semantic analysis result for a user or agent query."""

    model_config = ConfigDict(frozen=True)

    raw_query: str = Field(description="Original unprocessed query")
    normalized_query: str = Field(description="Normalized query text")
    primary_intent: IntentType = Field(default=IntentType.INFORMATIONAL)
    secondary_intents: list[IntentType] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    entities: ExtractedEntities = Field(default_factory=ExtractedEntities)
    suggested_top_k: int = Field(default=10, ge=1, le=100)
    requires_neighbor_expansion: bool = Field(default=False)
    preferred_category: Optional[str] = Field(default=None)


class QueryAnalyzer:
    """Enterprise query analysis and entity extractor for refinery queries."""

    # Intent keyword triggers
    _SAFETY_KEYWORDS = {
        "safety", "hazard", "ppe", "fire", "emergency", "oisd", "explosion",
        "danger", "warning", "flammable", "toxic", "leak", "evacuation", "interlock",
    }
    _MAINTENANCE_KEYWORDS = {
        "maintenance", "overhaul", "repair", "replace", "pm", "preventive",
        "breakdown", "inspection", "ndt", "lubrication", "bearing", "vibration",
        "alignment", "work order", "schedule", "service",
    }
    _INSPECTION_KEYWORDS = {
        "inspection", "inspect", "ndt", "ultrasonic", "thickness", "radiography",
        "corrosion", "dye", "penetrant", "magnetic", "visual inspection", "report",
    }
    _PROCEDURE_KEYWORDS = {
        "sop", "procedure", "step", "steps", "protocol", "checklist", "sequence",
        "startup", "shutdown", "commissioning", "isolation",
    }
    _SPEC_KEYWORDS = {
        "specification", "rating", "capacity", "tolerance", "dimension",
        "material", "datasheet", "design", "flow rate", "head", "rpm", "kw", "power",
    }
    _TABLE_KEYWORDS = {
        "table", "matrix", "schedule", "tabulated", "chart", "bill of materials", "bom",
    }
    _TEMPLATE_KEYWORDS = {
        "template", "templates", "form", "forms", "checklist", "checklists",
        "approval note", "blank_form", "work order template", "format", "sample form",
    }
    _NUMERICAL_KEYWORDS = {
        "limit", "maximum", "minimum", "max", "min", "pressure", "temperature",
        "setpoint", "alarm", "trip", "threshold", "range", "flow",
    }
    _COMPARISON_KEYWORDS = {
        "compare", "difference", "versus", "vs", "better", "contrast",
    }
    _TROUBLESHOOTING_KEYWORDS = {
        "troubleshoot", "troubleshooting", "failure", "cause", "abnormal", "fault",
        "trip", "alarm", "cavitation", "overheating", "vibration", "problem",
        "solution", "issue", "diagnosis", "why",
    }

    # Date pattern: e.g. 2024-05-12, 12/05/2024, May 2023, 15-Jan-2024
    _DATE_REGEX = re.compile(
        r"\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[-_\s]?\d{1,2}[,/-]?\s*\d{2,4}|\d{4})\b",
        re.IGNORECASE,
    )

    # Engineer / signature mentions: e.g. Eng. Sharma, Approved by John, inspected by K. Rao
    _ENGINEER_REGEX = re.compile(
        r"\b(?:eng(?:ineer)?\.?|approved by|inspected by|operator|technician)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b",
        re.IGNORECASE,
    )

    def analyze(self, query: str) -> QueryIntent:
        """Analyze a natural language query and extract structured intent and entities."""
        from rag_engine.retrieval.query_normalizer import QueryNormalizer

        raw = query or ""
        norm_query = QueryNormalizer.normalize(raw)

        entities = self._extract_entities(norm_query)
        primary, secondary, top_k, expand, pref_cat = self._classify_intent(
            norm_query, entities
        )

        return QueryIntent(
            raw_query=raw,
            normalized_query=norm_query,
            primary_intent=primary,
            secondary_intents=secondary,
            confidence=0.95 if entities.equipment_tags or entities.standards else 0.85,
            entities=entities,
            suggested_top_k=top_k,
            requires_neighbor_expansion=expand,
            preferred_category=pref_cat,
        )

    def _extract_entities(self, text: str) -> ExtractedEntities:
        """Extract domain specific entities from normalized query text."""
        # Equipment tags
        equip_matches = []
        excluded_prefixes = {
            "OISD", "API", "ASME", "PNGRB", "ISO", "REV", "SEC", "PG", "FIG",
            "LINE", "L", "STD", "DOC", "SPEC", "FORM", "INSP", "REPORT", "ANNEXURE",
        }
        for m in EQUIPMENT_TAG_REGEX.finditer(text):
            prefix = m.group(1).upper()
            num = m.group(2).upper()
            tag = f"{prefix}-{num}"
            if prefix not in excluded_prefixes and (len(prefix) > 1 or len(num) >= 3):
                equip_matches.append(tag)

        # Plant units
        unit_matches = []
        for m in PLANT_UNIT_REGEX.finditer(text):
            unit_matches.append(m.group(1).upper().replace(" ", "-"))

        # Standards
        standard_matches = []
        for m in STANDARD_REGEX.finditer(text):
            std = re.sub(r"[-_\s]+", "-", m.group(1)).upper()
            standard_matches.append(std)

        # Pressures
        press_matches = []
        for m in PRESSURE_REGEX.finditer(text):
            press_matches.append(f"{m.group(1)} {m.group(2).lower()}")

        # Temperatures
        temp_matches = []
        for m in TEMPERATURE_REGEX.finditer(text):
            val = m.group(1) or m.group(3)
            unit = m.group(2) or "C"
            temp_matches.append(f"{val} °{unit.upper()}")

        # Line numbers
        line_matches = [m.group(0).upper() for m in LINE_NUMBER_REGEX.finditer(text)]

        # Dates
        date_matches = [m.group(0) for m in self._DATE_REGEX.finditer(text)]

        # Engineer names
        eng_matches = [m.group(1) for m in self._ENGINEER_REGEX.finditer(text)]

        # Revision numbers
        rev_matches = [m.group(1) for m in REVISION_REGEX.finditer(text)]

        # Section references
        sec_matches = [m.group(1) for m in SECTION_REF_REGEX.finditer(text)]

        # Page references
        page_matches = [m.group(1) for m in PAGE_REF_REGEX.finditer(text)]

        return ExtractedEntities(
            equipment_tags=sorted(list(set(equip_matches))),
            plant_units=sorted(list(set(unit_matches))),
            standards=sorted(list(set(standard_matches))),
            pressure_values=sorted(list(set(press_matches))),
            temperature_values=sorted(list(set(temp_matches))),
            line_numbers=sorted(list(set(line_matches))),
            dates=sorted(list(set(date_matches))),
            engineer_names=sorted(list(set(eng_matches))),
            revision_numbers=sorted(list(set(rev_matches))),
            section_references=sorted(list(set(sec_matches))),
            page_references=sorted(list(set(page_matches))),
        )

    def _classify_intent(
        self, text: str, entities: ExtractedEntities
    ) -> tuple[IntentType, list[IntentType], int, bool, Optional[str]]:
        """Classify query intent and determine adaptive execution parameters."""
        lower_tokens = set(re.findall(r"\b[a-z0-9-]+\b", text.lower()))
        matched_intents: list[IntentType] = []

        if lower_tokens & self._TEMPLATE_KEYWORDS:
            matched_intents.append(IntentType.TEMPLATE_LOOKUP)
        if lower_tokens & self._TROUBLESHOOTING_KEYWORDS:
            matched_intents.append(IntentType.TROUBLESHOOTING)
        if lower_tokens & self._INSPECTION_KEYWORDS:
            matched_intents.append(IntentType.INSPECTION_LOOKUP)
        if lower_tokens & self._SAFETY_KEYWORDS or entities.standards:
            matched_intents.append(IntentType.SAFETY_LOOKUP)
        if lower_tokens & self._PROCEDURE_KEYWORDS:
            matched_intents.append(IntentType.PROCEDURE_LOOKUP)
        if lower_tokens & self._MAINTENANCE_KEYWORDS:
            matched_intents.append(IntentType.MAINTENANCE_LOOKUP)
        if lower_tokens & self._COMPARISON_KEYWORDS:
            matched_intents.append(IntentType.COMPARISON)
        if lower_tokens & self._TABLE_KEYWORDS:
            matched_intents.append(IntentType.TABLE_LOOKUP)
        if (
            lower_tokens & self._NUMERICAL_KEYWORDS
            or entities.pressure_values
            or entities.temperature_values
        ):
            matched_intents.append(IntentType.NUMERICAL_LOOKUP)
        if lower_tokens & self._SPEC_KEYWORDS:
            matched_intents.append(IntentType.SPECIFICATION_LOOKUP)
        if entities.equipment_tags:
            matched_intents.append(IntentType.EQUIPMENT_LOOKUP)

        if not matched_intents:
            primary = IntentType.INFORMATIONAL
            secondary: list[IntentType] = []
        else:
            primary = matched_intents[0]
            secondary = matched_intents[1:]

        # Adaptive parameters: top_k and neighbor expansion
        top_k = 10
        expand = False
        pref_cat = None

        if primary == IntentType.TEMPLATE_LOOKUP:
            top_k = 12
            expand = True
            pref_cat = "templates"
        elif primary == IntentType.TROUBLESHOOTING:
            top_k = 25
            expand = True
            pref_cat = "manuals"
        elif primary == IntentType.INSPECTION_LOOKUP:
            top_k = 15
            expand = True
            pref_cat = "inspection"
        elif primary == IntentType.PROCEDURE_LOOKUP:
            top_k = 12
            expand = True
            pref_cat = "sops"
        elif primary == IntentType.SAFETY_LOOKUP:
            top_k = 15
            expand = True
            pref_cat = "safety"
        elif primary == IntentType.MAINTENANCE_LOOKUP:
            top_k = 15
            expand = True
            pref_cat = "maintenance"
        elif primary == IntentType.TABLE_LOOKUP:
            top_k = 8
            expand = True
        elif primary == IntentType.EQUIPMENT_LOOKUP:
            top_k = 10
            expand = True
            pref_cat = "manuals"
        elif primary == IntentType.NUMERICAL_LOOKUP:
            top_k = 8
            expand = False

        return primary, secondary, top_k, expand, pref_cat
