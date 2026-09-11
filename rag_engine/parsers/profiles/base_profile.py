"""Base profile interface for refinery domain rules, equipment tags, and operational standards."""

from __future__ import annotations

import re
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field

from rag_engine.schemas.parsed_document import EquipmentType


class RefineryProfile(BaseModel):
    """Abstract configuration profile defining refinery-specific nomenclature and rules."""

    model_config = ConfigDict(frozen=True)

    profile_name: str = Field(description="Unique profile identifier, e.g. 'mrpl', 'generic'")
    refinery_name: str = Field(description="Display name of the refinery organization")
    plant_units: list[str] = Field(
        default_factory=list,
        description="Known plant units (e.g. ['CDU-1', 'CDU-2', 'VDU', 'DCU', 'PFCC', 'CCR', 'SRU'])",
    )
    equipment_tag_patterns: list[str] = Field(
        default_factory=list,
        description="Regex pattern strings matching equipment tags",
    )
    equipment_type_prefixes: dict[str, EquipmentType] = Field(
        default_factory=dict,
        description="Mapping from tag prefix to EquipmentType (e.g. {'P': EquipmentType.PUMP})",
    )
    equipment_type_keywords: dict[str, EquipmentType] = Field(
        default_factory=dict,
        description="Mapping from keyword in text to EquipmentType (e.g. {'compressor': EquipmentType.COMPRESSOR})",
    )
    standard_names: list[str] = Field(
        default_factory=lambda: ["OISD", "PNGRB", "API", "ASME", "ISO", "ASTM", "IEC", "IEEE", "IBR"],
        description="Recognized engineering and safety standards",
    )
    pressure_units: list[str] = Field(
        default_factory=lambda: ["bar", "barg", "bar(g)", "psi", "psig", "kPa", "MPa", "kg/cm2", "kg/cm²", "atm"],
        description="Physical pressure units",
    )
    temperature_units: list[str] = Field(
        default_factory=lambda: ["°C", "deg C", "degC", "C", "°F", "deg F", "degF", "F", "K"],
        description="Physical temperature units",
    )
    safety_keywords: dict[str, str] = Field(
        default_factory=lambda: {
            "DANGER": "DANGER",
            "WARNING": "WARNING",
            "CAUTION": "CAUTION",
            "NOTICE": "NOTICE",
            "IMPORTANT": "NOTICE",
            "PRECAUTION": "CAUTION",
        },
        description="Keywords mapping to SafetyWarningSeverity",
    )
    drawing_title_block_keys: list[str] = Field(
        default_factory=lambda: [
            "DRAWING NO", "DWG NO", "TITLE", "REV", "REVISION", "DATE", "DRAWN BY",
            "CHECKED BY", "APPROVED BY", "SCALE", "PROJECT", "CLIENT", "PLANT"
        ],
        description="Standard title block headers for engineering drawings",
    )

    def matches_equipment_tag(self, text: str) -> bool:
        """Check whether candidate token matches any equipment pattern."""
        for pattern in self.equipment_tag_patterns:
            if re.fullmatch(pattern, text):
                return True
        return False

    def resolve_equipment_type(self, tag: str, context: Optional[str] = None) -> EquipmentType:
        """Deterministically map tag or surrounding context to EquipmentType."""
        clean_tag = tag.upper().strip()
        # 1. Try prefix mapping, prioritizing longer prefixes first (e.g. 'PSV-' before 'P')
        sorted_prefixes = sorted(
            self.equipment_type_prefixes.items(), key=lambda item: len(item[0]), reverse=True
        )
        for prefix, eq_type in sorted_prefixes:
            if clean_tag.startswith(prefix.upper()):
                return eq_type
        # 2. Try context keywords if provided
        if context:
            ctx_lower = context.lower()
            for kw, eq_type in self.equipment_type_keywords.items():
                if kw.lower() in ctx_lower:
                    return eq_type
        return EquipmentType.OTHER

