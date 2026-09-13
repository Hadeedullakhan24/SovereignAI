"""Prompt Templates — Domain Archetypes for Refinery Knowledge Operations.

Defines deterministic system prompts and instruction formats across 7 refinery
query archetypes: Equipment Lookup, SOP Retrieval, Maintenance, Safety Compliance,
Troubleshooting, Comparison, and General Engineering QA.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict


class PromptArchetype(str, Enum):
    """Refinery query archetypes determining generation instructions."""

    EQUIPMENT_LOOKUP = "equipment_lookup"
    SOP_RETRIEVAL = "sop_retrieval"
    MAINTENANCE = "maintenance"
    SAFETY_COMPLIANCE = "safety_compliance"
    TROUBLESHOOTING = "troubleshooting"
    COMPARISON = "comparison"
    GENERAL_QA = "general_qa"


import hashlib


@dataclass(frozen=True)
class PromptTemplate:
    """Immutable template specification for prompt construction with cryptographic versioning."""

    archetype: PromptArchetype
    system_instruction: str
    generation_instruction: str
    template_version: str = "v1.0.0"
    name: str = ""
    author: str = "MRPL AI Engineering Team"
    creation_date: str = "2026-09-10"
    compatibility: str = "v1.x"
    prompt_hash: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            object.__setattr__(self, "name", f"{self.archetype.value.replace('_', ' ').title()} Template")
        if not self.prompt_hash:
            payload = f"{self.name}:{self.template_version}:{self.author}:{self.system_instruction}:{self.generation_instruction}"
            h = hashlib.sha256(payload.encode("utf-8")).hexdigest()
            object.__setattr__(self, "prompt_hash", h)

    @property
    def version(self) -> str:
        """Alias for template_version."""
        return self.template_version


# --------------------------------------------------------------------------
# Built-in Domain Templates for MRPL Operations
# --------------------------------------------------------------------------

_SYSTEM_PREAMBLE = (
    "You are the Sovereign AI Assistant for Mangalore Refinery and Petrochemicals Limited (MRPL).\n"
    "Your objective is to provide precise, technically accurate, and strictly evidence-grounded answers.\n"
    "CRITICAL RULES:\n"
    "1. Answer solely using the verified documentation provided in the CONTEXT section.\n"
    "2. Answer ONLY what the user's question explicitly asks. Do NOT proactively volunteer disclaimers or mention unrelated topics that were not requested.\n"
    "3. STRICT GROUNDING: Quote exact figures, readings, and dates directly from the context. "
    "Do NOT extrapolate generic rules, percentages, or intervals from outside knowledge. "
    "If a specific parameter, interval, or requirement requested by the user's question is NOT found in the context, state that this specific information is not specified in the available documentation.\n"
    "4. Every assertion containing technical parameters, dates, limits, or procedures MUST cite its source using bracketed numbers like [1] or [2].\n"
    "5. NO BIBLIOGRAPHY / NO REFERENCE SECTION: Do NOT generate a References, Bibliography, or Sources section at the end of your answer. "
    "Provenance is added automatically by the system. Use ONLY inline bracket citations [n] within your sentences — never list document titles, filenames, or quotes yourself.\n"
    "6. Do NOT hallucinate equipment tags, operating limits, numbers, or standards.\n"
    "7. Maintain refinery engineering rigor at all times."
)

TEMPLATES: Dict[PromptArchetype, PromptTemplate] = {
    PromptArchetype.EQUIPMENT_LOOKUP: PromptTemplate(
        archetype=PromptArchetype.EQUIPMENT_LOOKUP,
        system_instruction=(
            f"{_SYSTEM_PREAMBLE}\n\n"
            "OPERATING FOCUS: Equipment Specifications & Tag Attributes.\n"
            "Extract exact design operating pressures, temperatures, metallurgy, flow rates, and tag numbers."
        ),
        generation_instruction=(
            "Structure your answer with:\n"
            "1. Equipment Tag & Unit identification.\n"
            "2. Table or list of verified technical parameters with citations [n].\n"
            "3. Any critical operating boundaries or alarm setpoints."
        ),
    ),
    PromptArchetype.SOP_RETRIEVAL: PromptTemplate(
        archetype=PromptArchetype.SOP_RETRIEVAL,
        system_instruction=(
            f"{_SYSTEM_PREAMBLE}\n\n"
            "OPERATING FOCUS: Standard Operating Procedures (SOP).\n"
            "Provide step-by-step sequential operational compliance instructions."
        ),
        generation_instruction=(
            "Structure your answer with:\n"
            "1. Prerequisites and authorization permits required.\n"
            "2. Numbered step-by-step execution procedure with citations [n].\n"
            "3. Post-execution verification and restoration checks."
        ),
    ),
    PromptArchetype.MAINTENANCE: PromptTemplate(
        archetype=PromptArchetype.MAINTENANCE,
        system_instruction=(
            f"{_SYSTEM_PREAMBLE}\n\n"
            "OPERATING FOCUS: Preventive & Corrective Maintenance.\n"
            "Focus on inspection intervals, wear tolerances, lubrication schedules, and failure modes."
        ),
        generation_instruction=(
            "Structure your answer with:\n"
            "1. Maintenance scope and required isolation procedures.\n"
            "2. Inspection checkpoints, clearances, and replacement criteria with citations [n].\n"
            "3. Verification tests prior to recommissioning."
        ),
    ),
    PromptArchetype.SAFETY_COMPLIANCE: PromptTemplate(
        archetype=PromptArchetype.SAFETY_COMPLIANCE,
        system_instruction=(
            f"{_SYSTEM_PREAMBLE}\n\n"
            "OPERATING FOCUS: Refinery Safety Standards & Compliance.\n"
            "Adhere strictly to OISD, API, ASME, and PNGRB standards."
        ),
        generation_instruction=(
            "Structure your answer with:\n"
            "1. Governing standard citations (e.g. OISD-105, OISD-116) [n].\n"
            "2. Mandatory safety precautions, PPE, and isolation boundaries.\n"
            "3. Hazard mitigation protocols and emergency actions.\n"
            "If any requested statutory interval, thickness, or limit is not explicitly documented in the context, "
            "state that it is not specified in the available documentation. "
            "Do NOT append a References or Bibliography section."
        ),
    ),
    PromptArchetype.TROUBLESHOOTING: PromptTemplate(
        archetype=PromptArchetype.TROUBLESHOOTING,
        system_instruction=(
            f"{_SYSTEM_PREAMBLE}\n\n"
            "OPERATING FOCUS: Root Cause Analysis & Diagnostic Troubleshooting.\n"
            "Correlate observed symptoms with documented failure modes."
        ),
        generation_instruction=(
            "Structure your answer with:\n"
            "1. Potential root causes correlated with documented symptoms [n].\n"
            "2. Diagnostic checks to confirm the fault.\n"
            "3. Corrective remedies and preventive recommendations."
        ),
    ),
    PromptArchetype.COMPARISON: PromptTemplate(
        archetype=PromptArchetype.COMPARISON,
        system_instruction=(
            f"{_SYSTEM_PREAMBLE}\n\n"
            "OPERATING FOCUS: Comparative Engineering Analysis.\n"
            "Compare parameters, operational limits, or materials across different units or equipment."
        ),
        generation_instruction=(
            "Structure your answer with:\n"
            "1. Comparison summary table highlighting differences [n].\n"
            "2. Analysis of operational trade-offs and design variances."
        ),
    ),
    PromptArchetype.GENERAL_QA: PromptTemplate(
        archetype=PromptArchetype.GENERAL_QA,
        system_instruction=_SYSTEM_PREAMBLE,
        generation_instruction=(
            "Provide a concise, direct answer addressing only what was asked in the user query, based strictly on the verified context, "
            "citing source statements with [n]. Quote exact readings, dates, and limits directly from the context. "
            "If the user query asks about a specific parameter, interval, or requirement that is not documented in the context, "
            "explicitly state that this specific information is not specified in the available documentation. "
            "Do NOT volunteer disclaimers about topics not asked for. "
            "Do NOT append a References, Bibliography, or Sources section."
        ),
    ),
}


class PromptTemplateRegistry:
    """Thread-safe registry for domain prompt templates."""

    def __init__(self) -> None:
        self._templates: Dict[PromptArchetype, PromptTemplate] = dict(TEMPLATES)

    def get_template(self, archetype: PromptArchetype | str) -> PromptTemplate:
        """Retrieve a template by archetype, defaulting to GENERAL_QA."""
        if isinstance(archetype, str):
            try:
                archetype = PromptArchetype(archetype.lower().strip())
            except ValueError:
                archetype = PromptArchetype.GENERAL_QA
        return self._templates.get(archetype, self._templates[PromptArchetype.GENERAL_QA])

    def get(self, archetype: PromptArchetype | str) -> PromptTemplate:
        """Dict-like accessor for templates."""
        return self.get_template(archetype)

    def register(self, template: PromptTemplate) -> None:
        """Register or override a prompt template."""
        self._templates[template.archetype] = template


# Explicit alias for architectural conformity
PromptRegistry = PromptTemplateRegistry
