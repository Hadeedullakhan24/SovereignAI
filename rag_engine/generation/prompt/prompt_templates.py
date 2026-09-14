"""Prompt Templates — Domain Archetypes for Refinery Knowledge Operations.

Defines deterministic system prompts and instruction formats across refinery query
and document generation archetypes: Equipment Lookup, SOP Retrieval, Maintenance,
Safety Compliance, Troubleshooting, Comparison, General Engineering QA, Email Drafting,
Report Generation, Approval Notes, Summaries, Data Extraction, and Document Analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import re
from typing import Dict


class PromptArchetype(str, Enum):
    """Refinery query and task archetypes determining generation instructions."""

    EQUIPMENT_LOOKUP = "equipment_lookup"
    SOP_RETRIEVAL = "sop_retrieval"
    MAINTENANCE = "maintenance"
    SAFETY_COMPLIANCE = "safety_compliance"
    TROUBLESHOOTING = "troubleshooting"
    COMPARISON = "comparison"
    GENERAL_QA = "general_qa"
    EMAIL = "email"
    REPORT = "report"
    APPROVAL_NOTE = "approval_note"
    SUMMARY = "summary"
    EXTRACTION = "extraction"
    PROCEDURE = "procedure"
    ANALYSIS = "analysis"
    SPECIFICATION = "specification"


@dataclass(frozen=True)
class PromptTemplate:
    """Immutable template specification for prompt construction with cryptographic versioning."""

    archetype: PromptArchetype
    system_instruction: str
    generation_instruction: str
    template_version: str = "v1.1.0"
    name: str = ""
    author: str = "MRPL AI Engineering Team"
    creation_date: str = "2026-09-14"
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
    "Your objective is to provide precise, direct, well-structured, and strictly evidence-grounded answers.\n"
    "CRITICAL OPERATIONAL RULES:\n"
    "1. DIRECT ANSWER FIRST: Directly answer the user's core question in the opening sentence without fluff, conversational filler, or preambles.\n"
    "2. STRICT GROUNDING: Answer solely using the verified documentation provided in the CONTEXT section. Use only facts explicitly supported by retrieved evidence.\n"
    "3. SYNTHESIS & RELEVANCE: Synthesize and combine details across multiple relevant chunks when necessary. Do not simply copy or dump the first retrieved chunk. Avoid irrelevant retrieved material that does not directly address the query.\n"
    "4. STRUCTURE & HEADINGS: Use clear markdown headings, concise paragraphs, and bullet points (•) where appropriate for readability.\n"
    "5. NO INTERNAL RAG/ENGINE JARGON: Never mention embeddings, Qdrant, vector databases, BM25, RRF, reranking, chunks, retrieval latency, or internal execution metadata to the end user.\n"
    "6. TASK INTEGRITY & NO FALSE LABELS: Never call or label the response an 'Approval Note' unless the user explicitly requested to draft or generate an approval note.\n"
    "7. STATUS PRESERVATION & ACTION DISCIPLINE:\n"
    "   - Preserve exact recorded statuses: 'Action Required', 'Proposed Action', 'Recommended', 'Approved', 'Scheduled', 'Planned', 'Open', vs 'Completed'.\n"
    "   - NEVER convert or upgrade a required, proposed, recommended, or planned action into a completed action.\n"
    "   - Explicitly distinguish what documents establish from what they do not establish.\n"
    "8. NO CONTRADICTIONS: Never state that 'no action was recorded' if proposed, required, or open actions exist in the retrieved evidence.\n"
    "9. INCOMPLETE EVIDENCE: If the retrieved evidence does not contain sufficient details to answer a question or a specific parameter, clearly state that the available source material does not provide that information.\n"
    "10. CITATIONS & NO BIBLIOGRAPHY: In technical QA, summaries, or extraction, support statements with concise inline citations [n]. Never output a separate References, Bibliography, or Sources block at the end; provenance is rendered by the system interface.\n"
    "11. Maintain refinery engineering rigor at all times."
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
            "2. Direct verified technical parameters and operating limits with citations [n].\n"
            "3. If a requested parameter is not documented in the context, explicitly state that it is not specified in the available records.\n"
            "Do NOT append a References or Bibliography section."
        ),
    ),
    PromptArchetype.SPECIFICATION: PromptTemplate(
        archetype=PromptArchetype.SPECIFICATION,
        system_instruction=(
            f"{_SYSTEM_PREAMBLE}\n\n"
            "OPERATING FOCUS: Technical Specifications & Parameter Lookup.\n"
            "Extract exact design operating pressures, temperatures, metallurgy, flow rates, and tag numbers."
        ),
        generation_instruction=(
            "Structure your answer with:\n"
            "1. Direct verified parameter and unit (e.g. Design Pressure: 10.5 kg/cm²g) with citation [n].\n"
            "2. Supporting context from the source document (report number, equipment ID, inspection date).\n"
            "3. If the parameter is not documented in the retrieved records, explicitly state that it is not specified in the available documentation.\n"
            "Do NOT append a References or Bibliography section."
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
            "3. Post-execution verification and restoration checks.\n"
            "Do NOT append a References or Bibliography section."
        ),
    ),
    PromptArchetype.PROCEDURE: PromptTemplate(
        archetype=PromptArchetype.PROCEDURE,
        system_instruction=(
            f"{_SYSTEM_PREAMBLE}\n\n"
            "OPERATING FOCUS: Operational Procedures & Instructions.\n"
            "Provide step-by-step sequential instructions based solely on verified SOPs."
        ),
        generation_instruction=(
            "Structure your answer with:\n"
            "1. Prerequisites and required work permits [n].\n"
            "2. Numbered sequential operational steps.\n"
            "3. Safety warnings, interlocks, and control measures.\n"
            "Do NOT append a References or Bibliography section."
        ),
    ),
    PromptArchetype.MAINTENANCE: PromptTemplate(
        archetype=PromptArchetype.MAINTENANCE,
        system_instruction=(
            f"{_SYSTEM_PREAMBLE}\n\n"
            "OPERATING FOCUS: Preventive & Corrective Maintenance.\n"
            "Focus on inspection intervals, wear tolerances, lubrication schedules, and action statuses."
        ),
        generation_instruction=(
            "Structure your answer with:\n"
            "1. Equipment tag and direct summary of recorded actions with citations [n].\n"
            "2. Specific action details, conditions, and exact recorded status (e.g. required, proposed, approved, or completed).\n"
            "3. Clear distinction between what the documents establish vs what they do not establish (e.g. whether maintenance was completed).\n"
            "Do NOT append a References or Bibliography section."
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
            "2. Mandatory safety precautions, permits, PPE, and isolation boundaries specified in the context.\n"
            "3. Hazard mitigation protocols and emergency actions.\n"
            "If any requested statutory interval, thickness, or limit is not explicitly documented in the context, "
            "state that it is not specified in the available documentation.\n"
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
            "3. Corrective remedies and preventive recommendations.\n"
            "Do NOT append a References or Bibliography section."
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
            "1. A clean markdown comparison table highlighting differences and attributes across the compared items.\n"
            "2. Explanatory analysis of verified parameters and operating limits.\n"
            "3. If any parameter is not documented for an item, state 'Not specified in available documentation' in the table.\n"
            "Do NOT append a References or Bibliography section."
        ),
    ),
    PromptArchetype.EMAIL: PromptTemplate(
        archetype=PromptArchetype.EMAIL,
        system_instruction=(
            f"{_SYSTEM_PREAMBLE}\n\n"
            "OPERATING FOCUS: Professional Email Drafting.\n"
            "Draft a professional, well-structured business email based solely on retrieved documentation."
        ),
        generation_instruction=(
            "Draft a clean, professional email adhering strictly to retrieved facts:\n"
            "Subject: [Concise and informative subject line]\n\n"
            "Dear [Recipient / Team / Management],\n\n"
            "[Body paragraphs explaining background, equipment condition, and exact proposed/required scope]\n\n"
            "Best regards,\n"
            "[Engineering / Operations Team]\n\n"
            "EMAIL FORMATTING RULES:\n"
            "- Do NOT include bracket citations [n], bibliography, or debug tokens in the email body.\n"
            "- Do NOT claim files or attachments are attached unless confirmed in the evidence.\n"
            "- Preserve exact status: If the request asks for an email regarding a proposed action, draft it as requesting approval for the proposed action.\n"
            "- If the user asks to state that maintenance is completed, but the documents do NOT confirm completion, DO NOT falsely claim completion. State clearly in the email that the records document a proposed/required scope but completion is not established, and request confirmation."
        ),
    ),
    PromptArchetype.APPROVAL_NOTE: PromptTemplate(
        archetype=PromptArchetype.APPROVAL_NOTE,
        system_instruction=(
            f"{_SYSTEM_PREAMBLE}\n\n"
            "OPERATING FOCUS: Formal Approval & Recommendation Notes.\n"
            "Draft a formal approval/recommendation note with strict status preservation."
        ),
        generation_instruction=(
            "Format the response as a formal Approval / Recommendation Note:\n"
            "**Equipment:** [Equipment ID and Type]\n"
            "**Area / Unit:** [Plant Area / Unit]\n"
            "**Observed Condition / Issue:** [Issue identified in records]\n"
            "**Proposed Action:** [Proposed inspection or maintenance scope]\n"
            "**Required Decision / Approval:** [Specific scope requiring authorization]\n"
            "**Controls & Execution Window:** [Controls and agreed window]\n"
            "**Current Status:** [Status, e.g. Pending Approval / Proposed]\n\n"
            "Strictly maintain that this is a proposed action requiring approval and does NOT represent completed maintenance."
        ),
    ),
    PromptArchetype.SUMMARY: PromptTemplate(
        archetype=PromptArchetype.SUMMARY,
        system_instruction=(
            f"{_SYSTEM_PREAMBLE}\n\n"
            "OPERATING FOCUS: Executive & Technical Summaries.\n"
            "Provide a concise, structured summary synthesizing verified evidence."
        ),
        generation_instruction=(
            "Provide a structured summary with clear bullet points:\n"
            "- **Key Findings & Equipment Context:** [Summary of verified facts]\n"
            "- **Recorded Actions & Statuses:** [Action required, proposed, or approved, preserving status]\n"
            "- **Key Limits / Conditions:** [Operating limits, dates, parameters]\n"
            "- **Unestablished / Pending Information:** [What the documents do not establish, e.g. completion status]\n"
            "Do NOT append a References or Bibliography section."
        ),
    ),
    PromptArchetype.REPORT: PromptTemplate(
        archetype=PromptArchetype.REPORT,
        system_instruction=(
            f"{_SYSTEM_PREAMBLE}\n\n"
            "OPERATING FOCUS: Technical & Inspection Reports.\n"
            "Structure formal technical reports with executive summaries and findings."
        ),
        generation_instruction=(
            "Structure the report with appropriate markdown sections:\n"
            "# [Report Title]\n"
            "## 1. Executive Summary\n"
            "## 2. Technical Findings & Condition Assessment\n"
            "## 3. Proposed Actions & Controls (preserving exact source status)\n"
            "## 4. Conclusion & Recommendations\n"
            "Do NOT append a References or Bibliography section."
        ),
    ),
    PromptArchetype.EXTRACTION: PromptTemplate(
        archetype=PromptArchetype.EXTRACTION,
        system_instruction=(
            f"{_SYSTEM_PREAMBLE}\n\n"
            "OPERATING FOCUS: Structured Data Extraction & Template Requirements.\n"
            "Extract requested fields, parameters, or template requirements into clean structured bullet points or tables."
        ),
        generation_instruction=(
            "Structure your answer cleanly:\n"
            "1. State what fields or details are required by the specified template or document directly in the opening statement.\n"
            "2. Group fields under clear markdown subheadings (e.g. general fields, approval details, sections) using bullet points (•).\n"
            "3. Only include fields and details explicitly supported by the retrieved documentation.\n"
            "4. Do NOT label or format this as an 'Approval Note'.\n"
            "Do NOT append a References or Bibliography section."
        ),
    ),
    PromptArchetype.ANALYSIS: PromptTemplate(
        archetype=PromptArchetype.ANALYSIS,
        system_instruction=(
            f"{_SYSTEM_PREAMBLE}\n\n"
            "OPERATING FOCUS: Document & Evidence Analysis.\n"
            "Analyze what retrieved records establish versus what remains unresolved."
        ),
        generation_instruction=(
            "Structure your analysis with:\n"
            "- **Established Facts & Scope:** [What the documents explicitly confirm with citations [n]]\n"
            "- **Recorded Actions & Status:** [Action required, proposed, or approved]\n"
            "- **Unresolved / Missing Information:** [What the documents do not establish]\n"
            "Do NOT append a References or Bibliography section."
        ),
    ),
    PromptArchetype.GENERAL_QA: PromptTemplate(
        archetype=PromptArchetype.GENERAL_QA,
        system_instruction=_SYSTEM_PREAMBLE,
        generation_instruction=(
            "Provide a comprehensive, direct, and well-structured answer addressing the user query based strictly on all relevant retrieved evidence, "
            "citing source statements with inline citations [n].\n"
            "- Directly state the core answer in the opening sentence, synthesizing all relevant retrieved records.\n"
            "- When multiple documents provide relevant details (e.g. an alert email and an approval note), describe each recorded item, its scope, and its recorded status (e.g. 'Action Required' [1], 'Proposed Action' [2]).\n"
            "- Explicitly distinguish what the documents establish from what they do not establish (e.g. state what actions were required or proposed, while noting that the documents do not establish whether the action was completed).\n"
            "- Strictly preserve source status distinctions without internal contradictions.\n"
            "- Do NOT append a References, Bibliography, or Sources section."
        ),
    ),
}


def detect_task_type(query: str) -> PromptArchetype:
    """Classify user query into appropriate task archetype based on intent and keywords."""
    q_lower = query.lower().strip()

    # 1. Extraction / Required Fields / Template Details
    if any(k in q_lower for k in ["what fields", "list all fields", "extract the following", "required fields", "fields are required", "fields required", "template requires", "in the template", "template fields", "approval note template"]):
        return PromptArchetype.EXTRACTION

    # 2. Email detection
    if any(k in q_lower for k in ["draft an email", "write an email", "compose an email", "send an email", "draft email"]):
        return PromptArchetype.EMAIL

    # 3. Approval Note Drafting / Generation (only if explicitly asked to draft or create)
    if any(k in q_lower for k in ["draft an approval note", "draft approval note", "prepare an approval note", "generate an approval note", "create an approval note", "write an approval note", "draft recommendation note"]):
        return PromptArchetype.APPROVAL_NOTE

    # 4. Report
    if any(k in q_lower for k in ["prepare a report", "inspection report", "maintenance report", "generate a report", "write a report", "technical report"]):
        return PromptArchetype.REPORT

    # 5. Summary
    if any(k in q_lower for k in ["summarize", "summary", "short summary", "brief summary", "overview"]):
        return PromptArchetype.SUMMARY

    # 6. Comparison
    if any(k in q_lower for k in ["compare", "comparison", "differences between", "versus", " vs "]):
        return PromptArchetype.COMPARISON

    # 7. Safety / Compliance
    if any(k in q_lower for k in ["safety requirements", "safety precautions", "hot work", "work permit", "ppe", "loto", "oisd-std-", "oisd", "safety standard", "statutory requirement"]):
        return PromptArchetype.SAFETY_COMPLIANCE

    # 8. Procedure / SOP
    if any(k in q_lower for k in ["procedure", "sop", "steps to", "how to perform", "operational steps", "instructions for"]):
        return PromptArchetype.SOP_RETRIEVAL

    # 9. Technical Specifications / Equipment Lookup
    if any(k in q_lower for k in ["design pressure", "design temperature", "operating pressure", "operating temperature", "rated capacity", "flow rate", "metallurgy", "specs of", "specification", "knockout drum"]):
        return PromptArchetype.SPECIFICATION

    # 10. Maintenance / Inspection
    if any(k in q_lower for k in ["maintenance action", "maintenance recorded", "inspection action", "lubrication schedule", "overhaul"]):
        return PromptArchetype.MAINTENANCE

    # 11. Troubleshooting / RCA
    if any(k in q_lower for k in ["troubleshoot", "root cause", "failure mode", "diagnostic check", "abnormal vibration"]):
        return PromptArchetype.TROUBLESHOOTING

    # 12. Document Analysis
    if any(k in q_lower for k in ["analyze this", "review the inspection", "what does this document establish", "document analysis"]):
        return PromptArchetype.ANALYSIS

    return PromptArchetype.GENERAL_QA


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
