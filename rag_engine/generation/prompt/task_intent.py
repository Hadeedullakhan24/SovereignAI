"""Task Intent & Operation Classifier.

Provides deterministic, fine-grained classification of user requests into:
1. Requested Operation (Answer, Summarize, Compare, Extract, Explain, List, Calculate,
   Draft Email, Draft Approval Request, Draft Notification, Draft Action Request,
   Draft Recommendation, Generate Report, Generate Briefing, Generate Procedure, Generate Checklist)
2. Output Format (Direct QA, Email, Report, Approval Note, Procedure, Summary, Checklist, Table, List, Extraction)
3. Email Purpose (Summary, Approval Request, Confirmation Request, Action Request, Notification, Recommendation, General)
4. Target Recipient Extraction (e.g. "Site Safety Team", "Maintenance Manager")
5. Subject & Topic Extraction
6. Explicit Entity Extraction (Equipment Tags, Units, Standards)
7. Scope Discipline (General Topic vs Entity-Specific)
8. Strict Grounding Mode Detection ("Use only provided documentation", etc.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Optional, Sequence

from rag_engine.generation.prompt.prompt_templates import PromptArchetype
from rag_engine.retrieval.retrieval_utils import EQUIPMENT_TAG_REGEX, normalize_whitespace


class TaskOperation(str, Enum):
    """User-requested operational task."""

    ANSWER = "answer"
    SUMMARIZE = "summarize"
    COMPARE = "compare"
    EXTRACT = "extract"
    EXPLAIN = "explain"
    LIST = "list"
    CALCULATE = "calculate"
    DRAFT_EMAIL = "draft_email"
    DRAFT_APPROVAL_REQUEST = "draft_approval_request"
    DRAFT_NOTIFICATION = "draft_notification"
    DRAFT_ACTION_REQUEST = "draft_action_request"
    DRAFT_RECOMMENDATION = "draft_recommendation"
    GENERATE_REPORT = "generate_report"
    GENERATE_BRIEFING = "generate_briefing"
    GENERATE_PROCEDURE = "generate_procedure"
    GENERATE_CHECKLIST = "generate_checklist"
    GENERATE_DOCUMENT = "generate_document"
    GENERATE_IMAGE = "generate_image"


class ArtifactFormat(str, Enum):
    """Target output file artifact format."""

    PDF = "pdf"
    DOCX = "docx"
    XLSX = "xlsx"
    PPTX = "pptx"
    PNG = "png"


class OutputFormat(str, Enum):
    """Output structural presentation format."""

    DIRECT_QA = "direct_qa"
    EMAIL = "email"
    REPORT = "report"
    APPROVAL_NOTE = "approval_note"
    PROCEDURE = "procedure"
    SUMMARY = "summary"
    CHECKLIST = "checklist"
    TABLE = "table"
    LIST = "list"
    EXTRACTION = "extraction"
    BRIEFING = "briefing"
    ARTIFACT = "artifact"
    IMAGE = "image"


class EmailPurpose(str, Enum):
    """Specific communicative purpose for email generation."""

    NONE = "none"
    SUMMARY = "summary"
    APPROVAL_REQUEST = "approval_request"
    CONFIRMATION_REQUEST = "confirmation_request"
    ACTION_REQUEST = "action_request"
    NOTIFICATION = "notification"
    RECOMMENDATION = "recommendation"
    GENERAL = "general"


@dataclass(frozen=True)
class TaskIntent:
    """Structured classification of user intent and task constraints."""

    operation: TaskOperation
    output_format: OutputFormat
    email_purpose: EmailPurpose = EmailPurpose.NONE
    recipient: Optional[str] = None
    subject_topic: str = ""
    explicit_entities: list[str] = field(default_factory=list)
    is_general_query: bool = True
    strict_grounding: bool = False
    archetype: PromptArchetype = PromptArchetype.GENERAL_QA
    artifact_format: Optional[ArtifactFormat] = None
    raw_query: str = ""

    @property
    def is_artifact_request(self) -> bool:
        """True if the user explicitly requested a downloadable document or image artifact."""
        return (
            self.artifact_format is not None
            or self.operation in (TaskOperation.GENERATE_DOCUMENT, TaskOperation.GENERATE_IMAGE)
            or self.output_format in (OutputFormat.ARTIFACT, OutputFormat.IMAGE)
        )

    def get_directive_instructions(self) -> str:
        """Generate explicit, enforceable instructions tailored to the classified intent."""
        lines: list[str] = []

        if self.output_format == OutputFormat.EMAIL:
            lines.append("TASK OBJECTIVE: DRAFT A PROFESSIONAL EMAIL")
            salutation_target = self.recipient or "Team"
            lines.append(f"- Salutation: Use 'Dear {salutation_target},' (do NOT invent unrelated managers or personas).")

            if self.email_purpose == EmailPurpose.SUMMARY:
                subj = f"{self.subject_topic} — Summary" if self.subject_topic else "Summary of Requirements"
                lines.append(f"- Subject Line: 'Subject: {subj}' (reflects a summary, NOT a request for approval or confirmation).")
                lines.append("- Primary Purpose: Clearly communicate verified, documented information.")
                lines.append("- STRICT TASK DISCIPLINE:")
                lines.append("  * DO NOT ask the recipient to confirm, approve, inspect, verify, complete, arrange, or provide documents.")
                lines.append("  * DO NOT use phrases like 'We require confirmation', 'Please confirm', 'Immediate attention is required', 'Please arrange these checks', or 'Please provide additional approvals'.")
                lines.append("  * Communicate the documented requirements as informational points.")

            elif self.email_purpose == EmailPurpose.APPROVAL_REQUEST:
                subj = f"Request for Approval — {self.subject_topic}" if self.subject_topic else "Request for Approval"
                lines.append(f"- Subject Line: 'Subject: {subj}'")
                lines.append("- Primary Purpose: Request formal authorization/approval for the documented proposed scope.")

            elif self.email_purpose == EmailPurpose.CONFIRMATION_REQUEST:
                subj = f"Request for Confirmation — {self.subject_topic}" if self.subject_topic else "Request for Confirmation"
                lines.append(f"- Subject Line: 'Subject: {subj}'")
                lines.append("- Primary Purpose: Request confirmation regarding documented records or status.")

            elif self.email_purpose == EmailPurpose.NOTIFICATION:
                subj = f"Notification — {self.subject_topic}" if self.subject_topic else "Notification"
                lines.append(f"- Subject Line: 'Subject: {subj}'")
                lines.append("- Primary Purpose: Inform recipients about documented facts, schedules, or events.")
                lines.append("- DO NOT ask for approval or immediate action unless explicitly requested.")

            elif self.email_purpose == EmailPurpose.ACTION_REQUEST:
                subj = f"Action Required — {self.subject_topic}" if self.subject_topic else "Action Required"
                lines.append(f"- Subject Line: 'Subject: {subj}'")
                lines.append("- Primary Purpose: Request recipients to take the specified documented action.")

            else:
                subj = self.subject_topic or "Refinery Operations Update"
                lines.append(f"- Subject Line: 'Subject: {subj}'")

            lines.append("- Formatting: Do NOT include inline bracket citations [n], references sections, or attachment claims.")

        elif self.output_format == OutputFormat.SUMMARY or self.operation == TaskOperation.SUMMARIZE:
            lines.append("TASK OBJECTIVE: GENERATE A CONCISE, STRUCTURED SUMMARY")
            lines.append("- Primary Purpose: Synthesize verified facts directly answering the query.")
            lines.append("- DO NOT convert a summary request into an action request, confirmation request, or approval note.")

        elif self.output_format == OutputFormat.APPROVAL_NOTE or self.operation == TaskOperation.DRAFT_APPROVAL_REQUEST:
            lines.append("TASK OBJECTIVE: DRAFT A FORMAL APPROVAL / RECOMMENDATION NOTE")

        elif self.output_format == OutputFormat.REPORT or self.operation == TaskOperation.GENERATE_REPORT:
            lines.append("TASK OBJECTIVE: GENERATE A FORMAL TECHNICAL REPORT")

        elif self.output_format == OutputFormat.PROCEDURE or self.operation == TaskOperation.GENERATE_PROCEDURE:
            lines.append("TASK OBJECTIVE: GENERATE STEP-BY-STEP OPERATIONAL PROCEDURE")

        elif self.output_format == OutputFormat.CHECKLIST or self.operation == TaskOperation.GENERATE_CHECKLIST:
            lines.append("TASK OBJECTIVE: GENERATE A PRACTICAL STEP-BY-STEP CHECKLIST")

        elif self.output_format == OutputFormat.TABLE or self.operation == TaskOperation.COMPARE:
            lines.append("TASK OBJECTIVE: GENERATE A STRUCTURED COMPARISON TABLE")

        elif self.output_format == OutputFormat.EXTRACTION or self.operation == TaskOperation.EXTRACT:
            lines.append("TASK OBJECTIVE: EXTRACT REQUESTED FIELDS AND DATA INTO CLEAN BULLET POINTS")

        if self.is_artifact_request:
            fmt_str = self.artifact_format.value.upper() if self.artifact_format else "DOCUMENT"
            lines.append(f"DOCUMENT ARTIFACT OBJECTIVE: GENERATE GROUNDED STRUCTURED CONTENT FOR {fmt_str}")
            lines.append("- Present a clear document title, structured section headings, factual body paragraphs, bullet points, and source references.")
            lines.append("- Only report facts, findings, and parameters explicitly supported by the retrieved documentation.")
            lines.append("- If specific maintenance actions, inspection details, or parameters are not documented for the equipment, explicitly state: 'Not documented in available records.'")
            lines.append("- Do NOT invent unrecorded maintenance history, hypothetical condition assessments, or unverified numbers.")
            lines.append("- Do NOT output simulated web URLs, fake download links (e.g. example.com), or dummy attachment claims.")
            lines.append("- The physical file artifact will be compiled and validated directly by the system tool executor.")

        # Entity scope discipline
        if self.is_general_query:
            lines.append("- SCOPE DISCIPLINE: The user asked a general question without specifying equipment tags.")
            lines.append("  * DO NOT narrow the subject, title, salutation, or core summary to a specific equipment tag (e.g. E-330, C-118, P-203) that merely appears in retrieved context.")
            lines.append("  * Present the general requirements/facts applicable to the requested topic.")
        else:
            tags_str = ", ".join(self.explicit_entities)
            lines.append(f"- SCOPE DISCIPLINE: Focus specifically on the requested equipment: {tags_str}.")

        # Strict grounding mode
        if self.strict_grounding:
            lines.append("- STRICT GROUNDING MODE ACTIVE: Use ONLY facts explicitly supported by the provided documentation.")
            lines.append("  * DO NOT add generic outside industry knowledge or unverified safety filler.")
            lines.append("  * If details are missing from records, state that they are not specified in the available documentation.")

        # Modality preservation
        lines.append("- MODALITY PRESERVATION: Preserve exact obligation levels from sources (do not upgrade 'should' or 'recommended' to 'must').")

        return "\n".join(lines)


class TaskIntentClassifier:
    """Classifies user queries into structured operational intent, output format, and constraints."""

    _STRICT_GROUNDING_PATTERNS = [
        re.compile(r"\buse\s+only\s+(?:the\s+)?(?:provided\s+)?(?:documentation|documents|records|context)\b", re.IGNORECASE),
        re.compile(r"\bbased\s+only\s+on\s+(?:the\s+)?(?:provided\s+)?(?:documentation|documents|records|context)\b", re.IGNORECASE),
        re.compile(r"\baccording\s+to\s+(?:the\s+)?(?:provided\s+)?(?:documents|documentation|records)\b", re.IGNORECASE),
        re.compile(r"\b(?:don'?t|do\s+not)\s+use\s+outside\s+information\b", re.IGNORECASE),
        re.compile(r"\b(?:don'?t|do\s+not)\s+invent\b", re.IGNORECASE),
        re.compile(r"\bstrictly\s+from\s+(?:the\s+)?(?:provided\s+)?(?:documents|documentation|records)\b", re.IGNORECASE),
        re.compile(r"\bstrictly\s+based\s+on\b", re.IGNORECASE),
    ]

    _DOC_ACTION_VERBS = r"(?:create|generate|export|prepare|save|output|produce|build|draft|write|download|make)"

    _PDF_GEN_PATTERNS = [
        re.compile(rf"\b{_DOC_ACTION_VERBS}\b.*?\b(?:a|an|the|as|into)?\s*(?:actual\s+)?pdf\b(?:\s+(?:file|document|report|artifact))?", re.IGNORECASE),
        re.compile(r"\bpdf\s+(?:file|document|report|artifact|export)\b", re.IGNORECASE),
        re.compile(r"\bexport\s+(?:as|to)?\s*pdf\b", re.IGNORECASE),
        re.compile(r"\bas\s+(?:an?\s+)?(?:actual\s+)?pdf\s+artifact\b", re.IGNORECASE),
        re.compile(r"\bdownloadable\s+(?:pdf|report)\b", re.IGNORECASE),
    ]

    _DOCX_GEN_PATTERNS = [
        re.compile(rf"\b{_DOC_ACTION_VERBS}\b.*?\b(?:a|an|the|as|into)?\s*(?:actual\s+)?(?:word\s+doc(?:ument)?|docx)\b(?:\s+(?:file|document|report|artifact))?", re.IGNORECASE),
        re.compile(r"\b(?:docx|word\s+document)\s+(?:file|document|report|artifact|export)\b", re.IGNORECASE),
        re.compile(r"\bexport\s+(?:as|to)?\s*(?:docx|word)\b", re.IGNORECASE),
        re.compile(r"\bas\s+(?:an?\s+)?(?:actual\s+)?(?:docx|word)\s+artifact\b", re.IGNORECASE),
    ]

    _XLSX_GEN_PATTERNS = [
        re.compile(rf"\b{_DOC_ACTION_VERBS}\b.*?\b(?:a|an|the|as|into)?\s*(?:actual\s+)?(?:excel|spreadsheet|xlsx)\b(?:\s+(?:file|sheet|spreadsheet|table|artifact))?", re.IGNORECASE),
        re.compile(r"\b(?:xlsx|excel\s+(?:sheet|spreadsheet))\s+(?:file|sheet|spreadsheet|artifact|export)\b", re.IGNORECASE),
        re.compile(r"\bexport\s+(?:as|to)?\s*(?:xlsx|excel)\b", re.IGNORECASE),
        re.compile(r"\bas\s+(?:an?\s+)?(?:actual\s+)?(?:xlsx|excel)\s+artifact\b", re.IGNORECASE),
    ]

    _PPTX_GEN_PATTERNS = [
        re.compile(rf"\b{_DOC_ACTION_VERBS}\b.*?\b(?:a|an|the|as|into)?\s*(?:actual\s+)?(?:powerpoint|presentation|pptx|slide\s+deck)\b(?:\s+(?:file|presentation|deck|artifact))?", re.IGNORECASE),
        re.compile(r"\b(?:pptx|powerpoint)\s+(?:file|presentation|artifact|export)\b", re.IGNORECASE),
        re.compile(r"\bexport\s+(?:as|to)?\s*(?:pptx|powerpoint)\b", re.IGNORECASE),
        re.compile(r"\bas\s+(?:an?\s+)?(?:actual\s+)?(?:pptx|powerpoint)\s+artifact\b", re.IGNORECASE),
    ]

    _IMAGE_GEN_PATTERNS = [
        re.compile(r"\b(?:generate|create|make|render|draw|produce|synthesize|output|build|paint|sketch|illustrate)\b.*?\b(?:an?\s+)?(?:image|picture|photo|photograph|illustration|diagram|rendering|graphic|visual\s+representation|visual|schematic\s+image)\b", re.IGNORECASE),
        re.compile(r"\b(?:text-to-image|txt2img|stable\s*diffusion|diffusion\s*image|diffusion\s*model)\b", re.IGNORECASE),
        re.compile(r"\b(?:visual\s+representation\s+of)\b", re.IGNORECASE),
    ]

    _RECIPIENT_PATTERNS = [
        re.compile(r"\b(?:for|to|addressed to|send to)\s+(?:the\s+)?([A-Za-z0-9\s\-]+?)(?:\.|\,|$|\s+(?:using|use\s+only|based\s+on|according\s+to|strictly))", re.IGNORECASE),
    ]

    _EXCLUDED_RECIPIENTS = {
        "the following", "each", "all", "clarity", "details", "review", "reference",
        "documentation", "records", "context", "safe operation", "compliance",
    }

    @classmethod
    def classify(cls, query: str) -> TaskIntent:
        """Analyze and classify a user query into structured TaskIntent."""
        raw = query or ""
        q_clean = normalize_whitespace(raw)
        q_lower = q_clean.lower()

        # 1. Strict Grounding Mode Detection
        strict_grounding = any(pat.search(q_clean) for pat in cls._STRICT_GROUNDING_PATTERNS)

        # 2. Extract Explicit Equipment Entities
        excluded_prefixes = {
            "OISD", "API", "ASME", "PNGRB", "ISO", "REV", "SEC", "PG", "FIG",
            "LINE", "L", "STD", "DOC", "SPEC", "FORM", "INSP", "REPORT", "ANNEXURE",
        }
        equip_matches = []
        for m in EQUIPMENT_TAG_REGEX.finditer(q_clean):
            prefix = m.group(1).upper()
            num = m.group(2).upper()
            tag = f"{prefix}-{num}"
            if prefix not in excluded_prefixes and (len(prefix) > 1 or len(num) >= 3):
                equip_matches.append(tag)
        explicit_entities = sorted(list(set(equip_matches)))
        is_general_query = len(explicit_entities) == 0

        # 3. Detect Explicit Document / Image Artifact Generation Request
        detected_artifact: Optional[ArtifactFormat] = None
        if any(pat.search(q_clean) for pat in cls._IMAGE_GEN_PATTERNS):
            detected_artifact = ArtifactFormat.PNG
        elif any(pat.search(q_clean) for pat in cls._PDF_GEN_PATTERNS):
            detected_artifact = ArtifactFormat.PDF
        elif any(pat.search(q_clean) for pat in cls._DOCX_GEN_PATTERNS):
            detected_artifact = ArtifactFormat.DOCX
        elif any(pat.search(q_clean) for pat in cls._XLSX_GEN_PATTERNS):
            detected_artifact = ArtifactFormat.XLSX
        elif any(pat.search(q_clean) for pat in cls._PPTX_GEN_PATTERNS):
            detected_artifact = ArtifactFormat.PPTX

        # 4. Detect Format & Operation
        is_email = any(k in q_lower for k in [
            "draft an email", "draft email", "write an email", "write email",
            "compose an email", "compose email", "send an email", "send email", "prepare an email",
        ])

        is_approval_note = any(k in q_lower for k in [
            "draft an approval note", "draft approval note", "prepare an approval note",
            "generate an approval note", "create an approval note", "write an approval note",
            "draft recommendation note", "prepare recommendation note",
        ])

        is_report = any(k in q_lower for k in [
            "prepare a report", "generate a report", "write a report", "inspection report",
            "maintenance report", "technical report", "create a report",
        ])

        is_checklist = any(k in q_lower for k in [
            "generate a checklist", "prepare a checklist", "create a checklist",
            "make a checklist", "checklist for",
        ])

        is_procedure = any(k in q_lower for k in [
            "generate a procedure", "step-by-step procedure", "sop for", "procedure for",
            "operational procedure", "standard operating procedure",
        ])

        is_comparison = any(k in q_lower for k in [
            "compare", "comparison", "differences between", "versus", " vs ",
        ])

        is_extraction = any(k in q_lower for k in [
            "what fields", "list all fields", "extract the following", "required fields",
            "fields are required", "fields required", "template requires", "in the template",
            "template fields", "approval note template",
        ])

        # 5. Extract Recipient and Subject Topic
        recipient = None
        email_purpose = EmailPurpose.NONE
        subject_topic = cls._extract_topic(q_clean)

        # 6. Branch for Explicit Artifact Generation vs Text Tasks
        if detected_artifact is not None:
            if detected_artifact == ArtifactFormat.PNG:
                output_format = OutputFormat.IMAGE
                operation = TaskOperation.GENERATE_IMAGE
                archetype = PromptArchetype.GENERAL_QA
            else:
                output_format = OutputFormat.ARTIFACT
                operation = TaskOperation.GENERATE_DOCUMENT

                if is_email:
                    archetype = PromptArchetype.EMAIL
                    recipient = cls._extract_recipient(q_clean)
                elif is_approval_note:
                    archetype = PromptArchetype.APPROVAL_NOTE
                elif is_comparison:
                    archetype = PromptArchetype.COMPARISON
                elif any(k in q_lower for k in ["safety requirements", "safety precautions", "hot work", "work permit", "ppe", "loto", "oisd"]):
                    archetype = PromptArchetype.SAFETY_COMPLIANCE
                elif is_procedure or any(k in q_lower for k in ["sop", "procedure", "step-by-step"]):
                    archetype = PromptArchetype.SOP_RETRIEVAL
                elif any(k in q_lower for k in ["summarize", "summary", "overview"]):
                    archetype = PromptArchetype.SUMMARY
                elif any(k in q_lower for k in ["design pressure", "operating pressure", "specs of", "specification", "knockout drum"]):
                    archetype = PromptArchetype.SPECIFICATION
                elif any(k in q_lower for k in ["troubleshoot", "root cause", "failure mode"]):
                    archetype = PromptArchetype.TROUBLESHOOTING
                else:
                    archetype = PromptArchetype.REPORT

            return TaskIntent(
                operation=operation,
                output_format=output_format,
                email_purpose=email_purpose,
                recipient=recipient,
                subject_topic=subject_topic,
                explicit_entities=explicit_entities,
                is_general_query=is_general_query,
                strict_grounding=strict_grounding,
                archetype=archetype,
                artifact_format=detected_artifact,
                raw_query=raw,
            )

        if is_email:
            output_format = OutputFormat.EMAIL
            recipient = cls._extract_recipient(q_clean)

            if any(k in q_lower for k in ["summariz", "summary", "overview"]):
                operation = TaskOperation.SUMMARIZE
                email_purpose = EmailPurpose.SUMMARY
                archetype = PromptArchetype.EMAIL
            elif any(k in q_lower for k in ["requesting approval", "request approval", "ask for approval", "seek approval", "obtain approval"]):
                operation = TaskOperation.DRAFT_APPROVAL_REQUEST
                email_purpose = EmailPurpose.APPROVAL_REQUEST
                archetype = PromptArchetype.EMAIL
            elif any(k in q_lower for k in ["requesting confirmation", "request confirmation", "ask to confirm", "confirming"]):
                operation = TaskOperation.DRAFT_EMAIL
                email_purpose = EmailPurpose.CONFIRMATION_REQUEST
                archetype = PromptArchetype.EMAIL
            elif any(k in q_lower for k in ["notifying", "notify", "inform", "notification"]):
                operation = TaskOperation.DRAFT_NOTIFICATION
                email_purpose = EmailPurpose.NOTIFICATION
                archetype = PromptArchetype.EMAIL
            elif any(k in q_lower for k in ["take action", "action required", "requesting action", "ask the team to"]):
                operation = TaskOperation.DRAFT_ACTION_REQUEST
                email_purpose = EmailPurpose.ACTION_REQUEST
                archetype = PromptArchetype.EMAIL
            elif any(k in q_lower for k in ["recommending", "recommendation"]):
                operation = TaskOperation.DRAFT_RECOMMENDATION
                email_purpose = EmailPurpose.RECOMMENDATION
                archetype = PromptArchetype.EMAIL
            else:
                operation = TaskOperation.DRAFT_EMAIL
                email_purpose = EmailPurpose.GENERAL
                archetype = PromptArchetype.EMAIL

        elif is_approval_note:
            output_format = OutputFormat.APPROVAL_NOTE
            operation = TaskOperation.DRAFT_APPROVAL_REQUEST
            archetype = PromptArchetype.APPROVAL_NOTE

        elif is_report:
            output_format = OutputFormat.REPORT
            operation = TaskOperation.GENERATE_REPORT
            archetype = PromptArchetype.REPORT

        elif is_checklist:
            output_format = OutputFormat.CHECKLIST
            operation = TaskOperation.GENERATE_CHECKLIST
            archetype = PromptArchetype.PROCEDURE

        elif is_procedure:
            output_format = OutputFormat.PROCEDURE
            operation = TaskOperation.GENERATE_PROCEDURE
            archetype = PromptArchetype.SOP_RETRIEVAL

        elif is_comparison:
            output_format = OutputFormat.TABLE
            operation = TaskOperation.COMPARE
            archetype = PromptArchetype.COMPARISON

        elif is_extraction:
            output_format = OutputFormat.EXTRACTION
            operation = TaskOperation.EXTRACT
            archetype = PromptArchetype.EXTRACTION

        elif any(k in q_lower for k in ["summarize", "summary", "brief overview", "short summary"]):
            output_format = OutputFormat.SUMMARY
            operation = TaskOperation.SUMMARIZE
            archetype = PromptArchetype.SUMMARY

        elif any(k in q_lower for k in ["list", "give a list", "enumerate"]):
            output_format = OutputFormat.LIST
            operation = TaskOperation.LIST
            archetype = PromptArchetype.GENERAL_QA

        elif any(k in q_lower for k in ["explain", "how does", "why does", "describe"]):
            output_format = OutputFormat.DIRECT_QA
            operation = TaskOperation.EXPLAIN
            archetype = PromptArchetype.GENERAL_QA

        elif any(k in q_lower for k in ["calculate", "compute", "calculate the"]):
            output_format = OutputFormat.DIRECT_QA
            operation = TaskOperation.CALCULATE
            archetype = PromptArchetype.GENERAL_QA

        elif any(k in q_lower for k in ["safety requirements", "safety precautions", "hot work", "work permit", "ppe", "loto", "oisd"]):
            output_format = OutputFormat.DIRECT_QA
            operation = TaskOperation.ANSWER
            archetype = PromptArchetype.SAFETY_COMPLIANCE

        elif any(k in q_lower for k in ["design pressure", "operating pressure", "specs of", "specification", "knockout drum"]):
            output_format = OutputFormat.DIRECT_QA
            operation = TaskOperation.ANSWER
            archetype = PromptArchetype.SPECIFICATION

        elif any(k in q_lower for k in ["maintenance action", "maintenance recorded", "inspection action"]):
            output_format = OutputFormat.DIRECT_QA
            operation = TaskOperation.ANSWER
            archetype = PromptArchetype.MAINTENANCE

        elif any(k in q_lower for k in ["troubleshoot", "root cause", "failure mode"]):
            output_format = OutputFormat.DIRECT_QA
            operation = TaskOperation.ANSWER
            archetype = PromptArchetype.TROUBLESHOOTING

        else:
            output_format = OutputFormat.DIRECT_QA
            operation = TaskOperation.ANSWER
            archetype = PromptArchetype.GENERAL_QA

        return TaskIntent(
            operation=operation,
            output_format=output_format,
            email_purpose=email_purpose,
            recipient=recipient,
            subject_topic=subject_topic,
            explicit_entities=explicit_entities,
            is_general_query=is_general_query,
            strict_grounding=strict_grounding,
            archetype=archetype,
            raw_query=raw,
        )

    _RECIPIENT_PATTERNS = [
        re.compile(r"\b(?:for|to)\s+(?:the\s+)?((?:[A-Za-z0-9\-]+\s+){0,4}(?:team|manager|personnel|operators?|engineers?|department|staff|crew|group|officers?|in-charge))\b", re.IGNORECASE),
        re.compile(r"\baddressed\s+to\s+(?:the\s+)?((?:[A-Za-z0-9\-]+\s+){0,4}[A-Za-z0-9\-]+)(?:\.|\,|$|\s+(?:using|use\s+only|based\s+on|according\s+to|strictly))", re.IGNORECASE),
        re.compile(r"\bnotifying\s+(?:the\s+)?((?:[A-Za-z0-9\-]+\s+){0,4}[A-Za-z0-9\-]+)(?:\.|\,|$|\s+(?:about|regarding|of|on|that|using|use\s+only|based\s+on))", re.IGNORECASE),
        re.compile(r"\basking\s+(?:the\s+)?((?:[A-Za-z0-9\-]+\s+){0,4}[A-Za-z0-9\-]+)\s+to\b", re.IGNORECASE),
        re.compile(r"\bto\s+(?:the\s+)?([A-Z][a-zA-Z0-9\-]+(?:\s+[A-Z][a-zA-Z0-9\-]+){0,3})(?:\.|\,|$|\s+(?:using|use\s+only|based\s+on|according\s+to|strictly|regarding|about|on))", re.IGNORECASE),
    ]

    _EXCLUDED_RECIPIENTS = {
        "the following", "each", "all", "clarity", "details", "review", "reference",
        "documentation", "records", "context", "safe operation", "compliance",
        "hot work", "cold work", "inspection", "maintenance", "isolation",
    }

    @classmethod
    def _extract_recipient(cls, query: str) -> Optional[str]:
        """Extract designated recipient from query (e.g. 'for the site safety team' -> 'Site Safety Team')."""
        for pat in cls._RECIPIENT_PATTERNS:
            m = pat.search(query)
            if m:
                raw_rec = m.group(1).strip()
                clean_rec = re.sub(r"[,\.]+$", "", raw_rec).strip()
                if clean_rec.lower() not in cls._EXCLUDED_RECIPIENTS and len(clean_rec) >= 3:
                    words = clean_rec.split()
                    if len(words) <= 5:
                        return clean_rec.title()
        return None

    @classmethod
    def _extract_topic(cls, query: str) -> str:
        """Extract core subject topic from natural language query."""
        text = query.strip()

        # Strip common document creation / artifact generation prefixes
        text = re.sub(
            r"^(?:create|generate|prepare|export|draft|write|output|build|produce)\s+(?:an?\s+)?(?:actual\s+|professional\s+|downloadable\s+)?(?:pdf|docx|xlsx|pptx|word|excel|spreadsheet|presentation|powerpoint|slide\s+deck|report|file|document|table|artifact|summary|briefing|checklist|email)?\s*(?:report|file|document|presentation|spreadsheet|summary|table|artifact)?\s*(?:containing|summarizing|comparing|detailing|describing|covering|on|about|of|for)?\s*(?:a\s+concise\s+summary\s+of\s+)?(?:the\s+documented\s+|the\s+available\s+|the\s+)?",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()

        # Strip email drafting prefixes
        text = re.sub(
            r"^(?:draft\s+(?:an?\s+)?email\s+(?:summarizing|requesting\s+approval\s+(?:for|of)|requesting\s+confirmation\s+(?:for|of)|notifying\s+(?:the\s+team\s+about|about)|regarding|on)|summarize|write\s+(?:an?\s+)?email\s+(?:about|on|summarizing)|prepare\s+(?:an?\s+)?report\s+on)\s+",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()

        # Strip leading "the"
        text = re.sub(r"^the\s+", "", text, flags=re.IGNORECASE).strip()

        # Strip trailing instructions, constraints, and download demands
        text = re.sub(r"\s+(?:after\s+creating|after\s+generating|generate\s+and\s+return|and\s+return\s+the\s+real|generate\s+the\s+report\s+as|return\s+the\s+generated\s+file|so\s+the\s+user\s+can\s+download).*", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"\s+(?:using|use\s+only|based\s+on|according\s+to|strictly|include\s+the|include\s+source|include\s+a\s+title|do\s+not\s+use|keep\s+the\s+two).*", "", text, flags=re.IGNORECASE).strip()

        # Clean punctuation
        text = re.sub(r"[\?\.\!]+$", "", text).strip()

        if not text:
            return "Engineering & Compliance Report"

        words = text.split()
        if len(words) <= 8:
            return " ".join(w.capitalize() for w in words)
        return text[:50].title()


# Alias for backward compatibility and clean API
TaskClassifier = TaskIntentClassifier
