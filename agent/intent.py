"""Central Authoritative Intent Classification for Sovereign AI.

Determines the user's intended:
1. ACTION: create/generate/produce/depict vs analyze/inspect/read/extract vs retrieve/answer vs calculate vs code
2. OUTPUT MODALITY: newly synthesized visual vs analysis of existing visual vs document file vs textual answer vs calculation vs code
3. TARGET STATE: new synthesis vs existing artifact vs document corpus

Provides a single authoritative IntentDecision consumed by TaskRouter,
RAGPipeline, SovereignAgent, and CLI scripts, eliminating fragmented re-classification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import List, Optional, Tuple


class ActionType(str, Enum):
    """The fundamental action the user wants the system to perform."""
    CREATE_NEW = "create_new"                  # generate, create, make, render, depict, visualize, draw, paint, sketch, synthesize, illustrate
    ANALYZE_EXISTING = "analyze_existing"      # analyze, inspect, examine, read, extract, describe defects, check, ocr, transcribe, find similar
    RETRIEVE_INFO = "retrieve_info"            # what is, explain, describe process, SOP, safety standard, list, summarize, lookup
    CALCULATE = "calculate"                    # calculate, compute, formula, sizing, MAWP, thickness
    DRAFT_COMMUNICATION = "draft_communication"# draft email, approval note, notification
    WRITE_CODE = "write_code"                  # write python script, generate code
    UNKNOWN = "unknown"


class OutputModality(str, Enum):
    """The format or presentation of output the user expects."""
    NEW_VISUAL = "new_visual"                                 # newly synthesized image, picture, rendering, visual scene
    EXISTING_VISUAL_ANALYSIS = "existing_visual_analysis"     # textual analysis/extraction of an existing image, drawing, or scan
    DOCUMENT_FILE = "document_file"                           # downloadable PDF, DOCX, XLSX, PPTX artifact
    TEXT_ANSWER = "text_answer"                               # document-grounded textual answer / explanation / SOP
    NUMERICAL_RESULT = "numerical_result"                     # engineering computation result with reduction steps
    CODE = "code"                                             # executable script / Python code


class TargetState(str, Enum):
    """Whether the target of the query is a new artifact to synthesize or an existing artifact/corpus."""
    NEW_SYNTHESIS = "new_synthesis"            # user wants a new visual/file created from scratch
    EXISTING_ARTIFACT = "existing_artifact"    # user points to an existing image, drawing, P&ID, photo, or file path
    DOCUMENT_CORPUS = "document_corpus"        # user queries the refinery document knowledge base


@dataclass(frozen=True)
class IntentDecision:
    """Structured, authoritative classification of user intent."""
    action: ActionType
    output_modality: OutputModality
    target_state: TargetState
    capability_name: str                       # maps to agent.router.Capability value ("image_generation", "vision", "rag", etc.)
    tool_name: str                             # "image_generator", "vision_inspector", "rag_pipeline", etc.
    use_rag_context: bool                      # whether document retrieval is required
    reason: str
    matched_cues: List[str] = field(default_factory=list)

    @property
    def is_image_generation(self) -> bool:
        """True if the intent is to synthesize a new visual artifact."""
        return (
            self.output_modality == OutputModality.NEW_VISUAL
            and self.action == ActionType.CREATE_NEW
            and self.target_state == TargetState.NEW_SYNTHESIS
        )

    @property
    def is_existing_visual_analysis(self) -> bool:
        """True if the intent is to inspect/analyze an existing visual or document image."""
        return (
            self.output_modality == OutputModality.EXISTING_VISUAL_ANALYSIS
            and self.action == ActionType.ANALYZE_EXISTING
            and self.target_state == TargetState.EXISTING_ARTIFACT
        )


# ── Linguistic and Semantic Recognition Structures ─────────────────────────

# 1. Existing visual markers (reference to an already existing image or document file)
_EXISTING_IMAGE_MARKERS = [
    re.compile(r"\b(?:this|the|that|attached|uploaded|scanned|sample)\s+(?:[\w\s\-]+\s+)?(?:image|picture|photo|photograph|drawing|p\s*&?\s*id|pid|diagram|scan|radiograph|nameplate|plate|document)\b", re.IGNORECASE),
    re.compile(r"\b(?:scanned\s+(?:document|drawing|image|file|report|copy))\b", re.IGNORECASE),
    re.compile(r"\b(?:visible\s+in|shown\s+in|from|on)\s+(?:this|the|that)\s+(?:[\w\s\-]+\s+)?(?:image|picture|photo|drawing|p\s*&?\s*id|pid|diagram|scan|radiograph)\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+does\s+this\s+(?:p\s*&?\s*id|pid|drawing|image|photo|diagram|scan)\s+show\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+(?:anomalies?|defects?|features?)\s+are\s+shown\s+in\b", re.IGNORECASE),
    re.compile(r"\b(?:read|extract)\s+(?:the\s+)?nameplate\b", re.IGNORECASE),
    re.compile(r"\bfind\s+(?:images?|drawings?|photos?)\s+similar\s+to\b", re.IGNORECASE),
    re.compile(r"\b[\w\-./\\]+\.(?:png|jpg|jpeg|tiff|bmp|pdf)\b", re.IGNORECASE),
    re.compile(r"\b(?:datasets|images|drawings|scans)/[\w\-./\\]+", re.IGNORECASE),
]

# 2. Analytical action verbs targeting images/drawings
_ANALYTICAL_ACTIONS = [
    re.compile(r"\b(?:analyze|analyse|inspect|examine|read|extract|transcribe|ocr|detect|identify|check|audit|review|find)\b", re.IGNORECASE),
    re.compile(r"\bdescribe\s+(?:the\s+)?(?:defects?|flaws?|anomalies?|corrosion|cracks?|features?)\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+does\s+(?:this|the)\b.*?\bshow\b", re.IGNORECASE),
    re.compile(r"\bwhat\b.*?\bshown\s+in\b", re.IGNORECASE),
]

# 3. Creative/generative visual appearance constructs (asking how something looks)
_APPEARANCE_VISUAL_CONSTRUCTS = [
    # "show me what ... looks like / might look like / would look like / could look like"
    re.compile(r"\bshow\s+(?:me|us)\s+what\b.*?\b(?:looks?|might\s+look|would\s+look|could\s+look)\s+like\b", re.IGNORECASE),
    # "what would / might / does / could ... look like"
    re.compile(r"\bwhat\s+(?:would|might|does|could)\b.*?\blook\s+like\b", re.IGNORECASE),
    # "how would / might / does / could ... look"
    re.compile(r"\bhow\s+(?:would|might|does|could)\b.*?\blook\b", re.IGNORECASE),
    # "can you show me how ... looks"
    re.compile(r"\bshow\s+(?:me|us)\s+how\b.*?\blooks?\b", re.IGNORECASE),
    # "can you picture / picture what ... looks like"
    re.compile(r"\b(?:can\s+you\s+)?picture\s+(?:what|how)\b.*?\blook", re.IGNORECASE),
    # "picture a [scene/equipment]" (imperative/conversational picture)
    re.compile(r"\b(?:can\s+you\s+)?picture\s+(?:an?|the)\s+[\w\s]+(?:inside|in|around|at|with)\b", re.IGNORECASE),
]

# 4. Perspective and scene depiction phrases (requesting visual views of physical scenes)
_PERSPECTIVE_VIEW_CONSTRUCTS = [
    re.compile(r"\b(?:an?\s+)?(?:aerial\s+view|bird'?s[\s\-]eye\s+view|top[\s\-]down\s+view|satellite\s+view|isometric\s+view|panoramic\s+view|3d\s+render(?:ing)?|3d\s+visualization)\s+of\b", re.IGNORECASE),
    re.compile(r"\b(?:from\s+above|from\s+the\s+air|from\s+an\s+aerial\s+perspective)\b", re.IGNORECASE),
]

# 5. Generative verbs that intrinsically create a visual representation of a physical subject
# "visualize a distillation tower", "depict a control room", "render a pump", "illustrate the flare stack"
_INTRINSIC_VISUALIZE_VERBS = [
    re.compile(r"\b(?:can\s+you\s+)?(?:visualize|visualise)\b.*?\b(?:an?|the|what|how)\b", re.IGNORECASE),
    re.compile(r"\b(?:can\s+you\s+)?depict\b.*?\b(?:an?|the|what|how)\b", re.IGNORECASE),
    re.compile(r"\b(?:can\s+you\s+)?render\b.*?\b(?:an?|the|what)\b", re.IGNORECASE),
    re.compile(r"\b(?:can\s+you\s+)?illustrate\b.*?\b(?:an?|the)\b", re.IGNORECASE),
    re.compile(r"\b(?:can\s+you\s+)?(?:draw|sketch|paint)\b.*?\b(?:an?|the|what)\b", re.IGNORECASE),
]

# 6. Action Verb + Visual Artifact/Medium noun OR visual modifier
# "generate an image", "create a picture", "make a photo", "synthesize a visual", etc.
_CREATIVE_VERBS = r"(?:generate|create|make|produce|synthesize|render|draw|paint|sketch|illustrate|depict|output|build)"
_VISUAL_NOUNS = r"(?:image|picture|photo|photograph|illustration|diagram|rendering|graphic|visual\s+representation|visual|artwork|drawing|sketch|scene|concept\s+art|3d\s+render)"

_ACTION_PLUS_VISUAL_NOUN = re.compile(
    rf"\b{_CREATIVE_VERBS}\b.*?\b(?:an?\s+)?{_VISUAL_NOUNS}\b", re.IGNORECASE
)

# "generate a realistic industrial centrifugal pump..."
_ACTION_PLUS_VISUAL_MODIFIER = re.compile(
    rf"\b{_CREATIVE_VERBS}\b.*?\b(?:an?\s+)?(?:realistic|photorealistic|hyperrealistic|detailed\s+visual|3d|isometric|cinematic)\b.*?\b(?:pump|compressor|tower|column|vessel|tank|exchanger|refinery|pipeline|unit|terminal|platform|stack|valve|flare|facility|station|plant)\b",
    re.IGNORECASE,
)

# 7. Explicit diffusion / text-to-image indicators
_DIFFUSION_INDICATORS = [
    re.compile(r"\b(?:text[\s\-]to[\s\-]image|txt2img|stable\s*diffusion|diffusion\s*image|diffusion\s*model)\b", re.IGNORECASE),
    re.compile(r"\b(?:visual\s+representation\s+of)\b", re.IGNORECASE),
]

# 8. Document file artifact generation
_DOC_VERBS = r"(?:create|generate|export|prepare|save|output|produce|build|draft|write|download|make)"
_DOC_PATTERNS = [
    re.compile(rf"\b{_DOC_VERBS}\b.*?\b(?:a|an|the|as|into)?\s*(?:actual\s+)?(?:pdf|docx|word\s+doc(?:ument)?|excel|spreadsheet|xlsx|powerpoint|pptx)\b", re.IGNORECASE),
    re.compile(r"\b(?:pdf|docx|xlsx|pptx)\s+(?:file|document|report|artifact|export)\b", re.IGNORECASE),
    re.compile(r"\bexport\s+(?:as|to)?\s*(?:pdf|docx|word|excel|xlsx|pptx|powerpoint)\b", re.IGNORECASE),
    re.compile(r"\bas\s+(?:an?\s+)?(?:actual\s+)?(?:pdf|docx|xlsx|pptx)\s+artifact\b", re.IGNORECASE),
]

# 9. Engineering calculations
_CALC_PATTERNS = [
    re.compile(r"\b(?:calculate|compute|verify\s+calculation|check\s+calculation|stress\s+calculation|pressure\s+calculation|flow\s+rate\s+calculation|relief\s+valve\s+sizing|pipe\s+sizing|thickness\s+calculation|mawp|burst\s+pressure|factor\s+of\s+safety|corrosion\s+allowance)\b", re.IGNORECASE),
]

# 10. Code generation
_CODE_PATTERNS = [
    re.compile(r"\b(?:write\s+code|generate\s+code|write\s+(?:a\s+)?script|write\s+python|write\s+sql|generate\s+python|create\s+a\s+function|code\s+snippet|automate\s+with\s+script)\b", re.IGNORECASE),
]


def classify_intent(query: str, **kwargs) -> IntentDecision:
    """Classify user query into an authoritative IntentDecision.

    Evaluates:
    - Action: CREATE_NEW vs ANALYZE_EXISTING vs RETRIEVE_INFO vs CALCULATE vs WRITE_CODE
    - Output Modality: NEW_VISUAL vs EXISTING_VISUAL_ANALYSIS vs DOCUMENT_FILE vs TEXT_ANSWER vs NUMERICAL_RESULT
    - Target State: NEW_SYNTHESIS vs EXISTING_ARTIFACT vs DOCUMENT_CORPUS

    Guarantees:
    - Queries requesting a new visual (e.g. 'Show me what a modern oil refinery might look like from above.',
      'Can you visualize a distillation tower and the equipment around it?') are classified as
      CREATE_NEW + NEW_VISUAL + NEW_SYNTHESIS -> Capability: image_generation, Tool: image_generator,
      use_rag_context=False.
    - Queries analyzing an existing image/drawing (e.g. 'Analyze this image', 'What does this P&ID show?')
      are classified as ANALYZE_EXISTING + EXISTING_VISUAL_ANALYSIS + EXISTING_ARTIFACT -> Capability: vision,
      Tool: vision_inspector, use_rag_context=False.
    """
    raw = (query or "").strip()
    if not raw:
        return IntentDecision(
            action=ActionType.UNKNOWN,
            output_modality=OutputModality.TEXT_ANSWER,
            target_state=TargetState.DOCUMENT_CORPUS,
            capability_name="rag",
            tool_name="rag_pipeline",
            use_rag_context=True,
            reason="Empty query defaulted to RAG.",
        )

    matched_cues: List[str] = []

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 1: Check for Analytical Visual Inspection on an Existing Artifact
    # ─────────────────────────────────────────────────────────────────────────
    has_existing_marker = any(pat.search(raw) for pat in _EXISTING_IMAGE_MARKERS)
    has_analytical_action = any(pat.search(raw) for pat in _ANALYTICAL_ACTIONS)
    has_explicit_file_path = bool(
        kwargs.get("vision_file_path")
        or re.search(r"\b[\w\-./\\]+\.(?:png|jpg|jpeg|tiff|bmp)\b", raw, re.IGNORECASE)
    )

    if (has_existing_marker and has_analytical_action) or (has_explicit_file_path and has_analytical_action):
        for pat in _EXISTING_IMAGE_MARKERS + _ANALYTICAL_ACTIONS:
            m = pat.search(raw)
            if m:
                matched_cues.append(m.group(0))
        return IntentDecision(
            action=ActionType.ANALYZE_EXISTING,
            output_modality=OutputModality.EXISTING_VISUAL_ANALYSIS,
            target_state=TargetState.EXISTING_ARTIFACT,
            capability_name="vision",
            tool_name="vision_inspector",
            use_rag_context=False,
            reason="User requested analytical visual inspection or extraction from an existing image/drawing.",
            matched_cues=matched_cues,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 2: Check for Generative Request to Synthesize a New Visual
    # ─────────────────────────────────────────────────────────────────────────
    is_creative_visual = False

    # Check 2A: Appearance depiction ("show me what ... looks like / might look like")
    for pat in _APPEARANCE_VISUAL_CONSTRUCTS:
        m = pat.search(raw)
        if m:
            is_creative_visual = True
            matched_cues.append(m.group(0))
            break

    # Check 2B: Intrinsic visual generation verbs ("visualize [object]", "depict [object]", "render [object]")
    if not is_creative_visual:
        for pat in _INTRINSIC_VISUALIZE_VERBS:
            m = pat.search(raw)
            if m:
                is_creative_visual = True
                matched_cues.append(m.group(0))
                break

    # Check 2C: Perspective rendering requests ("aerial view of [site]", "bird's eye view of [unit]")
    if not is_creative_visual:
        for pat in _PERSPECTIVE_VIEW_CONSTRUCTS:
            m = pat.search(raw)
            if m:
                # Ensure it's not an analysis of an existing aerial photo
                if not has_analytical_action:
                    is_creative_visual = True
                    matched_cues.append(m.group(0))
                    break

    # Check 2D: Action verb + visual artifact noun or visual modifier ("create a picture", "generate an image", "generate a realistic pump")
    if not is_creative_visual:
        m = _ACTION_PLUS_VISUAL_NOUN.search(raw) or _ACTION_PLUS_VISUAL_MODIFIER.search(raw)
        if m:
            is_creative_visual = True
            matched_cues.append(m.group(0))

    # Check 2E: Text-to-image / diffusion technical terms
    if not is_creative_visual:
        for pat in _DIFFUSION_INDICATORS:
            m = pat.search(raw)
            if m:
                is_creative_visual = True
                matched_cues.append(m.group(0))
                break

    # If identified as generative visual request and NOT an analytical inspection
    if is_creative_visual and not (has_existing_marker and has_analytical_action):
        return IntentDecision(
            action=ActionType.CREATE_NEW,
            output_modality=OutputModality.NEW_VISUAL,
            target_state=TargetState.NEW_SYNTHESIS,
            capability_name="image_generation",
            tool_name="image_generator",
            use_rag_context=False,
            reason="User requested generation of a newly synthesized visual/scene.",
            matched_cues=matched_cues,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 3: Check for Document File Generation (PDF, DOCX, XLSX, PPTX)
    # ─────────────────────────────────────────────────────────────────────────
    for pat in _DOC_PATTERNS:
        m = pat.search(raw)
        if m:
            matched_cues.append(m.group(0))
            t_lower = raw.lower()
            if any(k in t_lower for k in ["xlsx", "excel", "spreadsheet"]):
                tool = "xlsx_generator"
            elif any(k in t_lower for k in ["pptx", "powerpoint", "slide"]):
                tool = "pptx_generator"
            elif any(k in t_lower for k in ["docx", "word"]):
                tool = "document_generator"
            else:
                tool = "pdf_generator"

            return IntentDecision(
                action=ActionType.CREATE_NEW,
                output_modality=OutputModality.DOCUMENT_FILE,
                target_state=TargetState.NEW_SYNTHESIS,
                capability_name="document_generation",
                tool_name=tool,
                use_rag_context=True,
                reason="User requested generation of a structured document artifact.",
                matched_cues=matched_cues,
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 4: Check for Engineering Calculations
    # ─────────────────────────────────────────────────────────────────────────
    for pat in _CALC_PATTERNS:
        m = pat.search(raw)
        if m:
            matched_cues.append(m.group(0))
            return IntentDecision(
                action=ActionType.CALCULATE,
                output_modality=OutputModality.NUMERICAL_RESULT,
                target_state=TargetState.DOCUMENT_CORPUS,
                capability_name="calculation",
                tool_name="calculator",
                use_rag_context=True,
                reason="User requested an engineering calculation with formula reduction.",
                matched_cues=matched_cues,
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 5: Check for Code Generation
    # ─────────────────────────────────────────────────────────────────────────
    for pat in _CODE_PATTERNS:
        m = pat.search(raw)
        if m:
            matched_cues.append(m.group(0))
            return IntentDecision(
                action=ActionType.WRITE_CODE,
                output_modality=OutputModality.CODE,
                target_state=TargetState.NEW_SYNTHESIS,
                capability_name="coding",
                tool_name="code_interpreter",
                use_rag_context=False,
                reason="User requested code generation or scripting.",
                matched_cues=matched_cues,
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 6: Default to Information Retrieval & QA (RAG)
    # ─────────────────────────────────────────────────────────────────────────
    return IntentDecision(
        action=ActionType.RETRIEVE_INFO,
        output_modality=OutputModality.TEXT_ANSWER,
        target_state=TargetState.DOCUMENT_CORPUS,
        capability_name="rag",
        tool_name="rag_pipeline",
        use_rag_context=True,
        reason="User requested factual information, standard, or procedural explanation from document corpus.",
        matched_cues=matched_cues,
    )
