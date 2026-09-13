"""Multimodal intelligence orchestration foundation for Sovereign AI Workbench (MRPL).

This module coordinates:
1. Image Preprocessing (member3_ocr.image_preprocessing)
2. Text OCR (member3_ocr.ocr_pipeline)
3. Visual Intelligence (member3_ocr.vision_pipeline)
4. Structured Document Parsing (member3_ocr.document_parser)
5. Engineering Drawing Analysis (member3_ocr.drawing_analyzer)

Primary Architectural Role:
The MultimodalProcessor is the SINGLE TOP-LEVEL ENTRY POINT for all Member 3 capabilities.
It takes a raw input file (image or PDF) of unknown type, routes it through the correct
combination of pipelines using a fast, non-VLM heuristic classifier, and outputs a unified,
RAG/Agent-ready envelope (MultimodalProcessingResult).

Guarantees:
- Completely Offline: Zero network access, zero cloud APIs, zero automatic downloads.
- Non-Destructive: Source images and datasets/ are never modified or written to.
- Fast Routing: Plain documents bypass the expensive VLM entirely (< 2s total latency).
- Fault-Tolerant: Sub-pipeline failures produce structured ProcessingError records, never fatal crashes.
- Provenance-Preserving: Every piece of combined information retains its source modality.
"""
from __future__ import annotations


import argparse
import hashlib
import json
import logging
import re
import sys
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from enum import Enum
try:
    from enum import StrEnum
except ImportError:
    class StrEnum(str, Enum):  # type: ignore[no-redef]
        pass

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
from pydantic import BaseModel

from .document_parser import (
    AgentQueryableDocument,
    DocumentParser,
    DocumentParserError,
    export_for_agent as export_document_for_agent,
    export_for_rag as export_document_for_rag,
)
from .drawing_analyzer import (
    AgentQueryableDrawing,
    DrawingAnalysisRecord,
    DrawingAnalysisResult,
    DrawingAnalyzer,
    DrawingAnalyzerConfig,
    DrawingLabel,
    DrawingMetadata,
    DrawingRegion,
    DrawingType,
    MockDrawingBackend,
    export_for_agent as export_drawing_for_agent,
    export_for_rag as export_drawing_for_rag,
)
from .image_preprocessing import (
    SUPPORTED_EXTENSIONS,
    ImagePreprocessingError,
    PreprocessingOptions,
    PreprocessingResult,
    preprocess_image,
)
from .ocr_pipeline import (
    BackendCapabilities as OCRBackendCapabilities,
    BackendInfo as OCRBackendInfo,
    BackendRecognition,
    BoundingBox,
    OCRBackend,
    OCRBackendExecutionError,
    OCRDocumentResult,
    OCRPageResult,
    OCRPipeline,
    TextBlock,
    TextLine,
    Word,
)
from .pdf_rendering import render_pdf_pages
from .vision_pipeline import (
    AgentQueryableVisionResult,
    MockVisionBackend,
    VisionAnalysisRecord,
    VisionPipeline,
    VisionResult,
    create_vision_analysis_record,
    export_for_agent as export_vision_for_agent,
    export_for_rag as export_vision_for_rag,
)
from rag_engine.schemas.chunk import Chunk, ChunkMetadata
from rag_engine.schemas.parsed_document import ParsedDocument

LOGGER = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"
PROCESSOR_VERSION = "2.0.0"

# Drawing keyword pattern for fast routing (case-insensitive)
_DRAWING_KEYWORD_RE = re.compile(
    r"\b("
    r"P&ID|PFD|PIPING\s+AND\s+INSTRUMENTATION|PROCESS\s+FLOW\s+DIAGRAM|"
    r"SINGLE\s+LINE\s+DIAGRAM|SLD|SCHEMATIC|FLOW\s+SHEET|"
    r"DWG\s*(?:NO|\.)?|DRAWING\s*(?:NO|\.)?|SCALE\s*:|REVISION\s*:|"
    r"LINE\s+NO|NOZZLE\s+SCHEDULE|INSTRUMENT\s+SCHEDULE|"
    r"SUCTION|DISCHARGE|CENTRIFUGAL\s+PUMP|HEAT\s+EXCHANGER|PRESSURE\s+VESSEL"
    r")\b",
    re.IGNORECASE,
)

# Plain document keyword pattern for fast routing (case-insensitive)
_DOCUMENT_KEYWORD_RE = re.compile(
    r"\b("
    r"MEMORANDUM|MEMO|REPORT|INVOICE|FORM|PURCHASE\s+ORDER|"
    r"TABLE\s+\d|SECTION\s+\d|PARAGRAPH|SIGNATURE|EXECUTIVE\s+SUMMARY|"
    r"DATE\s*:|SUBJECT\s*:|TO\s*:|FROM\s*:|ATTENTION\s*:|DEAR\s+"
    r")\b",
    re.IGNORECASE,
)

# Standard alphanumeric equipment tag pattern (e.g., P-203, TK-101, FIC-202)
_TAG_PATTERN_RE = re.compile(r"\b([A-Z]{1,4}-\d{2,5}[A-Z]?|[A-Z]{2,3}\s*\d{2,4}[A-Z]?)\b")


# ──────────────────────────────────────────────────────────────────────────────
# Enums & Statuses
# ──────────────────────────────────────────────────────────────────────────────

class ModalityStatus(StrEnum):
    """Execution status of an individual modality within the orchestration pipeline."""

    NOT_REQUESTED = "NOT_REQUESTED"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class RoutingDecision(StrEnum):
    """Routing path determined by the fast classifier or force-route override."""

    PLAIN_DOCUMENT = "plain_document"
    ENGINEERING_DRAWING = "engineering_drawing"
    VISUAL_INSPECTION = "visual_inspection"
    MIXED = "mixed"


# ──────────────────────────────────────────────────────────────────────────────
# Exceptions
# ──────────────────────────────────────────────────────────────────────────────

class MultimodalProcessorError(RuntimeError):
    """Base exception for multimodal processor errors."""


class MultimodalConfigurationError(MultimodalProcessorError):
    """Raised for invalid processor configuration or offline policy violations."""


# ──────────────────────────────────────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ProcessingError:
    """Structured capture of sub-pipeline partial failures without crashing."""

    stage: str
    code: str
    message: str
    severity: str = "error"  # "error", "warning", "info"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MultimodalIssue:
    """Structured issue, warning, conflict, or error in multimodal processing (legacy-compatible)."""

    code: str
    message: str
    severity: str = "warning"  # "error", "warning", "info"
    modality: str = "multimodal"
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MultimodalProcessorConfig:
    """Configuration options for multimodal orchestration."""

    enable_preprocessing: bool = True
    enable_ocr: bool = True
    enable_vision: bool = True
    enable_drawing_analysis: bool = True
    enable_document_parsing: bool = False
    force_route: str | None = None  # None, "plain_document", "engineering_drawing", "both" / "mixed"
    fail_fast: bool = False
    output_directory: Path | str | None = None
    is_mock: bool = False

    def validate(self) -> None:
        """Validate configuration settings."""
        if self.output_directory is not None:
            p = Path(self.output_directory)
            if "datasets" in p.parts:
                raise MultimodalConfigurationError("Writing output inside datasets/ is forbidden.")
        if self.force_route is not None:
            valid_routes = {"plain_document", "engineering_drawing", "visual_inspection", "inspection", "both", "mixed"}
            if self.force_route.lower().strip() not in valid_routes:
                raise MultimodalConfigurationError(
                    f"Invalid force_route '{self.force_route}'. Valid options: {sorted(list(valid_routes))}"
                )


@dataclass(frozen=True)
class MultimodalProcessingResult:
    """Unified top-level envelope for RAG and Agent downstream consumption (Contract 6 & 7)."""

    document_id: str
    source_path: str
    routing_decision: str  # "plain_document" | "engineering_drawing" | "visual_inspection" | "mixed"
    routing_signals: dict[str, Any]
    parsed_document: ParsedDocument | None = None
    drawing_analysis: DrawingAnalysisRecord | None = None
    vision_analysis: VisionAnalysisRecord | None = None
    processing_errors: tuple[ProcessingError, ...] = ()
    pipeline_versions: dict[str, str] = field(default_factory=dict)
    processing_metadata: dict[str, Any] = field(default_factory=dict)
    modality_statuses: dict[str, str] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION
    success: bool = True

    @property
    def has_document(self) -> bool:
        return self.parsed_document is not None

    @property
    def has_drawing(self) -> bool:
        return self.drawing_analysis is not None

    @property
    def has_vision_analysis(self) -> bool:
        return self.vision_analysis is not None

    @property
    def has_errors(self) -> bool:
        return any(e.severity == "error" for e in self.processing_errors)

    def to_dict(self) -> dict[str, Any]:
        """Convert multimodal result to a pure JSON-serializable dictionary."""
        return _to_serializable(self)

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize result as UTF-8 JSON text."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def get_full_text(self) -> str:
        """Return canonical document, drawing, or visual inspection text."""
        parts: list[str] = []
        if self.parsed_document is not None:
            parts.append(self.parsed_document.get_full_text())
        if self.drawing_analysis is not None:
            dwg = self.drawing_analysis
            dwg_lines = [
                f"Drawing Number: {dwg.title_block.drawing_number or 'Unknown'}",
                f"Title: {dwg.title_block.title or ''}",
                f"Drawing Type: {dwg.drawing_type.value if hasattr(dwg.drawing_type, 'value') else str(dwg.drawing_type)}",
            ]
            if dwg.equipment:
                dwg_lines.append("Equipment: " + ", ".join(f"{e.tag} ({e.equipment_type})" for e in dwg.equipment))
            if dwg.instruments:
                dwg_lines.append("Instruments: " + ", ".join(f"{i.tag} ({i.instrument_type})" for i in dwg.instruments))
            if dwg.connections:
                dwg_lines.append("Connections: " + ", ".join(f"{c.from_tag} -> {c.to_tag}" for c in dwg.connections))
            parts.append("\n".join(dwg_lines))
        if self.vision_analysis is not None:
            vis = self.vision_analysis
            vis_lines = [
                f"Visual Inspection Analysis: {vis.image_id}",
                f"Image Type: {vis.image_type}",
                f"Caption: {vis.caption or 'None'}",
            ]
            if vis.observations:
                vis_lines.append("Observations: " + "; ".join(o.description for o in vis.observations))
            if vis.equipment:
                vis_lines.append("Equipment: " + ", ".join(f"{e.equipment_type} [{e.name_or_tag}]" if e.name_or_tag else e.equipment_type for e in vis.equipment))
            parts.append("\n".join(vis_lines))
        return "\n\n".join(parts)


@dataclass(frozen=True)
class MultimodalResult:
    """Complete, normalized result of legacy multimodal processing (backward-compatible)."""

    success: bool
    source_path: str
    image_dimensions: tuple[int, int] | None = None
    preprocessing_result: Any | None = None
    ocr_result: OCRDocumentResult | None = None
    vision_result: VisionResult | None = None
    drawing_result: DrawingAnalysisResult | None = None
    parsed_document: ParsedDocument | None = None
    modality_statuses: dict[str, ModalityStatus] = field(default_factory=dict)
    combined_text: list[dict[str, Any]] = field(default_factory=list)
    combined_structured: dict[str, Any] = field(default_factory=dict)
    issues: tuple[MultimodalIssue, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)
    processing_metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return _to_serializable(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_rag_payload(self) -> dict[str, Any]:
        """Prepare legacy payload suitable for downstream RAG ingestion."""
        chunks: list[dict[str, Any]] = []
        for item in self.combined_text:
            chunks.append({
                "text": item.get("text", ""),
                "modality": item.get("source", "unknown"),
                "source_path": self.source_path,
            })

        entities: list[dict[str, Any]] = []
        if self.vision_result:
            for det in self.vision_result.detections:
                entities.append({
                    "label": det.label,
                    "confidence": det.confidence,
                    "modality": "vision",
                    "bbox": {
                        "left": det.bbox.left,
                        "top": det.bbox.top,
                        "right": det.bbox.right,
                        "bottom": det.bbox.bottom,
                        "coordinate_space": det.bbox.coordinate_space,
                    },
                    "attributes": det.attributes,
                })
        if self.drawing_result:
            for reg in self.drawing_result.regions:
                entities.append({
                    "label": reg.label or reg.region_type,
                    "region_type": reg.region_type,
                    "confidence": reg.confidence,
                    "modality": "drawing",
                    "bbox": {
                        "left": reg.bbox.left,
                        "top": reg.bbox.top,
                        "right": reg.bbox.right,
                        "bottom": reg.bbox.bottom,
                        "coordinate_space": reg.bbox.coordinate_space,
                    },
                    "attributes": reg.attributes,
                })

        relations: list[dict[str, Any]] = []
        if self.drawing_result:
            for r in self.drawing_result.spatial_relations:
                relations.append({
                    "relation_id": r.relation_id,
                    "source_id": r.source_id,
                    "relation": r.relation.value if hasattr(r.relation, "value") else str(r.relation),
                    "target_id": r.target_id,
                    "confidence": r.confidence,
                    "evidence": r.evidence,
                })

        return {
            "source_identifier": self.source_path,
            "success": self.success,
            "image_dimensions": list(self.image_dimensions) if self.image_dimensions else None,
            "text_chunks": chunks,
            "entities": entities,
            "drawing_relations": relations,
            "structured_metadata": _to_serializable(self.combined_structured.get("metadata", {})),
            "provenance": _to_serializable(self.provenance),
            "modality_statuses": {k: v.value for k, v in self.modality_statuses.items()},
        }


# ──────────────────────────────────────────────────────────────────────────────
# Serialization Helper
# ──────────────────────────────────────────────────────────────────────────────

def _to_serializable(value: Any) -> Any:
    """Recursively convert dataclasses, Pydantic models, and numpy objects to JSON primitives."""
    if isinstance(value, bool):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, BaseModel):
        return value.model_dump()
    if is_dataclass(value):
        return {k: _to_serializable(v) for k, v in asdict(value).items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    if isinstance(value, dict):
        return {str(k) if not isinstance(k, str) else k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


# ──────────────────────────────────────────────────────────────────────────────
# Deterministic Mock OCR Backend for Offline Testing
# ──────────────────────────────────────────────────────────────────────────────

class MockOCRBackend(OCRBackend):
    """Deterministic offline mock OCR backend for testing and default orchestration."""

    def __init__(
        self,
        *,
        text: str | None = None,
        fail: bool = False,
    ) -> None:
        self.text = text
        self.fail = fail
        self.initialized = False

    @property
    def backend_info(self) -> OCRBackendInfo:
        return OCRBackendInfo(
            name="mock_ocr_backend",
            version="1.0.0",
            model_ids=("mock_ocr_weights",),
            device="cpu",
            capabilities=OCRBackendCapabilities(),
        )

    def initialize(self) -> None:
        self.initialized = True

    def recognize(self, image: np.ndarray) -> BackendRecognition:
        if self.fail:
            raise OCRBackendExecutionError("Simulated mock OCR execution failure")
        h, w = image.shape[:2] if hasattr(image, "shape") else (100, 100)
        aspect = float(w) / float(h) if h > 0 else 1.0

        if self.text is not None:
            active_text = self.text
        elif aspect >= 1.35:
            active_text = "P&ID Schematic Diagram\nP-101 Centrifugal Pump\nFV-201 Control Valve"
        else:
            active_text = "Asset Data Block\nIdentifier: 101\nRecord Information"

        bbox = BoundingBox(10.0, 10.0, min(200.0, float(w - 10)), min(50.0, float(h - 10)))
        lines = []
        for line_text in active_text.splitlines():
            if not line_text.strip():
                continue
            word = Word(text=line_text, confidence=0.98, bbox=bbox)
            tl = TextLine(text=line_text, confidence=0.98, bbox=bbox, words=(word,))
            lines.append(tl)
        block = TextBlock(id="tb_mock_0", text=active_text, confidence=0.98, bbox=bbox, lines=tuple(lines))
        return BackendRecognition(blocks=(block,), warnings=())


# ──────────────────────────────────────────────────────────────────────────────
# Task 1: Fast Heuristic Routing Classifier (Zero VLM Calls)
# ──────────────────────────────────────────────────────────────────────────────

def classify_routing(
    file_path: Path | str,
    *,
    image_width: int = 0,
    image_height: int = 0,
    ocr_result: OCRDocumentResult | None = None,
) -> tuple[RoutingDecision, dict[str, Any]]:
    """Determine initial pipeline routing path using cheap, fast, non-VLM signals.

    Signals analyzed:
    1. Aspect ratio and geometric format (wide panoramic vs tall portrait).
    2. OCR text density and distribution (scattered tags vs continuous paragraphs).
    3. Drawing keywords vs Document keywords.
    4. Path/filename hints.

    Returns (RoutingDecision, signals_dict).
    """
    path = Path(file_path)
    stem = path.stem.lower()
    path_str = str(path).lower().replace("\\", "/")

    # 1. Geometry / Aspect Ratio
    aspect_ratio = float(image_width) / float(image_height) if image_height > 0 else 1.0
    is_panoramic = aspect_ratio >= 1.35
    is_portrait = aspect_ratio <= 0.90 and aspect_ratio > 0.0
    is_large_format = max(image_width, image_height) >= 2000 or (image_width >= 1600 and is_panoramic)

    # 2. Text tokens from OCR
    all_text_blocks: list[str] = []
    if ocr_result is not None:
        for page in ocr_result.pages:
            for block in page.blocks:
                if block.text.strip():
                    all_text_blocks.append(block.text.strip())

    combined_text = " ".join(all_text_blocks)
    text_length = len(combined_text)

    # Keyword searches
    drawing_keywords = _DRAWING_KEYWORD_RE.findall(combined_text)
    document_keywords = _DOCUMENT_KEYWORD_RE.findall(combined_text)
    equipment_tags = _TAG_PATTERN_RE.findall(combined_text)

    # Paragraph density check (sentences ending with . ! ?)
    paragraph_count = sum(
        1 for b in all_text_blocks
        if len(b) >= 70 and any(b.rstrip().endswith(p) for p in (".", "!", "?"))
    )

    # Path clues
    path_drawing_hint = any(tok in path_str for tok in ("engineering_drawings", "pid", "pfd", "drawing", "dwg", "diagram", "schematic"))
    path_doc_hint = any(tok in path_str for tok in ("funsd", "handwritten_notes", "safety_docs", "synthetic_ocr_dataset", "form", "table", "invoice", "report", "doc", "letter", "manual", "checklist"))
    path_inspection_hint = any(tok in path_str for tok in (
        "neu_surface_defect", "pcb_defect", "infrared", "inspection",
        "defect", "corrosion", "anomaly", "crazing", "pitting", "scratch",
        "inclusion", "patches", "rolled-in_scale", "surface"
    ))

    # Scoring
    drawing_score = 0.0
    doc_score = 0.0

    if path_drawing_hint:
        drawing_score += 4.0
    if drawing_keywords:
        drawing_score += min(len(drawing_keywords), 4) * 2.0
    if equipment_tags:
        drawing_score += min(len(equipment_tags), 5) * 1.0
    if is_panoramic:
        drawing_score += 2.5
    if is_large_format:
        drawing_score += 1.5

    if path_doc_hint:
        doc_score += 4.0
    if document_keywords:
        doc_score += min(len(document_keywords), 4) * 2.0
    if paragraph_count > 0:
        doc_score += min(paragraph_count, 5) * 2.0
    if is_portrait and not path_drawing_hint:
        doc_score += 2.0
    if text_length >= 400 and paragraph_count >= 1 and not is_panoramic:
        doc_score += 2.0

    # Decision calculation
    if path_inspection_hint and not path_drawing_hint and not path_doc_hint:
        decision = RoutingDecision.VISUAL_INSPECTION
        reasoning = "Visual defect/inspection image identified from path hints (surface defect, PCB flaw, or infrared)"
    elif drawing_score >= 3.0 and doc_score >= 3.0 and paragraph_count >= 2:
        decision = RoutingDecision.MIXED
        reasoning = f"Mixed content detected: drawing signals (score={drawing_score:.1f}) and narrative text (score={doc_score:.1f})"
    elif drawing_score >= 3.0 and drawing_score > doc_score:
        decision = RoutingDecision.ENGINEERING_DRAWING
        reasoning = f"Engineering drawing identified (score={drawing_score:.1f} vs doc_score={doc_score:.1f})"
    elif doc_score >= 2.0 and doc_score >= drawing_score:
        decision = RoutingDecision.PLAIN_DOCUMENT
        reasoning = f"Plain document identified (score={doc_score:.1f} vs drawing_score={drawing_score:.1f})"
    elif is_panoramic:
        decision = RoutingDecision.ENGINEERING_DRAWING
        reasoning = f"Panoramic aspect ratio ({aspect_ratio:.2f}) indicates drawing/diagram layout"
    else:
        decision = RoutingDecision.PLAIN_DOCUMENT
        reasoning = "Conservative default to plain document (avoids unnecessary VLM CPU invocation)"

    signals = {
        "aspect_ratio": round(aspect_ratio, 3),
        "is_panoramic": is_panoramic,
        "is_large_format": is_large_format,
        "text_blocks_count": len(all_text_blocks),
        "text_length": text_length,
        "drawing_keywords_matched": list(set(k.upper() for k in drawing_keywords))[:8],
        "document_keywords_matched": list(set(k.upper() for k in document_keywords))[:8],
        "equipment_tags_count": len(equipment_tags),
        "paragraphs_with_sentence_end": paragraph_count,
        "drawing_score": round(drawing_score, 2),
        "document_score": round(doc_score, 2),
        "reasoning": reasoning,
    }

    return decision, signals


# Note: AgentQueryableDocument is concretely implemented in member3_ocr.document_parser (Contract 3).


class AgentQueryableMultimodalDocument:
    """Unified queryable interface for multimodal documents, drawings, and visual inspections (Contract 6 & 7)."""

    def __init__(self, result: MultimodalProcessingResult) -> None:
        self._res = result
        self._doc_agent = AgentQueryableDocument(result.parsed_document) if result.parsed_document else None
        self._drw_agent = export_drawing_for_agent(result.drawing_analysis) if result.drawing_analysis else None
        self._vis_agent = export_vision_for_agent(result.vision_analysis) if result.vision_analysis else None

    @property
    def document_id(self) -> str:
        return self._res.document_id

    @property
    def routing_decision(self) -> str:
        return self._res.routing_decision

    @property
    def has_document(self) -> bool:
        return self._doc_agent is not None

    @property
    def has_drawing(self) -> bool:
        return self._drw_agent is not None

    @property
    def has_vision_analysis(self) -> bool:
        return self._vis_agent is not None

    @property
    def document(self) -> AgentQueryableDocument | None:
        return self._doc_agent

    @property
    def drawing(self) -> AgentQueryableDrawing | None:
        return self._drw_agent

    @property
    def visual_inspection(self) -> AgentQueryableVisionResult | None:
        return self._vis_agent

    def get_full_text(self) -> str:
        return self._res.get_full_text()

    # --- Document-oriented methods (pass-through) ---
    def get_sections(self) -> list[dict[str, Any]]:
        return self._doc_agent.get_sections() if self._doc_agent else []

    def get_tables(self) -> list[dict[str, Any]]:
        return self._doc_agent.get_tables() if self._doc_agent else []

    def get_key_value_fields(self) -> list[dict[str, Any]]:
        return self._doc_agent.get_key_value_fields() if self._doc_agent else []

    def citation_for_section(self, section_id: str) -> dict[str, Any] | None:
        return self._doc_agent.citation_for_section(section_id) if self._doc_agent else None

    # --- Drawing-oriented methods (pass-through) ---
    def get_equipment_list(self) -> list[dict[str, Any]]:
        if self._drw_agent:
            return self._drw_agent.get_equipment_list()
        if self._vis_agent:
            return self._vis_agent.get_equipment_list()
        return []

    def get_connections_for(self, tag: str) -> list[dict[str, Any]]:
        return self._drw_agent.get_connections_for(tag) if self._drw_agent else []

    def get_instrument_readings(self) -> list[dict[str, Any]]:
        return self._drw_agent.get_instrument_readings() if self._drw_agent else []

    def citation_for_equipment(self, tag: str) -> dict[str, Any] | None:
        return self._drw_agent.citation_for_equipment(tag) if self._drw_agent else None

    # --- Visual Inspection methods (pass-through) ---
    def get_visual_observations(self) -> list[dict[str, Any]]:
        return self._vis_agent.get_observations() if self._vis_agent else []

    def has_defects(self) -> bool:
        return self._vis_agent.has_defects() if self._vis_agent else False

    def get_summary(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "routing_decision": self.routing_decision,
            "has_document": self.has_document,
            "has_drawing": self.has_drawing,
            "has_vision_analysis": self.has_vision_analysis,
            "section_count": len(self.get_sections()),
            "table_count": len(self.get_tables()),
            "equipment_count": len(self.get_equipment_list()),
            "observation_count": len(self.get_visual_observations()),
            "error_count": len(self._res.processing_errors),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Task 4: Unified RAG Export Functions
# ──────────────────────────────────────────────────────────────────────────────

# Backwards compatibility alias: Contract 2 implementation lives in member3_ocr.document_parser
export_parsed_document_to_chunks = export_document_for_rag


def export_for_rag(result: MultimodalProcessingResult) -> list[Chunk]:
    """Unified RAG Chunk export delegate (Contract 6 & 7)."""
    chunks: list[Chunk] = []
    is_mixed = (result.has_document and result.has_drawing) or (result.has_vision_analysis and (result.has_document or result.has_drawing))

    if result.parsed_document is not None:
        doc_chunks = export_document_for_rag(result.parsed_document, prefix_modality=is_mixed)
        chunks.extend(doc_chunks)

    if result.drawing_analysis is not None:
        drw_chunks = export_drawing_for_rag(result.drawing_analysis)
        if is_mixed:
            # Re-index with 'Drawing' prefix
            prefixed_chunks = []
            for c in drw_chunks:
                hpath = ["Drawing"] + list(c.metadata.heading_path)
                updated_meta = ChunkMetadata(
                    document_id=c.metadata.document_id,
                    chunk_index=c.metadata.chunk_index,
                    document_name=c.metadata.document_name,
                    page_number=c.metadata.page_number,
                    section_title=c.metadata.section_title,
                    heading_path=hpath,
                    category=c.metadata.category,
                    plant_unit=c.metadata.plant_unit,
                    table_id=c.metadata.table_id,
                    is_table_chunk=c.metadata.is_table_chunk,
                    chunk_strategy=c.metadata.chunk_strategy,
                    char_start=c.metadata.char_start,
                    char_end=c.metadata.char_end,
                )
                prefixed_chunks.append(Chunk(
                    chunk_id=c.chunk_id,
                    content=c.content,
                    token_count=c.token_count,
                    metadata=updated_meta,
                ))
            chunks.extend(prefixed_chunks)
        else:
            chunks.extend(drw_chunks)

    if result.vision_analysis is not None:
        vis_chunks = export_vision_for_rag(result.vision_analysis)
        if is_mixed:
            prefixed_vis = []
            for c in vis_chunks:
                hpath = ["VisualInspection"] + list(c.metadata.heading_path)
                updated_meta = ChunkMetadata(
                    document_id=c.metadata.document_id,
                    chunk_index=c.metadata.chunk_index,
                    document_name=c.metadata.document_name,
                    page_number=c.metadata.page_number,
                    section_title=c.metadata.section_title,
                    heading_path=hpath,
                    category=c.metadata.category,
                    plant_unit=c.metadata.plant_unit,
                    table_id=c.metadata.table_id,
                    is_table_chunk=c.metadata.is_table_chunk,
                    chunk_strategy=c.metadata.chunk_strategy,
                    char_start=c.metadata.char_start,
                    char_end=c.metadata.char_end,
                )
                prefixed_vis.append(Chunk(
                    chunk_id=c.chunk_id,
                    content=c.content,
                    token_count=c.token_count,
                    metadata=updated_meta,
                ))
            chunks.extend(prefixed_vis)
        else:
            chunks.extend(vis_chunks)

    return chunks


def export_for_agent(result: MultimodalProcessingResult) -> AgentQueryableMultimodalDocument:
    """Unified Agent query interface delegate (Contract 6)."""
    return AgentQueryableMultimodalDocument(result)


# ──────────────────────────────────────────────────────────────────────────────
# Multimodal Processor Class
# ──────────────────────────────────────────────────────────────────────────────

class MultimodalProcessor:
    """Orchestrates image preprocessing, OCR, vision, drawing analysis, and document parsing."""

    def __init__(
        self,
        config: MultimodalProcessorConfig | None = None,
        *,
        ocr_pipeline: OCRPipeline | None = None,
        vision_pipeline: VisionPipeline | None = None,
        drawing_analyzer: DrawingAnalyzer | None = None,
        document_parser: DocumentParser | None = None,
    ) -> None:
        self.config = config or MultimodalProcessorConfig()
        self.config.validate()

        self.ocr_pipeline = ocr_pipeline or OCRPipeline(backend=MockOCRBackend())
        self.vision_pipeline = vision_pipeline or VisionPipeline(backend=MockVisionBackend())
        self.drawing_analyzer = drawing_analyzer or DrawingAnalyzer(backend=MockDrawingBackend())
        self.document_parser = document_parser or DocumentParser()

    # ── Modern Contract 6 Orchestrator ────────────────────────────────────────

    def orchestrate(
        self,
        source: str | Path | PreprocessingResult,
        *,
        force_route: str | None = None,
        ocr_result: OCRDocumentResult | None = None,
        vision_result: VisionResult | None = None,
        drawing_type: DrawingType | str | None = None,
        document_title: str | None = None,
        document_category: str = "Unknown",
        preprocessing_options: PreprocessingOptions | None = None,
    ) -> MultimodalProcessingResult:
        """Top-level intelligent routing entry point producing MultimodalProcessingResult."""
        start_ns = time.perf_counter_ns()
        errors: list[ProcessingError] = []
        pipeline_versions: dict[str, str] = {
            "processor": PROCESSOR_VERSION,
            "document_parser": getattr(self.document_parser, "version", "1.0.0"),
            "drawing_analyzer": "1.0.0",
        }

        # 1. Resolve source and dimensions
        orig_w, orig_h = 100, 100
        rendered_raster_path: Path | None = None

        if isinstance(source, PreprocessingResult):
            source_path = str(source.original_path)
            orig_w, orig_h = source.processed_width, source.processed_height
            prep_result: PreprocessingResult | None = source
        else:
            path = Path(source)
            source_path = str(path)
            prep_result = None

            if not path.exists():
                err = ProcessingError(
                    stage="input_validation",
                    code="FILE_NOT_FOUND",
                    message=f"Input file does not exist: {path}",
                    severity="error",
                )
                return MultimodalProcessingResult(
                    document_id=path.stem,
                    source_path=source_path,
                    routing_decision="unknown",
                    routing_signals={"error": "file_not_found"},
                    processing_errors=(err,),
                    pipeline_versions=pipeline_versions,
                )

            # Handle PDF inputs
            if path.suffix.lower() == ".pdf":
                try:
                    pages = render_pdf_pages(path, dpi=150)
                    if pages and pages[0].image is not None:
                        orig_h, orig_w = pages[0].image.shape[:2]
                        scratch_dir = Path("member3_ocr/output/evaluation/rendered_pages")
                        scratch_dir.mkdir(parents=True, exist_ok=True)
                        rendered_raster_path = scratch_dir / f"{path.stem}_p1_orchestrator.png"
                        import cv2
                        cv2.imwrite(str(rendered_raster_path), pages[0].image)
                except Exception as exc:
                    errors.append(ProcessingError(
                        stage="pdf_rendering",
                        code="PDF_RENDER_FAILED",
                        message=f"PDF rendering failed: {exc}",
                        severity="warning",
                    ))
            elif path.suffix.lower() in SUPPORTED_EXTENSIONS:
                try:
                    with Image.open(path) as pil_img:
                        orig_w, orig_h = pil_img.size
                except Exception as exc:
                    errors.append(ProcessingError(
                        stage="input_validation",
                        code="CORRUPT_IMAGE",
                        message=f"Image could not be read: {exc}",
                        severity="error",
                    ))
            else:
                errors.append(ProcessingError(
                    stage="input_validation",
                    code="UNSUPPORTED_FORMAT",
                    message=f"Unsupported format '{path.suffix}'",
                    severity="error",
                ))

            # Optional Preprocessing
            if self.config.enable_preprocessing and path.suffix.lower() in SUPPORTED_EXTENSIONS and not path.suffix.lower() == ".pdf":
                try:
                    opts = preprocessing_options or PreprocessingOptions.document_ocr()
                    prep_result = preprocess_image(path, opts, save_output=False)
                    orig_w, orig_h = prep_result.processed_width, prep_result.processed_height
                except Exception as exc:
                    errors.append(ProcessingError(
                        stage="image_preprocessing",
                        code="PREPROCESSING_FAILED",
                        message=f"Preprocessing failed: {exc}",
                        severity="warning",
                    ))

        working_input = prep_result if prep_result is not None else (rendered_raster_path or source_path)
        doc_id = Path(source_path).stem

        # 2. OCR Stage
        active_ocr: OCRDocumentResult | None = None
        if ocr_result is not None:
            active_ocr = ocr_result
        elif self.config.enable_ocr:
            try:
                if Path(source_path).suffix.lower() == ".pdf":
                    active_ocr = self.ocr_pipeline.process_pdf(source_path)
                else:
                    active_ocr = self.ocr_pipeline.process_image(working_input)
                if active_ocr.errors:
                    for e in active_ocr.errors:
                        errors.append(ProcessingError(
                            stage="ocr",
                            code=f"OCR_{e.code}",
                            message=e.message,
                            severity=e.severity,
                        ))
            except Exception as exc:
                errors.append(ProcessingError(
                    stage="ocr",
                    code="OCR_EXECUTION_FAILED",
                    message=f"OCR execution failed: {exc}",
                    severity="error",
                ))

        # 3. Routing Classification (Fast, non-VLM)
        effective_force = force_route or self.config.force_route
        if effective_force is not None:
            norm_force = effective_force.lower().strip()
            if norm_force in ("plain_document", "doc"):
                decision = RoutingDecision.PLAIN_DOCUMENT
            elif norm_force in ("engineering_drawing", "drawing"):
                decision = RoutingDecision.ENGINEERING_DRAWING
            elif norm_force in ("visual_inspection", "inspection", "visual"):
                decision = RoutingDecision.VISUAL_INSPECTION
            elif norm_force in ("both", "mixed"):
                decision = RoutingDecision.MIXED
            else:
                decision = RoutingDecision.PLAIN_DOCUMENT
            signals = {"forced": True, "force_value": effective_force, "reasoning": "Explicit force_route override"}
        else:
            decision, signals = classify_routing(
                source_path,
                image_width=orig_w,
                image_height=orig_h,
                ocr_result=active_ocr,
            )

        # 4. Route Execution
        parsed_doc: ParsedDocument | None = None
        drawing_rec: DrawingAnalysisRecord | None = None
        vision_rec: VisionAnalysisRecord | None = None
        active_vision: VisionResult | None = vision_result

        # Route A: Plain Document Path
        if decision in (RoutingDecision.PLAIN_DOCUMENT, RoutingDecision.MIXED) and self.config.enable_document_parsing:
            if active_ocr is not None and active_ocr.pages:
                try:
                    parsed_doc = self.document_parser.parse_ocr_result(
                        active_ocr,
                        raw_document_id=doc_id,
                        title=document_title,
                        category=document_category,
                    )
                except Exception as exc:
                    errors.append(ProcessingError(
                        stage="document_parser",
                        code="DOCUMENT_PARSER_FAILED",
                        message=f"DocumentParser failed: {exc}",
                        severity="error",
                    ))
            elif active_ocr is not None and not active_ocr.pages:
                signals["empty_ocr"] = True

        # Route B: Engineering Drawing Path
        if decision in (RoutingDecision.ENGINEERING_DRAWING, RoutingDecision.MIXED) and self.config.enable_drawing_analysis:
            drawing_raster = rendered_raster_path or working_input
            if isinstance(drawing_raster, PreprocessingResult):
                drawing_raster = drawing_raster.original_path

            active_vision = vision_result
            if active_vision is None and self.config.enable_vision:
                try:
                    active_vision = self.vision_pipeline.process_image(drawing_raster)
                except Exception as exc:
                    errors.append(ProcessingError(
                        stage="vision_pipeline",
                        code="VISION_FAILED",
                        message=f"VisionPipeline execution failed: {exc}",
                        severity="warning",
                    ))

            try:
                drawing_rec = self.drawing_analyzer.analyze_structured_drawing(
                    drawing_raster,
                    drawing_type=drawing_type,
                    ocr_result=active_ocr,
                    vlm_result=active_vision,
                    drawing_id=doc_id,
                )
            except Exception as exc:
                errors.append(ProcessingError(
                    stage="drawing_analyzer",
                    code="DRAWING_ANALYZER_FAILED",
                    message=f"DrawingAnalyzer failed: {exc}",
                    severity="error",
                ))

        # Route C: Visual Inspection Path (Defects, surface condition, thermal, components)
        if decision in (RoutingDecision.VISUAL_INSPECTION, RoutingDecision.MIXED) and self.config.enable_vision:
            inspection_raster = rendered_raster_path or working_input
            if isinstance(inspection_raster, PreprocessingResult):
                inspection_raster = inspection_raster.original_path

            if active_vision is None:
                try:
                    active_vision = self.vision_pipeline.process_image(inspection_raster)
                except Exception as exc:
                    errors.append(ProcessingError(
                        stage="vision_pipeline",
                        code="VISION_FAILED",
                        message=f"VisionPipeline execution failed: {exc}",
                        severity="error",
                    ))

            if active_vision is not None:
                try:
                    vision_rec = create_vision_analysis_record(
                        active_vision,
                        image_id=doc_id,
                        image_type="visual_inspection",
                    )
                except Exception as exc:
                    errors.append(ProcessingError(
                        stage="vision_pipeline",
                        code="VISION_RECORD_FAILED",
                        message=f"Failed to create VisionAnalysisRecord: {exc}",
                        severity="error",
                    ))

        elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
        metadata: dict[str, Any] = {
            "total_processing_time_ms": round(elapsed_ms, 2),
            "dimensions": (orig_w, orig_h),
            "ocr_blocks_count": len(active_ocr.pages[0].blocks) if active_ocr and active_ocr.pages else 0,
            "routing_reasoning": signals.get("reasoning", ""),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        modality_statuses = {
            "ocr": "success" if active_ocr and not active_ocr.errors else ("failed" if any(e.stage == "ocr" for e in errors) else "skipped"),
            "vision": "success" if (active_vision or vision_rec) else ("failed" if any(e.stage == "vision_pipeline" for e in errors) else "skipped"),
            "drawing": "success" if drawing_rec else ("failed" if any(e.stage == "drawing_analyzer" for e in errors) else "skipped"),
            "document": "success" if parsed_doc else ("failed" if any(e.stage == "document_parser" for e in errors) else "skipped"),
        }

        return MultimodalProcessingResult(
            document_id=doc_id,
            source_path=source_path,
            routing_decision=decision.value,
            routing_signals=signals,
            parsed_document=parsed_doc,
            drawing_analysis=drawing_rec,
            vision_analysis=vision_rec,
            processing_errors=tuple(errors),
            pipeline_versions=pipeline_versions,
            processing_metadata=metadata,
            modality_statuses=modality_statuses,
        )

    # ── Legacy-Compatible Process Method (Maintains 100% existing test pass) ──

    def process(
        self,
        source: str | Path | PreprocessingResult,
        *,
        ocr_result: OCRDocumentResult | None = None,
        vision_result: VisionResult | None = None,
        drawing_result: DrawingAnalysisResult | None = None,
        preprocessing_options: PreprocessingOptions | None = None,
    ) -> MultimodalResult:
        """Legacy low-level API. New integrations should use .orchestrate() instead — see Contract 6.

        This method executes low-level modality fusion using manual boolean flags on
        MultimodalProcessorConfig without automatic document classification or routing.
        It is retained strictly for backward compatibility with pre-existing internal callers
        and tests. It should not be used for new integrations, since it bypasses routing
        and always incurs the cost of whichever pipelines are manually enabled.
        """
        start_ns = time.perf_counter_ns()
        issues: list[MultimodalIssue] = []
        statuses: dict[str, ModalityStatus] = {
            "preprocessing": ModalityStatus.NOT_REQUESTED,
            "ocr": ModalityStatus.NOT_REQUESTED,
            "vision": ModalityStatus.NOT_REQUESTED,
            "drawing": ModalityStatus.NOT_REQUESTED,
            "document_parser": ModalityStatus.NOT_REQUESTED,
        }
        provenance: dict[str, Any] = {}
        processing_metadata: dict[str, Any] = {}

        if isinstance(source, PreprocessingResult):
            source_path = str(source.original_path)
            prep_result: PreprocessingResult | None = source
            width = source.processed_width
            height = source.processed_height
            statuses["preprocessing"] = ModalityStatus.SUCCESS
            provenance["preprocessing"] = {
                "source": "precomputed",
                "operations_applied": list(source.operations_applied),
            }
        else:
            path = Path(source)
            source_path = str(path)

            if not path.exists():
                return self._fatal_error(
                    source_path=source_path,
                    code="missing_image",
                    message=f"Image file does not exist: {path}",
                    statuses=statuses,
                )

            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                return self._fatal_error(
                    source_path=source_path,
                    code="unsupported_format",
                    message=f"Image format '{path.suffix}' is not supported.",
                    statuses=statuses,
                )

            try:
                with Image.open(path) as pil_img:
                    orig_w, orig_h = pil_img.size
            except Exception as exc:
                return self._fatal_error(
                    source_path=source_path,
                    code="corrupt_image",
                    message=f"Image is corrupt or unreadable: {exc}",
                    statuses=statuses,
                )

            width, height = orig_w, orig_h

            if self.config.enable_preprocessing:
                opts = preprocessing_options or PreprocessingOptions.general_vision()
                try:
                    prep_result = preprocess_image(path, opts, save_output=False)
                    width = prep_result.processed_width
                    height = prep_result.processed_height
                    statuses["preprocessing"] = ModalityStatus.SUCCESS
                    provenance["preprocessing"] = {
                        "source": "preprocess_image",
                        "operations_applied": list(prep_result.operations_applied),
                    }
                except ImagePreprocessingError as exc:
                    statuses["preprocessing"] = ModalityStatus.FAILED
                    issues.append(MultimodalIssue(
                        code="preprocessing_failed",
                        message=f"Image preprocessing failed: {exc}",
                        severity="error",
                        modality="preprocessing",
                    ))
                    if self.config.fail_fast:
                        return self._build_result(
                            success=False,
                            source_path=source_path,
                            dims=(width, height),
                            statuses=statuses,
                            issues=issues,
                            provenance=provenance,
                            metadata=processing_metadata,
                        )
                    prep_result = None
            else:
                prep_result = None

        working_input = prep_result if prep_result is not None else source_path

        # OCR
        active_ocr: OCRDocumentResult | None = None
        if ocr_result is not None:
            active_ocr = ocr_result
            statuses["ocr"] = ModalityStatus.SUCCESS
            provenance["ocr"] = {"source": "precomputed", "document_id": ocr_result.document_id}
        elif self.config.enable_ocr:
            try:
                res = self.ocr_pipeline.process_image(working_input)
                active_ocr = res
                if res.errors:
                    statuses["ocr"] = ModalityStatus.FAILED
                    for err in res.errors:
                        issues.append(MultimodalIssue(
                            code=f"ocr_{err.code}",
                            message=err.message,
                            severity=err.severity,
                            modality="ocr",
                        ))
                    if self.config.fail_fast:
                        return self._build_result(
                            success=False,
                            source_path=source_path,
                            dims=(width, height),
                            statuses=statuses,
                            issues=issues,
                            provenance=provenance,
                            metadata=processing_metadata,
                            ocr=active_ocr,
                        )
                else:
                    statuses["ocr"] = ModalityStatus.SUCCESS
                    provenance["ocr"] = {
                        "backend_name": res.backend.name,
                        "document_id": res.document_id,
                        "page_count": len(res.pages),
                    }
            except Exception as exc:
                statuses["ocr"] = ModalityStatus.FAILED
                issues.append(MultimodalIssue(
                    code="ocr_execution_failed",
                    message=f"OCR execution failed: {exc}",
                    severity="error",
                    modality="ocr",
                ))
                if self.config.fail_fast:
                    return self._build_result(
                        success=False,
                        source_path=source_path,
                        dims=(width, height),
                        statuses=statuses,
                        issues=issues,
                        provenance=provenance,
                        metadata=processing_metadata,
                    )

        # Vision
        active_vision: VisionResult | None = None
        if vision_result is not None:
            active_vision = vision_result
            statuses["vision"] = ModalityStatus.SUCCESS
            provenance["vision"] = {"source": "precomputed"}
        elif self.config.enable_vision:
            try:
                res_vis = self.vision_pipeline.process_image(working_input)
                active_vision = res_vis
                if not res_vis.success:
                    statuses["vision"] = ModalityStatus.FAILED
                    for iss in res_vis.issues:
                        issues.append(MultimodalIssue(
                            code=f"vision_{iss.code}",
                            message=iss.message,
                            severity=iss.severity,
                            modality="vision",
                        ))
                    if self.config.fail_fast:
                        return self._build_result(
                            success=False,
                            source_path=source_path,
                            dims=(width, height),
                            statuses=statuses,
                            issues=issues,
                            provenance=provenance,
                            metadata=processing_metadata,
                            ocr=active_ocr,
                            vision=active_vision,
                        )
                else:
                    statuses["vision"] = ModalityStatus.SUCCESS
                    provenance["vision"] = {
                        "backend_name": res_vis.backend.name,
                        "classifications_count": len(res_vis.classifications),
                        "detections_count": len(res_vis.detections),
                    }
            except Exception as exc:
                statuses["vision"] = ModalityStatus.FAILED
                issues.append(MultimodalIssue(
                    code="vision_execution_failed",
                    message=f"Vision execution failed: {exc}",
                    severity="error",
                    modality="vision",
                ))
                if self.config.fail_fast:
                    return self._build_result(
                        success=False,
                        source_path=source_path,
                        dims=(width, height),
                        statuses=statuses,
                        issues=issues,
                        provenance=provenance,
                        metadata=processing_metadata,
                        ocr=active_ocr,
                    )

        # Drawing Analysis
        active_drawing: DrawingAnalysisResult | None = None
        if drawing_result is not None:
            active_drawing = drawing_result
            statuses["drawing"] = ModalityStatus.SUCCESS
            provenance["drawing"] = {"source": "precomputed"}
        elif self.config.enable_drawing_analysis:
            try:
                res_drw = self.drawing_analyzer.analyze_drawing(
                    working_input,
                    ocr_result=active_ocr,
                )
                active_drawing = res_drw
                if not res_drw.success:
                    statuses["drawing"] = ModalityStatus.FAILED
                    for diss in res_drw.issues:
                        issues.append(MultimodalIssue(
                            code=f"drawing_{diss.code}",
                            message=diss.message,
                            severity=diss.severity,
                            modality="drawing",
                        ))
                    if self.config.fail_fast:
                        return self._build_result(
                            success=False,
                            source_path=source_path,
                            dims=(width, height),
                            statuses=statuses,
                            issues=issues,
                            provenance=provenance,
                            metadata=processing_metadata,
                            ocr=active_ocr,
                            vision=active_vision,
                            drawing=active_drawing,
                        )
                else:
                    statuses["drawing"] = ModalityStatus.SUCCESS
                    provenance["drawing"] = {
                        "backend_name": res_drw.backend.name,
                        "drawing_type": res_drw.drawing_type.value,
                        "regions_count": len(res_drw.regions),
                        "labels_count": len(res_drw.labels),
                    }
            except Exception as exc:
                statuses["drawing"] = ModalityStatus.FAILED
                issues.append(MultimodalIssue(
                    code="drawing_execution_failed",
                    message=f"Drawing analysis failed: {exc}",
                    severity="error",
                    modality="drawing",
                ))
                if self.config.fail_fast:
                    return self._build_result(
                        success=False,
                        source_path=source_path,
                        dims=(width, height),
                        statuses=statuses,
                        issues=issues,
                        provenance=provenance,
                        metadata=processing_metadata,
                        ocr=active_ocr,
                        vision=active_vision,
                    )

        # Document Parser
        active_parsed: ParsedDocument | None = None
        if self.config.enable_document_parsing:
            if active_ocr is not None and active_ocr.pages:
                try:
                    parsed_doc = self.document_parser.parse_ocr_result(active_ocr)
                    active_parsed = parsed_doc
                    statuses["document_parser"] = ModalityStatus.SUCCESS
                    provenance["document_parser"] = {
                        "document_id": parsed_doc.document_id,
                        "section_count": len(parsed_doc.sections),
                    }
                except Exception as exc:
                    statuses["document_parser"] = ModalityStatus.FAILED
                    issues.append(MultimodalIssue(
                        code="document_parser_failed",
                        message=f"Document parsing failed: {exc}",
                        severity="error",
                        modality="document_parser",
                    ))
            else:
                statuses["document_parser"] = ModalityStatus.SKIPPED
                issues.append(MultimodalIssue(
                    code="DOCUMENT_PARSE_REQUIRES_OCR",
                    message="Document parsing requested but no valid OCR result is available.",
                    severity="warning",
                    modality="document_parser",
                ))

        conflict_issues = self._detect_conflicts(active_ocr, active_vision, active_drawing)
        issues.extend(conflict_issues)

        combined_text = self._aggregate_text(active_ocr, active_vision, active_drawing)
        combined_structured = self._aggregate_structured(active_ocr, active_vision, active_drawing)

        requested_modalities = [
            m for m, stat in statuses.items()
            if stat in (ModalityStatus.SUCCESS, ModalityStatus.FAILED)
        ]
        any_success = any(statuses[m] == ModalityStatus.SUCCESS for m in requested_modalities)
        overall_success = any_success if requested_modalities else True

        elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
        processing_metadata["total_processing_time_ms"] = round(elapsed_ms, 3)

        return self._build_result(
            success=overall_success,
            source_path=source_path,
            dims=(width, height),
            statuses=statuses,
            issues=issues,
            provenance=provenance,
            metadata=processing_metadata,
            prep=prep_result,
            ocr=active_ocr,
            vision=active_vision,
            drawing=active_drawing,
            parsed=active_parsed,
            combined_text=combined_text,
            combined_structured=combined_structured,
        )

    # ── Legacy Aggregation Helpers ────────────────────────────────────────────

    @staticmethod
    def _aggregate_text(
        ocr: OCRDocumentResult | None,
        vision: VisionResult | None,
        drawing: DrawingAnalysisResult | None,
    ) -> list[dict[str, Any]]:
        combined: list[dict[str, Any]] = []
        seen_texts: set[str] = set()

        def add(text: str, source: str) -> None:
            t = text.strip()
            if not t:
                return
            t_lower = t.lower()
            if t_lower in seen_texts:
                return
            seen_texts.add(t_lower)
            combined.append({"text": t, "source": source})

        if ocr is not None:
            for page in ocr.pages:
                for block in page.blocks:
                    if block.lines:
                        for line in block.lines:
                            add(line.text, "ocr")
                    else:
                        for sub_text in block.text.splitlines():
                            add(sub_text, "ocr")

        if vision is not None:
            for cap in vision.captions:
                add(cap.text, "vision")

        if drawing is not None and drawing.description:
            add(drawing.description, "drawing")

        if drawing is not None:
            for lbl in drawing.labels:
                add(lbl.text, "drawing_label")

        return combined

    @staticmethod
    def _aggregate_structured(
        ocr: OCRDocumentResult | None,
        vision: VisionResult | None,
        drawing: DrawingAnalysisResult | None,
    ) -> dict[str, Any]:
        structured: dict[str, Any] = {
            "classifications": [],
            "detections": [],
            "captions": [],
            "drawing_type": None,
            "drawing_labels": [],
            "regions": [],
            "spatial_relations": [],
            "metadata": {},
        }

        if vision is not None:
            structured["classifications"] = [
                {"label": c.label, "confidence": c.confidence} for c in vision.classifications
            ]
            structured["detections"] = [
                {
                    "label": d.label,
                    "confidence": d.confidence,
                    "bbox": asdict(d.bbox),
                    "attributes": d.attributes,
                }
                for d in vision.detections
            ]
            structured["captions"] = [
                {"text": cap.text, "confidence": cap.confidence} for cap in vision.captions
            ]

        if drawing is not None:
            structured["drawing_type"] = drawing.drawing_type.value
            structured["drawing_labels"] = [
                {
                    "label_id": lbl.label_id,
                    "text": lbl.text,
                    "label_type": lbl.label_type,
                    "confidence": lbl.confidence,
                    "bbox": asdict(lbl.bbox),
                }
                for lbl in drawing.labels
            ]
            structured["regions"] = [
                {
                    "region_id": r.region_id,
                    "region_type": r.region_type,
                    "label": r.label,
                    "confidence": r.confidence,
                    "bbox": asdict(r.bbox),
                }
                for r in drawing.regions
            ]
            structured["spatial_relations"] = [
                {
                    "relation_id": s.relation_id,
                    "source_id": s.source_id,
                    "relation": s.relation.value if hasattr(s.relation, "value") else str(s.relation),
                    "target_id": s.target_id,
                    "confidence": s.confidence,
                }
                for s in drawing.spatial_relations
            ]
            structured["metadata"] = asdict(drawing.metadata)

        return structured

    @staticmethod
    def _detect_conflicts(
        ocr: OCRDocumentResult | None,
        vision: VisionResult | None,
        drawing: DrawingAnalysisResult | None,
    ) -> list[MultimodalIssue]:
        conflicts: list[MultimodalIssue] = []
        if ocr is None or drawing is None:
            return conflicts

        eq_re = re.compile(r"\b([A-Z]{1,4}-\d{2,5}[A-Z]?)\b")
        ocr_tags: set[str] = set()
        for page in ocr.pages:
            for block in page.blocks:
                for match in eq_re.finditer(block.text):
                    ocr_tags.add(match.group(1).upper())

        drw_tags: set[str] = set()
        for lbl in drawing.labels:
            if getattr(lbl, "source", None) == "ocr":
                continue
            if lbl.label_type == "equipment_tag" or eq_re.match(lbl.text):
                drw_tags.add(lbl.text.upper())
        for reg in drawing.regions:
            tag_attr = reg.attributes.get("tag")
            if tag_attr and isinstance(tag_attr, str):
                drw_tags.add(tag_attr.upper())
            elif reg.label and eq_re.match(reg.label):
                drw_tags.add(reg.label.upper())

        if ocr_tags and drw_tags and ocr_tags.isdisjoint(drw_tags):
            conflicts.append(MultimodalIssue(
                code="MULTIMODAL_CONFLICT",
                message=(
                    f"Equipment tag mismatch between OCR and Drawing: "
                    f"OCR={sorted(ocr_tags)} vs Drawing={sorted(drw_tags)}"
                ),
                severity="warning",
                modality="multimodal",
                details={
                    "modality_a": "ocr",
                    "modality_b": "drawing",
                    "value_a": sorted(ocr_tags),
                    "value_b": sorted(drw_tags),
                    "evidence": "Equipment tag extraction produced disjoint identifiers",
                },
            ))

        return conflicts

    def _fatal_error(
        self,
        source_path: str,
        code: str,
        message: str,
        statuses: dict[str, ModalityStatus],
    ) -> MultimodalResult:
        return MultimodalResult(
            success=False,
            source_path=source_path,
            image_dimensions=(0, 0),
            modality_statuses=statuses,
            issues=(MultimodalIssue(code=code, message=message, severity="error"),),
        )

    def _build_result(
        self,
        success: bool,
        source_path: str,
        dims: tuple[int, int],
        statuses: dict[str, ModalityStatus],
        issues: list[MultimodalIssue],
        provenance: dict[str, Any],
        metadata: dict[str, Any],
        prep: Any | None = None,
        ocr: OCRDocumentResult | None = None,
        vision: VisionResult | None = None,
        drawing: DrawingAnalysisResult | None = None,
        parsed: ParsedDocument | None = None,
        combined_text: list[dict[str, Any]] | None = None,
        combined_structured: dict[str, Any] | None = None,
    ) -> MultimodalResult:
        return MultimodalResult(
            success=success,
            source_path=source_path,
            image_dimensions=dims,
            preprocessing_result=prep,
            ocr_result=ocr,
            vision_result=vision,
            drawing_result=drawing,
            parsed_document=parsed,
            modality_statuses=statuses,
            combined_text=combined_text or [],
            combined_structured=combined_structured or {},
            issues=tuple(issues),
            provenance=provenance,
            processing_metadata=metadata,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Public Module-Level Orchestrator Function
# ──────────────────────────────────────────────────────────────────────────────

def process_document(
    source: str | Path | PreprocessingResult,
    *,
    force_route: str | None = None,
    ocr_result: OCRDocumentResult | None = None,
    vision_result: Any | None = None,
    config: MultimodalProcessorConfig | None = None,
    document_title: str | None = None,
    document_category: str = "Unknown",
    drawing_type: DrawingType | str | None = None,
) -> MultimodalProcessingResult:
    """Convenience functional orchestrator returning a MultimodalProcessingResult."""
    cfg = config or MultimodalProcessorConfig()
    processor = MultimodalProcessor(config=cfg)
    return processor.orchestrate(
        source,
        force_route=force_route,
        ocr_result=ocr_result,
        vision_result=vision_result,
        document_title=document_title,
        document_category=document_category,
        drawing_type=drawing_type,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Command Line Interface
# ──────────────────────────────────────────────────────────────────────────────

def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for running multimodal intelligence orchestration."""
    parser = argparse.ArgumentParser(
        description="Offline Multimodal Processor for Sovereign AI Workbench"
    )
    parser.add_argument("--input", "-i", required=True, help="Path to input image or PDF file")
    parser.add_argument(
        "--force-route",
        choices=["plain_document", "engineering_drawing", "both"],
        default=None,
        help="Explicitly override auto-classification routing path",
    )
    parser.add_argument(
        "--export-format",
        choices=["rag", "agent", "both", "none"],
        default="none",
        help="Format to export downstream: rag (chunks), agent (queryable), both, or none",
    )
    parser.add_argument("--json", action="store_true", help="Output result envelope as JSON")
    parser.add_argument("--no-ocr", action="store_true", help="Disable OCR modality")
    parser.add_argument("--no-vision", action="store_true", help="Disable Vision modality")
    parser.add_argument("--no-drawing", action="store_true", help="Disable Drawing analysis modality")
    parser.add_argument("--parse-document", action="store_true", help="Enable DocumentParser integration")
    parser.add_argument("--preprocess", action="store_true", help="Enable image preprocessing")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first modality failure")

    args = parser.parse_args(argv)

    config = MultimodalProcessorConfig(
        enable_preprocessing=args.preprocess,
        enable_ocr=not args.no_ocr,
        enable_vision=not args.no_vision,
        enable_drawing_analysis=not args.no_drawing,
        enable_document_parsing=args.parse_document or True,
        force_route=args.force_route,
        fail_fast=args.fail_fast,
    )

    processor = MultimodalProcessor(config=config)
    result = processor.orchestrate(args.input, force_route=args.force_route)

    if args.json:
        print(result.to_json(indent=2))
    else:
        status_str = "FAILED" if result.has_errors else "SUCCESS"
        print(f"Multimodal Processing [{status_str}]: {result.source_path}")
        print(f"  Routing Decision: {result.routing_decision}")
        print(f"  Signals: {result.routing_signals.get('reasoning', '')}")
        if result.parsed_document:
            print(f"  Parsed Document: {len(result.parsed_document.sections)} sections, {len(result.parsed_document.tables)} tables")
        if result.drawing_analysis:
            dwg = result.drawing_analysis
            print(f"  Drawing Analysis: {len(dwg.equipment)} equipment, {len(dwg.instruments)} instruments, {len(dwg.connections)} connections")
        if result.processing_errors:
            print("  Processing Errors / Warnings:")
            for err in result.processing_errors:
                print(f"    - [{err.severity.upper()}] ({err.stage}) {err.code}: {err.message}")

    if args.export_format in ("rag", "both"):
        chunks = export_for_rag(result)
        print(f"\nExported {len(chunks)} RAG Chunks:")
        for idx, c in enumerate(chunks[:5]):
            print(f"  [{idx}] {c.metadata.chunk_strategy.upper()} - {c.metadata.section_title} ({c.token_count} tokens)")
        if len(chunks) > 5:
            print(f"  ... and {len(chunks) - 5} more chunks")

    if args.export_format in ("agent", "both"):
        agent_doc = export_for_agent(result)
        summary = agent_doc.get_summary()
        print(f"\nAgent Queryable Interface Summary: {json.dumps(summary, indent=2)}")

    return 1 if result.has_errors else 0


if __name__ == "__main__":
    sys.exit(main())
