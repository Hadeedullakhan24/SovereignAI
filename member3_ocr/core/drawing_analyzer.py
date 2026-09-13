"""Robust, model-agnostic engineering drawing analysis foundation for Sovereign AI Workbench (MRPL).

This module provides a local, offline-only architecture for analyzing industrial
drawings (P&ID, PFD, Equipment, Electrical, Instrumentation, Plant Layout, etc.).

FOUNDATION CAPABILITIES:
- Drawing type abstraction and conservative classification (metadata/backend hints)
- Explicit region representation (title blocks, equipment areas, notes, annotations)
- OCR coordination and industrial tag extraction (equipment tags, instrument tags)
- Deterministic geometric/spatial analysis (LEFT_OF, RIGHT_OF, ABOVE, BELOW, OVERLAPS, NEAR)
- Strict provenance tracking for auditable industrial AI workflows
- Offline-only guarantees: zero model downloads, zero external API dependencies
- Non-destructive image handling: source files and datasets/ are never modified

CAPABILITIES REQUIRING FUTURE VALIDATED MODELS:
- Engineering-grade P&ID symbol recognition
- Guaranteed pipe connectivity and flow tracing
- Automated valve/instrument state identification
- 100% equipment boundary detection

NOTE:
CONNECTED_TO relations are NEVER manufactured from physical proximity alone.
Geometry calculations produce spatial relations (LEFT_OF, RIGHT_OF, ABOVE, BELOW,
OVERLAPS, NEAR). CONNECTED_TO is reserved for explicit model/graph evidence.

Bounding Box Coordinate Convention:
Bounding box coordinates in ``DrawingRegion``, ``DrawingLabel``, and ``BoundingBox`` use
axis-aligned pixel coordinates:
    left   : float (X minimum >= 0)
    top    : float (Y minimum >= 0)
    right  : float (X maximum >= left)
    bottom : float (Y maximum >= top)
    coordinate_space : "processed_pixels"
"""
from __future__ import annotations


import argparse
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
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
from PIL import Image

try:
    from rag_engine.schemas.chunk import Chunk, ChunkMetadata
except ImportError:
    Chunk = None  # type: ignore[assignment, misc]
    ChunkMetadata = None  # type: ignore[assignment, misc]

from .common_validators import (
    DRAWING_SYNONYMS,
    RE_DIMENSION,
    RE_PRESSURE,
    RE_PROXIMITY_CONNECTION,
    RE_SAFETY_CLAIM,
    RE_TEMPERATURE,
    check_drawing_type_match,
    check_hallucinations,
    inspect_pid_outputs,
)
from .image_preprocessing import (
    SUPPORTED_EXTENSIONS,
    ImagePreprocessingError,
    PreprocessingOptions,
    PreprocessingResult,
    preprocess_image,
)
from .ocr_pipeline import BoundingBox, OCRDocumentResult

LOGGER = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"
ANALYZER_VERSION = "1.0.0"

VECTOR_EXTENSIONS = frozenset({".svg", ".pdf", ".dwg", ".dxf"})


# ──────────────────────────────────────────────────────────────────────────────
# Drawing Types & Enums
# ──────────────────────────────────────────────────────────────────────────────

class DrawingType(StrEnum):
    """Refinery and industrial engineering drawing classifications."""

    UNKNOWN = "UNKNOWN"
    PID = "PID"
    PFD = "PFD"
    EQUIPMENT = "EQUIPMENT"
    INSTRUMENTATION = "INSTRUMENTATION"
    ELECTRICAL = "ELECTRICAL"
    PLANT_LAYOUT = "PLANT_LAYOUT"
    PUMP = "PUMP"
    CAD = "CAD"
    OTHER = "OTHER"


class SpatialRelationType(StrEnum):
    """Spatial and topological relationships between drawing regions."""

    LEFT_OF = "LEFT_OF"
    RIGHT_OF = "RIGHT_OF"
    ABOVE = "ABOVE"
    BELOW = "BELOW"
    INSIDE = "INSIDE"
    NEAR = "NEAR"
    OVERLAPS = "OVERLAPS"
    CONNECTED_TO = "CONNECTED_TO"


# ──────────────────────────────────────────────────────────────────────────────
# Exceptions
# ──────────────────────────────────────────────────────────────────────────────

class DrawingAnalyzerError(RuntimeError):
    """Base exception for drawing analyzer failures."""


class DrawingConfigurationError(DrawingAnalyzerError):
    """Raised when analyzer configuration is invalid or violates offline constraints."""


class MissingLocalModelError(DrawingConfigurationError):
    """Raised when an explicitly configured local model path does not exist."""


class DrawingBackendUnavailableError(DrawingAnalyzerError):
    """Raised when a configured drawing backend cannot be loaded."""


class DrawingBackendExecutionError(DrawingAnalyzerError):
    """Raised when a drawing backend fails during inference."""


# ──────────────────────────────────────────────────────────────────────────────
# Core Data Structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DrawingAnalyzerConfig:
    """Configuration for drawing analysis foundation.

    Enforces strict offline operation. Network downloads are unconditionally forbidden.
    """

    model_path: Path | str | None = None
    model_name: str = "local_drawing_model"
    device: str = "cpu"
    confidence_threshold: float = 0.5
    allow_downloads: bool = False
    preprocess: bool = False
    enable_ocr: bool = True
    enable_geometry_analysis: bool = True
    near_distance_threshold: float = 50.0

    def __post_init__(self) -> None:
        if self.allow_downloads:
            raise DrawingConfigurationError(
                "Model downloads are strictly forbidden by this offline drawing analyzer. "
                "Set allow_downloads=False and provide an explicit local model_path."
            )

    def validate(self) -> None:
        """Validate settings and verify local model path existence."""
        if self.allow_downloads:
            raise DrawingConfigurationError(
                "Model downloads are strictly forbidden by this offline drawing analyzer."
            )
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise DrawingConfigurationError(
                f"confidence_threshold must be between 0.0 and 1.0, got {self.confidence_threshold}"
            )
        if self.model_path is not None:
            path = Path(self.model_path)
            if not path.exists():
                raise MissingLocalModelError(
                    f"Configured local model path does not exist: {path}. "
                    "Ensure models are staged on the local filesystem."
                )


@dataclass(frozen=True)
class DrawingIssue:
    """Structured diagnostic issue, warning, or error."""

    code: str
    message: str
    severity: str = "warning"  # "error", "warning", "info"
    details: dict[str, Any] | None = None


@dataclass(frozen=True)
class DrawingRegion:
    """A detected or segmented region of an engineering drawing."""

    region_id: str
    region_type: str  # title_block, equipment, pipe, instrument, annotation, legend, table, notes, unknown
    bbox: BoundingBox
    label: str = ""
    confidence: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DrawingLabel:
    """An extracted text label, tag, or note from a drawing."""

    label_id: str
    text: str
    bbox: BoundingBox
    confidence: float | None = None
    label_type: str = "unknown"  # equipment_tag, instrument_tag, pipe_tag, title, note, dimension, unknown
    source: str = "ocr"  # ocr, backend, metadata


@dataclass(frozen=True)
class SpatialRelation:
    """Spatial or geometric relation between two drawing regions.

    IMPORTANT: CONNECTED_TO must NOT be inferred solely from geometric proximity.
    """

    relation_id: str
    source_id: str
    relation: SpatialRelationType
    target_id: str
    confidence: float | None = None
    evidence: str = "geometry"  # geometry, backend, ocr


@dataclass(frozen=True)
class DrawingMetadata:
    """Operational and structural metadata extracted from an engineering drawing."""

    drawing_type: DrawingType = DrawingType.UNKNOWN
    title: str | None = None
    drawing_number: str | None = None
    revision: str | None = None
    date: str | None = None
    plant_unit: str | None = None
    dimensions: tuple[int, int] = (0, 0)  # (width, height)
    units: str | None = None
    source: str = "unknown"  # caller, metadata, backend, ocr
    equipment_count: int = 0
    label_count: int = 0
    region_count: int = 0


@dataclass(frozen=True)
class DrawingBackendCapabilities:
    """Capabilities provided by a drawing analysis backend."""

    drawing_classification: bool = False
    symbol_detection: bool = False
    region_segmentation: bool = False
    text_association: bool = False
    spatial_reasoning: bool = False
    connectivity_tracing: bool = False


@dataclass(frozen=True)
class BackendInfo:
    """Provenance and metadata describing the drawing backend."""

    name: str
    version: str | None = None
    model_path: str | None = None
    device: str = "cpu"
    capabilities: DrawingBackendCapabilities = field(default_factory=DrawingBackendCapabilities)


@dataclass(frozen=True)
class DrawingAnalysisPayload:
    """Normalized payload returned by a drawing analysis backend adapter."""

    drawing_type: DrawingType = DrawingType.UNKNOWN
    regions: tuple[DrawingRegion, ...] = ()
    labels: tuple[DrawingLabel, ...] = ()
    spatial_relations: tuple[SpatialRelation, ...] = ()
    metadata: DrawingMetadata = field(default_factory=DrawingMetadata)
    description: str = ""
    issues: tuple[DrawingIssue, ...] = ()


@dataclass(frozen=True)
class DrawingAnalysisResult:
    """Normalized, JSON-serializable engineering drawing analysis result."""

    success: bool
    source_path: str
    drawing_type: DrawingType
    metadata: DrawingMetadata
    regions: tuple[DrawingRegion, ...] = ()
    labels: tuple[DrawingLabel, ...] = ()
    spatial_relations: tuple[SpatialRelation, ...] = ()
    description: str = ""
    issues: tuple[DrawingIssue, ...] = ()
    backend: BackendInfo = field(default_factory=lambda: BackendInfo(name="unknown"))
    processing_metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Convert result into a pure JSON-serializable dictionary."""
        return _to_serializable(self)

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize result as UTF-8 JSON text."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


# ──────────────────────────────────────────────────────────────────────────────
# Standardized Output Schema: DrawingAnalysisRecord & Entities
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TitleBlockInfo:
    """Standard title block information extracted from an engineering drawing."""

    drawing_number: str | None = None
    revision: str | None = None
    title: str | None = None
    date: str | None = None
    scale: str | None = None
    plant_unit: str | None = None
    source: str = "unknown"


@dataclass(frozen=True)
class EquipmentEntry:
    """Consolidated equipment entry with cross-referenced OCR and VLM evidence."""

    tag: str
    equipment_type: str
    source: str  # "ocr" | "vlm" | "both"
    confidence: float | None = None
    bbox: BoundingBox | None = None
    evidence: str = ""


@dataclass(frozen=True)
class InstrumentEntry:
    """Instrument bubble entry with parsed ISA-5.1 functional identification."""

    tag: str
    instrument_type: str
    function_code: str
    loop_number: str
    source: str = "ocr"  # "ocr" | "vlm" | "both"
    confidence: float | None = None
    bbox: BoundingBox | None = None
    evidence: str = ""


@dataclass(frozen=True)
class ConnectionEntry:
    """Evidenced topological connection between equipment or instruments.

    Non-negotiable rule: Proximity alone is NEVER a connection. An edge must be
    substantiated by an explicit textual statement or a drawn line trace.
    """

    from_tag: str
    to_tag: str
    line_tag: str | None = None
    evidence_type: str = "explicit_label"  # "drawn_line" | "explicit_label"
    source_evidence: str = ""
    confidence: float | None = None


@dataclass(frozen=True)
class DrawingCrossReference:
    """Drawing reference to standards, related drawings, or documentation."""

    target: str
    ref_type: str  # "drawing" | "standard" | "figure" | "table" | "section"
    source_text: str = ""
    confidence: float | None = None


@dataclass(frozen=True)
class DrawingAnalysisRecord:
    """Standardized, authoritative single-source-of-truth representation of an engineering drawing.

    Feeds evaluation harnesses, RAG chunk export, and Agent query interfaces.
    """

    drawing_id: str
    source_path: str
    drawing_type: DrawingType
    title_block: TitleBlockInfo
    equipment: tuple[EquipmentEntry, ...] = ()
    instruments: tuple[InstrumentEntry, ...] = ()
    connections: tuple[ConnectionEntry, ...] = ()
    cross_references: tuple[DrawingCrossReference, ...] = ()
    extraction_metadata: dict[str, Any] = field(default_factory=dict)
    hallucination_flags: tuple[str, ...] = ()
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Convert record into a pure JSON-serializable dictionary."""
        return _to_serializable(self)

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize record as UTF-8 JSON text."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def aggregate_confidence(self, entity_type: str = "equipment") -> float | None:
        """Compute mean confidence across entities, safely ignoring unmeasured (None) entries."""
        entities = getattr(self, entity_type, ())
        return compute_aggregate_confidence(entities)


def compute_aggregate_confidence(
    items: Sequence[Any],
    *,
    field_name: str = "confidence",
) -> float | None:
    """Calculate mean confidence across entries, strictly excluding unmeasured (None) entries.

    Returns None if no entries have a measured confidence value or items is empty.
    """
    measured: list[float] = []
    for item in items:
        val = getattr(item, field_name, None) if hasattr(item, field_name) else (item.get(field_name) if isinstance(item, dict) else None)
        if val is not None:
            measured.append(float(val))
    if not measured:
        return None
    return round(float(np.mean(measured)), 4)


def is_trusted_confidence(confidence: float | None, threshold: float) -> bool:
    """Determine if a confidence score satisfies a trust threshold.

    None represents an unmeasured value and is NEVER trusted (returns False).
    """
    if confidence is None:
        return False
    return confidence >= threshold


class AgentQueryableDrawing:
    """Read-only view of a DrawingAnalysisRecord optimized for agent tool execution."""

    def __init__(self, record: DrawingAnalysisRecord) -> None:
        self._record = record

    @property
    def drawing_id(self) -> str:
        return self._record.drawing_id

    @property
    def drawing_number(self) -> str | None:
        return self._record.title_block.drawing_number

    @property
    def drawing_type(self) -> str:
        dt = self._record.drawing_type
        return dt.value if hasattr(dt, "value") else str(dt)

    def get_title_block(self) -> dict[str, Any]:
        """Return title block metadata as a dictionary."""
        return _to_serializable(self._record.title_block)

    def get_equipment_list(self) -> list[dict[str, Any]]:
        """Return list of all registered equipment entries."""
        return [_to_serializable(e) for e in self._record.equipment]

    def get_trusted_equipment(self, threshold: float = 0.5) -> list[dict[str, Any]]:
        """Return registered equipment entries meeting the trust threshold.

        Entries with unmeasured confidence (None) are treated as untrusted and excluded.
        """
        results: list[dict[str, Any]] = []
        for e in self._record.equipment:
            if is_trusted_confidence(e.confidence, threshold):
                results.append(_to_serializable(e))
        return results

    def get_connections_for(self, tag: str) -> list[dict[str, Any]]:
        """Return all verified topological connections involving the specified tag."""
        tag_norm = tag.strip().upper()
        results: list[dict[str, Any]] = []
        for c in self._record.connections:
            if c.from_tag.upper() == tag_norm or c.to_tag.upper() == tag_norm:
                results.append(_to_serializable(c))
        return results

    def get_instrument_readings(self) -> list[dict[str, Any]]:
        """Return list of all identified instruments and control loops."""
        return [_to_serializable(i) for i in self._record.instruments]

    def get_instrument_list(self) -> list[dict[str, Any]]:
        """Return list of all identified instruments and control loops."""
        return self.get_instrument_readings()

    def get_trusted_instruments(self, threshold: float = 0.5) -> list[dict[str, Any]]:
        """Return registered instrument entries meeting the trust threshold.

        Entries with unmeasured confidence (None) are treated as untrusted and excluded.
        """
        results: list[dict[str, Any]] = []
        for i in self._record.instruments:
            if is_trusted_confidence(i.confidence, threshold):
                results.append(_to_serializable(i))
        return results

    def citation_for_equipment(self, tag: str) -> dict[str, Any] | None:
        """Return provenance citation and bounding box for a given equipment/instrument tag."""
        tag_norm = tag.strip().upper()
        for e in self._record.equipment:
            if e.tag.upper() == tag_norm:
                return {
                    "tag": e.tag,
                    "equipment_type": e.equipment_type,
                    "source": e.source,
                    "confidence": e.confidence,
                    "confidence_available": e.confidence is not None,
                    "bbox": _to_serializable(e.bbox) if e.bbox else None,
                    "evidence": e.evidence,
                }
        for i in self._record.instruments:
            if i.tag.upper() == tag_norm:
                return {
                    "tag": i.tag,
                    "instrument_type": i.instrument_type,
                    "source": i.source,
                    "confidence": i.confidence,
                    "confidence_available": i.confidence is not None,
                    "bbox": _to_serializable(i.bbox) if i.bbox else None,
                    "evidence": i.evidence,
                }
        return None

    def get_drawing_summary(self) -> str:
        """Return a natural language summary of the drawing for LLM context injection."""
        tb = self._record.title_block
        lines = [
            f"Drawing {tb.drawing_number or self._record.drawing_id} ({self.drawing_type})",
            f"Title: {tb.title or 'Untitled'}",
            f"Equipment ({len(self._record.equipment)}): " + (", ".join(e.tag for e in self._record.equipment[:10]) or "None"),
            f"Instruments ({len(self._record.instruments)}): " + (", ".join(i.tag for i in self._record.instruments[:10]) or "None"),
            f"Verified Connections ({len(self._record.connections)}): " + (", ".join(f"{c.from_tag}->{c.to_tag}" for c in self._record.connections[:5]) or "None"),
        ]
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Serialization Helper
# ──────────────────────────────────────────────────────────────────────────────

def _to_serializable(value: Any) -> Any:
    """Recursively convert dataclasses, enums, and numpy objects to standard primitives."""
    if isinstance(value, bool):
        return value
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {k: _to_serializable(v) for k, v in asdict(value).items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
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
# Geometry & Spatial Calculations
# ──────────────────────────────────────────────────────────────────────────────

def compute_box_distance(box1: BoundingBox, box2: BoundingBox) -> float:
    """Compute Euclidean distance between the boundaries of two axis-aligned boxes.

    If boxes overlap or touch, the distance is 0.0.
    """
    dx = max(0.0, box1.left - box2.right, box2.left - box1.right)
    dy = max(0.0, box1.top - box2.bottom, box2.top - box1.bottom)
    return float(np.hypot(dx, dy))


def is_left_of(source: BoundingBox, target: BoundingBox) -> bool:
    """True if source is strictly to the left of target."""
    return source.right <= target.left


def is_right_of(source: BoundingBox, target: BoundingBox) -> bool:
    """True if source is strictly to the right of target."""
    return source.left >= target.right


def is_above(source: BoundingBox, target: BoundingBox) -> bool:
    """True if source is strictly above target (Y increases downwards in image space)."""
    return source.bottom <= target.top


def is_below(source: BoundingBox, target: BoundingBox) -> bool:
    """True if source is strictly below target."""
    return source.top >= target.bottom


def is_overlapping(source: BoundingBox, target: BoundingBox) -> bool:
    """True if source and target bounding boxes intersect with positive overlap area."""
    return not (
        source.right <= target.left
        or source.left >= target.right
        or source.bottom <= target.top
        or source.top >= target.bottom
    )


def is_near(source: BoundingBox, target: BoundingBox, threshold: float = 50.0) -> bool:
    """True if distance between boxes is <= threshold and they do not overlap."""
    if is_overlapping(source, target):
        return False
    return compute_box_distance(source, target) <= threshold


def compute_spatial_relations(
    regions: Sequence[DrawingRegion],
    *,
    near_threshold: float = 50.0,
) -> list[SpatialRelation]:
    """Compute deterministic pairwise spatial relations between regions.

    CRITICAL GUARANTEE:
    CONNECTED_TO is never generated by geometric analysis alone. Only geometric
    spatial relations (LEFT_OF, RIGHT_OF, ABOVE, BELOW, OVERLAPS, NEAR) are computed.
    """
    relations: list[SpatialRelation] = []
    for i, r1 in enumerate(regions):
        for j, r2 in enumerate(regions):
            if i == j:
                continue
            b1, b2 = r1.bbox, r2.bbox

            if is_overlapping(b1, b2):
                relations.append(SpatialRelation(
                    relation_id=f"rel_{r1.region_id}_OVERLAPS_{r2.region_id}",
                    source_id=r1.region_id,
                    relation=SpatialRelationType.OVERLAPS,
                    target_id=r2.region_id,
                    confidence=None,
                    evidence="geometry",
                ))
            else:
                if is_left_of(b1, b2):
                    relations.append(SpatialRelation(
                        relation_id=f"rel_{r1.region_id}_LEFT_OF_{r2.region_id}",
                        source_id=r1.region_id,
                        relation=SpatialRelationType.LEFT_OF,
                        target_id=r2.region_id,
                        confidence=None,
                        evidence="geometry",
                    ))
                elif is_right_of(b1, b2):
                    relations.append(SpatialRelation(
                        relation_id=f"rel_{r1.region_id}_RIGHT_OF_{r2.region_id}",
                        source_id=r1.region_id,
                        relation=SpatialRelationType.RIGHT_OF,
                        target_id=r2.region_id,
                        confidence=None,
                        evidence="geometry",
                    ))

                if is_above(b1, b2):
                    relations.append(SpatialRelation(
                        relation_id=f"rel_{r1.region_id}_ABOVE_{r2.region_id}",
                        source_id=r1.region_id,
                        relation=SpatialRelationType.ABOVE,
                        target_id=r2.region_id,
                        confidence=None,
                        evidence="geometry",
                    ))
                elif is_below(b1, b2):
                    relations.append(SpatialRelation(
                        relation_id=f"rel_{r1.region_id}_BELOW_{r2.region_id}",
                        source_id=r1.region_id,
                        relation=SpatialRelationType.BELOW,
                        target_id=r2.region_id,
                        confidence=None,
                        evidence="geometry",
                    ))

                if is_near(b1, b2, threshold=near_threshold):
                    relations.append(SpatialRelation(
                        relation_id=f"rel_{r1.region_id}_NEAR_{r2.region_id}",
                        source_id=r1.region_id,
                        relation=SpatialRelationType.NEAR,
                        target_id=r2.region_id,
                        confidence=None,
                        evidence="geometry",
                    ))
    return relations


# ──────────────────────────────────────────────────────────────────────────────
# Tag & Metadata Extraction from OCR
# ──────────────────────────────────────────────────────────────────────────────

_EQUIPMENT_TAG_RE = re.compile(r"\b([A-Z]{1,4}-\d{2,5}[A-Z]?)\b")
_INSTRUMENT_TAG_RE = re.compile(r"\b((?:PI|TI|PT|TT|FCV|MOV|LT|FIC|XV|PSV|PDT)-\d{2,5}[A-Z]?)\b")
_DRAWING_NUM_RE = re.compile(r"\b(?:DWG(?:[-_]NO)?|DRAWING(?:[-_]NO)?|NO\.?)[:\s]*([A-Z0-9_-]{4,25})\b", re.IGNORECASE)
_REV_RE = re.compile(r"\b(?:REV(?:ISION)?)[.:\s]*([A-Z0-9]{1,5})\b", re.IGNORECASE)


def extract_labels_from_ocr(ocr_result: OCRDocumentResult) -> list[DrawingLabel]:
    """Extract candidate drawing labels from an OCRDocumentResult."""
    labels: list[DrawingLabel] = []
    label_idx = 0
    for page in ocr_result.pages:
        for block in page.blocks:
            text = block.text.strip()
            if not text:
                continue

            inst_match = _INSTRUMENT_TAG_RE.search(text)
            eq_match = _EQUIPMENT_TAG_RE.search(text)

            if inst_match:
                tag = inst_match.group(1)
                labels.append(DrawingLabel(
                    label_id=f"label_{label_idx}",
                    text=tag,
                    bbox=block.bbox,
                    confidence=block.confidence,
                    label_type="instrument_tag",
                    source="ocr",
                ))
                label_idx += 1
            elif eq_match:
                tag = eq_match.group(1)
                labels.append(DrawingLabel(
                    label_id=f"label_{label_idx}",
                    text=tag,
                    bbox=block.bbox,
                    confidence=block.confidence,
                    label_type="equipment_tag",
                    source="ocr",
                ))
                label_idx += 1
            else:
                labels.append(DrawingLabel(
                    label_id=f"label_{label_idx}",
                    text=text,
                    bbox=block.bbox,
                    confidence=block.confidence,
                    label_type="note",
                    source="ocr",
                ))
                label_idx += 1
    return labels


def extract_metadata_from_ocr(
    ocr_result: OCRDocumentResult,
    default_metadata: DrawingMetadata | None = None,
) -> DrawingMetadata:
    """Extract conservative metadata (drawing number, revision) from OCR blocks."""
    meta = default_metadata or DrawingMetadata()
    dwg_num = meta.drawing_number
    rev = meta.revision
    title = meta.title

    for page in ocr_result.pages:
        for block in page.blocks:
            text = block.text.strip()
            if not dwg_num:
                m = _DRAWING_NUM_RE.search(text)
                if m:
                    dwg_num = m.group(1).strip()
            if not rev:
                m = _REV_RE.search(text)
                if m:
                    rev = m.group(1).strip()

    return DrawingMetadata(
        drawing_type=meta.drawing_type,
        title=title,
        drawing_number=dwg_num,
        revision=rev,
        date=meta.date,
        plant_unit=meta.plant_unit,
        dimensions=meta.dimensions,
        units=meta.units,
        source=meta.source,
        equipment_count=meta.equipment_count,
        label_count=meta.label_count,
        region_count=meta.region_count,
    )


# ──────────────────────────────────────────────────────────────────────────────
# ISA-5.1 Instrumentation & Equipment Mapping
# ──────────────────────────────────────────────────────────────────────────────

ISA_FIRST_LETTER: dict[str, str] = {
    "P": "Pressure",
    "T": "Temperature",
    "F": "Flow",
    "L": "Level",
    "A": "Analysis",
    "D": "Differential",
    "V": "Vibration",
    "E": "Voltage",
    "I": "Current",
    "J": "Power",
    "M": "Moisture",
    "S": "Speed",
    "Z": "Position",
    "W": "Weight",
}

ISA_SUCCEEDING_LETTERS: dict[str, str] = {
    "I": "Indicator",
    "T": "Transmitter",
    "C": "Controller",
    "V": "Valve",
    "S": "Switch",
    "A": "Alarm",
    "G": "Gauge",
    "R": "Recorder",
    "E": "Element",
    "Y": "Relay",
}

SPECIAL_INSTRUMENT_TYPES: dict[str, str] = {
    "FCV": "Flow Control Valve",
    "PCV": "Pressure Control Valve",
    "TCV": "Temperature Control Valve",
    "LCV": "Level Control Valve",
    "MOV": "Motor Operated Valve",
    "XV": "Shutdown Valve",
    "PSV": "Pressure Safety Valve",
    "PRV": "Pressure Relief Valve",
    "TSV": "Temperature Safety Valve",
    "FIC": "Flow Indicating Controller",
    "PIC": "Pressure Indicating Controller",
    "TIC": "Temperature Indicating Controller",
    "LIC": "Level Indicating Controller",
    "PDT": "Differential Pressure Transmitter",
    "PDI": "Differential Pressure Indicator",
    "FIT": "Flow Indicating Transmitter",
    "PIT": "Pressure Indicating Transmitter",
    "TIT": "Temperature Indicating Transmitter",
    "LIT": "Level Indicating Transmitter",
}

EQUIPMENT_PREFIX_MAP: dict[str, str] = {
    "P": "Pump",
    "V": "Vessel",
    "TK": "Tank",
    "T": "Tank",
    "E": "Heat Exchanger",
    "C": "Column",
    "K": "Compressor",
    "B": "Boiler",
    "F": "Furnace",
    "R": "Reactor",
    "M": "Motor",
    "ST": "Strainer",
    "FL": "Filter",
    "D": "Drum",
    "HX": "Heat Exchanger",
    "HE": "Heat Exchanger",
}

_ISA_INSTRUMENT_RE = re.compile(r"\b([A-Z]{2,4})[-_ ]?(\d{2,5}[A-Z]?)\b")

_RE_CROSS_REF_DRAWING = re.compile(
    r"\b(?:SEE\s+|REFER\s+TO\s+)?((?:DWG|DRAWING)[-_ ]?(?:NO\.?)?[-_ ]?[A-Z0-9_-]{2,30}|[A-Z]{2,4}-PID-[0-9]{3,6})\b",
    re.IGNORECASE,
)
_RE_CROSS_REF_STANDARD = re.compile(
    r"\b(API\s*[-_]?\d+[A-Z]*|ASME\s+[A-Z0-9.]+|OISD(?:-STD)?-\d+|ISO\s*\d+|BS\s*\d+|IEEE\s*\d+|NEMA\s*\d+)\b",
    re.IGNORECASE,
)
_RE_CROSS_REF_DOC = re.compile(
    r"\b(Fig(?:ure)?\.?\s*\d+(?:\.\d+)*|Table\s*\d+(?:\.\d+)*|Section\s*\d+(?:\.\d+)*)\b",
    re.IGNORECASE,
)

_RE_EXPLICIT_CONNECTION = re.compile(
    r"\b([A-Z]{1,4}[-_]?\d{2,5}[A-Z]?)\s+"
    r"(?:connects to|flows to|discharges to|feeds into|leads to|piped to|inlet of|outlet to)\s+"
    r"([A-Z]{1,4}[-_]?\d{2,5}[A-Z]?)"
    r"(?:\s+(?:via|through|along)\s+(?:line|stream|pipe)?\s*([A-Z0-9\"_-]+))?",
    re.IGNORECASE,
)
_RE_PREFIX_LINE_CONNECTION = re.compile(
    r"\b(?:Line|Stream|Pipe)\s+([A-Z0-9\"_-]+)\s+connects\s+([A-Z]{1,4}[-_]?\d{2,5}[A-Z]?)\s+to\s+([A-Z]{1,4}[-_]?\d{2,5}[A-Z]?)\b",
    re.IGNORECASE,
)
_RE_LINE_CONNECTION_LABEL = re.compile(
    r"\b(?:LINE|STREAM)?\s*([A-Z0-9\"_-]+)\s*:\s*([A-Z]{1,4}[-_]?\d{2,5})\s*->\s*([A-Z]{1,4}[-_]?\d{2,5})\b",
    re.IGNORECASE,
)

_SCALE_RE = re.compile(r"\bSCALE[:\s]+([0-9]+:[0-9]+|NTS|NONE|AS NOTED)\b", re.IGNORECASE)
_DATE_RE = re.compile(r"\b(?:DATE)[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b", re.IGNORECASE)
_UNIT_RE = re.compile(r"\b(?:UNIT|PLANT|AREA)[:\s]+([A-Z0-9_-]{2,20})\b", re.IGNORECASE)


def parse_isa_instrument_tag(tag: str) -> tuple[str, str, str]:
    """Parse an ISA-5.1 instrument tag into (description, function_code, loop_number)."""
    clean_tag = tag.strip().upper()
    m = _ISA_INSTRUMENT_RE.search(clean_tag)
    if not m:
        return "Instrument", clean_tag, ""

    code = m.group(1)
    loop = m.group(2)

    if code in SPECIAL_INSTRUMENT_TYPES:
        return SPECIAL_INSTRUMENT_TYPES[code], code, loop

    first = code[0]
    var_desc = ISA_FIRST_LETTER.get(first, first)
    modifiers = [ISA_SUCCEEDING_LETTERS.get(c, c) for c in code[1:]]
    desc = f"{var_desc} " + " ".join(modifiers)
    return desc.strip(), code, loop


def extract_equipment_registry(
    ocr_labels: Sequence[DrawingLabel],
    vlm_equipment: Sequence[Any],
) -> list[EquipmentEntry]:
    """Consolidate equipment detected by OCR and VLM, cross-referencing and deduplicating."""
    entries: list[EquipmentEntry] = []
    seen_tags: set[str] = set()

    # Build OCR tag map
    ocr_tags: dict[str, DrawingLabel] = {}
    for lbl in ocr_labels:
        if lbl.label_type == "equipment_tag" or _EQUIPMENT_TAG_RE.search(lbl.text):
            m = _EQUIPMENT_TAG_RE.search(lbl.text)
            if m:
                tag = m.group(1).upper()
                if not _INSTRUMENT_TAG_RE.match(tag):
                    ocr_tags[tag] = lbl

    # Process VLM equipment items
    for item in vlm_equipment:
        eq_type = getattr(item, "equipment_type", None) or (item.get("equipment_type") if isinstance(item, dict) else "Equipment")
        tag = getattr(item, "name_or_tag", None) or (item.get("name_or_tag") if isinstance(item, dict) else None)
        if hasattr(item, "confidence"):
            raw_conf = getattr(item, "confidence")
        elif isinstance(item, dict):
            raw_conf = item.get("confidence")
        else:
            raw_conf = None
        conf = float(raw_conf) if raw_conf is not None else None
        evidence = getattr(item, "evidence", "") if hasattr(item, "evidence") else (item.get("evidence", "") if isinstance(item, dict) else "")
        bbox = getattr(item, "bbox", None) if hasattr(item, "bbox") else (item.get("bbox") if isinstance(item, dict) else None)

        if tag:
            tag_clean = tag.strip().upper()
            if tag_clean in ocr_tags:
                ocr_lbl = ocr_tags[tag_clean]
                ocr_conf = float(ocr_lbl.confidence) if ocr_lbl.confidence is not None else None
                if conf is not None and ocr_conf is not None:
                    joint_conf: float | None = round(max(conf, ocr_conf), 4)
                elif conf is not None:
                    joint_conf = round(conf, 4)
                elif ocr_conf is not None:
                    joint_conf = round(ocr_conf, 4)
                else:
                    joint_conf = None

                entries.append(
                    EquipmentEntry(
                        tag=tag_clean,
                        equipment_type=eq_type,
                        source="both",
                        confidence=joint_conf,
                        bbox=ocr_lbl.bbox or bbox,
                        evidence=f"OCR verified tag '{tag_clean}'; VLM identified as '{eq_type}'.",
                    )
                )
                seen_tags.add(tag_clean)
            else:
                entries.append(
                    EquipmentEntry(
                        tag=tag_clean,
                        equipment_type=eq_type,
                        source="vlm",
                        confidence=round(conf, 4) if conf is not None else None,
                        bbox=bbox,
                        evidence=evidence or f"VLM detection of {eq_type}",
                    )
                )
                seen_tags.add(tag_clean)
        else:
            entries.append(
                EquipmentEntry(
                    tag=f"UNKNOWN_{eq_type.upper().replace(' ', '_')}",
                    equipment_type=eq_type,
                    source="vlm",
                    confidence=round(conf, 4) if conf is not None else None,
                    bbox=bbox,
                    evidence=evidence or f"VLM identified general {eq_type}",
                )
            )

    # Process remaining OCR tags
    for tag_str, lbl in ocr_tags.items():
        if tag_str not in seen_tags:
            prefix = tag_str.split("-")[0].strip().upper()
            inferred_type = EQUIPMENT_PREFIX_MAP.get(prefix, f"{prefix} Equipment")
            lbl_conf = round(float(lbl.confidence), 4) if lbl.confidence is not None else None
            entries.append(
                EquipmentEntry(
                    tag=tag_str,
                    equipment_type=inferred_type,
                    source="ocr",
                    confidence=lbl_conf,
                    bbox=lbl.bbox,
                    evidence=f"OCR detected equipment tag '{tag_str}' on drawing.",
                )
            )
            seen_tags.add(tag_str)

    return entries


def extract_instrument_registry(
    ocr_labels: Sequence[DrawingLabel],
    vlm_labels: Sequence[Any] = (),
) -> list[InstrumentEntry]:
    """Extract instrument bubbles and control loop tags with ISA-5.1 decoding."""
    entries: list[InstrumentEntry] = []
    seen_tags: set[str] = set()

    for lbl in ocr_labels:
        text = lbl.text.strip()
        m = _INSTRUMENT_TAG_RE.search(text) or _ISA_INSTRUMENT_RE.search(text)
        if m:
            tag = m.group(0).upper().replace(" ", "-")
            if tag not in seen_tags:
                desc, code, loop = parse_isa_instrument_tag(tag)
                lbl_conf = round(float(lbl.confidence), 4) if lbl.confidence is not None else None
                entries.append(
                    InstrumentEntry(
                        tag=tag,
                        instrument_type=desc,
                        function_code=code,
                        loop_number=loop,
                        source="ocr",
                        confidence=lbl_conf,
                        bbox=lbl.bbox,
                        evidence=f"OCR detected instrument bubble tag '{tag}'.",
                    )
                )
                seen_tags.add(tag)

    for item in vlm_labels:
        txt = getattr(item, "text", "") if hasattr(item, "text") else (item.get("text", "") if isinstance(item, dict) else str(item))
        m = _INSTRUMENT_TAG_RE.search(txt)
        if m:
            tag = m.group(1).upper()
            if tag in seen_tags:
                for idx, entry in enumerate(entries):
                    if entry.tag == tag:
                        entries[idx] = InstrumentEntry(
                            tag=entry.tag,
                            instrument_type=entry.instrument_type,
                            function_code=entry.function_code,
                            loop_number=entry.loop_number,
                            source="both",
                            confidence=entry.confidence,
                            bbox=entry.bbox,
                            evidence=entry.evidence + " Cross-confirmed by VLM visible text.",
                        )
            else:
                desc, code, loop = parse_isa_instrument_tag(tag)
                if hasattr(item, "confidence"):
                    raw_conf = getattr(item, "confidence")
                elif isinstance(item, dict):
                    raw_conf = item.get("confidence")
                else:
                    raw_conf = None
                conf = round(float(raw_conf), 4) if raw_conf is not None else None
                entries.append(
                    InstrumentEntry(
                        tag=tag,
                        instrument_type=desc,
                        function_code=code,
                        loop_number=loop,
                        source="vlm",
                        confidence=conf,
                        evidence=f"VLM observed instrument label '{tag}'.",
                    )
                )
                seen_tags.add(tag)

    return entries


def extract_connectivity(
    text_blocks: Sequence[str],
    spatial_relations: Sequence[SpatialRelation] = (),
    regions: Sequence[DrawingRegion] = (),
) -> tuple[list[ConnectionEntry], list[str]]:
    """Extract evidenced connections while actively guarding against proximity hallucinations."""
    connections: list[ConnectionEntry] = []
    hallucination_flags: list[str] = []
    seen_pairs: set[tuple[str, str]] = set()
    region_map = {r.region_id: r for r in regions}

    for text in text_blocks:
        text_lower = text.lower()
        if (
            RE_PROXIMITY_CONNECTION.search(text)
            or ("proximity" in text_lower)
            or ("adjacent" in text_lower and any(w in text_lower for w in ("connect", "flow", "lead", "pipe")))
            or ("assumed connection" in text_lower)
            or ("inferred connection" in text_lower)
        ):
            hallucination_flags.append("unsupported_connected_to_relationship")
            continue

        for m in _RE_EXPLICIT_CONNECTION.finditer(text):
            from_tag = m.group(1).upper()
            to_tag = m.group(2).upper()
            line_tag = m.group(3).strip() if m.group(3) else None
            pair = (from_tag, to_tag)
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                connections.append(
                    ConnectionEntry(
                        from_tag=from_tag,
                        to_tag=to_tag,
                        line_tag=line_tag,
                        evidence_type="explicit_label",
                        source_evidence=m.group(0).strip(),
                        confidence=0.95,
                    )
                )

        for m in _RE_PREFIX_LINE_CONNECTION.finditer(text):
            line_tag = m.group(1).strip()
            from_tag = m.group(2).upper()
            to_tag = m.group(3).upper()
            pair = (from_tag, to_tag)
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                connections.append(
                    ConnectionEntry(
                        from_tag=from_tag,
                        to_tag=to_tag,
                        line_tag=line_tag,
                        evidence_type="explicit_label",
                        source_evidence=m.group(0).strip(),
                        confidence=0.95,
                    )
                )

        for m in _RE_LINE_CONNECTION_LABEL.finditer(text):
            line_tag = m.group(1).strip()
            from_tag = m.group(2).upper()
            to_tag = m.group(3).upper()
            pair = (from_tag, to_tag)
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                connections.append(
                    ConnectionEntry(
                        from_tag=from_tag,
                        to_tag=to_tag,
                        line_tag=line_tag,
                        evidence_type="explicit_label",
                        source_evidence=m.group(0).strip(),
                        confidence=0.95,
                    )
                )

    for rel in spatial_relations:
        if rel.relation == SpatialRelationType.CONNECTED_TO:
            if rel.evidence in ("geometry", "proximity"):
                hallucination_flags.append("unsupported_connected_to_relationship")
                continue
            src_reg = region_map.get(rel.source_id)
            tgt_reg = region_map.get(rel.target_id)
            from_tag = src_reg.label if src_reg and src_reg.label else rel.source_id
            to_tag = tgt_reg.label if tgt_reg and tgt_reg.label else rel.target_id
            pair = (from_tag, to_tag)
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                connections.append(
                    ConnectionEntry(
                        from_tag=from_tag,
                        to_tag=to_tag,
                        evidence_type="drawn_line",
                        source_evidence=f"Backend line tracing evidence: {rel.evidence}",
                        confidence=rel.confidence,
                    )
                )

    return connections, sorted(list(set(hallucination_flags)))


def extract_drawing_cross_references(text_blocks: Sequence[str]) -> list[DrawingCrossReference]:
    """Extract references to other engineering drawings, standards (API, ASME, OISD), and figures."""
    cross_refs: list[DrawingCrossReference] = []
    seen: set[tuple[str, str]] = set()

    for text in text_blocks:
        for m in _RE_CROSS_REF_DRAWING.finditer(text):
            raw_target = m.group(1).strip()
            target = (
                re.sub(r"^(?:DRAWING|DWG)\s*(?:NO\.?)?\s*[:\-_]?\s*", "", raw_target, flags=re.IGNORECASE).strip()
                if ("NO." in raw_target.upper() or raw_target.upper().startswith("DRAWING "))
                else raw_target
            )
            pair = (target, "drawing")
            if pair not in seen and len(target) >= 3:
                seen.add(pair)
                cross_refs.append(
                    DrawingCrossReference(
                        target=target,
                        ref_type="drawing",
                        source_text=m.group(0).strip(),
                        confidence=0.92,
                    )
                )

        for m in _RE_CROSS_REF_STANDARD.finditer(text):
            target = m.group(1).strip()
            pair = (target, "standard")
            if pair not in seen:
                seen.add(pair)
                cross_refs.append(
                    DrawingCrossReference(
                        target=target,
                        ref_type="standard",
                        source_text=m.group(0).strip(),
                        confidence=0.95,
                    )
                )

        for m in _RE_CROSS_REF_DOC.finditer(text):
            target = m.group(1).strip()
            pair = (target, "document")
            if pair not in seen:
                seen.add(pair)
                cross_refs.append(
                    DrawingCrossReference(
                        target=target,
                        ref_type="document",
                        source_text=m.group(0).strip(),
                        confidence=0.90,
                    )
                )

    return cross_refs


def extract_drawing_title_block(
    text_blocks: Sequence[str],
    existing_meta: DrawingMetadata | None = None,
) -> TitleBlockInfo:
    """Extract standard title block attributes from OCR text blocks and drawing metadata."""
    dwg_num = existing_meta.drawing_number if existing_meta else None
    rev = existing_meta.revision if existing_meta else None
    title = existing_meta.title if existing_meta else None
    date_val = existing_meta.date if existing_meta else None
    plant_unit = existing_meta.plant_unit if existing_meta else None
    scale_val = None

    for text in text_blocks:
        if not dwg_num:
            m = _DRAWING_NUM_RE.search(text)
            if m:
                dwg_num = m.group(1).strip()
        if not rev:
            m = _REV_RE.search(text)
            if m:
                rev = m.group(1).strip()
        if not scale_val:
            m = _SCALE_RE.search(text)
            if m:
                scale_val = m.group(1).strip()
        if not date_val:
            m = _DATE_RE.search(text)
            if m:
                date_val = m.group(1).strip()
        if not plant_unit:
            m = _UNIT_RE.search(text)
            if m:
                plant_unit = m.group(1).strip()

    source = "ocr" if (dwg_num or rev or scale_val) else (existing_meta.source if existing_meta else "unknown")
    return TitleBlockInfo(
        drawing_number=dwg_num,
        revision=rev,
        title=title,
        date=date_val,
        scale=scale_val,
        plant_unit=plant_unit,
        source=source,
    )


def export_for_rag(record: DrawingAnalysisRecord) -> list[Any]:
    """Serialize DrawingAnalysisRecord into retrievable text chunks for downstream RAG vectorization."""
    if Chunk is None:
        raise RuntimeError("rag_engine.schemas.chunk.Chunk is not available in environment.")

    chunks: list[Any] = []
    doc_id = record.drawing_id
    doc_name = Path(record.source_path).name if record.source_path else f"{record.drawing_id}.png"
    dwg_type = record.drawing_type.value if hasattr(record.drawing_type, "value") else str(record.drawing_type)
    dwg_num = record.title_block.drawing_number or record.drawing_id
    heading_path = [dwg_type, dwg_num]
    chunk_idx = 0

    # 1. Title Block Chunk
    tb = record.title_block
    tb_lines = [
        "ENGINEERING DRAWING TITLE BLOCK:",
        f"Drawing Number: {tb.drawing_number or 'N/A'}",
        f"Drawing Title: {tb.title or 'N/A'}",
        f"Drawing Type: {dwg_type}",
        f"Revision: {tb.revision or 'N/A'}",
        f"Date: {tb.date or 'N/A'}",
        f"Plant Unit: {tb.plant_unit or 'N/A'}",
        f"Scale: {tb.scale or 'N/A'}",
    ]
    chunks.append(
        Chunk.create(
            document_id=doc_id,
            content="\n".join(tb_lines),
            chunk_index=chunk_idx,
            document_name=doc_name,
            source_path=record.source_path,
            page_number=1,
            section_title="Title Block",
            section_id=f"sec_title_block_{doc_id}",
            heading_path=heading_path + ["Title Block"],
            category=dwg_type,
            plant_unit=tb.plant_unit,
            chunk_strategy="title_block",
        )
    )
    chunk_idx += 1

    # 2. Equipment & Instrument Inventory Chunk
    inv_lines = ["EQUIPMENT & INSTRUMENT INVENTORY:"]
    if record.equipment:
        inv_lines.append("Equipment:")
        for eq in record.equipment:
            conf_str = f"conf: {eq.confidence:.2f}" if eq.confidence is not None else "conf: null (unmeasured)"
            inv_lines.append(f"- Tag: {eq.tag}, Type: {eq.equipment_type} (source: {eq.source}, {conf_str})")
    else:
        inv_lines.append("Equipment: None identified")

    if record.instruments:
        inv_lines.append("Instrumentation & Control Loops:")
        for inst in record.instruments:
            conf_str = f"conf: {inst.confidence:.2f}" if inst.confidence is not None else "conf: null (unmeasured)"
            inv_lines.append(
                f"- Tag: {inst.tag}, Type: {inst.instrument_type}, Function: {inst.function_code}, Loop: {inst.loop_number} (source: {inst.source}, {conf_str})"
            )
    else:
        inv_lines.append("Instrumentation: None identified")

    chunks.append(
        Chunk.create(
            document_id=doc_id,
            content="\n".join(inv_lines),
            chunk_index=chunk_idx,
            document_name=doc_name,
            source_path=record.source_path,
            page_number=1,
            section_title="Equipment & Instrumentation",
            section_id=f"sec_equipment_{doc_id}",
            heading_path=heading_path + ["Equipment & Instrumentation"],
            category=dwg_type,
            plant_unit=tb.plant_unit,
            equipment_entities=[e.tag for e in record.equipment],
            chunk_strategy="equipment_list",
        )
    )
    chunk_idx += 1

    # 3. Process Connectivity & Flow Topology Chunk
    conn_lines = ["PROCESS CONNECTIVITY & FLOW TOPOLOGY:"]
    if record.connections:
        for c in record.connections:
            line_info = f" via Line {c.line_tag}" if c.line_tag else ""
            conn_lines.append(
                f"- Connection: {c.from_tag} connects to {c.to_tag}{line_info} (evidence_type: {c.evidence_type}, evidence: '{c.source_evidence}')"
            )
    else:
        conn_lines.append("No explicit line connections confirmed on drawing.")

    chunks.append(
        Chunk.create(
            document_id=doc_id,
            content="\n".join(conn_lines),
            chunk_index=chunk_idx,
            document_name=doc_name,
            source_path=record.source_path,
            page_number=1,
            section_title="Process Connectivity",
            section_id=f"sec_connectivity_{doc_id}",
            heading_path=heading_path + ["Process Connectivity"],
            category=dwg_type,
            plant_unit=tb.plant_unit,
            chunk_strategy="connectivity",
        )
    )
    chunk_idx += 1

    # 4. Cross-References Chunk (if any)
    if record.cross_references:
        ref_lines = ["DRAWING CROSS-REFERENCES & REFERENCED STANDARDS:"]
        for ref in record.cross_references:
            ref_lines.append(f"- Reference: {ref.target} (type: {ref.ref_type}, evidence: '{ref.source_text}')")
        chunks.append(
            Chunk.create(
                document_id=doc_id,
                content="\n".join(ref_lines),
                chunk_index=chunk_idx,
                document_name=doc_name,
                source_path=record.source_path,
                page_number=1,
                section_title="Cross-References",
                section_id=f"sec_cross_refs_{doc_id}",
                heading_path=heading_path + ["Cross-References"],
                category=dwg_type,
                plant_unit=tb.plant_unit,
                chunk_strategy="cross_references",
            )
        )

    return chunks


def export_for_agent(record: DrawingAnalysisRecord) -> AgentQueryableDrawing:
    """Wrap DrawingAnalysisRecord in an AgentQueryableDrawing helper."""
    return AgentQueryableDrawing(record)


# ──────────────────────────────────────────────────────────────────────────────
# Drawing Type Classification
# ──────────────────────────────────────────────────────────────────────────────

_FILENAME_TYPE_MAP = {
    "pid": DrawingType.PID,
    "p_id": DrawingType.PID,
    "p&id": DrawingType.PID,
    "pfd": DrawingType.PFD,
    "equipment": DrawingType.EQUIPMENT,
    "pump": DrawingType.PUMP,
    "instrument": DrawingType.INSTRUMENTATION,
    "electrical": DrawingType.ELECTRICAL,
    "layout": DrawingType.PLANT_LAYOUT,
    "cad": DrawingType.CAD,
}


def infer_drawing_type_from_path(path: Path | str) -> tuple[DrawingType, str]:
    """Conservative heuristic mapping filename/path keywords to DrawingType.

    Returns (DrawingType, source_description).
    """
    p = Path(path)
    stem = p.stem.lower()
    parent = p.parent.name.lower()
    combined = f"{parent}_{stem}"

    for kw, dtype in _FILENAME_TYPE_MAP.items():
        if kw in combined:
            return dtype, "metadata"
    return DrawingType.UNKNOWN, "unclassified"


# ──────────────────────────────────────────────────────────────────────────────
# Backend Architecture
# ──────────────────────────────────────────────────────────────────────────────

@runtime_checkable
class DrawingAnalysisBackend(Protocol):
    """Protocol implemented by local drawing analysis backends."""

    @property
    def backend_info(self) -> BackendInfo:
        """Return backend provenance and metadata."""
        ...

    def capabilities(self) -> DrawingBackendCapabilities:
        """Return capability flags for this backend."""
        ...

    def initialize(self) -> None:
        """Initialize the local backend (validating weights and local paths)."""
        ...

    def analyze(self, image: np.ndarray) -> DrawingAnalysisPayload:
        """Perform visual analysis on an RGB image array."""
        ...


class LocalDrawingBackendBase:
    """Base class for local offline drawing analysis backends."""

    def __init__(self, config: DrawingAnalyzerConfig) -> None:
        self.config = config
        self._initialized = False

    def capabilities(self) -> DrawingBackendCapabilities:
        return self.backend_info.capabilities

    def initialize(self) -> None:
        if self._initialized:
            return
        self.config.validate()
        self._load_model()
        self._initialized = True

    def _load_model(self) -> None:
        """Subclasses override this to load local weights."""
        pass


class MockDrawingBackend(LocalDrawingBackendBase):
    """Deterministic mock backend for architecture testing and offline foundation.

    Returns fixed, synthetic drawing analysis results clearly tagged as mock provenance.
    """

    def __init__(
        self,
        config: DrawingAnalyzerConfig | None = None,
        *,
        drawing_type: DrawingType = DrawingType.PID,
        regions: Sequence[DrawingRegion] | None = None,
        labels: Sequence[DrawingLabel] | None = None,
        spatial_relations: Sequence[SpatialRelation] | None = None,
        description: str = "Mock P&ID drawing showing pump and valve assembly.",
        issues: Sequence[DrawingIssue] | None = None,
        capabilities: DrawingBackendCapabilities | None = None,
        should_fail: bool = False,
        invalid_output: bool = False,
    ) -> None:
        super().__init__(config or DrawingAnalyzerConfig(model_name="mock_drawing_backend"))
        self._drawing_type = drawing_type
        self._regions = tuple(regions) if regions is not None else (
            DrawingRegion(
                region_id="region_0",
                region_type="equipment",
                bbox=BoundingBox(left=50.0, top=100.0, right=200.0, bottom=250.0),
                label="Centrifugal Pump P-203",
                confidence=0.95,
                attributes={"tag": "P-203", "source": "mock"},
            ),
            DrawingRegion(
                region_id="region_1",
                region_type="instrument",
                bbox=BoundingBox(left=250.0, top=100.0, right=350.0, bottom=200.0),
                label="Pressure Indicator PI-101",
                confidence=0.90,
                attributes={"tag": "PI-101", "source": "mock"},
            ),
            DrawingRegion(
                region_id="region_2",
                region_type="title_block",
                bbox=BoundingBox(left=500.0, top=600.0, right=780.0, bottom=780.0),
                label="Title Block",
                confidence=0.99,
                attributes={"source": "mock"},
            ),
        )
        self._labels = tuple(labels) if labels is not None else (
            DrawingLabel(
                label_id="label_0",
                text="P-203",
                bbox=BoundingBox(left=50.0, top=80.0, right=120.0, bottom=100.0),
                confidence=0.98,
                label_type="equipment_tag",
                source="mock",
            ),
        )
        self._spatial_relations = tuple(spatial_relations or ())
        self._description = description
        self._issues = tuple(issues or ())
        self._capabilities = capabilities or DrawingBackendCapabilities(
            drawing_classification=True,
            symbol_detection=True,
            region_segmentation=True,
            spatial_reasoning=True,
        )
        self._should_fail = should_fail
        self._invalid_output = invalid_output

    @property
    def backend_info(self) -> BackendInfo:
        return BackendInfo(
            name="mock_drawing_backend",
            version="1.0.0",
            model_path=str(self.config.model_path) if self.config.model_path else None,
            device=self.config.device,
            capabilities=self._capabilities,
        )

    def analyze(self, image: np.ndarray) -> DrawingAnalysisPayload:
        if self._should_fail:
            raise DrawingBackendExecutionError("Simulated mock backend execution failure")
        if self._invalid_output:
            return "not_a_valid_payload"  # type: ignore[return-value]

        return DrawingAnalysisPayload(
            drawing_type=self._drawing_type,
            regions=self._regions,
            labels=self._labels,
            spatial_relations=self._spatial_relations,
            metadata=DrawingMetadata(
                drawing_type=self._drawing_type,
                title="Synthetic MRPL P&ID Unit 2",
                drawing_number="DWG-P-203-01",
                revision="A",
                source="mock",
                equipment_count=1,
                label_count=len(self._labels),
                region_count=len(self._regions),
            ),
            description=self._description,
            issues=self._issues,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Drawing Analyzer Pipeline
# ──────────────────────────────────────────────────────────────────────────────

class DrawingAnalyzer:
    """High-level orchestrator for engineering drawing analysis."""

    def __init__(
        self,
        backend: DrawingAnalysisBackend | None = None,
        config: DrawingAnalyzerConfig | None = None,
        *,
        default_preprocessing: PreprocessingOptions | None = None,
    ) -> None:
        self.config = config or DrawingAnalyzerConfig()
        self.config.validate()
        if backend is None:
            backend = MockDrawingBackend(config=self.config)
        elif not isinstance(backend, DrawingAnalysisBackend):
            raise TypeError("backend must implement DrawingAnalysisBackend protocol")
        self.backend = backend
        self.default_preprocessing = default_preprocessing or PreprocessingOptions.drawing()

    def analyze_drawing(
        self,
        source: str | Path | PreprocessingResult,
        *,
        ocr_result: OCRDocumentResult | None = None,
        drawing_type: DrawingType | str | None = None,
        preprocessing_options: PreprocessingOptions | None = None,
    ) -> DrawingAnalysisResult:
        """Analyze an engineering drawing image or preprocessed result.

        Parameters
        ----------
        source:
            Filesystem path to the drawing, or an existing PreprocessingResult.
        ocr_result:
            Optional pre-computed OCRDocumentResult from .ocr_pipeline.
        drawing_type:
            Explicit drawing type to override or enforce.
        preprocessing_options:
            Optional preprocessing overrides.

        Returns
        -------
        DrawingAnalysisResult:
            Deterministic, structured, JSON-safe analysis record.
        """
        start_ns = time.perf_counter_ns()
        issues: list[DrawingIssue] = []
        processing_metadata: dict[str, Any] = {}

        # 1. Handle PreprocessingResult
        if isinstance(source, PreprocessingResult):
            source_path = str(source.original_path)
            image_array = source.image
            width = source.processed_width
            height = source.processed_height
            processing_metadata["source_type"] = "preprocessing_result"
            processing_metadata["operations_applied"] = list(source.operations_applied)
            inferred_type, type_source = (
                (DrawingType(drawing_type), "caller")
                if drawing_type
                else infer_drawing_type_from_path(source_path)
            )
        else:
            path = Path(source)
            source_path = str(path)
            processing_metadata["source_type"] = "image_path"
            processing_metadata["source_path"] = source_path

            # 2. Check path existence
            if not path.exists():
                if isinstance(self.backend, MockDrawingBackend):
                    image_array = np.zeros((100, 100, 3), dtype=np.uint8)
                    width, height = 100, 100
                    processing_metadata["operations_applied"] = []
                    inferred_type = DrawingType(drawing_type) if drawing_type else infer_drawing_type_from_path(source_path)[0]
                    type_source = "caller" if drawing_type else "path"
                else:
                    return DrawingAnalysisResult(
                        success=False,
                        source_path=source_path,
                        drawing_type=DrawingType(drawing_type) if drawing_type else DrawingType.UNKNOWN,
                        metadata=DrawingMetadata(drawing_type=DrawingType(drawing_type) if drawing_type else DrawingType.UNKNOWN),
                        issues=(
                            DrawingIssue(
                                code="missing_drawing",
                                message=f"Drawing file does not exist: {path}",
                                severity="error",
                            ),
                        ),
                        backend=self.backend.backend_info,
                        processing_metadata=processing_metadata,
                    )
            else:
                # 3. Check for vector / PDF extensions requiring rasterization
                ext = path.suffix.lower()
                if ext in VECTOR_EXTENSIONS:
                    return DrawingAnalysisResult(
                        success=False,
                        source_path=source_path,
                        drawing_type=DrawingType.UNKNOWN,
                        metadata=DrawingMetadata(drawing_type=DrawingType.UNKNOWN),
                        issues=(
                            DrawingIssue(
                                code="VECTOR_RENDER_REQUIRED",
                                message=(
                                    f"Vector or multi-page drawing format '{ext}' requires "
                                    "pre-rendering or rasterization before visual analysis."
                                ),
                                severity="error",
                            ),
                        ),
                        backend=self.backend.backend_info,
                        processing_metadata=processing_metadata,
                    )

                # 4. Check supported raster extensions
                if ext not in SUPPORTED_EXTENSIONS:
                    return DrawingAnalysisResult(
                        success=False,
                        source_path=source_path,
                        drawing_type=DrawingType.UNKNOWN,
                        metadata=DrawingMetadata(drawing_type=DrawingType.UNKNOWN),
                        issues=(
                            DrawingIssue(
                                code="UNSUPPORTED_DRAWING_FORMAT",
                                message=(
                                    f"Drawing format '{ext}' is not supported. "
                                    f"Supported formats: {sorted(SUPPORTED_EXTENSIONS)}"
                                ),
                                severity="error",
                            ),
                        ),
                        backend=self.backend.backend_info,
                        processing_metadata=processing_metadata,
                    )

                # 5. Check image readability without modifying the file
                try:
                    with Image.open(path) as pil_img:
                        pil_img.verify()
                    with Image.open(path) as pil_img:
                        orig_width, orig_height = pil_img.size
                except Exception as exc:
                    return DrawingAnalysisResult(
                        success=False,
                        source_path=source_path,
                        drawing_type=DrawingType.UNKNOWN,
                        metadata=DrawingMetadata(drawing_type=DrawingType.UNKNOWN),
                        issues=(
                            DrawingIssue(
                                code="corrupt_image",
                                message=f"Drawing image is corrupt or unreadable: {exc}",
                                severity="error",
                            ),
                        ),
                        backend=self.backend.backend_info,
                        processing_metadata=processing_metadata,
                    )

                # 6. Preprocessing (if enabled in config or requested)
                if self.config.preprocess:
                    opts = preprocessing_options or self.default_preprocessing
                    try:
                        prep = preprocess_image(path, opts, save_output=False)
                        image_array = prep.image
                        width = prep.processed_width
                        height = prep.processed_height
                        processing_metadata["operations_applied"] = list(prep.operations_applied)
                    except ImagePreprocessingError as exc:
                        return DrawingAnalysisResult(
                            success=False,
                            source_path=source_path,
                            drawing_type=DrawingType.UNKNOWN,
                            metadata=DrawingMetadata(drawing_type=DrawingType.UNKNOWN),
                            issues=(
                                DrawingIssue(
                                    code="preprocessing_failed",
                                    message=f"Drawing preprocessing failed: {exc}",
                                    severity="error",
                                ),
                            ),
                            backend=self.backend.backend_info,
                            processing_metadata=processing_metadata,
                        )
                else:
                    with Image.open(path) as pil_img:
                        rgb_img = pil_img.convert("RGB")
                        image_array = np.array(rgb_img)
                        width, height = orig_width, orig_height
                    processing_metadata["operations_applied"] = []

                # Determine initial drawing type
                if drawing_type:
                    inferred_type = DrawingType(drawing_type)
                    type_source = "caller"
                else:
                    inferred_type, type_source = infer_drawing_type_from_path(source_path)

        # 7. Initialize backend
        try:
            self.backend.initialize()
        except (MissingLocalModelError, DrawingConfigurationError):
            raise
        except Exception as exc:
            return DrawingAnalysisResult(
                success=False,
                source_path=source_path,
                drawing_type=inferred_type,
                metadata=DrawingMetadata(drawing_type=inferred_type, dimensions=(width, height)),
                issues=(
                    DrawingIssue(
                        code="backend_init_failed",
                        message=f"Drawing backend initialization failed: {exc}",
                        severity="error",
                    ),
                ),
                backend=self.backend.backend_info,
                processing_metadata=processing_metadata,
            )

        # 8. Execute backend visual analysis
        try:
            payload = self.backend.analyze(image_array)
            if not isinstance(payload, DrawingAnalysisPayload):
                return DrawingAnalysisResult(
                    success=False,
                    source_path=source_path,
                    drawing_type=inferred_type,
                    metadata=DrawingMetadata(drawing_type=inferred_type, dimensions=(width, height)),
                    issues=(
                        DrawingIssue(
                            code="invalid_backend_output",
                            message=f"Backend returned invalid payload type: {type(payload).__name__}",
                            severity="error",
                        ),
                    ),
                    backend=self.backend.backend_info,
                    processing_metadata=processing_metadata,
                )
        except Exception as exc:
            return DrawingAnalysisResult(
                success=False,
                source_path=source_path,
                drawing_type=inferred_type,
                metadata=DrawingMetadata(drawing_type=inferred_type, dimensions=(width, height)),
                issues=(
                    DrawingIssue(
                        code="backend_execution_failed",
                        message=f"Drawing backend execution failed: {exc}",
                        severity="error",
                    ),
                ),
                backend=self.backend.backend_info,
                processing_metadata=processing_metadata,
            )

        # 9. Reconcile drawing type (caller > backend > metadata hint)
        resolved_type = inferred_type
        if drawing_type:
            resolved_type = DrawingType(drawing_type)
        elif payload.drawing_type != DrawingType.UNKNOWN:
            resolved_type = payload.drawing_type
            type_source = "backend"

        # 10. Combine backend regions with deterministic IDs
        regions: list[DrawingRegion] = []
        for idx, reg in enumerate(payload.regions):
            region_id = reg.region_id if reg.region_id else f"region_{idx}"
            regions.append(DrawingRegion(
                region_id=region_id,
                region_type=reg.region_type,
                bbox=reg.bbox,
                label=reg.label,
                confidence=reg.confidence,
                attributes=dict(reg.attributes),
            ))

        # 11. Coordinate with OCR if enabled/supplied
        labels: list[DrawingLabel] = list(payload.labels)
        ocr_provenance: dict[str, Any] = {}
        if self.config.enable_ocr and ocr_result is not None:
            ocr_labels = extract_labels_from_ocr(ocr_result)
            labels.extend(ocr_labels)
            ocr_provenance = {
                "ocr_document_id": ocr_result.document_id,
                "ocr_page_count": len(ocr_result.pages),
                "ocr_labels_extracted": len(ocr_labels),
            }
            processing_metadata["ocr_coordination"] = ocr_provenance

        # Ensure all labels have deterministic IDs
        normalized_labels: list[DrawingLabel] = []
        for idx, lbl in enumerate(labels):
            label_id = lbl.label_id if lbl.label_id else f"label_{idx}"
            normalized_labels.append(DrawingLabel(
                label_id=label_id,
                text=lbl.text,
                bbox=lbl.bbox,
                confidence=lbl.confidence,
                label_type=lbl.label_type,
                source=lbl.source,
            ))

        # 12. Spatial / Geometric analysis
        spatial_relations: list[SpatialRelation] = list(payload.spatial_relations)
        if self.config.enable_geometry_analysis and regions:
            geom_relations = compute_spatial_relations(
                regions,
                near_threshold=self.config.near_distance_threshold,
            )
            # Merge relations avoiding duplicates
            seen_rel_ids = {r.relation_id for r in spatial_relations}
            for rel in geom_relations:
                if rel.relation_id not in seen_rel_ids:
                    spatial_relations.append(rel)
                    seen_rel_ids.add(rel.relation_id)

        # 13. Reconcile metadata
        metadata = payload.metadata
        if self.config.enable_ocr and ocr_result is not None:
            metadata = extract_metadata_from_ocr(ocr_result, metadata)

        # Count equipment, labels, and regions
        equipment_count = sum(1 for r in regions if r.region_type == "equipment")
        final_metadata = DrawingMetadata(
            drawing_type=resolved_type,
            title=metadata.title,
            drawing_number=metadata.drawing_number,
            revision=metadata.revision,
            date=metadata.date,
            plant_unit=metadata.plant_unit,
            dimensions=(width, height),
            units=metadata.units,
            source=type_source,
            equipment_count=equipment_count,
            label_count=len(normalized_labels),
            region_count=len(regions),
        )

        issues.extend(payload.issues)

        elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
        processing_metadata["processing_time_ms"] = round(elapsed_ms, 3)
        processing_metadata["drawing_type_source"] = type_source

        return DrawingAnalysisResult(
            success=True,
            source_path=source_path,
            drawing_type=resolved_type,
            metadata=final_metadata,
            regions=tuple(regions),
            labels=tuple(normalized_labels),
            spatial_relations=tuple(spatial_relations),
            description=payload.description,
            issues=tuple(issues),
            backend=self.backend.backend_info,
            processing_metadata=processing_metadata,
        )

    def analyze_structured_drawing(
        self,
        image_path: Path | str,
        *,
        drawing_type: DrawingType | str | None = None,
        ocr_result: OCRDocumentResult | None = None,
        vlm_result: Any | None = None,
        raw_vlm_text: str | None = None,
        drawing_id: str | None = None,
    ) -> DrawingAnalysisRecord:
        """Execute comprehensive, structured engineering drawing analysis.

        Fuses OCR text, spatial relations, and VLM task outputs into a unified,
        hallucination-checked DrawingAnalysisRecord.
        """
        path = Path(image_path)
        stable_id = drawing_id or path.stem

        # Execute base drawing analysis for spatial relations, metadata, and OCR coordination
        base_res = self.analyze_drawing(path, drawing_type=drawing_type, ocr_result=ocr_result)

        # Collect all text sources
        text_blocks: list[str] = []
        if ocr_result is not None:
            for page in ocr_result.pages:
                for block in page.blocks:
                    if block.text.strip():
                        text_blocks.append(block.text.strip())

        for lbl in base_res.labels:
            if lbl.text.strip() and lbl.text.strip() not in text_blocks:
                text_blocks.append(lbl.text.strip())

        if raw_vlm_text:
            text_blocks.append(raw_vlm_text.strip())

        if vlm_result is not None:
            if getattr(vlm_result, "caption", ""):
                text_blocks.append(vlm_result.caption.strip())
            for obs in getattr(vlm_result, "observations", ()):
                desc = getattr(obs, "description", "")
                if desc:
                    text_blocks.append(desc.strip())
            for vt in getattr(vlm_result, "visible_text", ()):
                vt_text = getattr(vt, "text", "")
                if vt_text:
                    text_blocks.append(vt_text.strip())

        # 1. Equipment registry consolidation
        vlm_eq = getattr(vlm_result, "equipment", ()) if vlm_result else ()
        equipment_entries = extract_equipment_registry(base_res.labels, vlm_eq)

        # 2. Instrument registry extraction with ISA-5.1 decoding
        vlm_vt = getattr(vlm_result, "visible_text", ()) if vlm_result else ()
        instrument_entries = extract_instrument_registry(base_res.labels, vlm_vt)

        # 3. Connectivity extraction with strict proximity guard
        connections, conn_flags = extract_connectivity(
            text_blocks,
            base_res.spatial_relations,
            base_res.regions,
        )

        # 4. Cross-references
        cross_refs = extract_drawing_cross_references(text_blocks)

        # 5. Title block
        title_block = extract_drawing_title_block(text_blocks, base_res.metadata)

        # 6. Comprehensive hallucination check
        all_text = "\n".join(text_blocks)
        eq_dicts = [
            {"name_or_tag": e.tag, "equipment_type": e.equipment_type, "evidence": e.evidence}
            for e in equipment_entries
        ]
        known_tags = [e.tag for e in equipment_entries] + [i.tag for i in instrument_entries]
        h_flags = check_hallucinations(
            all_text,
            eq_dicts,
            expected_category=base_res.drawing_type.value if hasattr(base_res.drawing_type, "value") else str(base_res.drawing_type),
            known_tags=known_tags,
        )

        # Check for connectivity claims without explicit evidence
        for rel in base_res.spatial_relations:
            if rel.relation == SpatialRelationType.CONNECTED_TO and rel.evidence in ("geometry", "proximity"):
                conn_flags.append("unsupported_connected_to_relationship")

        all_hallucination_flags = sorted(list(set(h_flags + conn_flags)))

        return DrawingAnalysisRecord(
            drawing_id=stable_id,
            source_path=str(path),
            drawing_type=base_res.drawing_type,
            title_block=title_block,
            equipment=tuple(equipment_entries),
            instruments=tuple(instrument_entries),
            connections=tuple(connections),
            cross_references=tuple(cross_refs),
            extraction_metadata={
                "backend": base_res.backend.name,
                "confidence_method": "fused_ocr_vlm_provenance",
                "ocr_enabled": self.config.enable_ocr and ocr_result is not None,
                "analyzer_version": ANALYZER_VERSION,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            hallucination_flags=tuple(all_hallucination_flags),
        )


def analyze_engineering_drawing(
    image_path: Path | str,
    *,
    drawing_type: DrawingType | str | None = None,
    ocr_result: OCRDocumentResult | None = None,
    ocr_pipeline: Any | None = None,
    vlm_result: Any | None = None,
    vision_pipeline: Any | None = None,
    raw_vlm_text: str | None = None,
    config: DrawingAnalyzerConfig | None = None,
    drawing_id: str | None = None,
) -> DrawingAnalysisRecord:
    """Top-level convenience orchestrator for engineering drawing analysis.

    Optionally runs OCR and Vision pipelines if provided and pre-computed results are missing.
    """
    cfg = config or DrawingAnalyzerConfig()
    backend = MockDrawingBackend(config=cfg)
    analyzer = DrawingAnalyzer(backend=backend, config=cfg)

    actual_ocr = ocr_result
    if actual_ocr is None and ocr_pipeline is not None:
        try:
            actual_ocr = ocr_pipeline.process_image(image_path)
        except Exception as exc:
            LOGGER.warning("Failed to run OCR on %s: %s", image_path, exc)

    actual_vlm = vlm_result
    actual_raw_vlm = raw_vlm_text
    if actual_vlm is None and vision_pipeline is not None:
        try:
            actual_vlm = vision_pipeline.analyze(image_path, tasks=("drawing", "equipment"))
            actual_raw_vlm = actual_raw_vlm or getattr(actual_vlm, "caption", "")
        except Exception as exc:
            LOGGER.warning("Failed to run vision pipeline on %s: %s", image_path, exc)

    return analyzer.analyze_structured_drawing(
        image_path,
        drawing_type=drawing_type,
        ocr_result=actual_ocr,
        vlm_result=actual_vlm,
        raw_vlm_text=actual_raw_vlm,
        drawing_id=drawing_id,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Command Line Interface
# ──────────────────────────────────────────────────────────────────────────────

def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for running local engineering drawing analysis."""
    parser = argparse.ArgumentParser(
        description="Offline Engineering Drawing Analyzer for Sovereign AI Workbench"
    )
    parser.add_argument("--input", "-i", required=True, help="Path to engineering drawing image")
    parser.add_argument(
        "--model-path",
        "-m",
        default=None,
        help="Local directory containing pre-staged drawing model weights",
    )
    parser.add_argument(
        "--drawing-type",
        "-t",
        default=None,
        choices=[d.value for d in DrawingType],
        help="Explicit drawing type (e.g. PID, PFD, EQUIPMENT)",
    )
    parser.add_argument("--device", default="cpu", help="Compute device (default: cpu)")
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Detection confidence threshold (0.0 to 1.0)",
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Enable drawing-optimized preprocessing before analysis",
    )
    parser.add_argument("--json", action="store_true", help="Output results as JSON")

    args = parser.parse_args(argv)

    config = DrawingAnalyzerConfig(
        model_name="cli_drawing_model",
        model_path=args.model_path,
        device=args.device,
        confidence_threshold=args.confidence,
        allow_downloads=False,
        preprocess=args.preprocess,
    )

    try:
        config.validate()
    except (MissingLocalModelError, DrawingConfigurationError) as exc:
        LOGGER.error("Configuration error: %s", exc)
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    # In foundation phase, execute with MockDrawingBackend
    backend = MockDrawingBackend(config=config)
    analyzer = DrawingAnalyzer(backend=backend, config=config)

    result = analyzer.analyze_drawing(args.input, drawing_type=args.drawing_type)

    if args.json:
        print(result.to_json(indent=2))
    else:
        status_str = "SUCCESS" if result.success else "FAILED"
        print(f"Drawing Analysis [{status_str}]: {result.source_path}")
        print(f"  Drawing Type: {result.drawing_type.value} (source: {result.metadata.source})")
        print(f"  Dimensions: {result.metadata.dimensions[0]} x {result.metadata.dimensions[1]}")
        if result.metadata.title:
            print(f"  Title: {result.metadata.title}")
        if result.metadata.drawing_number:
            print(f"  Drawing Number: {result.metadata.drawing_number} (Rev: {result.metadata.revision or '-'})")
        print(f"  Regions: {len(result.regions)} | Labels: {len(result.labels)} | Relations: {len(result.spatial_relations)}")
        if result.description:
            print(f"  Description: {result.description}")
        if result.issues:
            print("  Issues:")
            for issue in result.issues:
                print(f"    - [{issue.severity.upper()}] {issue.code}: {issue.message}")

    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
