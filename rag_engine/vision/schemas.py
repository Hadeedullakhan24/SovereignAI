"""Small, provenance-first payload record for image vectors."""
from __future__ import annotations
import hashlib
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

def _compact(value: Any, limit: int = 4000) -> str: return str(value or "").strip()[:limit]
def _box(value: Any) -> dict[str, float] | None:
    if value is None: return None
    if is_dataclass(value): value = asdict(value)
    if isinstance(value, dict): return {str(k): float(v) for k, v in value.items() if isinstance(v, (int, float))}
    return None

@dataclass(frozen=True)
class VisionIndexRecord:
    document_id: str; image_id: str; source_file: str; file_path: str; page_number: int | None = None; image_type: str = "unknown"; routing_decision: str = "unknown"; drawing_number: str | None = None; title: str | None = None; equipment_tags: tuple[str, ...] = (); ocr_text: str = ""; vlm_description: str = ""; visual_observations: tuple[dict[str, Any], ...] = (); bounding_boxes: tuple[dict[str, Any], ...] = (); confidence: float | None = None; provenance: dict[str, Any] = field(default_factory=dict); model_name: str = ""; created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    @property
    def point_id(self) -> str: return "vision_" + hashlib.sha256(f"{self.document_id}|{self.image_id}|{self.file_path}|{self.page_number}".encode()).hexdigest()[:24]
    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self); payload.update(equipment_tags=list(self.equipment_tags), visual_observations=list(self.visual_observations), bounding_boxes=list(self.bounding_boxes), point_id=self.point_id); return payload
    @classmethod
    def from_multimodal_result(cls, result: Any, *, page_number: int | None = None) -> "VisionIndexRecord | None":
        drawing, vision = getattr(result, "drawing_analysis", None), getattr(result, "vision_analysis", None)
        if drawing is None and vision is None: return None
        source_path, metadata = str(getattr(result, "source_path", "")), dict(getattr(result, "processing_metadata", {}) or {})
        status = dict(metadata.get("vision", {}) or {})
        if drawing is not None:
            equipment = tuple(e.tag for e in drawing.equipment if getattr(e, "tag", "")); boxes = tuple({"tag": e.tag, "bbox": _box(getattr(e, "bbox", None))} for e in drawing.equipment if getattr(e, "bbox", None) is not None); block = drawing.title_block
            return cls(result.document_id, drawing.drawing_id, Path(source_path).name, source_path, page_number, "engineering_drawing", result.routing_decision, block.drawing_number, block.title, equipment, _compact(result.get_full_text(), 8000), _compact(status.get("caption", "")), (), boxes, None, {"vision_status": status, "drawing_metadata": drawing.extraction_metadata}, str(status.get("model_name", "")))
        observations = tuple({"description": o.description, "confidence": o.confidence} for o in vision.observations[:20]); boxes = tuple({"equipment_type": e.equipment_type, "tag": e.name_or_tag, "bbox": _box(e.bbox)} for e in vision.equipment if e.bbox is not None)
        return cls(result.document_id, vision.image_id, Path(source_path).name, source_path, page_number, vision.image_type, result.routing_decision, None, None, tuple(e.name_or_tag for e in vision.equipment if e.name_or_tag), "", _compact(vision.caption), observations, boxes, vision.confidence, dict(vision.extraction_metadata), str(vision.extraction_metadata.get("model_name", "")))
