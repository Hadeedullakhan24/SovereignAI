"""Engineering drawing and visual asset parser extracting structural metadata without OCR/CV."""

from __future__ import annotations

import re
import struct
import uuid
from pathlib import Path
from typing import Any, Optional

from rag_engine.parsers.base_parser import BaseParser
from rag_engine.parsers.parser_registry import register_parser
from rag_engine.parsers.parser_utils import ParserUtils
from rag_engine.parsers.parsing_context import ParsingContext
from rag_engine.schemas.document import Document
from rag_engine.schemas.parsed_document import (
    DocumentStatistics,
    DrawingMetadata,
    EntityGraph,
    ParsedDocument,
    ParsedMetadata,
    Section,
)


@register_parser(
    extensions=[".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".dwg", ".dxf", ".svg"],
    categories=["engineering drawing", "drawing", "schematic", "image"],
)
class ImageMetadataParser(BaseParser):
    """Deterministic parser extracting dimensions, title blocks, and drawing references."""

    def can_parse(self, document: Document) -> bool:
        ext = (document.metadata.file_format or "").lower()
        return ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".dwg", ".dxf", ".svg"]

    def supported_formats(self) -> list[str]:
        return [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".dwg", ".dxf", ".svg"]

    def supported_categories(self) -> list[str]:
        return ["engineering drawing", "drawing", "schematic", "diagram", "blueprint"]

    def driver_name(self) -> str:
        return "native_drawing_metadata_parser"

    def _parse_document(self, document: Document, context: ParsingContext) -> ParsedDocument:
        content = document.content or ""
        source_path = Path(document.metadata.source_path)
        profile = context.profile

        file_size = document.metadata.file_size_bytes
        if file_size == 0 and source_path.exists():
            try:
                file_size = source_path.stat().st_size
            except Exception:
                pass

        # Extract dimensions from metadata or binary headers
        dims = (
            document.metadata.extra_metadata.get("width", 0),
            document.metadata.extra_metadata.get("height", 0),
        )
        if dims == (0, 0) and source_path.exists():
            dims = self._read_image_dimensions(source_path)

        # Parse title block from text or metadata
        title_block = self._extract_title_block(content, profile)

        drawing_name = title_block.get("TITLE") or document.metadata.file_name.rsplit(".", 1)[0].replace("_", " ").title()
        drawing_no = title_block.get("DRAWING NO") or title_block.get("DWG NO") or document.metadata.file_name
        revision = title_block.get("REV") or title_block.get("REVISION") or "0"

        drawing_metadata = DrawingMetadata(
            drawing_name=drawing_name,
            drawing_number=drawing_no,
            title_block=title_block,
            revision=revision,
            dimensions=dims,
            resolution_dpi=(
                document.metadata.extra_metadata.get("dpi_x", 72),
                document.metadata.extra_metadata.get("dpi_y", 72),
            ),
            file_size_bytes=file_size,
            image_reference=str(source_path.resolve()),
        )

        sections = [
            Section(
                section_id=f"sec_dwg_{uuid.uuid4().hex[:8]}",
                title=f"Drawing: {drawing_name} ({drawing_no})",
                level=1,
                content=(
                    f"Engineering drawing specification: {drawing_name}. "
                    f"Drawing Number: {drawing_no}, Rev: {revision}. "
                    f"Dimensions: {dims[0]}x{dims[1]} px. Size: {file_size} bytes."
                ),
                paragraphs=[
                    f"Image asset stored at: {source_path}",
                    f"Title Block attributes: {', '.join(f'{k}={v}' for k, v in title_block.items())}",
                ],
                page_number=1,
                confidence=0.99,
            )
        ]

        # Extract any equipment or safety mentions embedded in drawing text
        equipment = (
            ParserUtils.extract_equipment(content, profile=profile, page_number=1)
            if context.extract_equipment
            else []
        )
        entity_graph = (
            ParserUtils.build_entity_graph(content, equipment)
            if context.extract_relationships
            else EntityGraph()
        )
        warnings = (
            ParserUtils.extract_safety_warnings(content, profile=profile, page_number=1)
            if context.extract_safety
            else []
        )
        cross_refs = (
            ParserUtils.extract_cross_references(content, page_number=1)
            if context.extract_cross_references
            else []
        )
        op_meta = (
            ParserUtils.extract_operational_metadata(content, profile=profile)
            if context.extract_metadata
            else {}
        )

        parsed_meta = ParsedMetadata(
            title=drawing_name,
            category="Engineering Drawing",
            revision_numbers=[revision] if revision else op_meta.get("revisions", []),
            engineer_names=op_meta.get("engineers", []),
            inspection_dates=op_meta.get("dates", []),
            standards_referenced=op_meta.get("standards", []),
            extra_metadata={
                "drawing_number": drawing_no,
                "revision": revision,
                "dimensions": f"{dims[0]}x{dims[1]}",
                "file_size": file_size,
                "image_reference": str(source_path),
            },
        )

        return ParsedDocument(
            document_id=f"parsed_{uuid.uuid4().hex[:12]}",
            raw_document_id=document.doc_id,
            title=drawing_name,
            category="Engineering Drawing",
            sections=sections,
            tables=[],
            equipment=equipment,
            entity_graph=entity_graph,
            warnings=warnings,
            cross_references=cross_refs,
            drawing_metadata=drawing_metadata,
            metadata=parsed_meta,
            statistics=DocumentStatistics(total_pages=1),
            processing_history=[
                {"stage": "ImageMetadataParser", "status": "COMPLETED", "driver": self.driver_name()}
            ],
        )

    def _extract_title_block(self, text: str, profile: Any) -> dict[str, Any]:
        """Extract key-value pairs from drawing title blocks."""
        title_block: dict[str, Any] = {}
        for key in profile.drawing_title_block_keys:
            pattern = re.compile(rf"\b{re.escape(key)}\b[\s.:-]+([^\n,;]+)", re.IGNORECASE)
            match = pattern.search(text)
            if match:
                title_block[key] = match.group(1).strip()
        return title_block

    def _read_image_dimensions(self, image_path: Path) -> tuple[int, int]:
        """Deterministic binary header parsing for PNG/JPEG without external libraries."""
        try:
            with open(image_path, "rb") as f:
                header = f.read(32)
                # PNG check
                if header.startswith(b"\x89PNG\r\n\x1a\n") and len(header) >= 24:
                    w, h = struct.unpack(">II", header[16:24])
                    return w, h
                # JPEG check
                f.seek(0)
                data = f.read(2)
                if data == b"\xff\xd8":
                    while True:
                        marker = f.read(2)
                        if not marker or len(marker) < 2 or marker[0] != 0xFF:
                            break
                        if marker[1] in (0xC0, 0xC1, 0xC2, 0xC3):
                            f.read(3)
                            h, w = struct.unpack(">HH", f.read(4))
                            return w, h
                        else:
                            length_bytes = f.read(2)
                            if len(length_bytes) < 2:
                                break
                            length = struct.unpack(">H", length_bytes)[0]
                            f.seek(length - 2, 1)
        except Exception:
            pass
        return (0, 0)
