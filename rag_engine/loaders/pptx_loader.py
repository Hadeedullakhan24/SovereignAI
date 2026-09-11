"""
PPTX Loader — Microsoft PowerPoint Presentation Loading with Dual-Driver Resiliency.

Sovereign On-Premise Agentic AI Workbench (SIH26117 / MRPL)
Primary driver: python-pptx (if available)
Fallback driver: Standard library zipfile + xml.etree.ElementTree
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any
import xml.etree.ElementTree as ET
import zipfile

from rag_engine.loaders.base_loader import BaseLoader
from rag_engine.loaders.loader_registry import register_loader

try:
    import pptx
    _PPTX_AVAILABLE = True
except ImportError:
    _PPTX_AVAILABLE = False


@register_loader(".pptx", ".ppt")
class PPTXLoader(BaseLoader):
    """Universal loader for Microsoft PowerPoint presentations."""

    def supported_formats(self) -> list[str]:
        return [".pptx", ".ppt"]

    def is_primary_driver_available(self) -> bool:
        return _PPTX_AVAILABLE

    def dependencies(self) -> dict[str, bool]:
        return {"python-pptx": _PPTX_AVAILABLE}

    def _load_primary(self, file_path: Path) -> tuple[str, dict[str, Any]]:
        """Load PPTX using python-pptx."""
        prs = pptx.Presentation(str(file_path))
        slides_text: list[str] = []

        for idx, slide in enumerate(prs.slides):
            slide_lines: list[str] = []
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for paragraph in shape.text_frame.paragraphs:
                        text = paragraph.text.strip()
                        if text:
                            slide_lines.append(text)

            if slide_lines:
                slides_text.append(f"--- Slide {idx + 1} ---\n" + "\n".join(slide_lines))

        full_content = "\n\n".join(slides_text)
        meta: dict[str, Any] = {
            "slide_count": len(prs.slides),
            "page_count": len(prs.slides),
            "driver": "python-pptx",
        }
        return full_content, meta

    def _load_fallback(self, file_path: Path) -> tuple[str, dict[str, Any]]:
        """Air-gapped fallback: parse ppt/slides/slide*.xml inside PPTX ZIP archive."""
        slides_text: list[str] = []
        slide_count = 0
        meta: dict[str, Any] = {"driver": "stdlib_pptx_fallback"}

        with zipfile.ZipFile(file_path, "r") as zf:
            # Locate all slide XML files and sort them numerically
            slide_files = [f for f in zf.namelist() if re.match(r"ppt/slides/slide\d+\.xml", f)]
            # Sort by slide number
            slide_files.sort(key=lambda name: int(re.search(r"\d+", name).group()))
            slide_count = len(slide_files)

            ns = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}

            for idx, slide_file in enumerate(slide_files):
                xml_data = zf.read(slide_file)
                root = ET.fromstring(xml_data)
                texts = [t.text.strip() for t in root.findall(".//a:t", ns) if t.text and t.text.strip()]
                if texts:
                    slides_text.append(f"--- Slide {idx + 1} ---\n" + "\n".join(texts))

        full_content = "\n\n".join(slides_text)
        meta["slide_count"] = slide_count
        meta["page_count"] = slide_count

        return full_content, meta
