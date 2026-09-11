"""
DOCX Loader — Microsoft Word Document Loading with Dual-Driver Resiliency.

Sovereign On-Premise Agentic AI Workbench (SIH26117 / MRPL)
Primary driver: python-docx (if available)
Fallback driver: Standard library zipfile + xml.etree.ElementTree
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET
import zipfile

from rag_engine.loaders.base_loader import BaseLoader
from rag_engine.loaders.loader_registry import register_loader

try:
    import docx
    _DOCX_AVAILABLE = True
except ImportError:
    _DOCX_AVAILABLE = False


@register_loader(".docx", ".doc")
class DOCXLoader(BaseLoader):
    """Universal loader for Microsoft Word documents."""

    def supported_formats(self) -> list[str]:
        return [".docx", ".doc"]

    def is_primary_driver_available(self) -> bool:
        return _DOCX_AVAILABLE

    def dependencies(self) -> dict[str, bool]:
        return {"python-docx": _DOCX_AVAILABLE}

    def _load_primary(self, file_path: Path) -> tuple[str, dict[str, Any]]:
        """Load DOCX using python-docx."""
        doc = docx.Document(str(file_path))
        paragraphs: list[str] = []

        for p in doc.paragraphs:
            txt = p.text.strip()
            if txt:
                paragraphs.append(txt)

        for table in doc.tables:
            for row in table.rows:
                row_txt = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                if row_txt:
                    paragraphs.append(row_txt)

        full_content = "\n\n".join(paragraphs)

        meta: dict[str, Any] = {
            "paragraph_count": len(doc.paragraphs),
            "table_count": len(doc.tables),
            "driver": "python-docx",
        }

        # Core properties
        try:
            core = doc.core_properties
            if core.title:
                meta["title"] = core.title
            if core.author:
                meta["author"] = core.author
        except Exception:
            pass

        return full_content, meta

    def _load_fallback(self, file_path: Path) -> tuple[str, dict[str, Any]]:
        """Air-gapped fallback: parse word/document.xml inside docx ZIP container."""
        paragraphs: list[str] = []
        table_count = 0
        p_count = 0
        meta: dict[str, Any] = {"driver": "stdlib_docx_fallback"}

        with zipfile.ZipFile(file_path, "r") as zf:
            # 1. Parse document body
            if "word/document.xml" in zf.namelist():
                xml_data = zf.read("word/document.xml")
                root = ET.fromstring(xml_data)

                # XML namespaces
                ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

                for p_elem in root.findall(".//w:p", ns):
                    p_count += 1
                    texts = [t.text for t in p_elem.findall(".//w:t", ns) if t.text]
                    if texts:
                        paragraphs.append("".join(texts).strip())

                tables = root.findall(".//w:tbl", ns)
                table_count = len(tables)

            # 2. Parse core metadata
            if "docProps/core.xml" in zf.namelist():
                try:
                    core_xml = zf.read("docProps/core.xml")
                    core_root = ET.fromstring(core_xml)
                    dc_ns = {"dc": "http://purl.org/dc/elements/1.1/"}
                    title_elem = core_root.find(".//dc:title", dc_ns)
                    if title_elem is not None and title_elem.text:
                        meta["title"] = title_elem.text
                    creator_elem = core_root.find(".//dc:creator", dc_ns)
                    if creator_elem is not None and creator_elem.text:
                        meta["author"] = creator_elem.text
                except Exception:
                    pass

        full_content = "\n\n".join(p for p in paragraphs if p)
        meta["paragraph_count"] = p_count
        meta["table_count"] = table_count

        return full_content, meta
