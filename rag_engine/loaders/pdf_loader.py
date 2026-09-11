"""
PDF Loader — Universal PDF Document Loading with Dual-Driver Resiliency.

Sovereign On-Premise Agentic AI Workbench (SIH26117 / MRPL)
Primary driver: pypdf (if available)
Fallback driver: Standard library zlib + regex stream extractor (air-gapped)
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any
import zlib

from rag_engine.loaders.base_loader import BaseLoader
from rag_engine.loaders.loader_registry import register_loader

try:
    import pypdf
    _PYPDF_AVAILABLE = True
except ImportError:
    _PYPDF_AVAILABLE = False


@register_loader(".pdf")
class PDFLoader(BaseLoader):
    """Universal loader for PDF documents supporting refinery manuals, P&IDs, and reports."""

    def supported_formats(self) -> list[str]:
        return [".pdf"]

    def is_primary_driver_available(self) -> bool:
        return _PYPDF_AVAILABLE

    def dependencies(self) -> dict[str, bool]:
        return {"pypdf": _PYPDF_AVAILABLE}

    def _load_primary(self, file_path: Path) -> tuple[str, dict[str, Any]]:
        """Load PDF using pypdf reader."""
        reader = pypdf.PdfReader(str(file_path))
        page_count = len(reader.pages)
        pages_text: list[str] = []

        for idx, page in enumerate(reader.pages):
            try:
                txt = page.extract_text() or ""
                if txt.strip():
                    pages_text.append(f"--- Page {idx + 1} ---\n{txt}")
            except Exception:
                continue

        full_content = "\n\n".join(pages_text)

        meta: dict[str, Any] = {
            "page_count": page_count,
            "is_encrypted": reader.is_encrypted,
            "driver": "pypdf",
        }

        # Extract document info if present
        if reader.metadata:
            meta["title"] = reader.metadata.title or ""
            meta["author"] = reader.metadata.author or ""
            meta["producer"] = reader.metadata.producer or ""

        return full_content, meta

    def _load_fallback(self, file_path: Path) -> tuple[str, dict[str, Any]]:
        """Air-gapped fallback: parse raw PDF streams using zlib and regex."""
        with open(file_path, "rb") as fh:
            data = fh.read()

        # 1. Estimate page count from /Type /Page objects
        page_matches = re.findall(rb"/Type\s*/Page\b", data)
        page_count = max(1, len(page_matches))

        # 2. Extract and decompress text streams
        extracted_text_chunks: list[str] = []
        stream_pattern = re.compile(rb"stream[\r\n]+(.*?)[\r\n]+endstream", re.DOTALL)
        tj_pattern = re.compile(rb"\[(.*?)\]\s*TJ", re.DOTALL)
        lit_pattern = re.compile(rb"\(((?:[^()\\]|\\.)*)\)")

        for match in stream_pattern.finditer(data):
            stream_data = match.group(1)
            # Attempt zlib inflate
            try:
                decompressed = zlib.decompress(stream_data)
            except Exception:
                decompressed = stream_data

            # Check if this is a valid text content stream
            if b"BT" not in decompressed or b"ET" not in decompressed:
                continue
            # Skip embedded fonts and CID tables
            if any(marker in decompressed for marker in (b"/Font", b"/Type1", b"/TrueType", b"CIDInit")):
                continue

            # Extract within BT ... ET blocks
            for bt_match in re.finditer(rb"BT(.*?)ET", decompressed, re.DOTALL):
                bt_content = bt_match.group(1)
                found_tj = False
                for tj in tj_pattern.finditer(bt_content):
                    found_tj = True
                    parts = [
                        p.decode("latin1", errors="ignore").replace(r"\(", "(").replace(r"\)", ")")
                        for p in lit_pattern.findall(tj.group(1))
                    ]
                    combined = "".join(parts).strip()
                    if combined and "downloaded by" not in combined.lower():
                        alpha_count = sum(1 for c in combined if c.isalnum() or c.isspace())
                        if alpha_count / len(combined) >= 0.65:
                            extracted_text_chunks.append(combined)

                if not found_tj:
                    for ts in re.finditer(rb"\(((?:[^()\\]|\\.)*)\)\s*T[jJ]", bt_content):
                        line = ts.group(1).decode("latin1", errors="ignore").replace(r"\(", "(").replace(r"\)", ")").strip()
                        if line and "downloaded by" not in line.lower():
                            alpha_count = sum(1 for c in line if c.isalnum() or c.isspace())
                            if alpha_count / len(line) >= 0.65:
                                extracted_text_chunks.append(line)

        # Fallback if no BT...ET blocks were found
        if not extracted_text_chunks:
            for match in stream_pattern.finditer(data):
                try:
                    decompressed = zlib.decompress(match.group(1))
                except Exception:
                    decompressed = match.group(1)
                for lit in lit_pattern.findall(decompressed):
                    try:
                        s = lit.decode("utf-8", errors="ignore").strip()
                        if s and "downloaded by" not in s.lower() and len(s) > 2:
                            extracted_text_chunks.append(s)
                    except Exception:
                        continue

        full_content = "\n".join(extracted_text_chunks)

        meta: dict[str, Any] = {
            "page_count": page_count,
            "is_encrypted": b"/Encrypt" in data,
            "driver": "stdlib_pdf_fallback",
        }

        return full_content, meta
