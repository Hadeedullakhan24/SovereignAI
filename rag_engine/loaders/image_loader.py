"""
Image Loader — Visual Asset & Engineering Drawing Loader.

Sovereign On-Premise Agentic AI Workbench (SIH26117 / MRPL)
Primary driver: Pillow (PIL)
Fallback driver: Standard library struct binary header unpacker (air-gapped)

Note: In accordance with Milestone 3A boundaries, this loader indexes the image
reference, extracts dimensions and visual metadata, but does NOT perform deep OCR.
"""

from __future__ import annotations

from pathlib import Path
import struct
from typing import Any

from rag_engine.loaders.base_loader import BaseLoader
from rag_engine.loaders.loader_registry import register_loader

try:
    from PIL import Image
    _PILLOW_AVAILABLE = True
except ImportError:
    _PILLOW_AVAILABLE = False


@register_loader(".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif")
class ImageLoader(BaseLoader):
    """Universal loader for refinery engineering drawings, P&IDs, and inspection photos."""

    def supported_formats(self) -> list[str]:
        return [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"]

    def is_primary_driver_available(self) -> bool:
        return _PILLOW_AVAILABLE

    def dependencies(self) -> dict[str, bool]:
        return {"Pillow": _PILLOW_AVAILABLE}

    def _load_primary(self, file_path: Path) -> tuple[str, dict[str, Any]]:
        """Load visual metadata using Pillow."""
        with Image.open(str(file_path)) as img:
            width, height = img.size
            img_format = img.format or file_path.suffix.upper().lstrip(".")
            mode = img.mode

        meta: dict[str, Any] = {
            "width": width,
            "height": height,
            "dimensions": f"{width}x{height}",
            "color_mode": mode,
            "image_format": img_format,
            "image_reference": file_path.as_posix(),
            "driver": "pillow",
        }

        # Descriptive placeholder content until OCR pipeline (Milestone 3B / Member 2)
        content = (
            f"[Visual Asset: {file_path.name}]\n"
            f"Type: {img_format} Image\n"
            f"Resolution: {width}x{height}\n"
            f"Path: {file_path.as_posix()}"
        )
        return content, meta

    def _load_fallback(self, file_path: Path) -> tuple[str, dict[str, Any]]:
        """Air-gapped fallback: parse dimensions directly from image header bytes using struct."""
        width, height, img_format = self._parse_dimensions_from_bytes(file_path)

        meta: dict[str, Any] = {
            "width": width,
            "height": height,
            "dimensions": f"{width}x{height}" if width and height else "unknown",
            "image_format": img_format,
            "image_reference": file_path.as_posix(),
            "driver": "stdlib_struct_header_parser",
        }

        content = (
            f"[Visual Asset: {file_path.name}]\n"
            f"Type: {img_format} Image\n"
            f"Resolution: {width}x{height}\n"
            f"Path: {file_path.as_posix()}"
        )
        return content, meta

    def _parse_dimensions_from_bytes(self, file_path: Path) -> tuple[int, int, str]:
        """Extract dimensions without 3rd party libraries."""
        ext = file_path.suffix.lower()
        with open(file_path, "rb") as fh:
            data = fh.read(1024)

        # PNG
        if ext == ".png" and len(data) >= 24 and data.startswith(b"\x89PNG\r\n\x1a\n"):
            w, h = struct.unpack(">II", data[16:24])
            return w, h, "PNG"

        # BMP
        if ext == ".bmp" and len(data) >= 26 and data.startswith(b"BM"):
            w, h = struct.unpack("<ii", data[18:26])
            return abs(w), abs(h), "BMP"

        # JPEG
        if ext in {".jpg", ".jpeg"} and data.startswith(b"\xff\xd8"):
            # Search for SOF0 (0xFF, 0xC0) or SOF2 (0xFF, 0xC2)
            with open(file_path, "rb") as fh:
                full = fh.read(65536)  # SOF is usually within the first 64KB
            idx = 2
            while idx < len(full) - 9:
                if full[idx] == 0xFF and full[idx + 1] in {0xC0, 0xC1, 0xC2}:
                    # Header length is full[idx+2:idx+4]
                    # Precision is full[idx+4]
                    h, w = struct.unpack(">HH", full[idx + 5:idx + 9])
                    return w, h, "JPEG"
                idx += 1
            return 0, 0, "JPEG"

        return 0, 0, ext.upper().lstrip(".")
