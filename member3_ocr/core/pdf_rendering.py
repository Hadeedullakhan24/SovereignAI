"""Offline, local PDF rendering for Member 3 OCR.

Renders scanned and multi-page PDFs locally into image arrays using pypdfium2.
Strictly offline: no cloud services, no network requests, no model downloads.
Dataset-safe: strictly read-only, never modifies source files or writes to datasets/.
"""
from __future__ import annotations


import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import pypdfium2 as pdfium

LOGGER = logging.getLogger(__name__)


class PDFRenderingError(RuntimeError):
    """Base exception for PDF rendering failures."""


class PDFValidationError(PDFRenderingError):
    """Raised when the PDF path does not exist, is not a file, or has an invalid extension."""


class PDFUnreadableError(PDFRenderingError):
    """Raised when the PDF file cannot be opened, parsed, or is corrupt."""


class PDFZeroPageError(PDFRenderingError):
    """Raised when a PDF contains zero renderable pages."""


class PDFPageRenderError(PDFRenderingError):
    """Raised when a specific page within a PDF fails to render."""


@dataclass(frozen=True)
class RenderedPDFPage:
    """A rendered raster image from a single PDF page."""

    page_number: int  # 1-based page number
    image: np.ndarray | None  # OpenCV BGR uint8 array (H, W, 3) or None if render failed
    width: int  # Rendered pixel width
    height: int  # Rendered pixel height
    dpi: int  # Rendered DPI
    error: str | None = None  # Error message if page rendering failed


def validate_pdf_path(path: str | Path) -> Path:
    """Validate that a path points to an existing, regular .pdf file.

    Parameters
    ----------
    path:
        File path to validate.

    Returns
    -------
    Path
        Resolved, validated Path object.

    Raises
    ------
    PDFValidationError
        If file does not exist, is a directory, or does not end with .pdf.
    """
    pdf_path = Path(path).expanduser()
    if not pdf_path.exists():
        raise PDFValidationError(f"PDF file does not exist: {pdf_path}")
    if not pdf_path.is_file():
        raise PDFValidationError(f"Path is not a regular file: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise PDFValidationError(
            f"Unsupported file type '{pdf_path.suffix}'. Expected '.pdf'"
        )
    return pdf_path


def get_pdf_page_count(path: str | Path) -> int:
    """Get the page count of a PDF without rendering all pages."""
    pdf_path = validate_pdf_path(path)
    try:
        with pdfium.PdfDocument(pdf_path) as doc:
            return len(doc)
    except Exception as exc:
        raise PDFUnreadableError(f"Could not read PDF {pdf_path}: {exc}") from exc


def render_pdf_pages(
    path: str | Path,
    *,
    dpi: int = 200,
) -> list[RenderedPDFPage]:
    """Render all pages of a PDF into BGR image arrays.

    Parameters
    ----------
    path:
        Path to the PDF file.
    dpi:
        Target rendering resolution in dots per inch (default 200).
        Standard PDF resolution is 72 points per inch; render scale is dpi / 72.

    Returns
    -------
    list[RenderedPDFPage]
        Rendered page results with 1-based page numbering.

    Raises
    ------
    PDFValidationError
        If path is missing, not a file, or not a .pdf.
    PDFUnreadableError
        If the file is corrupt or unreadable.
    PDFZeroPageError
        If the document contains 0 pages.
    """
    if dpi <= 0:
        raise ValueError(f"dpi must be positive, got {dpi}")

    pdf_path = validate_pdf_path(path)
    scale = dpi / 72.0

    try:
        doc = pdfium.PdfDocument(pdf_path)
    except Exception as exc:
        raise PDFUnreadableError(f"Failed to open or parse PDF document {pdf_path}: {exc}") from exc

    try:
        total_pages = len(doc)
        if total_pages == 0:
            raise PDFZeroPageError(f"PDF document contains zero pages: {pdf_path}")

        rendered: list[RenderedPDFPage] = []
        for page_idx in range(total_pages):
            page_num = page_idx + 1
            try:
                page = doc[page_idx]
                bitmap = page.render(scale=scale)
                pil_img = bitmap.to_pil()
                # Convert PIL RGB to OpenCV BGR
                rgb_arr = np.asarray(pil_img)
                if rgb_arr.ndim == 2:
                    bgr_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_GRAY2BGR)
                elif rgb_arr.shape[2] == 4:
                    bgr_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGBA2BGR)
                else:
                    bgr_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)

                h, w = bgr_arr.shape[:2]
                rendered.append(
                    RenderedPDFPage(
                        page_number=page_num,
                        image=bgr_arr,
                        width=w,
                        height=h,
                        dpi=dpi,
                        error=None,
                    )
                )
            except Exception as exc:
                LOGGER.warning("Failed to render page %d of %s: %s", page_num, pdf_path, exc)
                rendered.append(
                    RenderedPDFPage(
                        page_number=page_num,
                        image=None,
                        width=0,
                        height=0,
                        dpi=dpi,
                        error=f"Page rendering failed: {exc}",
                    )
                )

        return rendered
    finally:
        doc.close()
