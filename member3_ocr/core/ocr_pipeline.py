"""Local, backend-neutral OCR pipeline foundation.

The primary adapter is PaddleOCR, but downstream callers consume only the
dataclasses in this module.  This module never downloads a package or model:
Paddle model directories must be supplied explicitly and exist locally.
"""
from __future__ import annotations


import argparse
import importlib
import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence, runtime_checkable

import cv2
import numpy as np

from .image_preprocessing import (
    ImagePreprocessingError,
    PreprocessingOptions,
    PreprocessingResult,
    preprocess_image,
)
from .pdf_rendering import (
    PDFPageRenderError,
    PDFRenderingError,
    PDFUnreadableError,
    PDFValidationError,
    PDFZeroPageError,
    RenderedPDFPage,
    render_pdf_pages,
)

LOGGER = logging.getLogger(__name__)
SCHEMA_VERSION = "1.0"


class OCRPipelineError(RuntimeError):
    """Base exception for OCR pipeline errors."""


class OCRConfigurationError(OCRPipelineError):
    """Raised for an invalid local OCR backend configuration."""


class MissingLocalModelError(OCRConfigurationError):
    """Raised when a required, explicitly configured model path is unavailable."""


class OCRBackendUnavailableError(OCRPipelineError):
    """Raised when the configured backend package cannot be imported."""


class OCRBackendExecutionError(OCRPipelineError):
    """Raised when a backend cannot process a valid preprocessed image."""


@dataclass(frozen=True)
class BoundingBox:
    """An axis-aligned bounding box in processed-image pixel coordinates."""

    left: float
    top: float
    right: float
    bottom: float
    coordinate_space: str = "processed_pixels"


@dataclass(frozen=True)
class Issue:
    """A serializable warning or error emitted during OCR processing."""

    code: str
    message: str
    severity: str = "warning"
    recoverable: bool = True
    backend_detail: str | None = None


@dataclass(frozen=True)
class Word:
    """A recognized word and its processed-image geometry."""

    text: str
    confidence: float | None
    bbox: BoundingBox
    polygon: tuple[tuple[float, float], ...] | None = None


@dataclass(frozen=True)
class TextLine:
    """A recognized line. Geometry is in processed-image pixel coordinates."""

    text: str
    confidence: float | None
    bbox: BoundingBox
    words: tuple[Word, ...] = ()
    polygon: tuple[tuple[float, float], ...] | None = None


@dataclass(frozen=True)
class TextBlock:
    """An OCR text region, not a semantic document-layout classification."""

    id: str
    text: str
    confidence: float | None
    bbox: BoundingBox
    lines: tuple[TextLine, ...] = ()
    polygon: tuple[tuple[float, float], ...] | None = None
    block_type: str = "text"


@dataclass(frozen=True)
class Table:
    """Structured table extracted from OCR output."""

    id: str
    bbox: BoundingBox
    confidence: float | None = None
    html: str | None = None
    markdown: str | None = None
    cells: tuple[Mapping[str, Any], ...] = ()
    page_number: int | None = None


@dataclass(frozen=True)
class KeyValueField:
    """Structured form key-value pair preserving geometry and OCR provenance."""

    key: str
    value: str
    confidence: float | None = None
    key_bbox: BoundingBox | None = None
    value_bbox: BoundingBox | None = None
    extraction_method: str = "backend"
    page_number: int | None = None


@dataclass(frozen=True)
class BackendCapabilities:
    """Features a backend can actually provide in this pipeline stage."""

    local_execution: bool = True
    text_recognition: bool = True
    word_boxes: bool = True
    line_boxes: bool = True
    polygons: bool = True
    confidence_scores: bool = True
    layout_analysis: bool = False
    table_extraction: bool = False
    key_value_extraction: bool = False


@dataclass(frozen=True)
class BackendInfo:
    """Backend and model provenance attached to every OCR document result."""

    name: str
    version: str | None
    model_ids: tuple[str, ...]
    device: str
    capabilities: BackendCapabilities


@dataclass(frozen=True)
class OCRPageResult:
    """Normalized OCR for one page/image.

    Every ``bbox`` and ``polygon`` in this result uses processed-image pixel
    coordinates. Original and processed dimensions are retained for callers
    that need to map results back to source imagery.
    """

    page_number: int
    original_width: int
    original_height: int
    processed_width: int
    processed_height: int
    text: str
    blocks: tuple[TextBlock, ...] = ()
    tables: tuple[Table, ...] = ()
    key_value_fields: tuple[KeyValueField, ...] = ()
    preprocessing_operations: tuple[str, ...] = ()
    preprocessing_time_ms: float = 0.0
    processing_time_ms: float = 0.0
    warnings: tuple[Issue, ...] = ()
    errors: tuple[Issue, ...] = ()


@dataclass(frozen=True)
class OCRDocumentResult:
    """Versioned OCR result suitable for document parsing and RAG ingestion."""

    document_id: str
    pages: tuple[OCRPageResult, ...]
    backend: BackendInfo
    provenance: Mapping[str, Any]
    schema_version: str = SCHEMA_VERSION
    warnings: tuple[Issue, ...] = ()
    errors: tuple[Issue, ...] = ()
    total_processing_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation without backend objects."""
        return _to_serializable(self)

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the stable schema as UTF-8-safe JSON text."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


@dataclass(frozen=True)
class PaddleOCRModelConfig:
    """Explicit local model configuration for PaddleOCR detection/recognition.

    No network path, package installation, or implicit model discovery is used.
    Both directories must point to pre-staged local Paddle inference models.
    """

    detection_model_dir: Path
    recognition_model_dir: Path
    detection_model_name: str = "PP-OCRv5_mobile_det"
    recognition_model_name: str = "PP-OCRv5_mobile_rec"
    language: str = "en"
    device: str = "cpu"
    allow_model_download: bool = False
    model_version: str | None = None

    def validate(self) -> None:
        """Fail early with an actionable message before PaddleOCR is imported."""
        if self.allow_model_download:
            raise OCRConfigurationError("Model download is forbidden by this offline OCR pipeline")
        missing = [
            path for path in (self.detection_model_dir, self.recognition_model_dir)
            if not Path(path).is_dir()
        ]
        if missing:
            listed = ", ".join(str(path) for path in missing)
            raise MissingLocalModelError(
                "Missing required local PaddleOCR model directory/directories: "
                f"{listed}. Stage approved inference models locally and configure "
                "detection_model_dir and recognition_model_dir explicitly. "
                "Automatic model download is disabled."
            )


@dataclass(frozen=True)
class BackendRecognition:
    """Backend-neutral internal recognition payload returned by adapters."""

    blocks: tuple[TextBlock, ...]
    warnings: tuple[Issue, ...] = ()


@runtime_checkable
class OCRBackend(Protocol):
    """Protocol implemented by local OCR backends.

    Implementations must return only normalized project dataclasses, never a
    backend SDK object or raw backend response.
    """

    @property
    def backend_info(self) -> BackendInfo: ...

    def initialize(self) -> None: ...

    def recognize(self, image: np.ndarray) -> BackendRecognition: ...


class PaddleOCRBackend:
    """Offline PaddleOCR adapter with project-schema normalization.

    This phase supports text recognition only. PP-StructureV3 layout, table,
    and key-value extraction are deliberately not initialized or advertised.
    """

    def __init__(self, config: PaddleOCRModelConfig, *, engine_factory: Any | None = None) -> None:
        self.config = config
        self._engine_factory = engine_factory
        self._engine: Any | None = None
        self._version: str | None = None

    @property
    def backend_info(self) -> BackendInfo:
        return BackendInfo(
            name="paddleocr",
            version=self._version or self.config.model_version,
            model_ids=(str(self.config.detection_model_dir), str(self.config.recognition_model_dir)),
            device=self.config.device,
            capabilities=BackendCapabilities(),
        )

    def initialize(self) -> None:
        """Create a PaddleOCR engine only after local model validation succeeds."""
        if self._engine is not None:
            return
        self.config.validate()
        try:
            if self._engine_factory is None:
                paddleocr = importlib.import_module("paddleocr")
                engine_factory = paddleocr.PaddleOCR
                self._version = getattr(paddleocr, "__version__", None)
            else:
                engine_factory = self._engine_factory
        except (ImportError, AttributeError) as exc:
            raise OCRBackendUnavailableError(
                "PaddleOCR is not installed. Install an approved, compatible local "
                "paddlepaddle/paddleocr wheel set before using PaddleOCRBackend."
            ) from exc
        try:
            # These PaddleOCR 3.x arguments disable optional orientation/unwarping
            # stages, which would otherwise introduce additional default models.
            self._engine = engine_factory(
                text_detection_model_name=self.config.detection_model_name,
                text_detection_model_dir=str(self.config.detection_model_dir),
                text_recognition_model_name=self.config.recognition_model_name,
                text_recognition_model_dir=str(self.config.recognition_model_dir),
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                device=self.config.device,
            )
        except Exception as exc:  # Backend exceptions vary between Paddle releases.
            raise OCRBackendUnavailableError(
                "PaddleOCR could not initialize the explicitly configured local models. "
                "Verify the approved PaddleOCR/PaddlePaddle versions and model layout."
            ) from exc

    def recognize(self, image: np.ndarray) -> BackendRecognition:
        """Recognize text and normalize only text boxes/lines/words for this phase."""
        self.initialize()
        if not isinstance(image, np.ndarray):
            raise OCRBackendExecutionError(
                f"Image must be a numpy ndarray, got {type(image).__name__}"
            )
        if image.ndim == 2:
            if image.dtype != np.uint8:
                raise OCRBackendExecutionError(
                    f"Unsupported image dtype: {image.dtype} (expected uint8)"
                )
            input_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3:
            if image.shape[2] != 3:
                raise OCRBackendExecutionError(
                    f"Unsupported image shape {image.shape}: 3-D images must have exactly 3 channels"
                )
            if image.dtype != np.uint8:
                raise OCRBackendExecutionError(
                    f"Unsupported image dtype: {image.dtype} (expected uint8)"
                )
            input_image = image
        else:
            raise OCRBackendExecutionError(
                f"Unsupported image dimensions: {image.ndim}D array with shape {image.shape} (expected 2-D or 3-D array)"
            )

        try:
            raw_result = list(self._engine.predict(input_image))
            return BackendRecognition(blocks=tuple(self.normalize_raw_output(raw_result)))
        except OCRPipelineError:
            raise
        except Exception as exc:  # SDK-specific failures must not escape downstream.
            raise OCRBackendExecutionError("PaddleOCR failed while recognizing the preprocessed image") from exc

    @classmethod
    def normalize_raw_output(cls, raw_result: Any) -> list[TextBlock]:
        """Normalize PaddleOCR 3.x payloads (and v2-style records) into dataclasses.

        Standard PaddleOCR 3.x ``rec_polys``/``rec_texts``/``rec_scores`` payloads
        and legacy ``[[polygon, (text, confidence)], …]`` records are accepted.
        Empty responses normalize to an empty list. Unknown SDK response formats
        fail explicitly instead of leaking untyped Paddle objects upstream.
        """
        records = cls._unwrap_records(raw_result)
        normalized: list[tuple[TextBlock, float, float]] = []
        for index, record in enumerate(records):
            try:
                polygon_raw, recognition = record
                text, confidence_raw = recognition
                polygon = _normalize_polygon(polygon_raw)
                bbox = _bbox_from_polygon(polygon)
                confidence = _normalize_confidence(confidence_raw)
            except (TypeError, ValueError) as exc:
                raise OCRBackendExecutionError(
                    "PaddleOCR returned an unsupported text-detection record; "
                    "pin a supported PaddleOCR release or update the adapter."
                ) from exc
            word = Word(text=str(text), confidence=confidence, bbox=bbox, polygon=polygon)
            line = TextLine(text=str(text), confidence=confidence, bbox=bbox, words=(word,), polygon=polygon)
            block = TextBlock(
                id=f"text-{index + 1}", text=str(text), confidence=confidence,
                bbox=bbox, lines=(line,), polygon=polygon,
            )
            normalized.append((block, bbox.top, bbox.left))
        # A simple top/left order is intentional: semantic document reading order
        # belongs to the later document-structure enrichment stage.
        normalized.sort(key=lambda item: (round(item[1] / 10) * 10, item[2]))
        return [item[0] for item in normalized]

    @staticmethod
    def _unwrap_records(raw_result: Any) -> list[Any]:
        if raw_result is None:
            return []
        if not isinstance(raw_result, (list, tuple)):
            raise OCRBackendExecutionError("PaddleOCR returned a non-list result unsupported by this adapter")
        if not raw_result:
            return []
        first = raw_result[0]
        # PaddleOCR 3.x returns an OCRResult payload with rec_polys/texts/scores.
        payload = _as_mapping(first)
        if payload is not None:
            payload = _as_mapping(payload.get("res", payload)) or payload
            # Do not use truthiness here: ``[]`` is a valid zero-detection OCR
            # result and must remain distinct from a field that is absent.
            polygons = payload.get("rec_polys") if "rec_polys" in payload else payload.get("dt_polys")
            texts = payload.get("rec_texts") if "rec_texts" in payload else None
            scores = payload.get("rec_scores") if "rec_scores" in payload else None
            if polygons is None or texts is None or scores is None:
                raise OCRBackendExecutionError("PaddleOCR 3.x result omitted text polygons, texts, or scores")
            if not (len(polygons) == len(texts) == len(scores)):
                raise OCRBackendExecutionError("PaddleOCR 3.x result arrays have inconsistent lengths")
            return [[polygon, (text, score)] for polygon, text, score in zip(polygons, texts, scores)]
        # PaddleOCR 2.x-compatible response: [ [ record, ... ] ].
        if isinstance(first, (list, tuple)) and len(first) == 2 and _is_polygon_like(first[0]):
            return list(raw_result)
        if len(raw_result) == 1 and isinstance(first, (list, tuple)):
            return list(first)
        raise OCRBackendExecutionError("PaddleOCR returned multiple/unsupported page payloads for a single image")


class OCRPipeline:
    """Orchestrate preprocessing and one local OCR backend for one image/page."""

    def __init__(
        self,
        backend: OCRBackend,
        *,
        default_preprocessing: PreprocessingOptions | None = None,
        extract_key_values: bool = True,
        extract_tables: bool = True,
    ) -> None:
        if not isinstance(backend, OCRBackend):
            raise TypeError("backend must implement OCRBackend")
        self.backend = backend
        self.default_preprocessing = default_preprocessing or PreprocessingOptions.document_ocr()
        self.extract_key_values = extract_key_values
        self.extract_tables = extract_tables

    def process_image(
        self,
        source: str | Path | PreprocessingResult,
        *,
        document_id: str | None = None,
        page_number: int = 1,
        preprocessing_options: PreprocessingOptions | None = None,
        extract_key_values: bool | None = None,
        extract_tables: bool | None = None,
    ) -> OCRDocumentResult:
        """Run OCR for an image path or existing preprocessing result.

        Invalid image input and backend execution failures become structured result
        errors. Configuration and missing-model failures are raised early because
        callers must explicitly remediate those deployment conditions.
        """
        if page_number < 1:
            raise ValueError("page_number must be one-based and positive")
        start = time.perf_counter()
        resolved_id = document_id or _document_id_from_source(source)
        try:
            prepared = source if isinstance(source, PreprocessingResult) else preprocess_image(
                source, preprocessing_options or self.default_preprocessing, save_output=False
            )
        except ImagePreprocessingError as exc:
            return OCRDocumentResult(
                document_id=resolved_id,
                pages=(), backend=self.backend.backend_info,
                provenance={"source_path": str(source), "source_type": "image_path"},
                errors=(Issue("invalid_image", str(exc), severity="error", recoverable=True),),
                total_processing_time_ms=_elapsed_ms(start),
            )

        self.backend.initialize()
        try:
            recognized = self.backend.recognize(prepared.image)
            page_errors: tuple[Issue, ...] = ()
            blocks = recognized.blocks
            warnings = recognized.warnings
        except OCRBackendExecutionError as exc:
            blocks = ()
            warnings = ()
            page_errors = (Issue("backend_execution_failed", str(exc), severity="error", recoverable=True),)
            LOGGER.warning("OCR backend failed for %s: %s", prepared.original_path, exc)

        should_extract_kv = self.extract_key_values if extract_key_values is None else extract_key_values
        kv_fields: tuple[KeyValueField, ...] = ()
        if should_extract_kv and blocks:
            from .form_extractor import extract_form_key_values
            kv_fields, kv_issues = extract_form_key_values(blocks, page_number=page_number)
            warnings = warnings + kv_issues

        should_extract_tbl = self.extract_tables if extract_tables is None else extract_tables
        tables: tuple[Table, ...] = ()
        if should_extract_tbl and blocks:
            from .table_extractor import extract_tables_from_blocks
            tables, tbl_issues = extract_tables_from_blocks(blocks, page_number=page_number)
            warnings = warnings + tbl_issues

        page = OCRPageResult(
            page_number=page_number,
            original_width=prepared.original_width,
            original_height=prepared.original_height,
            processed_width=prepared.processed_width,
            processed_height=prepared.processed_height,
            text="\n".join(block.text for block in blocks),
            blocks=blocks,
            tables=tables,
            key_value_fields=kv_fields,
            preprocessing_operations=tuple(prepared.operations_applied),
            preprocessing_time_ms=prepared.processing_time_seconds * 1000,
            processing_time_ms=_elapsed_ms(start),
            warnings=warnings,
            errors=page_errors,
        )
        return OCRDocumentResult(
            document_id=resolved_id,
            pages=(page,), backend=self.backend.backend_info,
            provenance={
                "source_path": str(prepared.original_path),
                "source_type": "preprocessing_result" if isinstance(source, PreprocessingResult) else "image_path",
                "original_mode": prepared.original_mode,
                "coordinate_space": "processed_pixels",
            },
            warnings=warnings,
            errors=page_errors,
            total_processing_time_ms=_elapsed_ms(start),
        )

    def process_pdf(
        self,
        source: str | Path,
        *,
        document_id: str | None = None,
        preprocessing_options: PreprocessingOptions | None = None,
        dpi: int = 200,
        extract_key_values: bool | None = None,
        extract_tables: bool | None = None,
    ) -> OCRDocumentResult:
        """Run OCR on all pages of a local scanned or digital PDF document.

        Parameters
        ----------
        source:
            Filesystem path to the .pdf document.
        document_id:
            Explicit document ID. Defaults to source filename stem.
        preprocessing_options:
            Overrides for page image preprocessing.
        dpi:
            Rendering resolution in dots per inch (default 200).

        Returns
        -------
        OCRDocumentResult
            Normalized result containing an OCRPageResult for each page in order.
        """
        start = time.perf_counter()
        resolved_id = document_id or _document_id_from_source(source)
        source_str = str(source)

        # 1. Path & file validation
        pdf_path = Path(source).expanduser()
        if not pdf_path.exists():
            return OCRDocumentResult(
                document_id=resolved_id,
                pages=(),
                backend=self.backend.backend_info,
                provenance={"source_path": source_str, "source_type": "pdf_path"},
                errors=(Issue("missing_file", f"PDF file does not exist: {source}", severity="error", recoverable=False),),
                total_processing_time_ms=_elapsed_ms(start),
            )
        if not pdf_path.is_file():
            return OCRDocumentResult(
                document_id=resolved_id,
                pages=(),
                backend=self.backend.backend_info,
                provenance={"source_path": source_str, "source_type": "pdf_path"},
                errors=(Issue("invalid_pdf", f"Path is not a regular file: {source}", severity="error", recoverable=False),),
                total_processing_time_ms=_elapsed_ms(start),
            )
        if pdf_path.suffix.lower() != ".pdf":
            return OCRDocumentResult(
                document_id=resolved_id,
                pages=(),
                backend=self.backend.backend_info,
                provenance={"source_path": source_str, "source_type": "pdf_path"},
                errors=(Issue("unsupported_format", f"Unsupported file type '{pdf_path.suffix}'. Expected '.pdf'", severity="error", recoverable=False),),
                total_processing_time_ms=_elapsed_ms(start),
            )

        # 2. Render pages locally
        try:
            rendered_pages = render_pdf_pages(pdf_path, dpi=dpi)
        except PDFZeroPageError as exc:
            return OCRDocumentResult(
                document_id=resolved_id,
                pages=(),
                backend=self.backend.backend_info,
                provenance={"source_path": source_str, "source_type": "pdf_path", "total_pages": 0, "rendered_dpi": dpi},
                errors=(Issue("zero_page_pdf", str(exc), severity="error", recoverable=False),),
                total_processing_time_ms=_elapsed_ms(start),
            )
        except PDFUnreadableError as exc:
            return OCRDocumentResult(
                document_id=resolved_id,
                pages=(),
                backend=self.backend.backend_info,
                provenance={"source_path": source_str, "source_type": "pdf_path"},
                errors=(Issue("invalid_pdf", str(exc), severity="error", recoverable=False),),
                total_processing_time_ms=_elapsed_ms(start),
            )
        except PDFRenderingError as exc:
            return OCRDocumentResult(
                document_id=resolved_id,
                pages=(),
                backend=self.backend.backend_info,
                provenance={"source_path": source_str, "source_type": "pdf_path"},
                errors=(Issue("pdf_rendering_error", str(exc), severity="error", recoverable=False),),
                total_processing_time_ms=_elapsed_ms(start),
            )

        # 3. Initialize backend
        self.backend.initialize()

        # 4. Process each page sequentially
        page_results: list[OCRPageResult] = []
        doc_warnings: list[Issue] = []
        doc_errors: list[Issue] = []

        for r_page in rendered_pages:
            page_start = time.perf_counter()
            p_num = r_page.page_number

            if r_page.error is not None or r_page.image is None:
                err_issue = Issue("page_render_failed", r_page.error or f"Page {p_num} render failed", severity="error", recoverable=True)
                doc_errors.append(err_issue)
                page_results.append(
                    OCRPageResult(
                        page_number=p_num,
                        original_width=r_page.width,
                        original_height=r_page.height,
                        processed_width=r_page.width,
                        processed_height=r_page.height,
                        text="",
                        blocks=(),
                        tables=(),
                        key_value_fields=(),
                        preprocessing_operations=(),
                        preprocessing_time_ms=0.0,
                        processing_time_ms=_elapsed_ms(page_start),
                        warnings=(),
                        errors=(err_issue,),
                    )
                )
                continue

            # Run existing preprocessing on rendered image array
            try:
                prepared = preprocess_image(
                    r_page.image,
                    preprocessing_options or self.default_preprocessing,
                    save_output=False,
                    source_path=pdf_path,
                )
            except ImagePreprocessingError as exc:
                err_issue = Issue("preprocessing_failed", str(exc), severity="error", recoverable=True)
                doc_errors.append(err_issue)
                page_results.append(
                    OCRPageResult(
                        page_number=p_num,
                        original_width=r_page.width,
                        original_height=r_page.height,
                        processed_width=r_page.width,
                        processed_height=r_page.height,
                        text="",
                        blocks=(),
                        tables=(),
                        key_value_fields=(),
                        preprocessing_operations=(),
                        preprocessing_time_ms=0.0,
                        processing_time_ms=_elapsed_ms(page_start),
                        warnings=(),
                        errors=(err_issue,),
                    )
                )
                continue

            # Run existing backend recognition
            try:
                recognized = self.backend.recognize(prepared.image)
                page_errors: tuple[Issue, ...] = ()
                blocks = recognized.blocks
                warnings = recognized.warnings
            except OCRBackendExecutionError as exc:
                blocks = ()
                warnings = ()
                page_errors = (Issue("backend_execution_failed", str(exc), severity="error", recoverable=True),)
                doc_errors.extend(page_errors)
                LOGGER.warning("OCR backend failed on PDF %s page %d: %s", pdf_path, p_num, exc)

            should_extract_kv = self.extract_key_values if extract_key_values is None else extract_key_values
            kv_fields: tuple[KeyValueField, ...] = ()
            if should_extract_kv and blocks:
                from .form_extractor import extract_form_key_values
                kv_fields, kv_issues = extract_form_key_values(blocks, page_number=p_num)
                warnings = warnings + kv_issues

            should_extract_tbl = self.extract_tables if extract_tables is None else extract_tables
            tables: tuple[Table, ...] = ()
            if should_extract_tbl and blocks:
                from .table_extractor import extract_tables_from_blocks
                tables, tbl_issues = extract_tables_from_blocks(blocks, page_number=p_num)
                warnings = warnings + tbl_issues

            doc_warnings.extend(warnings)

            # Construct OCRPageResult for this page
            page = OCRPageResult(
                page_number=p_num,
                original_width=prepared.original_width,
                original_height=prepared.original_height,
                processed_width=prepared.processed_width,
                processed_height=prepared.processed_height,
                text="\n".join(b.text for b in blocks),
                blocks=blocks,
                tables=tables,
                key_value_fields=kv_fields,
                preprocessing_operations=tuple(prepared.operations_applied),
                preprocessing_time_ms=prepared.processing_time_seconds * 1000,
                processing_time_ms=_elapsed_ms(page_start),
                warnings=warnings,
                errors=page_errors,
            )
            page_results.append(page)

        # 5. Assemble OCRDocumentResult representing the entire PDF
        return OCRDocumentResult(
            document_id=resolved_id,
            pages=tuple(page_results),
            backend=self.backend.backend_info,
            provenance={
                "source_path": str(pdf_path),
                "source_type": "pdf_path",
                "total_pages": len(rendered_pages),
                "rendered_dpi": dpi,
                "coordinate_space": "processed_pixels",
            },
            warnings=tuple(doc_warnings),
            errors=tuple(doc_errors),
            total_processing_time_ms=_elapsed_ms(start),
        )


def _normalize_polygon(value: Any) -> tuple[tuple[float, float], ...]:
    if not _is_polygon_like(value):
        raise ValueError("invalid polygon")
    polygon = tuple((float(point[0]), float(point[1])) for point in value)
    if len(polygon) < 4:
        raise ValueError("polygon must contain at least four points")
    return polygon


def _is_polygon_like(value: Any) -> bool:
    return isinstance(value, (list, tuple, np.ndarray)) and len(value) >= 4 and all(
        isinstance(point, (list, tuple, np.ndarray)) and len(point) >= 2 for point in value
    )


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    """Read a PaddleOCR 3.x result object locally without exposing it upstream."""
    if isinstance(value, Mapping):
        return value
    candidate = getattr(value, "json", None)
    if callable(candidate):
        candidate = candidate()
    if isinstance(candidate, str):
        try:
            candidate = json.loads(candidate)
        except json.JSONDecodeError:
            return None
    return candidate if isinstance(candidate, Mapping) else None


def _bbox_from_polygon(polygon: Sequence[tuple[float, float]]) -> BoundingBox:
    xs, ys = zip(*polygon)
    return BoundingBox(left=min(xs), top=min(ys), right=max(xs), bottom=max(ys))


def _normalize_confidence(value: Any) -> float | None:
    if value is None:
        return None
    confidence = float(value)
    if not math.isfinite(confidence):
        return None
    return confidence / 100 if confidence > 1 else max(0.0, confidence)


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 3)


def _document_id_from_source(source: str | Path | PreprocessingResult) -> str:
    if isinstance(source, PreprocessingResult):
        return source.original_path.stem
    return Path(source).stem


def _to_serializable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _to_serializable(item) for key, item in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _to_serializable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_to_serializable(item) for item in value]
    return value


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local PaddleOCR on an image or PDF with explicit local models.")
    parser.add_argument("--input", required=True, help="Input image or PDF path")
    parser.add_argument("--det-model-dir", required=True, help="Pre-staged local PaddleOCR detection-model directory")
    parser.add_argument("--rec-model-dir", required=True, help="Pre-staged local PaddleOCR recognition-model directory")
    parser.add_argument("--document-id", help="Explicit document ID")
    parser.add_argument("--page-number", type=int, default=1, help="Page number (for single image input)")
    parser.add_argument("--dpi", type=int, default=200, help="Rendering DPI for PDF inputs (default: 200)")
    parser.add_argument("--language", default="en")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", help="Optional JSON output path outside datasets/")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run an explicitly local, image or PDF PaddleOCR smoke test."""
    args = _build_cli_parser().parse_args(argv)
    config = PaddleOCRModelConfig(
        detection_model_dir=Path(args.det_model_dir), recognition_model_dir=Path(args.rec_model_dir),
        language=args.language, device=args.device,
    )
    pipeline = OCRPipeline(PaddleOCRBackend(config))
    try:
        if Path(args.input).suffix.lower() == ".pdf":
            result = pipeline.process_pdf(
                args.input, document_id=args.document_id, dpi=args.dpi
            )
        else:
            result = pipeline.process_image(
                args.input, document_id=args.document_id, page_number=args.page_number
            )
    except OCRPipelineError as exc:
        LOGGER.error("OCR configuration failed: %s", exc)
        return 2
    rendered = result.to_json()
    if args.output:
        output_path = Path(args.output).resolve()
        dataset_root = Path(__file__).resolve().parents[2] / "datasets"
        if output_path.is_relative_to(dataset_root.resolve()):
            LOGGER.error("Refusing to write OCR output inside datasets/")
            return 2
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
    else:
        print(rendered)
    return 1 if result.errors else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())
