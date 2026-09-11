"""
Base Loader — Abstract Base Class & Execution Template for Document Loaders.

Sovereign On-Premise Agentic AI Workbench (SIH26117 / MRPL)
Establishes the contract, multi-stage validation pipeline, telemetry collection,
and dual-driver fallback pattern for all document loaders.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
import logging
from pathlib import Path
import time
from typing import Any, Optional
import uuid

from rag_engine.config.logging_config import get_logger
from rag_engine.loaders.exceptions import (
    CorruptedDocumentError,
    LoaderError,
    UnsupportedFormatError,
    ValidationFailedError,
)
from rag_engine.loaders.loader_events import (
    CorruptedDocument,
    DocumentFailed,
    DocumentLoaded,
    FallbackActivated,
    ValidationFailed,
    global_event_bus,
)
from rag_engine.loaders.loader_health import LoaderHealthReport, LoaderHealthStatus
from rag_engine.loaders.loader_metrics import LoadMetrics, global_metrics
from rag_engine.loaders.loader_utils import (
    compute_file_sha256,
    count_characters,
    count_words,
    detect_mime_type,
    estimate_tokens,
    get_file_timestamps,
)
from rag_engine.loaders.processing_context import ProcessingContext
from rag_engine.schemas.document import Document, DocumentLifecycleState, DocumentMetadata

logger = get_logger("loaders.base_loader")


class BaseLoader(ABC):
    """Abstract Base Class for all universal document loaders."""

    def __init__(self, max_file_size_bytes: int = 500 * 1024 * 1024) -> None:
        """
        Initialize BaseLoader.

        Args:
            max_file_size_bytes: Maximum allowed file size in bytes (default: 500MB).
        """
        self.max_file_size_bytes = max_file_size_bytes

    # --- Core Contracts ---

    @abstractmethod
    def supported_formats(self) -> list[str]:
        """Return list of supported lowercase extensions with dot (e.g. ['.pdf'])."""
        ...

    @abstractmethod
    def _load_primary(self, file_path: Path) -> tuple[str, dict[str, Any]]:
        """
        Execute primary specialized driver.

        Returns:
            Tuple of (raw_text_content, format_specific_metadata).
        """
        ...

    @abstractmethod
    def _load_fallback(self, file_path: Path) -> tuple[str, dict[str, Any]]:
        """
        Execute air-gapped standard library fallback driver.

        Returns:
            Tuple of (raw_text_content, format_specific_metadata).
        """
        ...

    @abstractmethod
    def is_primary_driver_available(self) -> bool:
        """Return True if external driver dependencies are installed."""
        ...

    def version(self) -> str:
        """Return loader component version."""
        return "1.0.0"

    def dependencies(self) -> dict[str, bool]:
        """Return dictionary of dependency package names and availability."""
        return {"primary_driver": self.is_primary_driver_available()}

    def health(self) -> LoaderHealthReport:
        """Generate operational health report for this loader."""
        primary_avail = self.is_primary_driver_available()
        status = (
            LoaderHealthStatus.HEALTHY
            if primary_avail
            else LoaderHealthStatus.DEGRADED
        )
        msg = (
            "Primary driver operational"
            if primary_avail
            else "Primary driver unavailable; operating in air-gapped fallback mode"
        )
        return LoaderHealthReport(
            loader_name=self.__class__.__name__,
            status=status,
            version=self.version(),
            supported_formats=self.supported_formats(),
            primary_driver_available=primary_avail,
            fallback_driver_available=True,
            dependencies=self.dependencies(),
            message=msg,
        )

    # --- Validation Pipeline ---

    def can_load(self, file_path: Path | str) -> bool:
        """Check if this loader supports the given file extension."""
        p = Path(file_path)
        return p.suffix.lower() in {ext.lower() for ext in self.supported_formats()}

    def validate(self, file_path: Path | str) -> bool:
        """
        Execute multi-stage pre-load validation:
        1. File exists
        2. Read permissions
        3. Size bounds (non-empty & <= max_size)
        4. Supported format

        Raises:
            ValidationFailedError: On check failure.
        """
        p = Path(file_path).resolve()

        if not p.exists() or not p.is_file():
            raise ValidationFailedError(f"File does not exist or is not a regular file: {p}", str(p))

        try:
            stat = p.stat()
        except Exception as e:
            raise ValidationFailedError(f"Permission denied accessing file: {e}", str(p))

        if stat.st_size == 0:
            raise ValidationFailedError(f"File is empty (0 bytes): {p}", str(p))

        if stat.st_size > self.max_file_size_bytes:
            raise ValidationFailedError(
                f"File size ({stat.st_size} bytes) exceeds limit ({self.max_file_size_bytes} bytes)",
                str(p),
            )

        if not self.can_load(p):
            raise UnsupportedFormatError(
                f"Format '{p.suffix}' is not supported by {self.__class__.__name__}",
                str(p),
            )

        return True

    def extract_basic_metadata(self, file_path: Path | str) -> dict[str, Any]:
        """Extract baseline filesystem metadata."""
        p = Path(file_path).resolve()
        stat = p.stat()
        ctime_iso, mtime_iso = get_file_timestamps(p)
        checksum = compute_file_sha256(p)
        mime = detect_mime_type(p)

        return {
            "source_path": p.as_posix(),
            "file_name": p.name,
            "file_format": p.suffix.lower(),
            "mime_type": mime,
            "file_size_bytes": stat.st_size,
            "checksum_sha256": checksum,
            "created_at": ctime_iso,
            "modified_at": mtime_iso,
        }

    # --- Loading Workflow ---

    def load(
        self,
        file_path: Path | str,
        context: Optional[ProcessingContext] = None,
    ) -> Document:
        """
        Execute universal loading workflow with dual-driver fallback,
        multi-stage validation, telemetry metrics, and event notification.
        """
        path = Path(file_path).resolve()
        start_time = time.perf_counter()
        ctx = context or ProcessingContext.create_default()

        loader_name = self.__class__.__name__
        logger.info(f"Loading document [{loader_name}]: {path}")

        # 1. Multi-stage Validation
        try:
            self.validate(path)
        except ValidationFailedError as e:
            global_event_bus.publish(ValidationFailed(file_path=str(path), loader_name=loader_name, reason=str(e)))
            logger.error(f"Validation failed for {path}: {e}")
            raise
        except UnsupportedFormatError as e:
            global_event_bus.publish(ValidationFailed(file_path=str(path), loader_name=loader_name, reason=str(e)))
            logger.error(f"Unsupported format for {path}: {e}")
            raise

        # 2. Extract baseline filesystem metadata
        base_meta = self.extract_basic_metadata(path)

        # 3. Dual-Driver Execution
        raw_content = ""
        extra_meta: dict[str, Any] = {}
        primary_used = False
        fallback_used = False
        driver_name = "unknown"
        warnings: list[str] = []
        errors: list[str] = []

        if self.is_primary_driver_available():
            try:
                raw_content, extra_meta = self._load_primary(path)
                primary_used = True
                driver_name = "primary"
            except Exception as e:
                warnings.append(f"Primary driver failed: {e}. Attempting fallback driver.")
                global_event_bus.publish(
                    FallbackActivated(
                        file_path=str(path),
                        loader_name=loader_name,
                        primary_driver="primary",
                        fallback_driver="fallback",
                        reason=str(e),
                    )
                )
                logger.warning(f"Primary driver failed on {path}: {e}. Switching to fallback.")

        if not primary_used:
            try:
                raw_content, extra_meta = self._load_fallback(path)
                fallback_used = True
                driver_name = "fallback"
            except Exception as e:
                errors.append(f"Fallback driver failed: {e}")
                global_event_bus.publish(
                    CorruptedDocument(file_path=str(path), loader_name=loader_name, error_details=str(e))
                )
                global_event_bus.publish(
                    DocumentFailed(file_path=str(path), loader_name=loader_name, error=str(e))
                )
                logger.error(f"Fallback driver also failed on {path}: {e}")
                raise CorruptedDocumentError(f"Failed to load document via primary and fallback: {e}", str(path))

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # 4. Text and Token Metrics
        word_cnt = count_words(raw_content)
        char_cnt = count_characters(raw_content)
        tokens_est = estimate_tokens(raw_content)

        # 5. Build Unified Document Schema
        doc_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, base_meta["source_path"]))
        image_ref = extra_meta.get("image_reference")

        metadata = DocumentMetadata(
            source_path=base_meta["source_path"],
            file_name=base_meta["file_name"],
            file_format=base_meta["file_format"],
            mime_type=base_meta["mime_type"],
            file_size_bytes=base_meta["file_size_bytes"],
            checksum_sha256=base_meta["checksum_sha256"],
            category=ctx.category,
            subcategory=ctx.subcategory,
            created_at=base_meta["created_at"],
            modified_at=base_meta["modified_at"],
            loader_name=loader_name,
            language=extra_meta.get("language", "en"),
            page_count=extra_meta.get("page_count"),
            word_count=word_cnt,
            character_count=char_cnt,
            estimated_tokens=tokens_est,
            image_reference=image_ref,
            loading_status="SUCCESS",
            lifecycle_state=DocumentLifecycleState.LOADED,
            processing_history=[{
                "stage": "LOADED",
                "loader": loader_name,
                "driver": driver_name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "duration_ms": round(elapsed_ms, 2),
                "context": ctx.to_dict(),
            }],
            extra_metadata=extra_meta,
        )

        doc = Document(
            doc_id=doc_uuid,
            content=raw_content,
            metadata=metadata,
        )

        # 6. Collect Telemetry and Publish Event
        metrics = LoadMetrics(
            file_path=str(path),
            loader_name=loader_name,
            duration_ms=elapsed_ms,
            primary_driver_used=primary_used,
            fallback_driver_used=fallback_used,
            driver_name=driver_name,
            success=True,
            warnings=warnings,
            errors=errors,
        )
        global_metrics.record(metrics)

        global_event_bus.publish(
            DocumentLoaded(
                file_path=str(path),
                loader_name=loader_name,
                doc_id=doc_uuid,
                driver_used=driver_name,
            )
        )

        logger.info(f"Loaded {path.name} in {elapsed_ms:.2f}ms [{driver_name}] (chars: {char_cnt})")
        return doc
