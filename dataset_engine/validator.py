"""
Dataset Validator — Document Integrity, Format & Size Verification.

Sovereign On-Premise Agentic AI Workbench (SIH26117 / MRPL)
Performs deterministic multi-stage checks on files: existence, readability,
non-emptiness, allowed extension, size limits, and format-specific magic byte probes.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional
import xml.etree.ElementTree as ET
import zipfile

from rag_engine.schemas.manifest import ValidationStatus


@dataclass(frozen=True)
class ValidationResult:
    """Represents the outcome of a document validation check."""

    file_path: Path
    status: ValidationStatus
    is_valid: bool
    size_bytes: int
    error_message: Optional[str] = None


class DatasetValidator:
    """Validates files against refinery operational and RAG processing standards."""

    def __init__(
        self,
        allowed_extensions: Optional[set[str]] = None,
        max_file_size_mb: int = 500,
        min_file_size_bytes: int = 1,
        enable_deep_integrity_check: bool = True,
    ) -> None:
        """
        Initialize DatasetValidator.

        Args:
            allowed_extensions: Set of permitted extensions.
            max_file_size_mb: Maximum allowable file size in megabytes.
            min_file_size_bytes: Minimum allowable file size in bytes.
            enable_deep_integrity_check: Whether to perform format-specific header probes.
        """
        self.allowed_extensions = (
            {ext.lower() for ext in allowed_extensions}
            if allowed_extensions
            else None
        )
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self.min_file_size_bytes = min_file_size_bytes
        self.enable_deep_integrity_check = enable_deep_integrity_check

    def validate(self, file_path: Path | str) -> ValidationResult:
        """
        Run the complete validation suite on a file.

        Args:
            file_path: Path to target file.

        Returns:
            ValidationResult with detailed status and error messages.
        """
        path = Path(file_path).resolve()

        # 1. Existence check
        if not path.exists() or not path.is_file():
            return ValidationResult(
                file_path=path,
                status=ValidationStatus.CORRUPTED,
                is_valid=False,
                size_bytes=0,
                error_message="File does not exist or is not a regular file",
            )

        # 2. Extension check
        ext = path.suffix.lower()
        if self.allowed_extensions is not None and ext not in self.allowed_extensions:
            return ValidationResult(
                file_path=path,
                status=ValidationStatus.UNSUPPORTED,
                is_valid=False,
                size_bytes=0,
                error_message=f"Extension '{ext}' is not in allowed extensions list",
            )

        # 3. Readability & size checks
        try:
            size = path.stat().st_size
        except (OSError, PermissionError) as e:
            return ValidationResult(
                file_path=path,
                status=ValidationStatus.CORRUPTED,
                is_valid=False,
                size_bytes=0,
                error_message=f"Unable to access file metadata: {e}",
            )

        # Non-empty check
        if size < self.min_file_size_bytes:
            return ValidationResult(
                file_path=path,
                status=ValidationStatus.EMPTY,
                is_valid=False,
                size_bytes=size,
                error_message="File is empty (0 bytes)",
            )

        # Maximum size check
        if size > self.max_file_size_bytes:
            return ValidationResult(
                file_path=path,
                status=ValidationStatus.CORRUPTED,
                is_valid=False,
                size_bytes=size,
                error_message=(
                    f"File size ({size / (1024 * 1024):.2f} MB) exceeds "
                    f"limit ({self.max_file_size_bytes / (1024 * 1024):.2f} MB)"
                ),
            )

        # Test readability
        try:
            with open(path, "rb") as fh:
                sample = fh.read(min(size, 4096))
        except (PermissionError, OSError) as e:
            return ValidationResult(
                file_path=path,
                status=ValidationStatus.CORRUPTED,
                is_valid=False,
                size_bytes=size,
                error_message=f"Read permission denied or I/O failure: {e}",
            )

        # 4. Deep format-specific integrity probe
        if self.enable_deep_integrity_check:
            is_intact, integrity_error = self._probe_format_integrity(path, ext, sample)
            if not is_intact:
                return ValidationResult(
                    file_path=path,
                    status=ValidationStatus.CORRUPTED,
                    is_valid=False,
                    size_bytes=size,
                    error_message=integrity_error,
                )

        return ValidationResult(
            file_path=path,
            status=ValidationStatus.VALID,
            is_valid=True,
            size_bytes=size,
            error_message=None,
        )

    def _probe_format_integrity(
        self, path: Path, ext: str, sample_bytes: bytes
    ) -> tuple[bool, Optional[str]]:
        """
        Probe format-specific headers, magic bytes, or container structures.
        """
        try:
            # PDF validation
            if ext == ".pdf":
                if not sample_bytes.startswith(b"%PDF-"):
                    return False, "Invalid PDF header magic bytes (missing %PDF-)"
                # Quick probe for EOF marker in last 2048 bytes if file is large enough
                file_size = path.stat().st_size
                with open(path, "rb") as fh:
                    seek_offset = max(0, file_size - 2048)
                    fh.seek(seek_offset)
                    tail = fh.read()
                    if b"%%EOF" not in tail and b"%EOF" not in tail:
                        # Some linearized or updated PDFs lack trailing EOF, allow warning
                        pass
                return True, None

            # OpenXML / ZIP formats (DOCX, PPTX, XLSX)
            if ext in {".docx", ".pptx", ".xlsx"}:
                if not sample_bytes.startswith(b"PK\x03\x04") and not sample_bytes.startswith(b"PK\x05\x06"):
                    return False, f"Invalid {ext.upper()} container: Missing standard ZIP header magic bytes"
                try:
                    with zipfile.ZipFile(path, "r") as zf:
                        # Test CRC of first few items
                        bad_file = zf.testzip()
                        if bad_file:
                            return False, f"Corrupted archive member in {ext.upper()}: {bad_file}"
                except zipfile.BadZipFile as e:
                    return False, f"Corrupted {ext.upper()} archive: {e}"
                return True, None

            # JSON validation
            if ext == ".json":
                try:
                    with open(path, "r", encoding="utf-8", errors="replace") as fh:
                        json.load(fh)
                except Exception as e:
                    return False, f"Invalid JSON structure: {e}"
                return True, None

            # XML validation
            if ext == ".xml":
                try:
                    ET.parse(str(path))
                except Exception as e:
                    return False, f"Invalid XML structure: {e}"
                return True, None

            # PNG image validation
            if ext == ".png":
                if not sample_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
                    return False, "Invalid PNG magic signature"
                return True, None

            # JPEG image validation
            if ext in {".jpg", ".jpeg"}:
                if not sample_bytes.startswith(b"\xff\xd8\xff"):
                    return False, "Invalid JPEG magic signature"
                return True, None

            # Plain text / Markdown / CSV
            if ext in {".txt", ".md", ".csv"}:
                # Ensure it can be decoded into text
                try:
                    sample_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    try:
                        sample_bytes.decode("latin-1")
                    except UnicodeDecodeError as e:
                        return False, f"File content is not decodable text: {e}"
                return True, None

            # Other supported formats
            return True, None

        except Exception as e:
            return False, f"Integrity probe failed: {e}"
