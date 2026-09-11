"""
Dataset Orchestrator — End-to-End Pipeline Execution.

Sovereign On-Premise Agentic AI Workbench (SIH26117 / MRPL)
Coordinates scanner, validator, hash generator, duplicate detector,
manifest generator, and statistics generator into a deterministic offline workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import time
from typing import Any, Optional
import yaml

from dataset_engine.duplicate_detector import DuplicateDetector, DuplicateReport
from dataset_engine.hash_generator import HashGenerator
from dataset_engine.manifest_generator import ManifestGenerator
from dataset_engine.scanner import DatasetScanner, DiscoveredFile
from dataset_engine.statistics import StatisticsGenerator
from dataset_engine.validator import DatasetValidator, ValidationResult
from rag_engine.config.logging_config import get_logger
from rag_engine.config.paths import Paths
from rag_engine.schemas.manifest import ValidationStatus

logger = get_logger("dataset_engine.orchestrator")


@dataclass
class PipelineResult:
    """Result of running the dataset validation orchestrator."""

    total_files_scanned: int
    valid_files_count: int
    corrupted_files_count: int
    empty_files_count: int
    unsupported_files_count: int
    duplicate_instances_count: int
    health_score: float
    manifest_path: Path
    statistics_path: Path
    health_report_path: Path
    duration_seconds: float


class DatasetOrchestrator:
    """Coordinates the end-to-end dataset validation and indexing pipeline."""

    def __init__(self, config_path: Optional[Path | str] = None) -> None:
        """
        Initialize orchestrator with configuration rules.

        Args:
            config_path: Path to validation_rules.yaml. If None, auto-locates.
        """
        self.config_path = self._locate_config(config_path)
        self.config = self._load_config(self.config_path)

        # Extract config values
        dataset_cfg = self.config.get("dataset", {})
        val_cfg = self.config.get("validation", {})
        cat_cfg = self.config.get("categories", {})
        rep_cfg = self.config.get("reporting", {})

        supported_exts = set(dataset_cfg.get("supported_extensions", []))
        ignored_exts = set(dataset_cfg.get("ignored_extensions", []))
        ignored_dirs = set(dataset_cfg.get("ignored_directories", []))

        max_size_mb = val_cfg.get("max_file_size_mb", 500)
        min_size_bytes = val_cfg.get("min_file_size_bytes", 1)
        deep_integrity = val_cfg.get("enable_deep_integrity_check", True)
        chunk_size = val_cfg.get("hash_chunk_size_bytes", 65536)

        # Build category-to-doctype lookup
        category_to_doctype = {
            cat: data.get("doc_type", "general_document")
            for cat, data in cat_cfg.items()
        }

        # Initialize sub-modules
        self.scanner = DatasetScanner(
            supported_extensions=supported_exts,
            ignored_extensions=ignored_exts,
            ignored_directories=ignored_dirs,
        )
        self.validator = DatasetValidator(
            allowed_extensions=supported_exts,
            max_file_size_mb=max_size_mb,
            min_file_size_bytes=min_size_bytes,
            enable_deep_integrity_check=deep_integrity,
        )
        self.hash_generator = HashGenerator(chunk_size_bytes=chunk_size)
        self.duplicate_detector = DuplicateDetector()
        self.manifest_generator = ManifestGenerator(
            category_to_doctype_map=category_to_doctype
        )
        self.statistics_generator = StatisticsGenerator()

        self.output_dir = Path(rep_cfg.get("output_dir", "outputs"))
        self.manifest_filename = rep_cfg.get("manifest_filename", "manifest.json")
        self.statistics_filename = rep_cfg.get("statistics_filename", "dataset_statistics.json")
        self.health_report_filename = rep_cfg.get("health_report_filename", "dataset_health_report.md")

    def _locate_config(self, config_path: Optional[Path | str]) -> Path:
        """Find the validation rules YAML configuration file."""
        if config_path:
            p = Path(config_path).resolve()
            if p.exists():
                return p

        # Check candidate locations
        candidates = [
            Paths.PROJECT_ROOT / "config" / "validation_rules.yaml",
            Paths.PROJECT_ROOT / "rag_engine" / "config" / "validation_rules.yaml",
            Path.cwd() / "config" / "validation_rules.yaml",
        ]
        for cand in candidates:
            if cand.exists():
                return cand.resolve()

        # Fallback to local default if no file found
        return Paths.PROJECT_ROOT / "config" / "validation_rules.yaml"

    def _load_config(self, path: Path) -> dict[str, Any]:
        """Read and parse the YAML configuration file."""
        if not path.exists():
            logger.warning(f"Config file not found at {path}, using built-in defaults.")
            return {}
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return yaml.safe_load(fh) or {}
        except Exception as e:
            logger.error(f"Failed to parse config at {path}: {e}. Using defaults.")
            return {}

    def run(
        self,
        dataset_dir: Optional[Path | str] = None,
        output_dir: Optional[Path | str] = None,
    ) -> PipelineResult:
        """
        Execute the full validation pipeline.

        Args:
            dataset_dir: Path to dataset root (defaults to config / Paths.DATASETS).
            output_dir: Path to destination outputs (defaults to config / Paths.OUTPUTS).

        Returns:
            PipelineResult with paths and summary metrics.
        """
        start_time = time.time()
        logger.info("=" * 60)
        logger.info("Starting Dataset Validation Engine (Milestone 2)")
        logger.info("=" * 60)

        # Resolve paths
        target_dataset_dir = (
            Path(dataset_dir).resolve()
            if dataset_dir
            else Paths.DATASETS
        )
        target_output_dir = (
            Path(output_dir).resolve()
            if output_dir
            else (Paths.PROJECT_ROOT / self.output_dir)
        )
        target_output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Target dataset directory: {target_dataset_dir}")
        logger.info(f"Target output directory: {target_output_dir}")

        # 1. SCAN
        logger.info("Step 1/6: Recursively scanning dataset directory...")
        discovered_files = self.scanner.scan(target_dataset_dir)
        logger.info(f"Discovered {len(discovered_files)} total files.")

        # 2. VALIDATE & HASH
        logger.info("Step 2/6: Validating files and generating SHA-256 hashes...")
        self.duplicate_detector.reset()

        validated_records: list[dict[str, Any]] = []

        for df in discovered_files:
            if not df.is_supported:
                # Unsupported file
                rec = self.manifest_generator.build_document_entry(
                    file_path=df.file_path,
                    relative_path=df.relative_path,
                    category=df.category,
                    subcategory=df.subcategory,
                    extension=df.extension,
                    size_bytes=df.size_bytes,
                    sha256_hash="",
                    status=ValidationStatus.UNSUPPORTED,
                    last_modified=df.last_modified,
                    error_message=f"Unsupported file format '{df.extension}'",
                )
                validated_records.append(rec)
                continue

            # Run validator
            val_result = self.validator.validate(df.file_path)

            if not val_result.is_valid:
                # Corrupted or empty
                rec = self.manifest_generator.build_document_entry(
                    file_path=df.file_path,
                    relative_path=df.relative_path,
                    category=df.category,
                    subcategory=df.subcategory,
                    extension=df.extension,
                    size_bytes=val_result.size_bytes,
                    sha256_hash="",
                    status=val_result.status,
                    last_modified=df.last_modified,
                    error_message=val_result.error_message,
                )
                validated_records.append(rec)
                continue

            # Compute hash for valid file
            try:
                file_hash = self.hash_generator.generate_file_hash(df.file_path)
            except Exception as e:
                logger.error(f"Failed to hash {df.file_path}: {e}")
                rec = self.manifest_generator.build_document_entry(
                    file_path=df.file_path,
                    relative_path=df.relative_path,
                    category=df.category,
                    subcategory=df.subcategory,
                    extension=df.extension,
                    size_bytes=val_result.size_bytes,
                    sha256_hash="",
                    status=ValidationStatus.CORRUPTED,
                    last_modified=df.last_modified,
                    error_message=f"Hashing failed: {e}",
                )
                validated_records.append(rec)
                continue

            # Register in duplicate detector
            self.duplicate_detector.register(df.file_path, file_hash)

            rec = self.manifest_generator.build_document_entry(
                file_path=df.file_path,
                relative_path=df.relative_path,
                category=df.category,
                subcategory=df.subcategory,
                extension=df.extension,
                size_bytes=val_result.size_bytes,
                sha256_hash=file_hash,
                status=ValidationStatus.VALID,
                last_modified=df.last_modified,
            )
            validated_records.append(rec)

        # 3. DUPLICATE DETECTION
        logger.info("Step 3/6: Analyzing duplicates...")
        duplicate_report = self.duplicate_detector.analyze()
        logger.info(
            f"Found {len(duplicate_report.hash_duplicates)} exact content duplicate groups "
            f"({duplicate_report.total_duplicate_instances} redundant files)."
        )

        # Update duplicate flags in records
        for rec in validated_records:
            h = rec.get("hash")
            if h and self.duplicate_detector.is_duplicate_hash(h):
                canonical = self.duplicate_detector.get_canonical_for_hash(h)
                if canonical and canonical != rec["absolute_path"]:
                    rec["is_duplicate"] = True
                    rec["duplicate_of"] = canonical

        # 4. MANIFEST GENERATION
        logger.info("Step 4/6: Generating manifest.json...")
        manifest_out = target_output_dir / self.manifest_filename
        self.manifest_generator.generate_manifest(
            documents=validated_records,
            output_path=manifest_out,
            dataset_root=target_dataset_dir.as_posix(),
        )
        logger.info(f"Manifest written to: {manifest_out}")

        # 5. STATISTICS & REPORT GENERATION
        logger.info("Step 5/6: Generating dataset_statistics.json and health report...")
        elapsed_sec = time.time() - start_time
        stats = self.statistics_generator.compute_statistics(
            documents=validated_records,
            duplicate_report=duplicate_report,
            scan_duration_seconds=elapsed_sec,
        )

        stats_out = target_output_dir / self.statistics_filename
        self.statistics_generator.generate_statistics_json(stats, stats_out)
        logger.info(f"Statistics written to: {stats_out}")

        report_out = target_output_dir / self.health_report_filename
        self.statistics_generator.generate_health_report_md(stats, report_out)
        logger.info(f"Health report written to: {report_out}")

        # Also mirror outputs directly into datasets/ for immediate visibility if needed
        try:
            mirror_manifest = target_dataset_dir / self.manifest_filename
            mirror_stats = target_dataset_dir / self.statistics_filename
            self.manifest_generator.generate_manifest(validated_records, mirror_manifest, target_dataset_dir.as_posix())
            self.statistics_generator.generate_statistics_json(stats, mirror_stats)
        except Exception:
            pass

        logger.info("Step 6/6: Pipeline execution finished successfully!")
        logger.info(f"Total Duration: {elapsed_sec:.2f}s | Health Score: {stats['health_score']}%")

        return PipelineResult(
            total_files_scanned=stats["total_files"],
            valid_files_count=stats["valid_files"],
            corrupted_files_count=stats["corrupted_files"],
            empty_files_count=stats["empty_files"],
            unsupported_files_count=stats["unsupported_files"],
            duplicate_instances_count=duplicate_report.total_duplicate_instances,
            health_score=stats["health_score"],
            manifest_path=manifest_out,
            statistics_path=stats_out,
            health_report_path=report_out,
            duration_seconds=elapsed_sec,
        )


if __name__ == "__main__":
    orchestrator = DatasetOrchestrator()
    result = orchestrator.run()
    print(f"\nPipeline finished: Scanned {result.total_files_scanned} files in {result.duration_seconds:.2f}s")
    print(f"Health Score: {result.health_score}% | Manifest: {result.manifest_path}")
