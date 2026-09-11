"""
Dataset Engine — Offline Dataset Validation & Indexing System.

Sovereign On-Premise Agentic AI Workbench
Problem Statement: SIH26117 | Organization: MRPL
Member 1: Knowledge Base / RAG Engine

This package is responsible for scanning, validating, hashing, deduplicating,
and indexing all refinery documents before ingestion into the downstream RAG pipeline.
"""

from __future__ import annotations

from dataset_engine.scanner import DatasetScanner, DiscoveredFile
from dataset_engine.validator import DatasetValidator, ValidationResult
from dataset_engine.hash_generator import HashGenerator
from dataset_engine.duplicate_detector import DuplicateDetector, DuplicateReport
from dataset_engine.manifest_generator import ManifestGenerator
from dataset_engine.statistics import StatisticsGenerator
from dataset_engine.orchestrator import DatasetOrchestrator, PipelineResult

__all__ = [
    "DatasetScanner",
    "DiscoveredFile",
    "DatasetValidator",
    "ValidationResult",
    "HashGenerator",
    "DuplicateDetector",
    "DuplicateReport",
    "ManifestGenerator",
    "StatisticsGenerator",
    "DatasetOrchestrator",
    "PipelineResult",
]
