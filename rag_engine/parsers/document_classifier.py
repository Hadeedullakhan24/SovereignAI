"""Deterministic document classification engine.

Classifies incoming documents into operational categories prior to parser selection:
- Manual
- Inspection Report
- Safety Document
- Maintenance Record
- Email
- Engineering Drawing
- Template
- Unknown

NO AI / NO LLM / NO EMBEDDINGS.
Uses strictly deterministic rule-based analysis:
1. Directory / folder location tokens
2. File naming conventions and extensions
3. Metadata properties and tags
4. Structural keywords and header tokens in document text
"""

from __future__ import annotations

import re
from enum import StrEnum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from rag_engine.schemas.document import Document


class DocumentCategory(StrEnum):
    """Classified document categories."""

    MANUAL = "Manual"
    INSPECTION_REPORT = "Inspection Report"
    SAFETY_DOCUMENT = "Safety Document"
    MAINTENANCE_RECORD = "Maintenance Record"
    EMAIL = "Email"
    ENGINEERING_DRAWING = "Engineering Drawing"
    TEMPLATE = "Template"
    UNKNOWN = "Unknown"


class ClassificationResult(BaseModel):
    """Result of deterministic document classification."""

    model_config = ConfigDict(frozen=True)

    category: DocumentCategory = Field(description="Detected document category")
    confidence: float = Field(ge=0.0, le=1.0, description="Rule-based classification confidence")
    evidence: list[str] = Field(default_factory=list, description="Heuristic rules and tokens matched")


class DocumentClassifier:
    """Deterministic, air-gapped document classifier."""

    # Folder keywords mapped to categories
    FOLDER_KEYWORDS = {
        DocumentCategory.MANUAL: ["manual", "manuals", "o&m", "operations", "handbook", "guide"],
        DocumentCategory.INSPECTION_REPORT: ["inspection", "inspections", "inspection_reports", "ndt", "audit", "survey"],
        DocumentCategory.SAFETY_DOCUMENT: ["safety", "safety_docs", "oisd", "hse", "hazard", "msds", "sop", "ppe"],
        DocumentCategory.MAINTENANCE_RECORD: ["maintenance", "pm_records", "work_orders", "breakdown", "overhaul", "logs"],
        DocumentCategory.EMAIL: ["email", "emails", "correspondence", "memos", "communications"],
        DocumentCategory.ENGINEERING_DRAWING: ["drawing", "drawings", "engineering_drawings", "pid", "p&id", "pfd", "schematics", "blueprints"],
        DocumentCategory.TEMPLATE: ["template", "templates", "forms", "blank_forms", "checklists"],
    }

    # Filename tokens
    FILENAME_PATTERNS = {
        DocumentCategory.EMAIL: [r"\.eml$", r"\.msg$", r"^email_", r"^memo_"],
        DocumentCategory.ENGINEERING_DRAWING: [r"\.dwg$", r"\.dxf$", r"\.cad$", r"_pid_", r"_pfd_", r"drawing_", r"dwg_"],
        DocumentCategory.INSPECTION_REPORT: [r"inspection", r"ndt_report", r"thickness", r"hydrotest", r"survey_report"],
        DocumentCategory.MAINTENANCE_RECORD: [r"maintenance", r"pm_log", r"work_order", r"overhaul_report"],
        DocumentCategory.SAFETY_DOCUMENT: [r"oisd", r"safety", r"msds", r"sop_", r"hse_", r"hazard"],
        DocumentCategory.MANUAL: [r"manual", r"user_guide", r"handbook", r"specs"],
        DocumentCategory.TEMPLATE: [r"template", r"checklist", r"blank_form", r"standard_form"],
    }

    # Content keywords in top portion of document
    CONTENT_SIGNATURES = {
        DocumentCategory.EMAIL: [
            r"^from:\s+", r"^to:\s+", r"^subject:\s+", r"^sent:\s+", r"^date:\s+.*(?:am|pm|\d{4})"
        ],
        DocumentCategory.ENGINEERING_DRAWING: [
            r"title\s*block", r"drawing\s*no", r"dwg\s*no", r"p&id", r"piping\s*and\s*instrumentation",
            r"rev\s*no\.", r"designed\s*by", r"scale:\s*\d+:\d+"
        ],
        DocumentCategory.INSPECTION_REPORT: [
            r"inspection\s*report", r"ndt\s*report", r"visual\s*inspection", r"wall\s*thickness",
            r"ultrasonic\s*testing", r"radiographic\s*testing", r"dye\s*penetrant", r"hydro\s*test"
        ],
        DocumentCategory.SAFETY_DOCUMENT: [
            r"safety\s*standard", r"standard\s*operating\s*procedure", r"job\s*safety\s*analysis",
            r"material\s*safety\s*data\s*sheet", r"personal\s*protective\s*equipment", r"oisd-std-",
            r"permit\s*to\s*work", r"hazard\s*identification"
        ],
        DocumentCategory.MAINTENANCE_RECORD: [
            r"preventive\s*maintenance", r"work\s*order", r"equipment\s*log", r"overhaul\s*record",
            r"calibration\s*certificate", r"greasing\s*schedule", r"vibration\s*analysis"
        ],
        DocumentCategory.MANUAL: [
            r"operating\s*manual", r"instruction\s*manual", r"technical\s*manual",
            r"user\s*guide", r"installation\s*manual", r"maintenance\s*and\s*operating\s*instructions"
        ],
        DocumentCategory.TEMPLATE: [
            r"template\s*document", r"form\s*number:", r"checklist\s*template", r"sample\s*template"
        ],
    }

    def classify(self, doc: Document) -> ClassificationResult:
        """Classify document using deterministic multi-tier analysis."""
        scores: dict[DocumentCategory, float] = {cat: 0.0 for cat in DocumentCategory}
        evidence: dict[DocumentCategory, list[str]] = {cat: [] for cat in DocumentCategory}

        # 1. Inspect existing metadata category if present
        if doc.metadata.category:
            meta_cat = doc.metadata.category.lower().replace(" ", "_").replace("-", "_")
            for cat, folder_keys in self.FOLDER_KEYWORDS.items():
                if any(fk in meta_cat for fk in folder_keys):
                    scores[cat] += 40.0
                    evidence[cat].append(f"metadata.category='{doc.metadata.category}'")

        # 2. Inspect file path and folder structure
        source_path = doc.metadata.source_path.lower().replace("\\", "/")
        path_parts = source_path.split("/")
        for cat, folder_keys in self.FOLDER_KEYWORDS.items():
            for part in path_parts[:-1]:
                if any(fk in part for fk in folder_keys):
                    scores[cat] += 30.0
                    evidence[cat].append(f"folder='{part}'")
                    break

        # 3. Inspect filename
        filename = doc.metadata.file_name.lower()
        for cat, patterns in self.FILENAME_PATTERNS.items():
            for pat in patterns:
                if re.search(pat, filename):
                    scores[cat] += 25.0
                    evidence[cat].append(f"filename_pattern='{pat}'")
                    break

        # 4. Inspect content text (first 3000 chars for efficiency)
        sample_text = (doc.content[:3000] if doc.content else "").lower()
        if sample_text:
            for cat, signatures in self.CONTENT_SIGNATURES.items():
                match_count = 0
                for sig in signatures:
                    if re.search(sig, sample_text, re.MULTILINE):
                        match_count += 1
                        evidence[cat].append(f"content_signature='{sig}'")
                if match_count > 0:
                    # Scale based on matches
                    scores[cat] += min(match_count * 15.0, 45.0)

        # 5. Format-specific hints
        ext = doc.metadata.file_format.lower()
        if ext in [".dwg", ".dxf"]:
            scores[DocumentCategory.ENGINEERING_DRAWING] += 50.0
            evidence[DocumentCategory.ENGINEERING_DRAWING].append(f"extension='{ext}'")
        elif ext in [".eml", ".msg"]:
            scores[DocumentCategory.EMAIL] += 50.0
            evidence[DocumentCategory.EMAIL].append(f"extension='{ext}'")

        # Select highest scoring category
        best_cat = DocumentCategory.UNKNOWN
        best_score = 0.0

        for cat, score in scores.items():
            if cat == DocumentCategory.UNKNOWN:
                continue
            if score > best_score:
                best_score = score
                best_cat = cat

        if best_score >= 20.0:
            # Normalize confidence score between 0.50 and 0.99
            norm_confidence = min(0.50 + (best_score / 150.0) * 0.49, 0.99)
            return ClassificationResult(
                category=best_cat,
                confidence=round(norm_confidence, 2),
                evidence=evidence[best_cat],
            )

        return ClassificationResult(
            category=DocumentCategory.UNKNOWN,
            confidence=0.10,
            evidence=["Insufficient deterministic matches found"],
        )
