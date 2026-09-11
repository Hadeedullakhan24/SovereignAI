"""
RAG Engine — Centralized Path Registry.

All filesystem paths are resolved through this module.
No other module should hardcode paths.

Usage:
    from rag_engine.config.paths import Paths
    pdf_dir = Paths.DATASETS / "manuals"
"""

from __future__ import annotations

from pathlib import Path


class Paths:
    """Centralized filesystem path constants for the RAG Engine."""

    # Project root: d:\SovereignAI
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent

    # Existing dataset directories (READ-ONLY — never write to these)
    DATASETS: Path = PROJECT_ROOT / "datasets"
    DATASETS_VISION: Path = DATASETS / "Vision"
    DATASETS_CODING: Path = DATASETS / "coding"
    DATASETS_EMAILS: Path = DATASETS / "emails"
    DATASETS_ENGINEERING_DRAWINGS: Path = DATASETS / "engineering_drawings"
    DATASETS_HANDWRITTEN_NOTES: Path = DATASETS / "handwritten_notes"
    DATASETS_INSPECTION_REPORTS: Path = DATASETS / "inspection_reports"
    DATASETS_MAINTENANCE: Path = DATASETS / "maintenance"
    DATASETS_MANUALS: Path = DATASETS / "manuals"
    DATASETS_OCR: Path = DATASETS / "ocr"
    DATASETS_PIDQA: Path = DATASETS / "pidqa"
    DATASETS_SAFETY_DOCS: Path = DATASETS / "safety_docs"
    DATASETS_TEMPLATES: Path = DATASETS / "templates"

    # RAG Engine package root
    RAG_ENGINE_ROOT: Path = PROJECT_ROOT / "rag_engine"

    # Model weights (populated by scripts/download_models)
    MODELS: Path = PROJECT_ROOT / "models"
    MODELS_EMBEDDINGS: Path = MODELS / "embeddings"
    MODELS_LLMS: Path = MODELS / "llms"
    MODELS_RERANKERS: Path = MODELS / "rerankers"
    MODELS_ADAPTERS: Path = MODELS / "adapters"

    # Vector database persistence
    VECTOR_DB: Path = PROJECT_ROOT / "vector_db"
    VECTOR_DB_CHROMA: Path = VECTOR_DB / "chroma"
    VECTOR_DB_FAISS: Path = VECTOR_DB / "faiss"

    # Generated outputs
    OUTPUTS: Path = PROJECT_ROOT / "outputs"
    OUTPUTS_DOCX: Path = OUTPUTS / "docx"
    OUTPUTS_PDF: Path = OUTPUTS / "pdf"
    OUTPUTS_PPTX: Path = OUTPUTS / "pptx"
    OUTPUTS_XLSX: Path = OUTPUTS / "xlsx"
    OUTPUTS_PY: Path = OUTPUTS / "py"
    OUTPUTS_TMP: Path = OUTPUTS / "tmp"

    # Logs
    LOGS: Path = PROJECT_ROOT / "logs"
    LOGS_APP: Path = LOGS / "app"
    LOGS_RAG: Path = LOGS / "rag"
    LOGS_AUDIT: Path = LOGS / "audit"
    LOGS_ERRORS: Path = LOGS / "errors"

    @classmethod
    def ensure_directories(cls) -> None:
        """Create all required output directories if they do not exist."""
        directories = [
            cls.MODELS_EMBEDDINGS,
            cls.MODELS_LLMS,
            cls.MODELS_RERANKERS,
            cls.MODELS_ADAPTERS,
            cls.VECTOR_DB_CHROMA,
            cls.VECTOR_DB_FAISS,
            cls.OUTPUTS_DOCX,
            cls.OUTPUTS_PDF,
            cls.OUTPUTS_PPTX,
            cls.OUTPUTS_XLSX,
            cls.OUTPUTS_PY,
            cls.OUTPUTS_TMP,
            cls.LOGS_APP,
            cls.LOGS_RAG,
            cls.LOGS_AUDIT,
            cls.LOGS_ERRORS,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
