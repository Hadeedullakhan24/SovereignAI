# Sovereign On-Premise Agentic AI Workbench — RAG Engine (Milestone 1 Scaffolding Audit)

> **Document Purpose:** This document provides a complete, transparent, and auditable summary of all files, directories, interfaces, schemas, and configurations created for the `rag_engine` package (Member 1: Knowledge Base / RAG Engine) under Problem Statement **SIH26117** (Smart India Hackathon 2026) for **MRPL** (Mangalore Refinery and Petrochemicals Limited).
>
> You can feed this entire document to ChatGPT or any senior architect to audit the correctness of the package scaffolding, design choices, schema integrity, and architectural readiness before proceeding to Milestone 2 (Business Logic Implementation).

---

## 1. Project Context & Constraints

| Attribute | Specification |
|:---|:---|
| **Project Name** | Sovereign On-Premise Agentic AI Workbench |
| **Problem Statement** | SIH26117 (Smart India Hackathon 2026) |
| **Organization / Domain** | MRPL (Petroleum Refinery — heavy engineering, safety manuals, P&IDs, equipment specs, lab assays) |
| **Deployment Paradigm** | 100% Air-Gapped / Offline-First (No external SaaS, OpenAI API, or WAN egress) |
| **Current Task / Scope** | **Milestone 1: Project Scaffolding Only**. Establish directory structure, packaging, dependencies, centralized settings, immutable Pydantic v2 schemas, and abstract base classes (ABCs). No business logic written yet. |
| **Architecture Pattern** | Clean Architecture / Ports & Adapters (Inversion of Control) |

---

## 2. Directory Tree Structure

Below is the verified, exact directory tree created under `d:\SovereignAI\rag_engine\`:

```text
d:\SovereignAI\rag_engine\
├── .gitignore
├── pyproject.toml
├── README.md
├── requirements.txt
├── __init__.py
│
├── config/
│   ├── __init__.py
│   ├── constants.py
│   ├── logging_config.py
│   ├── paths.py
│   └── settings.py
│
├── interfaces/
│   ├── __init__.py
│   ├── base_chunker.py
│   ├── base_embedder.py
│   ├── base_loader.py
│   ├── base_retriever.py
│   └── base_vector_store.py
│
├── schemas/
│   ├── __init__.py
│   ├── chunk.py
│   ├── citation.py
│   ├── document.py
│   ├── embedding.py
│   └── retrieved_document.py
│
├── utils/
│   ├── __init__.py
│   ├── file_utils.py
│   ├── hash_utils.py
│   ├── logger.py
│   └── validators.py
│
├── preprocessing/
│   └── __init__.py
├── loaders/
│   └── __init__.py
├── parsers/
│   └── __init__.py
├── chunking/
│   └── __init__.py
├── metadata/
│   └── __init__.py
├── embeddings/
│   └── __init__.py
├── vector_store/
│   └── __init__.py
├── retrieval/
│   └── __init__.py
├── reranking/
│   └── __init__.py
├── citations/
│   └── __init__.py
├── evaluation/
│   └── __init__.py
├── pipeline/
│   └── __init__.py
└── cache/
    └── __init__.py
```

---

## 3. Module & Folder Breakdown

| Directory | Architectural Responsibility | Milestone 1 Status |
|:---|:---|:---|
| `rag_engine/` | Root package directory containing build and dependency specifications. | Fully initialized (`pyproject.toml`, `requirements.txt`, `.gitignore`, `README.md`). |
| `rag_engine/config/` | Application-wide settings, path registry, constants, and logging configuration. | Complete: Singleton `settings` via `pydantic-settings`, immutable `constants`, deterministic `Paths` registry, rotating logger. |
| `rag_engine/interfaces/` | Abstract Base Classes (ABCs) specifying strict contractual interfaces for all pipeline stages. | Complete: Contracts for `BaseLoader`, `BaseChunker`, `BaseEmbedder`, `BaseVectorStore`, and `BaseRetriever`. |
| `rag_engine/schemas/` | Immutable Data Transfer Objects (DTOs) built using Pydantic v2 (`ConfigDict(frozen=True)`). | Complete: `Document`, `Chunk`, `EmbeddingVector`, `Citation`, `RetrievedDocument`, `ScoredChunk`. |
| `rag_engine/utils/` | Shared utility helpers (logging proxy, file operations, hashing, input validators). | Complete: Logger utility configured; placeholders defined for file, hash, and validator functions. |
| `rag_engine/preprocessing/` | Text cleaning, unicode normalization, boilerplate stripping, and deduplication. | Scaffolded with `__init__.py` ready for M2 implementations. |
| `rag_engine/loaders/` | Concrete format-specific document loaders (PDF, DOCX, CSV, TXT, MD) implementing `BaseLoader`. | Scaffolded with `__init__.py`. |
| `rag_engine/parsers/` | Deep parsers for tabular data, section structures, and layout analysis. | Scaffolded with `__init__.py`. |
| `rag_engine/chunking/` | Chunking strategies (Fixed-size, Recursive Character, Semantic) implementing `BaseChunker`. | Scaffolded with `__init__.py`. |
| `rag_engine/metadata/` | Metadata extractors (refinery equipment tags like `Pump_P203`, unit IDs, P&ID numbers). | Scaffolded with `__init__.py`. |
| `rag_engine/embeddings/` | Offline local embedding wrappers (Sentence-Transformers, BAAI/bge, etc.) implementing `BaseEmbedder`. | Scaffolded with `__init__.py`. |
| `rag_engine/vector_store/` | On-premise vector index adapters (ChromaDB, FAISS) implementing `BaseVectorStore`. | Scaffolded with `__init__.py`. |
| `rag_engine/retrieval/` | Retrieval mechanisms (Dense vector, BM25 keyword, Hybrid fusion) implementing `BaseRetriever`. | Scaffolded with `__init__.py`. |
| `rag_engine/reranking/` | Local cross-encoder rerankers (e.g. `bge-reranker-large`) for high precision. | Scaffolded with `__init__.py`. |
| `rag_engine/citations/` | Page-accurate, document-accurate citation linkers and quotation verifiers. | Scaffolded with `__init__.py`. |
| `rag_engine/evaluation/` | Quantitative retrieval quality benchmarks (Recall@K, MRR, NDCG, hallucination checks). | Scaffolded with `__init__.py`. |
| `rag_engine/pipeline/` | Ingestion pipeline orchestration and query retrieval orchestration workflows. | Scaffolded with `__init__.py`. |
| `rag_engine/cache/` | In-memory and disk caching layer for repeated embeddings and query results. | Scaffolded with `__init__.py`. |

---

## 4. Full Source Code of All Implemented Files

### 4.1 Packaging, Build & Environment

#### `rag_engine/requirements.txt`
```text
# Core RAG dependencies
pydantic>=2.5.0
pydantic-settings>=2.1.0
pyyaml>=6.0.1

# Document loaders
pypdf>=4.0.0
python-docx>=1.1.0
openpyxl>=3.1.0

# Text processing
nltk>=3.8.1

# Embeddings & Vector DB
sentence-transformers>=2.2.2
chromadb>=0.4.22
faiss-cpu>=1.7.4

# Cross-encoder reranking
torch>=2.1.0

# Testing & linting
pytest>=7.4.0
pytest-asyncio>=0.23.0
black>=23.12.0
isort>=5.13.0
mypy>=1.8.0
ruff>=0.1.9
```

#### `rag_engine/pyproject.toml`
```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "rag-engine"
version = "1.0.0"
description = "Knowledge Base / RAG Engine for Sovereign On-Premise Agentic AI Workbench"
authors = [{ name = "SovereignAI Team — Member 1" }]
license = { text = "Proprietary" }
requires-python = ">=3.11"
dependencies = [
    "pydantic>=2.5.0",
    "pydantic-settings>=2.1.0",
    "pyyaml>=6.0.1",
    "pypdf>=4.0.0",
    "python-docx>=1.1.0",
    "openpyxl>=3.1.0",
    "sentence-transformers>=2.2.2",
    "chromadb>=0.4.22",
    "faiss-cpu>=1.7.4",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.23.0",
    "black>=23.12.0",
    "isort>=5.13.0",
    "mypy>=1.8.0",
    "ruff>=0.1.9",
]

[tool.black]
line-length = 88
target-version = ["py311"]

[tool.isort]
profile = "black"
line_length = 88

[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_configs = true

[tool.ruff]
line-length = 88
select = ["E", "F", "W", "I", "N", "UP", "B", "A", "C4", "SIM"]
```

#### `rag_engine/__init__.py`
```python
"""
RAG Engine — Knowledge Base / Retrieval-Augmented Generation Pipeline.

Sovereign On-Premise Agentic AI Workbench
Problem Statement: SIH26117 | Organization: MRPL

This package provides a complete, offline-first RAG pipeline for ingesting
refinery manuals, safety documents, inspection reports, P&ID diagrams, and
maintenance logs, and performing semantic search and retrieval.
"""

__version__ = "1.0.0"
```

---

### 4.2 Configuration Module (`rag_engine/config/`)

#### `rag_engine/config/settings.py`
```python
"""
RAG Engine — Centralized Settings.

Loads configuration from YAML files and environment variables
using Pydantic BaseSettings. Provides a single `settings` instance
that all modules import.

Usage:
    from rag_engine.config.settings import settings
    print(settings.rag.chunk_size)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class EmbeddingModelConfig(BaseModel):
    """Configuration for the embedding model."""

    name: str = Field(default="", description="Model name or path")
    device: str = Field(default="cpu", description="Device: cpu | cuda | mps")
    batch_size: int = Field(default=32, description="Batch size for embedding")
    max_length: int = Field(default=512, description="Maximum token length")


class LLMConfig(BaseModel):
    """Configuration for the LLM backend."""

    provider: str = Field(default="ollama", description="LLM provider")
    name: str = Field(default="", description="Model name or path")
    device: str = Field(default="cpu", description="Device: cpu | cuda | mps")
    max_tokens: int = Field(default=4096, description="Maximum generation tokens")
    temperature: float = Field(default=0.1, description="Sampling temperature")


class RerankerConfig(BaseModel):
    """Configuration for the reranker model."""

    name: str = Field(default="", description="Reranker model name or path")
    device: str = Field(default="cpu", description="Device: cpu | cuda | mps")
    top_k: int = Field(default=5, description="Top-K after reranking")


class OCRConfig(BaseModel):
    """Configuration for the OCR engine."""

    engine: str = Field(default="tesseract", description="OCR engine backend")
    language: str = Field(default="eng", description="OCR language")
    confidence_threshold: float = Field(default=0.6, description="Min confidence")


class RAGConfig(BaseModel):
    """Configuration for the RAG pipeline."""

    chunk_size: int = Field(default=512, description="Chunk size in tokens")
    chunk_overlap: int = Field(default=50, description="Overlap between chunks")
    chunking_strategy: str = Field(
        default="recursive", description="fixed | semantic | recursive"
    )
    top_k: int = Field(default=5, description="Top-K retrieval results")
    rerank: bool = Field(default=True, description="Enable reranking")
    rerank_top_k: int = Field(default=3, description="Top-K after reranking")
    similarity_threshold: float = Field(
        default=0.7, description="Minimum similarity score"
    )


class VectorDBConfig(BaseModel):
    """Configuration for the vector database."""

    backend: str = Field(default="chromadb", description="chromadb | faiss")
    collection_name: str = Field(
        default="sovereign_kb", description="Vector collection name"
    )
    persist_directory: str = Field(
        default="./vector_db/chroma", description="Persistence directory"
    )


class CacheConfig(BaseModel):
    """Configuration for caching."""

    enabled: bool = Field(default=True, description="Enable caching")
    backend: str = Field(default="memory", description="memory | disk")
    max_size_mb: int = Field(default=512, description="Maximum cache size in MB")
    ttl_seconds: int = Field(default=3600, description="Cache TTL in seconds")
    embedding_cache_enabled: bool = Field(
        default=True, description="Enable embedding cache"
    )


class LoggingConfig(BaseModel):
    """Configuration for logging."""

    level: str = Field(default="INFO", description="Log level")
    format: str = Field(default="json", description="json | text")
    rotation: str = Field(default="10 MB", description="Log file rotation size")
    retention: str = Field(default="30 days", description="Log retention period")


class Settings(BaseSettings):
    """Root settings object for the RAG Engine."""

    app_name: str = Field(
        default="Sovereign AI Workbench — RAG Engine",
        description="Application name",
    )
    app_version: str = Field(default="1.0.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")

    embedding: EmbeddingModelConfig = Field(default_factory=EmbeddingModelConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    vector_db: VectorDBConfig = Field(default_factory=VectorDBConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    model_config = {"env_prefix": "RAG_", "env_nested_delimiter": "__"}


# Singleton settings instance — import this everywhere.
settings = Settings()
```

#### `rag_engine/config/paths.py`
```python
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
```

#### `rag_engine/config/constants.py`
```python
"""
RAG Engine — Application Constants.

Immutable values used across the RAG pipeline.
These NEVER change at runtime — use config/settings.py for configurable values.
"""

from __future__ import annotations


# --- Package Metadata ---
APP_NAME: str = "rag_engine"
APP_VERSION: str = "1.0.0"

# --- Supported File Formats ---
SUPPORTED_DOCUMENT_FORMATS: frozenset[str] = frozenset({
    ".pdf", ".docx", ".doc", ".md", ".txt", ".csv", ".xlsx",
})
SUPPORTED_IMAGE_FORMATS: frozenset[str] = frozenset({
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif",
})

# --- Chunking Defaults ---
DEFAULT_CHUNK_SIZE: int = 512
DEFAULT_CHUNK_OVERLAP: int = 50
MIN_CHUNK_SIZE: int = 50
MAX_CHUNK_SIZE: int = 4096

# --- Retrieval Defaults ---
DEFAULT_TOP_K: int = 5
MAX_TOP_K: int = 100
DEFAULT_SIMILARITY_THRESHOLD: float = 0.7

# --- Embedding Defaults ---
DEFAULT_EMBEDDING_DIMENSION: int = 768
DEFAULT_EMBEDDING_BATCH_SIZE: int = 32
MAX_EMBEDDING_TEXT_LENGTH: int = 8192

# --- Cache Defaults ---
DEFAULT_CACHE_TTL_SECONDS: int = 3600
DEFAULT_CACHE_MAX_SIZE_MB: int = 512

# --- Logging ---
LOG_DATE_FORMAT: str = "%Y-%m-%dT%H:%M:%S.%fZ"
LOG_FORMAT_JSON: str = "json"
LOG_FORMAT_TEXT: str = "text"

# --- Vector Database ---
DEFAULT_COLLECTION_NAME: str = "sovereign_kb"
CHROMA_BACKEND: str = "chromadb"
FAISS_BACKEND: str = "faiss"
```

#### `rag_engine/config/logging_config.py`
```python
"""
RAG Engine — Logging Configuration.

Configures structured logging with rotation and multiple output channels.
All modules obtain loggers via: get_logger(__name__)

Usage:
    from rag_engine.config.logging_config import get_logger
    logger = get_logger(__name__)
    logger.info("Processing file", extra={"path": str(file_path)})
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path
from typing import Optional

from rag_engine.config.constants import APP_NAME, LOG_DATE_FORMAT
from rag_engine.config.paths import Paths


_CONFIGURED: bool = False


def setup_logging(
    log_level: str = "INFO",
    log_dir: Optional[Path] = None,
    log_to_file: bool = True,
) -> None:
    """Configure the root logger for the RAG engine."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    level = getattr(logging, log_level.upper(), logging.INFO)
    root_logger = logging.getLogger(APP_NAME)
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s",
        datefmt=LOG_DATE_FORMAT,
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler with rotation
    if log_to_file:
        target_dir = log_dir or Paths.LOGS_RAG
        target_dir.mkdir(parents=True, exist_ok=True)
        log_file = target_dir / "rag_engine.log"

        file_handler = logging.handlers.RotatingFileHandler(
            filename=str(log_file),
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger for the given module name."""
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(f"{APP_NAME}.{name}")
```

---

### 4.3 Schemas Module (`rag_engine/schemas/`)

#### `rag_engine/schemas/document.py`
```python
"""Schema: Document — Represents a loaded source document."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class DocumentMetadata(BaseModel):
    """Metadata associated with a source document."""

    model_config = ConfigDict(frozen=True)

    source_path: str = Field(description="Absolute path to the source file")
    file_name: str = Field(description="Original file name with extension")
    file_format: str = Field(description="File extension, e.g. '.pdf'")
    file_size_bytes: int = Field(default=0, description="File size in bytes")
    page_count: Optional[int] = Field(default=None, description="Number of pages")
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Ingestion timestamp"
    )
    category: Optional[str] = Field(
        default=None, description="Dataset category, e.g. 'manuals', 'safety_docs'"
    )
    language: str = Field(default="en", description="Document language code")


class Document(BaseModel):
    """A loaded document with its full content and metadata."""

    model_config = ConfigDict(frozen=True)

    doc_id: str = Field(description="Unique document identifier")
    content: str = Field(description="Full text content of the document")
    metadata: DocumentMetadata = Field(description="Document metadata")
```

#### `rag_engine/schemas/chunk.py`
```python
"""Schema: Chunk — Represents a text chunk derived from a Document."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class ChunkMetadata(BaseModel):
    """Metadata associated with a text chunk."""

    model_config = ConfigDict(frozen=True)

    source_doc_id: str = Field(description="ID of the parent Document")
    source_file: str = Field(description="Original source file name")
    page_number: Optional[int] = Field(default=None, description="Page number")
    section: Optional[str] = Field(default=None, description="Section heading")
    equipment_id: Optional[str] = Field(
        default=None, description="Equipment identifier (e.g., Pump_P203)"
    )
    chunk_index: int = Field(description="Position of this chunk in the document")
    char_start: Optional[int] = Field(
        default=None, description="Start character offset in source"
    )
    char_end: Optional[int] = Field(
        default=None, description="End character offset in source"
    )


class Chunk(BaseModel):
    """A text chunk with metadata, ready for embedding."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(description="Unique chunk identifier")
    content: str = Field(description="Text content of the chunk")
    token_count: int = Field(default=0, description="Approximate token count")
    metadata: ChunkMetadata = Field(description="Chunk metadata")
```

#### `rag_engine/schemas/embedding.py`
```python
"""Schema: Embedding — Represents dense vector embeddings."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class EmbeddingVector(BaseModel):
    """A dense embedding vector for a chunk of text."""

    model_config = ConfigDict(frozen=True)

    vector: list[float] = Field(description="Dense vector representation")
    dimension: int = Field(description="Vector dimensionality")
    model_name: str = Field(default="", description="Embedding model used")
    source_chunk_id: str = Field(default="", description="ID of the source chunk")


class EmbeddingRequest(BaseModel):
    """Request to generate embeddings for one or more texts."""

    model_config = ConfigDict(frozen=True)

    texts: list[str] = Field(description="Texts to embed")
    model_name: str = Field(default="", description="Model to use for embedding")
```

#### `rag_engine/schemas/citation.py`
```python
"""Schema: Citation — Represents a source reference for retrieved content."""

from __future__ import annotations

from enum import StrEnum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class SourceType(StrEnum):
    """Enumeration of supported source document types."""

    PDF = "pdf"
    DOCX = "docx"
    MARKDOWN = "markdown"
    CSV = "csv"
    TEXT = "text"
    IMAGE = "image"
    UNKNOWN = "unknown"


class Source(BaseModel):
    """A reference to the original source document."""

    model_config = ConfigDict(frozen=True)

    file_name: str = Field(description="Source file name")
    file_path: str = Field(description="Source file path")
    source_type: SourceType = Field(
        default=SourceType.UNKNOWN, description="Type of source document"
    )
    page_number: Optional[int] = Field(default=None, description="Page number")
    section: Optional[str] = Field(default=None, description="Section heading")


class Citation(BaseModel):
    """A citation linking a retrieved chunk back to its source."""

    model_config = ConfigDict(frozen=True)

    citation_id: str = Field(description="Unique citation identifier")
    source: Source = Field(description="Source document reference")
    chunk_id: str = Field(description="ID of the cited chunk")
    excerpt: str = Field(default="", description="Relevant text excerpt")
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Confidence score"
    )
```

#### `rag_engine/schemas/retrieved_document.py`
```python
"""Schema: RetrievedDocument — Represents retrieval results."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from rag_engine.schemas.chunk import Chunk
from rag_engine.schemas.citation import Citation


class ScoredChunk(BaseModel):
    """A chunk with its similarity score from retrieval."""

    model_config = ConfigDict(frozen=True)

    chunk: Chunk = Field(description="The retrieved chunk")
    score: float = Field(ge=0.0, le=1.0, description="Similarity score")
    rank: int = Field(default=0, ge=0, description="Rank position (0-indexed)")


class RetrievedDocument(BaseModel):
    """Aggregated retrieval result for a user query."""

    model_config = ConfigDict(frozen=True)

    query: str = Field(description="Original query text")
    chunks: list[ScoredChunk] = Field(
        default_factory=list, description="Ranked retrieved chunks"
    )
    citations: list[Citation] = Field(
        default_factory=list, description="Citations for the retrieved chunks"
    )
    total_retrieved: int = Field(default=0, description="Number of chunks retrieved")
    latency_ms: float = Field(default=0.0, description="Retrieval latency in ms")
```

---

### 4.4 Interfaces Module (`rag_engine/interfaces/`)

#### `rag_engine/interfaces/base_loader.py`
```python
"""
Interface: BaseLoader

Abstract contract for document loaders. Each file format (PDF, DOCX, MD,
CSV, TXT, etc.) provides a concrete implementation of this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag_engine.schemas.document import Document


class BaseLoader(ABC):
    """Abstract base class for document loaders."""

    @abstractmethod
    def load(self, file_path: Path) -> Document:
        """
        Load a single document from the given file path.

        Args:
            file_path: Absolute or relative path to the source file.

        Returns:
            A Document schema instance with content and metadata.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file format is not supported by this loader.
        """
        ...

    @abstractmethod
    def supported_formats(self) -> list[str]:
        """
        Return the list of file extensions this loader supports.

        Returns:
            List of lowercase extensions including the dot, e.g. [".pdf"].
        """
        ...

    def can_load(self, file_path: Path) -> bool:
        """
        Check if this loader supports the given file.

        Args:
            file_path: Path to check.

        Returns:
            True if the file extension is in supported_formats().
        """
        return file_path.suffix.lower() in self.supported_formats()
```

#### `rag_engine/interfaces/base_chunker.py`
```python
"""
Interface: BaseChunker

Abstract contract for text chunking strategies. Implementations include
fixed-size, semantic, and recursive chunking.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag_engine.schemas.chunk import Chunk
    from rag_engine.schemas.document import Document


class BaseChunker(ABC):
    """Abstract base class for document chunking strategies."""

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """
        Split a document into chunks.

        Args:
            document: The Document to chunk.

        Returns:
            Ordered list of Chunk instances.

        Raises:
            ValueError: If the document content is empty.
        """
        ...

    @abstractmethod
    def get_strategy_name(self) -> str:
        """
        Return the name of the chunking strategy.

        Returns:
            Strategy identifier string, e.g. "fixed", "semantic", "recursive".
        """
        ...
```

#### `rag_engine/interfaces/base_embedder.py`
```python
"""
Interface: BaseEmbedder

Abstract contract for embedding model wrappers. Implementations wrap
models like sentence-transformers, instructor-xl, etc.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag_engine.schemas.embedding import EmbeddingVector


class BaseEmbedder(ABC):
    """Abstract base class for text embedding models."""

    @abstractmethod
    def embed(self, text: str) -> EmbeddingVector:
        """
        Generate an embedding vector for a single text.

        Args:
            text: Input text to embed.

        Returns:
            EmbeddingVector with the dense vector representation.

        Raises:
            ValueError: If text is empty or exceeds max length.
        """
        ...

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[EmbeddingVector]:
        """
        Generate embedding vectors for a batch of texts.

        Args:
            texts: List of input texts to embed.

        Returns:
            List of EmbeddingVector instances, same order as input.
        """
        ...

    @abstractmethod
    def get_dimension(self) -> int:
        """
        Return the dimensionality of the embedding vectors.

        Returns:
            Integer dimension (e.g., 768, 1024).
        """
        ...

    @abstractmethod
    def get_model_name(self) -> str:
        """
        Return the name or path of the underlying embedding model.

        Returns:
            Model identifier string.
        """
        ...
```

#### `rag_engine/interfaces/base_vector_store.py`
```python
"""
Interface: BaseVectorStore

Abstract contract for vector database backends. Implementations include
ChromaDB and FAISS.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag_engine.schemas.chunk import Chunk
    from rag_engine.schemas.embedding import EmbeddingVector
    from rag_engine.schemas.retrieved_document import ScoredChunk


class BaseVectorStore(ABC):
    """Abstract base class for vector store backends."""

    @abstractmethod
    def add(self, chunks: list[Chunk], embeddings: list[EmbeddingVector]) -> int:
        """
        Add chunks and their embeddings to the vector store.

        Args:
            chunks: List of Chunk instances.
            embeddings: Corresponding embedding vectors (same order).

        Returns:
            Number of items successfully added.

        Raises:
            ValueError: If lengths of chunks and embeddings differ.
        """
        ...

    @abstractmethod
    def search(
        self, query_embedding: EmbeddingVector, top_k: int = 5
    ) -> list[ScoredChunk]:
        """
        Search for the most similar chunks to the query embedding.

        Args:
            query_embedding: The query vector.
            top_k: Number of results to return.

        Returns:
            List of ScoredChunk instances, sorted by descending similarity.
        """
        ...

    @abstractmethod
    def delete(self, chunk_ids: list[str]) -> int:
        """
        Delete chunks by their IDs.

        Args:
            chunk_ids: List of chunk identifiers to remove.

        Returns:
            Number of items successfully deleted.
        """
        ...

    @abstractmethod
    def count(self) -> int:
        """
        Return the total number of vectors in the store.

        Returns:
            Integer count.
        """
        ...

    @abstractmethod
    def clear(self) -> None:
        """Remove all vectors from the store."""
        ...
```

#### `rag_engine/interfaces/base_retriever.py`
```python
"""
Interface: BaseRetriever

Abstract contract for document retrieval strategies. Implementations
include dense retrieval, hybrid retrieval, and contextual retrieval.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag_engine.schemas.retrieved_document import RetrievedDocument


class BaseRetriever(ABC):
    """Abstract base class for retrieval strategies."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> RetrievedDocument:
        """
        Retrieve relevant chunks for the given query.

        Args:
            query: Natural language query string.
            top_k: Maximum number of chunks to retrieve.

        Returns:
            RetrievedDocument containing scored chunks and metadata.

        Raises:
            ValueError: If query is empty.
        """
        ...

    @abstractmethod
    def get_strategy_name(self) -> str:
        """
        Return the name of the retrieval strategy.

        Returns:
            Strategy identifier string, e.g. "dense", "hybrid", "contextual".
        """
        ...
```

---

### 4.5 Utilities Module (`rag_engine/utils/`)

#### `rag_engine/utils/logger.py`
```python
"""
Logger utility — Convenience re-export of the logging factory.

Usage:
    from rag_engine.utils.logger import get_logger
    logger = get_logger(__name__)
"""

from rag_engine.config.logging_config import get_logger

__all__ = ["get_logger"]
```

#### `rag_engine/utils/file_utils.py`
```python
"""
File utilities — Path manipulation, temporary files, format detection.

Placeholder for M2 implementation.
"""

from __future__ import annotations

# TODO: Implement in Milestone 2
# - safe_read_file(path) -> bytes
# - detect_format(path) -> str
# - get_temp_path(prefix) -> Path
# - list_files(directory, extensions) -> list[Path]
# - ensure_directory(path) -> Path
```

#### `rag_engine/utils/hash_utils.py`
```python
"""
Hash utilities — Content hashing for deduplication and cache keys.

Placeholder for M2 implementation.
"""

from __future__ import annotations

# TODO: Implement in Milestone 2
# - hash_text(text) -> str           (SHA-256 hex digest)
# - hash_file(path) -> str           (SHA-256 of file contents)
# - hash_embedding(vector) -> str    (Stable hash for embedding cache)
```

#### `rag_engine/utils/validators.py`
```python
"""
Validators — Input validation for files, schemas, and configurations.

Placeholder for M2 implementation.
"""

from __future__ import annotations

# TODO: Implement in Milestone 2
# - validate_file_exists(path) -> bool
# - validate_file_format(path, allowed) -> bool
# - validate_file_size(path, max_bytes) -> bool
# - validate_chunk_size(size, min_size, max_size) -> bool
# - validate_embedding_dimension(vector, expected_dim) -> bool
```

---

## 5. Architectural Correctness & Invariant Validation

1. **Strict Decoupling**:
   - Consumers in higher layers depend solely on `interfaces/` and `schemas/`.
   - Concrete implementations (such as ChromaDB, PyPDF, SentenceTransformers) will be injected via factory patterns in Milestone 2.
2. **Immutable Data Contracts**:
   - All Pydantic models inherit `ConfigDict(frozen=True)`, preventing unintended in-place mutation across concurrent threads or asynchronous pipelines.
3. **No Circular Imports**:
   - All cross-schema references in interfaces use `if TYPE_CHECKING:` guards and `from __future__ import annotations`.
4. **Offline Isolation**:
   - Zero remote network endpoints, cloud APIs, or external telemetry are referenced or configured.
   - All paths are relative to `PROJECT_ROOT` (`d:\SovereignAI`).

---

## 6. Prompt to Give ChatGPT for Auditing

You can copy and paste the prompt below along with this document into ChatGPT:

```markdown
Hello ChatGPT,

I am participating in Smart India Hackathon 2026 (Problem Statement: SIH26117) building an enterprise, 100% offline, on-premise Agentic AI Workbench for MRPL (Mangalore Refinery and Petrochemicals Limited).

I have completed Milestone 1 (Project Scaffolding & Architecture Foundation) for Member 1: Knowledge Base / RAG Engine (`rag_engine`).

Please review the attached architectural scaffolding document and evaluate:
1. Architectural Soundness: Does this structure strictly adhere to Clean Architecture, SOLID principles, and separation of concerns?
2. Pydantic v2 Correctness: Are the schemas (Document, Chunk, EmbeddingVector, Citation, RetrievedDocument) idiomatic, robust, and correctly typed?
3. Interface Contracts: Are the abstract base classes (`BaseLoader`, `BaseChunker`, `BaseEmbedder`, `BaseVectorStore`, `BaseRetriever`) sufficient to support PDF/DOCX/CSV loading, recursive/semantic chunking, Chroma/FAISS storage, and hybrid retrieval?
4. Completeness & Gaps: Are there any critical omissions or design flaws that could cause refactoring issues during Milestone 2 (business logic implementation)?
5. Recommended Next Step: What exact tasks should I assign to the developer for Milestone 2 (e.g., implementing specific document loaders, tokenizers, or vector store wrappers)?
```
