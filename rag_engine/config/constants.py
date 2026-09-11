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
    ".pdf", ".docx", ".doc", ".md", ".txt", ".csv", ".xlsx", ".pptx", ".json", ".xml",
})
SUPPORTED_IMAGE_FORMATS: frozenset[str] = frozenset({
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif",
})
SUPPORTED_ALL_FORMATS: frozenset[str] = SUPPORTED_DOCUMENT_FORMATS | SUPPORTED_IMAGE_FORMATS

# --- Validation Status ---
STATUS_VALID: str = "VALID"
STATUS_EMPTY: str = "EMPTY"
STATUS_CORRUPTED: str = "CORRUPTED"
STATUS_UNSUPPORTED: str = "UNSUPPORTED"
STATUS_DUPLICATE: str = "DUPLICATE"

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
