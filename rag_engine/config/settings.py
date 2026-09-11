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
