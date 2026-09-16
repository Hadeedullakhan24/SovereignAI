"""Master End-to-End Operational Pipeline Verification Test.

Validates the complete chain:
Dataset
  ↓
Loader
  ↓
Parser
  ↓
Cleaning
  ↓
Chunking
  ↓
Embedding
  ↓
Qdrant Vector Store
  ↓
Hybrid Retrieval
  ↓
Context Packing
  ↓
Prompt Builder
  ↓
RetrievedPrompt
  ↓
Generation Engine
  ↓
Verified RAGResponse
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import pytest

from rag_engine.chunking import get_chunk_factory
from rag_engine.embeddings.embedding_pipeline import EmbeddingPipeline
from rag_engine.generation.generation_config import GenerationConfig
from rag_engine.generation.generation_pipeline import GenerationPipeline
from rag_engine.generation.prompt import PromptArchetype
from rag_engine.loaders import LoaderFactory
from rag_engine.parsers import ParserFactory
from rag_engine.pipeline.rag_pipeline import RAGPipeline, RAGResponse
from rag_engine.preprocessing import CleaningPipeline
from rag_engine.retrieval.base_retriever import RetrievalResult
from rag_engine.retrieval.bm25_retriever import BM25Index
from rag_engine.retrieval.retrieval_pipeline import RetrievalPipeline
from rag_engine.schemas.chunk import Chunk
from rag_engine.schemas.prompt import RetrievedPrompt
from rag_engine.schemas.vector_store import DistanceMetric
from rag_engine.vector_store import (
    CollectionConfig,
    CollectionManager,
    IndexManager,
    QdrantVectorStore,
    VectorRepository,
    VectorStoreConfig,
)


def test_complete_rag_pipeline_end_to_end(tmp_path: Path):
    """Verify that every single stage in the RAG pipeline integrates seamlessly."""

    # 1. Dataset Creation (Simulated Technical Manual)
    doc_path = tmp_path / "centrifugal_pump_p101_spec.md"
    doc_content = (
        "# Centrifugal Pump P-101 Operating Manual\n\n"
        "## Technical Specifications\n"
        "Centrifugal pump P-101 operates at a maximum discharge pressure of 18.5 bar.\n"
        "Normal suction pressure is maintained at 2.4 bar with operating temperature of 75.0 °C.\n\n"
        "| Parameter | Specification | Compliance Standard |\n"
        "| Discharge Pressure | 18.5 bar | ASME B16.34 |\n"
        "| Suction Pressure | 2.4 bar | API 610 |\n"
        "| Operating Temp | 75.0 °C | OISD-105 |\n\n"
        "## Safety Procedures\n"
        "In case of seal failure or high vibration exceeding 4.5 mm/s, activate emergency isolation valve XV-101.\n"
    )
    doc_path.write_text(doc_content, encoding="utf-8")

    # 2. Loader
    loader_factory = LoaderFactory()
    loader = loader_factory.get_loader(file_path=doc_path)
    raw_doc = loader.load(file_path=doc_path)
    assert len(raw_doc.content) > 0

    # 3. Parser
    parser_factory = ParserFactory()
    parser, _ = parser_factory.get_parser(raw_doc)
    parsed_doc = parser.parse(raw_doc)
    assert len(parsed_doc.sections) >= 1

    # 4. Cleaning
    cleaner = CleaningPipeline(strict_token_verification=False)
    cleaned_doc = cleaner.clean(parsed_doc)
    assert len(cleaned_doc.get_full_text()) > 0

    # 5. Chunking
    chunk_factory = get_chunk_factory()
    chunker = chunk_factory.for_document(cleaned_doc)
    chunks = chunker.chunk(cleaned_doc)
    assert len(chunks) >= 1

    # 6. Embedding (Using Deterministic Test Embedder for air-gapped test)
    cache_db = tmp_path / "embed_cache.db"
    checkpoint_db = tmp_path / "checkpoints.db"
    embed_pipeline = EmbeddingPipeline(
        model_name="test-embedder-384",
        cache_db_path=cache_db,
        checkpoint_db_path=checkpoint_db,
    )
    embedded_chunks = embed_pipeline.embed_chunks(chunks)
    assert len(embedded_chunks) == len(chunks)
    assert len(embedded_chunks[0].embedding) == 384

    # 7. Qdrant Vector Store
    qdrant_dir = tmp_path / "qdrant_db"
    config = VectorStoreConfig(
        backend="qdrant",
        storage_path=qdrant_dir,
        enable_wal_journal=False,
    )
    store = QdrantVectorStore(config=config)
    col_name = "mrpl_test_pipeline_v1"
    store.create_collection(
        CollectionConfig(name=col_name, vector_size=384, distance=DistanceMetric.COSINE)
    )

    index_mgr = IndexManager(store)
    index_res = index_mgr.index_document_chunks(embedded_chunks, collection_name=col_name)
    assert index_res.inserted == len(embedded_chunks)

    # 8. Hybrid Retrieval (Sparse BM25 + Dense Qdrant)
    bm25_index = BM25Index()
    bm25_index.add_chunks(chunks)

    repo = VectorRepository(store=store)
    retrieval_pipe = RetrievalPipeline(
        strategy_name="hybrid",
        repository=repo,
        bm25_index=bm25_index,
        use_cache=False,
    )

    query = "What is the maximum discharge pressure of centrifugal pump P-101?"

    # 9. Context Packing & Prompt Builder -> RetrievedPrompt
    gen_config = GenerationConfig(
        default_model_name="deterministic_test",
        cache_enabled=False,
    )
    rag_pipe = RAGPipeline(
        retrieval_pipeline=retrieval_pipe,
        config=gen_config,
    )

    retrieved_prompt = rag_pipe.build_retrieved_prompt(
        query=query,
        archetype=PromptArchetype.EQUIPMENT_LOOKUP,
    )

    assert isinstance(retrieved_prompt, RetrievedPrompt)
    assert retrieved_prompt.query == query
    assert len(retrieved_prompt.prompt_hash) == 64
    assert "18.5 bar" in retrieved_prompt.prompt_text
    assert "[1]" in retrieved_prompt.prompt_text
    assert retrieved_prompt.context_tokens > 0

    # 10. Generation Engine Execution -> Verified RAGResponse
    rag_response = rag_pipe.query(
        query=query,
        archetype=PromptArchetype.EQUIPMENT_LOOKUP,
    )

    assert isinstance(rag_response, RAGResponse)
    assert rag_response.query == query
    assert "18.5 bar" in rag_response.answer
    assert len(rag_response.citations) >= 1
    assert rag_response.is_grounded is True
    assert rag_response.confidence_score > 0.70
    assert rag_response.total_latency_ms > 0.0

    store.close()
