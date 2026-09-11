"""Targeted Ingestion of Refinery Inspection Reports and Safety Standards.

Ensures real documents like centrifugal_pump_inspection_005.md (P-315B) and
OISD-STD-105.pdf are fully parsed, cleaned, chunked, embedded, and indexed into
both Qdrant ('mrpl_docs_v1') and the Okapi BM25 index.
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys
import time

# Ensure project root is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_engine.chunking import get_chunk_factory
from rag_engine.embeddings.embedding_pipeline import EmbeddingPipeline
from rag_engine.loaders import LoaderFactory
from rag_engine.parsers import ParserFactory
from rag_engine.preprocessing import CleaningPipeline
from rag_engine.retrieval.bm25_retriever import BM25Index
from rag_engine.schemas.chunk import Chunk
from rag_engine.schemas.embedding import EmbeddedChunk
from rag_engine.schemas.vector_store import DistanceMetric
from rag_engine.vector_store import (
    CollectionConfig,
    IndexManager,
    QdrantVectorStore,
    VectorStoreConfig,
)
from scripts.ingest_real_documents import extract_technical_entities

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("target_ingest")


def main() -> None:
    target_files = [
        ("inspection_reports", Path("datasets/inspection_reports/centrifugal_pump_inspection_005.md")),
        ("safety_docs", Path("datasets/safety_docs/OISD-STD-105.pdf")),
    ]

    # Also add remaining inspection reports if available
    insp_dir = Path("datasets/inspection_reports")
    if insp_dir.exists():
        for p in sorted(insp_dir.glob("*.md")):
            item = ("inspection_reports", p)
            if item not in target_files:
                target_files.append(item)

    logger.info("Target files to ingest: %d files", len(target_files))
    for cat, p in target_files:
        logger.info("  - [%s] %s (%d bytes)", cat, p.name, p.stat().st_size)

    loader_factory = LoaderFactory()
    parser_factory = ParserFactory()
    cleaner = CleaningPipeline(strict_token_verification=False)
    chunk_factory = get_chunk_factory()

    all_chunks: list[Chunk] = []

    for cat, file_path in target_files:
        raw_doc = loader_factory.get_loader(file_path=file_path).load(file_path=file_path)
        parser, _ = parser_factory.get_parser(raw_doc)
        parsed_doc = parser.parse(raw_doc)
        cleaned_doc = cleaner.clean(parsed_doc)
        chunker = chunk_factory.for_document(cleaned_doc)
        chunks = chunker.chunk(cleaned_doc)

        for c in chunks:
            eq_tags, stds, params = extract_technical_entities(c.content)
            meta = c.metadata
            object.__setattr__(meta, "category", cat)
            object.__setattr__(meta, "source_file", file_path.name)
            if "oisd" in file_path.name.lower():
                object.__setattr__(meta, "document_name", "OISD-STD-105: Work Permit System")
                stds = sorted(list(set(stds + ["OISD-STD-105", "OISD-105"])))
            elif "pump" in file_path.name.lower():
                object.__setattr__(meta, "document_name", "Inspection Report - Centrifugal Pump P-315B")
                eq_tags = sorted(list(set(eq_tags + ["P-315B"])))

            if eq_tags:
                existing_eq = list(meta.equipment_entities or [])
                object.__setattr__(meta, "equipment_entities", sorted(list(set(existing_eq + eq_tags))))
            if stds:
                existing_std = list(meta.safety_entities or [])
                object.__setattr__(meta, "safety_entities", sorted(list(set(existing_std + stds))))
            if params:
                existing_params = dict(meta.operating_parameters or {})
                existing_params.update(params)
                object.__setattr__(meta, "operating_parameters", existing_params)

        logger.info("  Processed '%s' -> %d chunks", file_path.name, len(chunks))
        all_chunks.extend(chunks)

    logger.info("Total generated chunks across target files: %d", len(all_chunks))

    # Embed chunks
    logger.info("Generating embeddings with BAAI/bge-small-en-v1.5...")
    embed_pipeline = EmbeddingPipeline(
        model_name="BAAI/bge-small-en-v1.5",
        models_dir="models/embeddings",
        cache_db_path="cache/embeddings/production_embedding_cache.db",
        enable_cache=True,
    )
    embedded_chunks = embed_pipeline.embed_chunks(all_chunks, job_id="target_eval_ingest")
    logger.info("Generated %d embedded chunks.", len(embedded_chunks))

    # Qdrant Vector Store
    storage_path = Path("vector_db/qdrant")
    store = QdrantVectorStore(config=VectorStoreConfig(storage_path=storage_path))
    index_mgr = IndexManager(store=store)

    for col in ["mrpl_docs_v1", "mrpl_general_v1"]:
        if not store.collection_exists(col):
            store.create_collection(
                CollectionConfig(name=col, vector_size=embed_pipeline.dimension, distance=DistanceMetric.COSINE)
            )
        res = index_mgr.index_document_chunks(embedded_chunks, collection_name=col)
        logger.info("Indexed into Qdrant collection '%s': %d inserted, %d updated.", col, res.inserted, res.updated)

    # BM25 Index
    bm25_path = Path("cache/retrieval/bm25_index.json")
    bm25_index = BM25Index(storage_path=bm25_path)
    if bm25_path.exists():
        bm25_index.load()
    bm25_index.add_chunks(all_chunks)
    bm25_index.save()
    logger.info("BM25 Index updated: %d total documents persisted to %s", bm25_index.total_docs, bm25_path)

    store.close()
    logger.info("Target document ingestion completed successfully.")


if __name__ == "__main__":
    main()
