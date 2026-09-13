"""Master End-to-End Refinery Dataset Ingestion & Indexing Pipeline.

Sovereign On-Premise Agentic AI Workbench (SIH26117 / MRPL)
Executes Stages 1 through 8 on real refinery datasets:
    1. Dataset Discovery
    2. Document Loading (Universal Loader Framework)
    3. Structural Parsing (Parsing Engine)
    4. Text & Entity Cleaning (Cleaning Pipeline)
    5. Enterprise Chunking (Chunking Engine with Equipment/OISD Extraction)
    6. Local Air-Gapped Embedding (BGE-Small-EN-v1.5)
    7. Qdrant Vector Store Setup (Embedded vector_db/qdrant)
    8. Dual-Channel Indexing (Qdrant Dense Vectors + BM25 Lexical Inverted Index)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import re
import sys
import time
from typing import Any, List

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("mrpl_ingest")


def extract_technical_entities(text: str) -> tuple[list[str], list[str], dict[str, str]]:
    """Extract equipment tags, standards, and numerical operating parameters from text."""
    # Equipment tags (e.g. P-315B, P-203, C-420A, V-2201, TK-610, B-101)
    eq_matches = re.findall(r"\b([A-Z]{1,4}-[0-9]{2,5}[A-Z]?)\b", text)
    equipment_tags = sorted(list(set(eq_matches)))

    # Standards (e.g. OISD-STD-105, OISD-105, API-610, ASME B31.3)
    std_matches = re.findall(r"\b(OISD(?:-STD)?-[0-9]{3}|API-[0-9]{3}|ASME\s+[A-Z0-9\.]+)\b", text, re.IGNORECASE)
    standards = sorted(list(set(std_matches)))

    # Operating parameters (pressure, temperature)
    params: dict[str, str] = {}
    p_match = re.search(r"([0-9]+(?:\.[0-9]+)?\s*(?:bar|barg|psi|kPa|MPa))\b", text, re.IGNORECASE)
    if p_match:
        params["pressure"] = p_match.group(1)

    t_match = re.search(r"([0-9]+(?:\.[0-9]+)?\s*(?:°C|degC|F))\b", text, re.IGNORECASE)
    if t_match:
        params["temperature"] = t_match.group(1)

    return equipment_tags, standards, params


def run_ingestion(
    dataset_dir: str | Path = "datasets",
    max_docs_per_category: int | None = None,
    collection_name: str = "mrpl_docs_v1",
) -> dict[str, Any]:
    dataset_path = Path(dataset_dir)
    print("=" * 80)
    print("  MRPL SOVEREIGN RAG ENGINE - PRODUCTION DATASET INGESTION")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # STAGE 1: DATASET DISCOVERY
    # -------------------------------------------------------------------------
    print("\n[STAGE 1/8] Discovering Real Refinery Documents...")
    if not dataset_path.exists() or not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset directory '{dataset_path}' does not exist.")

    categories = [
        "inspection_reports",
        "safety_docs",
        "manuals",
        "templates",
        "emails",
        "maintenance",
    ]

    discovered_files: list[tuple[str, Path]] = []
    supported_extensions = {".pdf", ".md", ".txt", ".docx", ".csv", ".xlsx"}

    for cat in categories:
        cat_dir = dataset_path / cat
        if not cat_dir.is_dir():
            continue
        all_candidate_files = sorted(cat_dir.rglob("*.*"))
        files = []
        for f in all_candidate_files:
            if f.name.startswith(".") or not f.is_file():
                continue
            if f.suffix.lower() not in supported_extensions:
                continue
            # Explicit exclusion for maintenance/CMaps non-pdf files (numeric sensor data)
            if cat == "maintenance" and "CMaps" in f.parts and f.suffix.lower() != ".pdf":
                continue
            files.append(f)

        if max_docs_per_category:
            files = files[:max_docs_per_category]
        for f in files:
            discovered_files.append((cat, f))

    print(f"  -> Discovered {len(discovered_files)} documents across {len(categories)} categories.")
    for cat in categories:
        count = sum(1 for c, _ in discovered_files if c == cat)
        print(f"     - {cat:20s}: {count} documents")

    if not discovered_files:
        raise RuntimeError(f"No valid documents found in {dataset_path}.")

    # -------------------------------------------------------------------------
    # STAGE 2: LOADING & STAGE 3: PARSING & STAGE 4: CLEANING & STAGE 5: CHUNKING
    # -------------------------------------------------------------------------
    print("\n[STAGE 2-5/8] Loading, Parsing, Cleaning & Chunking Documents...")
    loader_factory = LoaderFactory()
    parser_factory = ParserFactory()
    cleaner = CleaningPipeline(strict_token_verification=False)
    chunk_factory = get_chunk_factory()

    all_chunks: list[Chunk] = []
    failed_docs: list[str] = []
    doc_summary: list[dict[str, Any]] = []

    for cat, file_path in discovered_files:
        t0 = time.perf_counter()
        try:
            # Stage 2: Load
            loader = loader_factory.get_loader(file_path=file_path)
            raw_doc = loader.load(file_path=file_path)

            # Stage 3: Parse
            parser, _ = parser_factory.get_parser(raw_doc)
            parsed_doc = parser.parse(raw_doc)

            # Stage 4: Clean
            cleaned_doc = cleaner.clean(parsed_doc)

            # Stage 5: Chunk
            chunker = chunk_factory.for_document(cleaned_doc)
            chunks = chunker.chunk(cleaned_doc)

            # Enrich chunks with category and technical entities
            for c in chunks:
                eq_tags, stds, params = extract_technical_entities(c.content)
                meta = c.metadata
                # Update category
                object.__setattr__(meta, "category", cat)
                if eq_tags:
                    existing_eq = list(meta.equipment_entities or [])
                    merged_eq = sorted(list(set(existing_eq + eq_tags)))
                    object.__setattr__(meta, "equipment_entities", merged_eq)
                if stds:
                    existing_std = list(meta.safety_entities or [])
                    merged_std = sorted(list(set(existing_std + stds)))
                    object.__setattr__(meta, "safety_entities", merged_std)
                if params:
                    existing_params = dict(meta.operating_parameters or {})
                    existing_params.update(params)
                    object.__setattr__(meta, "operating_parameters", existing_params)

            all_chunks.extend(chunks)
            dur = time.perf_counter() - t0
            print(f"  [OK] {cat}/{file_path.name[:45]:45s} -> {len(chunks):3d} chunks ({dur:.2f}s)")
            doc_summary.append({
                "category": cat,
                "file": file_path.name,
                "chunks": len(chunks),
                "duration_s": round(dur, 2),
            })
        except Exception as e:
            failed_docs.append(f"{cat}/{file_path.name}: {e}")
            print(f"  [FAIL] {cat}/{file_path.name}: {e}")

    total_chunks = len(all_chunks)
    avg_chunk_chars = sum(len(c.content) for c in all_chunks) // max(1, total_chunks)
    print(f"\n  -> Successfully prepared {total_chunks} chunks from {len(discovered_files) - len(failed_docs)} documents.")
    print(f"  -> Average chunk character length: {avg_chunk_chars} chars")
    if failed_docs:
        print(f"  -> Failed documents ({len(failed_docs)}): {failed_docs}")

    # -------------------------------------------------------------------------
    # STAGE 6: EMBEDDING
    # -------------------------------------------------------------------------
    print("\n[STAGE 6/8] Generating Vector Embeddings with Local BGE Model...")
    t_embed_start = time.perf_counter()
    embed_pipeline = EmbeddingPipeline(
        model_name="BAAI/bge-small-en-v1.5",
        models_dir="models/embeddings",
        cache_db_path="cache/embeddings/production_embedding_cache.db",
        checkpoint_db_path="cache/embeddings/production_checkpoints.db",
        enable_cache=True,
        enable_checkpoints=True,
    )
    print(f"  -> Model: {embed_pipeline.model_name} (Dimension: {embed_pipeline.dimension}, Device: {embed_pipeline.embedder.get_device()})")

    embedded_chunks: list[EmbeddedChunk] = embed_pipeline.embed_chunks(
        all_chunks, job_id="mrpl_ingest_master_job"
    )
    embed_time = time.perf_counter() - t_embed_start
    print(f"  -> Embedded {len(embedded_chunks)} chunks in {embed_time:.2f}s ({len(embedded_chunks) / max(0.1, embed_time):.1f} chunks/sec)")

    # -------------------------------------------------------------------------
    # STAGE 7: QDRANT VECTOR STORE INITIALIZATION
    # -------------------------------------------------------------------------
    print("\n[STAGE 7/8] Initializing Local Qdrant Vector Store...")
    storage_path = Path("vector_db/qdrant")
    storage_path.mkdir(parents=True, exist_ok=True)

    config = VectorStoreConfig(
        backend="qdrant",
        storage_path=storage_path,
        enable_wal_journal=True,
    )
    store = QdrantVectorStore(config=config)

    # Ensure collection exists
    target_collections = [collection_name, "mrpl_general_v1"]
    for col in target_collections:
        if not store.collection_exists(col):
            print(f"  -> Creating Qdrant collection '{col}' (size={embed_pipeline.dimension}, metric=Cosine)...")
            store.create_collection(
                CollectionConfig(
                    name=col,
                    vector_size=embed_pipeline.dimension,
                    distance=DistanceMetric.COSINE,
                )
            )
        else:
            print(f"  -> Qdrant collection '{col}' already exists.")

    # -------------------------------------------------------------------------
    # STAGE 8: INDEXING INTO QDRANT & BM25
    # -------------------------------------------------------------------------
    print("\n[STAGE 8/8] Indexing Chunks into Qdrant & Building BM25 Index...")
    index_mgr = IndexManager(store=store)

    # Index into primary collection mrpl_docs_v1
    print(f"  -> Inserting {len(embedded_chunks)} vectors into Qdrant collection '{collection_name}'...")
    res_primary = index_mgr.index_document_chunks(embedded_chunks, collection_name=collection_name)
    print(f"     Indexed: {res_primary.inserted} inserted, {res_primary.updated} updated, {res_primary.skipped} skipped.")

    # Index into fallback collection mrpl_general_v1
    print(f"  -> Inserting into Qdrant fallback collection 'mrpl_general_v1'...")
    index_mgr.index_document_chunks(embedded_chunks, collection_name="mrpl_general_v1")

    # Verify Qdrant point count
    actual_points = store.count_points(collection_name)
    print(f"  -> Verified Qdrant points in '{collection_name}': {actual_points}")

    # Build and Persist BM25 Index
    bm25_path = Path("cache/retrieval/bm25_index.json")
    bm25_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  -> Building Okapi BM25 index on {len(all_chunks)} chunks...")
    bm25_index = BM25Index(storage_path=bm25_path)
    bm25_index.add_chunks(all_chunks)
    bm25_index.save()
    print(f"  -> Persisted BM25 index to: {bm25_path} (Docs: {bm25_index.total_docs})")

    store.close()

    print("\n" + "=" * 80)
    print("  INGESTION & INDEXING COMPLETE - RUNTIME IS READY FOR QUERIES")
    print("=" * 80)

    return {
        "discovered_documents": len(discovered_files),
        "total_chunks": len(all_chunks),
        "embedded_chunks": len(embedded_chunks),
        "vector_dimension": embed_pipeline.dimension,
        "qdrant_collection": collection_name,
        "qdrant_points": actual_points,
        "bm25_docs": bm25_index.total_docs,
        "bm25_path": str(bm25_path),
        "failed_documents": failed_docs,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest Real Refinery Documents into Qdrant & BM25")
    parser.add_argument("--dataset-dir", default="datasets", help="Path to datasets folder")
    parser.add_argument("--max-per-cat", type=int, default=None, help="Max docs per category (optional limit)")
    parser.add_argument("--collection", default="mrpl_docs_v1", help="Target Qdrant collection name")
    args = parser.parse_args()

    run_ingestion(
        dataset_dir=args.dataset_dir,
        max_docs_per_category=args.max_per_cat,
        collection_name=args.collection,
    )
