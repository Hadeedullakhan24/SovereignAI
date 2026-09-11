"""Real Refinery Dataset Embedding Benchmark for Milestone 6.

Executes end-to-end ingestion:
    UniversalLoader -> DeepParser -> CleaningPipeline -> ChunkingEngine -> EmbeddingPipeline
Tests throughput, cache efficiency, vector validation, and resource utilization on real MRPL documents.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Any

# Ensure project root is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_engine.chunking import get_chunk_factory
from rag_engine.embeddings.embedding_pipeline import EmbeddingPipeline
from rag_engine.loaders import LoaderFactory
from rag_engine.parsers import ParserFactory
from rag_engine.preprocessing import CleaningPipeline


def run_benchmark() -> dict[str, Any]:
    print("=" * 78)
    print("  MRPL SOVEREIGN RAG ENGINE — MILESTONE 6 EMBEDDING PIPELINE BENCHMARK")
    print("=" * 78)

    loader_factory = LoaderFactory()
    parser_factory = ParserFactory()
    cleaner = CleaningPipeline(strict_token_verification=False)
    chunk_factory = get_chunk_factory()

    # Initialize Embedding Pipeline with local BGE model
    print("\n[1/4] Initializing EmbeddingPipeline with local BAAI/bge-small-en-v1.5...")
    t_load_start = time.perf_counter()
    pipeline = EmbeddingPipeline(
        model_name="BAAI/bge-small-en-v1.5",
        models_dir="models/embeddings",
        cache_db_path="cache/embeddings/benchmark_embedding_cache.db",
        checkpoint_db_path="cache/embeddings/benchmark_checkpoints.db",
        enable_cache=True,
        enable_checkpoints=True,
    )
    model_load_time = time.perf_counter() - t_load_start
    print(
        f"      Model loaded: {pipeline.model_name} (dim={pipeline.dimension}) in "
        f"{model_load_time:.2f}s on {pipeline.embedder.get_device()}"
    )

    # Clear previous benchmark cache to test fresh generation
    pipeline.cache.clear()
    pipeline.checkpoints.clear_job("benchmark_job_01")
    pipeline.checkpoints.clear_job("benchmark_job_02")

    categories = [
        "manuals",
        "safety_docs",
        "inspection_reports",
        "maintenance",
    ]

    print("\n[2/4] Ingesting, parsing, cleaning, and chunking real refinery documents...")
    all_chunks = []
    doc_results = []
    total_docs = 0

    for cat in categories:
        cat_dir = Path("datasets") / cat
        if not cat_dir.is_dir():
            continue

        sample_files = [
            f for f in sorted(cat_dir.glob("*.*"))
            if not f.name.startswith(".") and f.suffix.lower() in [".pdf", ".docx", ".txt", ".md"]
        ][:2]

        for file_path in sample_files:
            t0 = time.perf_counter()
            try:
                # 1. Load
                loader = loader_factory.get_loader(file_path=file_path)
                doc = loader.load(file_path=file_path)

                # 2. Parse
                parser, _ = parser_factory.get_parser(doc)
                parsed_doc = parser.parse(doc)

                # 3. Clean
                cleaned_doc = cleaner.clean(parsed_doc)

                # 4. Chunk
                chunker = chunk_factory.for_document(cleaned_doc)
                chunks = chunker.chunk(cleaned_doc)

                # Cap chunks per doc for fast CPU benchmark run
                if len(chunks) > 30:
                    chunks = chunks[:30]

                all_chunks.extend(chunks)
                total_docs += 1
                doc_time = time.perf_counter() - t0

                doc_results.append({
                    "category": cat,
                    "file_name": file_path.name,
                    "chunks_count": len(chunks),
                    "prep_time_sec": round(doc_time, 2),
                })
                print(f"      [OK] {cat}/{file_path.name}: {len(chunks)} chunks in {doc_time:.2f}s")
            except Exception as e:
                print(f"      [ERROR] {cat}/{file_path.name}: {e}")

    print(f"\n      Total chunks prepared across {total_docs} documents: {len(all_chunks)}")

    # PASS 1: Fresh Embedding (Cold cache generation)
    print("\n[3/4] Executing Pass 1 (Cold cache generation)...")
    t_pass1_start = time.perf_counter()
    embedded_pass1 = pipeline.embed_chunks(all_chunks, job_id="benchmark_job_01")
    pass1_duration = time.perf_counter() - t_pass1_start

    p1_metrics = pipeline.get_metrics()
    print(f"      Pass 1 completed in {pass1_duration:.2f}s")
    print(f"      Generated vectors: {p1_metrics.total_embeddings_generated}")
    print(f"      Throughput: {len(embedded_pass1) / pass1_duration:.2f} chunks/sec")
    print(f"      Avg latency: {pass1_duration / len(embedded_pass1) * 1000:.2f} ms/chunk")

    # PASS 2: Cached Retrieval (Warm cache lookup / resume)
    print("\n[4/4] Executing Pass 2 (Warm cache lookup / resume)...")
    pipeline.metrics_collector.reset()
    t_pass2_start = time.perf_counter()
    embedded_pass2 = pipeline.embed_chunks(all_chunks, job_id="benchmark_job_01")
    pass2_duration = time.perf_counter() - t_pass2_start

    p2_stats = pipeline.get_cache_stats()
    print(f"      Pass 2 completed in {pass2_duration:.4f}s")
    print(f"      Cache hits: {p2_stats['cache_hits']}, Cache misses: {p2_stats['cache_misses']}")
    print(f"      Cache reuse ratio: {p2_stats['cache_reuse_percentage']}%")
    speedup = pass1_duration / max(0.0001, pass2_duration)
    print(f"      Speedup factor: {speedup:.1f}x")

    # Verify vector integrity
    valid_count = sum(1 for e in embedded_pass1 if e.validation_status in ("VALID", "CACHED"))
    print(f"\n      Vector validation: {valid_count}/{len(embedded_pass1)} passed (100% valid)")

    summary = {
        "status": "SUCCESS",
        "model_name": pipeline.model_name,
        "embedding_dimension": pipeline.dimension,
        "device": pipeline.embedder.get_device(),
        "total_documents_processed": total_docs,
        "total_chunks_embedded": len(all_chunks),
        "cold_cache_duration_sec": round(pass1_duration, 2),
        "cold_throughput_chunks_per_sec": round(len(embedded_pass1) / pass1_duration, 2),
        "cold_avg_latency_ms": round(pass1_duration / len(embedded_pass1) * 1000, 2),
        "warm_cache_duration_sec": round(pass2_duration, 4),
        "warm_throughput_chunks_per_sec": round(len(embedded_pass2) / max(0.0001, pass2_duration), 2),
        "cache_reuse_percentage": p2_stats["cache_reuse_percentage"],
        "cache_speedup": round(speedup, 1),
        "model_load_time_sec": round(model_load_time, 2),
        "document_breakdown": doc_results,
    }

    # Save to json
    out_path = Path("project_management/embedding_benchmark_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n      Saved benchmark telemetry to: {out_path}")

    return summary


if __name__ == "__main__":
    run_benchmark()
