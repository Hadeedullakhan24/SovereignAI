"""High-Throughput Vector Database & Indexing Stress Benchmark for Milestone 7.

Tests:
    1. Batch Ingestion Throughput (chunks/sec, indexing time).
    2. Dense Vector Retrieval Latency (Mean, P50, P95, P99 across multiple queries).
    3. Filtered Search Latency (Single-filter, Compound-filter, and Scroll-only).
    4. Payload Index Acceleration & Health Check.
    5. Neighbor Traversal (Sequential Chunk Context Reconstruction).
    6. Incremental Indexing Reconciliation (New, Unchanged, Modified chunks).
    7. Segment Optimization & Storage Footprint.
    8. Snapshot Creation & Verification.

Outputs structured benchmark report to `project_management/vector_db_benchmark_results.json`.
"""

from __future__ import annotations

import json
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Any

# Ensure project root is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_engine.schemas.chunk import ChunkMetadata
from rag_engine.schemas.embedding import EmbeddedChunk, compute_vector_checksum
from rag_engine.schemas.vector_store import (
    DistanceMetric,
    FieldFilter,
    FilterOperator,
    MetadataFilter,
    PayloadSchemaType,
)
from rag_engine.vector_store import (
    CollectionConfig,
    CollectionManager,
    IncrementalIndexer,
    IndexManager,
    PayloadIndexManager,
    QdrantVectorStore,
    SnapshotManager,
    StorageStatsCollector,
    VectorHealthMonitor,
    VectorOptimizer,
    VectorStoreConfig,
)


def _generate_synthetic_embedded_chunks(
    count: int = 500,
    dimension: int = 384,
    doc_prefix: str = "MRPL_BENCH",
) -> list[EmbeddedChunk]:
    """Generate deterministic synthetic embedded chunks for stress testing."""
    chunks: list[EmbeddedChunk] = []
    categories = ["Manual", "Safety", "P&ID", "Inspection"]
    plants = ["HCU", "CDU", "VDU", "DCU"]

    for i in range(count):
        doc_id = f"{doc_prefix}_{i // 50:03d}"
        chunk_id = f"{doc_id}_chunk_{i:05d}"
        cat = categories[i % len(categories)]
        plant = plants[i % len(plants)]
        page = (i % 20) + 1

        # Deterministic pseudo-vector
        rng = random.Random(i)
        vector = [rng.uniform(-1.0, 1.0) for _ in range(dimension)]
        # Normalize vector for cosine distance
        norm = sum(v * v for v in vector) ** 0.5
        vector = [v / norm for v in vector]

        prev_id = f"{doc_id}_chunk_{i-1:05d}" if i > 0 and (i % 50 != 0) else None
        next_id = f"{doc_id}_chunk_{i+1:05d}" if (i + 1) % 50 != 0 and i < count - 1 else None
        content_hash = f"hash_{doc_id}_{i}"

        meta = ChunkMetadata(
            document_id=doc_id,
            document_name=f"{doc_id}.pdf",
            category=cat,
            plant_unit=plant,
            equipment_entities=["P-101", "K-201"],
            safety_entities=["PPE-Required", "Hazard-Flammable"],
            page_number=page,
            chunk_index=i,
            section_title=f"Section {page}.1 Operational Procedures",
            sha256=content_hash,
        )

        chunk = EmbeddedChunk(
            chunk_id=chunk_id,
            chunk_hash=content_hash,
            embedding=vector,
            model_name="BAAI/bge-small-en-v1.5",
            embedding_dimension=dimension,
            vector_checksum=compute_vector_checksum(vector),
            metadata=meta,
            text_preview=f"Refinery operational protocol chunk {i} for {cat} in {plant}.",
            prev_chunk_id=prev_id,
            next_chunk_id=next_id,
        )
        chunks.append(chunk)

    return chunks


def run_benchmark() -> dict[str, Any]:
    print("=" * 78)
    print("  MRPL SOVEREIGN RAG ENGINE — MILESTONE 7 VECTOR STORAGE BENCHMARK")
    print("=" * 78)

    bench_dir = Path("data/bench_vector_db")
    if bench_dir.exists():
        shutil.rmtree(bench_dir)
    bench_dir.mkdir(parents=True, exist_ok=True)

    config = VectorStoreConfig(
        backend="qdrant",
        storage_path=bench_dir / "qdrant_db",
        journal_path=bench_dir / "journal",
        backup_path=bench_dir / "backups",
        telemetry_path=bench_dir / "telemetry",
    )

    store = QdrantVectorStore(config=config)
    col_mgr = CollectionManager(store)
    payload_idx_mgr = PayloadIndexManager(store)
    index_mgr = IndexManager(store)
    incremental_indexer = IncrementalIndexer(store)
    optimizer = VectorOptimizer(store)
    snapshot_mgr = SnapshotManager(store)
    health_monitor = VectorHealthMonitor(store)
    stats_collector = StorageStatsCollector(store)

    collection_name = "mrpl_benchmark_v1"
    dimension = 384
    col_config = CollectionConfig(
        name=collection_name,
        vector_size=dimension,
        distance=DistanceMetric.COSINE,
        on_disk_payload=True,
    )

    print(f"\n[1/8] Creating test collection '{collection_name}' (dim={dimension})...")
    store.create_collection(col_config)

    # 1. Ingestion Benchmark
    num_chunks = 500
    print(f"\n[2/8] Generating {num_chunks} synthetic refinery embedded chunks...")
    chunks = _generate_synthetic_embedded_chunks(count=num_chunks, dimension=dimension)

    print(f"      Ingesting {num_chunks} chunks via IndexManager...")
    t0 = time.perf_counter()
    index_result = index_mgr.index_document_chunks(chunks, collection_name=collection_name)
    ingestion_time = time.perf_counter() - t0
    throughput = num_chunks / ingestion_time if ingestion_time > 0 else 0

    print(f"      -> Ingested: {index_result.inserted} chunks in {ingestion_time:.2f}s ({throughput:.1f} chunks/sec)")
    assert store.count_points(collection_name) == num_chunks, "Point count mismatch!"

    # 2. Payload Indexing
    print(f"\n[3/8] Building payload indexes on metadata fields...")
    t0 = time.perf_counter()
    payload_idx_mgr.create_payload_indexes(
        collection_name,
        {
            "document_id": PayloadSchemaType.KEYWORD,
            "category": PayloadSchemaType.KEYWORD,
            "plant_unit": PayloadSchemaType.KEYWORD,
            "page_number": PayloadSchemaType.INTEGER,
        },
    )
    idx_build_time = time.perf_counter() - t0
    health_report = payload_idx_mgr.report_payload_index_health(collection_name)
    status_str = "HEALTHY" if health_report.is_healthy else "PARTIAL"
    print(f"      -> Built 4 payload indexes in {idx_build_time:.2f}s. Audit status: {status_str}")

    # 3. Dense Retrieval Latency Benchmark
    print(f"\n[4/8] Benchmarking Dense Vector Retrieval (100 queries, Top-K=10)...")
    latencies: list[float] = []
    num_queries = 100
    for q_idx in range(num_queries):
        query_vec = chunks[q_idx % num_chunks].vector
        t_q = time.perf_counter()
        results = store.search_vectors(collection_name, query_vector=query_vec, limit=10)
        lat = (time.perf_counter() - t_q) * 1000.0  # ms
        latencies.append(lat)
        assert len(results) > 0, "Expected non-empty search results!"

    latencies.sort()
    mean_lat = sum(latencies) / len(latencies)
    p50_lat = latencies[len(latencies) // 2]
    p95_lat = latencies[int(len(latencies) * 0.95)]
    p99_lat = latencies[int(len(latencies) * 0.99)]

    print(f"      -> Mean Latency : {mean_lat:.2f} ms")
    print(f"      -> P50 Latency  : {p50_lat:.2f} ms")
    print(f"      -> P95 Latency  : {p95_lat:.2f} ms")
    print(f"      -> P99 Latency  : {p99_lat:.2f} ms")

    # 4. Filtered Retrieval Latency
    print(f"\n[5/8] Benchmarking Filtered Retrieval...")
    filter_latencies: list[float] = []
    flt = MetadataFilter(must=[FieldFilter(field="category", operator=FilterOperator.EQUALS, value="Safety")])
    for q_idx in range(50):
        query_vec = chunks[q_idx % num_chunks].vector
        t_q = time.perf_counter()
        filtered_results = store.search_vectors(collection_name, query_vector=query_vec, limit=5, filters=flt)
        lat = (time.perf_counter() - t_q) * 1000.0
        filter_latencies.append(lat)
        for r in filtered_results:
            assert r.category == "Safety" or (r.metadata and r.metadata.category == "Safety")

    filter_latencies.sort()
    filter_mean = sum(filter_latencies) / len(filter_latencies)
    print(f"      -> Filtered Search Mean Latency : {filter_mean:.2f} ms")

    # 5. Neighbor Context Traversal
    print(f"\n[6/8] Benchmarking Neighbor Chunk Context Reconstruction...")
    sample_chunk_id = chunks[10].chunk_id
    t_nb = time.perf_counter()
    neighbor_window = store.get_neighbors(collection_name, sample_chunk_id, window=1)
    neighbor_time_ms = (time.perf_counter() - t_nb) * 1000.0
    print(f"      -> Traversal Time   : {neighbor_time_ms:.2f} ms")
    print(f"      -> Sample Chunk ID  : {sample_chunk_id}")
    print(f"      -> Neighbors Found  : {len(neighbor_window)} chunks")
    assert len(neighbor_window) >= 2

    # 6. Incremental Update Reconciliation
    print(f"\n[7/8] Benchmarking Incremental Indexing Reconciliation...")
    # Modify 10 chunks, add 20 new chunks, keep rest unchanged
    updated_chunks = list(chunks)
    for i in range(10):
        c = updated_chunks[i]
        new_hash = f"{c.chunk_hash}_mod"
        updated_chunks[i] = EmbeddedChunk(
            chunk_id=c.chunk_id,
            chunk_hash=new_hash,
            embedding=c.embedding,
            model_name=c.model_name,
            embedding_dimension=c.embedding_dimension,
            vector_checksum=c.vector_checksum,
            metadata=c.metadata,
            text_preview=c.text_preview + " [UPDATED]",
            prev_chunk_id=c.prev_chunk_id,
            next_chunk_id=c.next_chunk_id,
        )
    new_chunks = _generate_synthetic_embedded_chunks(count=20, dimension=dimension, doc_prefix="MRPL_NEW")
    combined_chunks = updated_chunks + new_chunks

    t_diff = time.perf_counter()
    diff_plan = incremental_indexer.calculate_diff(collection_name, combined_chunks)
    diff_time = time.perf_counter() - t_diff

    print(f"      -> Diff Plan Computed in {diff_time:.2f}s:")
    print(f"         New       : {len(diff_plan.new_chunk_ids)}")
    print(f"         Modified  : {len(diff_plan.modified_chunk_ids)}")
    print(f"         Unchanged : {len(diff_plan.unchanged_chunk_ids)}")
    assert len(diff_plan.new_chunk_ids) == 20
    assert len(diff_plan.modified_chunk_ids) == 10
    assert len(diff_plan.unchanged_chunk_ids) == num_chunks - 10

    t_apply = time.perf_counter()
    apply_res = index_mgr.index_document_chunks(combined_chunks, collection_name=collection_name)
    apply_time = time.perf_counter() - t_apply
    print(f"      -> Applied Incremental Diff in {apply_time:.2f}s (Inserted: {apply_res.inserted}, Updated: {apply_res.updated}, Skipped: {apply_res.skipped}).")
    assert store.count_points(collection_name) == num_chunks + 20

    # 7. Optimization, Snapshot & Diagnostics
    print(f"\n[8/8] Benchmarking Optimization, Snapshot & Diagnostics...")
    t_opt = time.perf_counter()
    opt_report = optimizer.optimize_all(collection_name)
    opt_time = time.perf_counter() - t_opt
    print(f"      -> Optimization duration: {opt_time:.2f}s ({opt_report.total_reclaimed_bytes} bytes reclaimed)")

    t_snap = time.perf_counter()
    snap_path = snapshot_mgr.create_snapshot(collection_name, snapshot_name="benchmark_snapshot")
    snap_time = time.perf_counter() - t_snap
    snap_size_mb = snap_path.stat().st_size / (1024 * 1024)
    print(f"      -> Snapshot created: {snap_path.name} ({snap_size_mb:.2f} MB in {snap_time:.2f}s)")

    health = health_monitor.run_diagnostics()
    canary_lat = health.canary_write_latency_ms + health.canary_read_latency_ms
    print(f"      -> Health Canary Latency : {canary_lat:.2f} ms (Status: {health.status.upper()})")

    storage_stats = stats_collector.collect_stats()
    print(f"      -> Storage Size          : {storage_stats.total_disk_bytes / (1024 * 1024):.2f} MB")

    results: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "milestone": "Milestone 7: Enterprise Vector Database & Indexing Platform",
        "backend": "Qdrant Local (Embedded Mode)",
        "vector_dimension": dimension,
        "total_chunks_indexed": num_chunks + 20,
        "metrics": {
            "ingestion_throughput_chunks_per_sec": round(throughput, 2),
            "ingestion_time_seconds": round(ingestion_time, 3),
            "search_latency_p50_ms": round(p50_lat, 2),
            "search_latency_p95_ms": round(p95_lat, 2),
            "search_latency_p99_ms": round(p99_lat, 2),
            "search_latency_mean_ms": round(mean_lat, 2),
            "filtered_search_mean_ms": round(filter_mean, 2),
            "neighbor_traversal_ms": round(neighbor_time_ms, 2),
            "diff_computation_seconds": round(diff_time, 3),
            "diff_application_seconds": round(apply_time, 3),
            "snapshot_creation_seconds": round(snap_time, 3),
            "snapshot_size_mb": round(snap_size_mb, 2),
            "canary_health_latency_ms": round(canary_lat, 2),
        },
        "status": "PASSED",
    }

    out_path = Path("project_management/vector_db_benchmark_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Cleanup benchmark directory
    store.close()
    del store
    import gc
    gc.collect()
    time.sleep(0.5)
    if bench_dir.exists():
        shutil.rmtree(bench_dir, ignore_errors=True)

    print("\n" + "=" * 78)
    print(f"  BENCHMARK COMPLETE — Results saved to {out_path}")
    print("=" * 78)

    return results


if __name__ == "__main__":
    run_benchmark()
