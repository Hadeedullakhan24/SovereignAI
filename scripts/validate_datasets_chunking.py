"""Dataset validation and benchmark runner for Milestone 5: Enterprise Chunking Engine.

Executes end-to-end processing:
Loader -> Parser -> Cleaning Pipeline -> Enterprise Chunking Engine
Across real refinery dataset categories:
- manuals
- safety_docs
- inspection_reports
- maintenance
- emails
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Ensure project root is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_engine.chunking import get_chunk_factory
from rag_engine.loaders import LoaderFactory
from rag_engine.parsers import ParserFactory
from rag_engine.preprocessing import CleaningPipeline


def main() -> None:
    loader_factory = LoaderFactory()
    parser_factory = ParserFactory()
    cleaner = CleaningPipeline(strict_token_verification=False)
    chunk_factory = get_chunk_factory()

    categories = [
        "manuals",
        "safety_docs",
        "inspection_reports",
        "maintenance",
        "emails",
    ]

    results: list[dict] = []
    total_docs = 0
    total_chunks = 0
    start_total_time = time.perf_counter()

    for cat in categories:
        cat_dir = Path("datasets") / cat
        if not cat_dir.is_dir():
            continue

        sample_files = [f for f in sorted(cat_dir.glob("*.*")) if not f.name.startswith(".")][:3]

        for file_path in sample_files:
            t0 = time.perf_counter()
            try:
                # 1. Load
                loader = loader_factory.get_loader(file_path=file_path)
                doc = loader.load(file_path=file_path)
                
                # 2. Parse
                parser, _ = parser_factory.get_parser(doc)
                parsed_doc = parser.parse(doc)

                # 3. Clean & Normalize
                cleaned_doc = cleaner.clean(parsed_doc)

                # 4. Adaptive Enterprise Chunking
                chunker = chunk_factory.for_document(cleaned_doc)
                chunks = chunker.chunk(cleaned_doc)

                t1 = time.perf_counter()
                elapsed_ms = round((t1 - t0) * 1000.0, 2)

                total_docs += 1
                total_chunks += len(chunks)

                tok_counts = [c.token_count for c in chunks]
                avg_toks = round(sum(tok_counts) / len(tok_counts), 1) if tok_counts else 0

                res_entry = {
                    "category": cat,
                    "file_name": file_path.name,
                    "file_size_bytes": file_path.stat().st_size,
                    "strategy_used": chunker.strategy_name,
                    "total_chunks": len(chunks),
                    "avg_tokens_per_chunk": avg_toks,
                    "min_tokens": min(tok_counts) if tok_counts else 0,
                    "max_tokens": max(tok_counts) if tok_counts else 0,
                    "table_chunks": sum(1 for c in chunks if c.metadata.is_table_chunk),
                    "list_chunks": sum(1 for c in chunks if c.metadata.is_list_chunk),
                    "elapsed_ms": elapsed_ms,
                    "status": "SUCCESS",
                }
                results.append(res_entry)
                print(f"[{cat.upper()}] {file_path.name}: {len(chunks)} chunks, avg {avg_toks} toks, strat={chunker.strategy_name} ({elapsed_ms} ms)")

            except Exception as exc:
                print(f"[{cat.upper()}] ERROR processing {file_path.name}: {exc}")
                results.append({
                    "category": cat,
                    "file_name": file_path.name,
                    "status": f"FAILED: {exc}",
                })

    total_time_sec = round(time.perf_counter() - start_total_time, 3)
    print(f"\nCompleted: {total_docs} files processed into {total_chunks} retrieval-optimized chunks in {total_time_sec}s.")

    output_path = Path("project_management") / "chunking_benchmark_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "total_documents": total_docs,
                "total_chunks": total_chunks,
                "total_time_seconds": total_time_sec,
                "throughput_chunks_per_sec": round(total_chunks / max(0.001, total_time_sec), 1),
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"Benchmark results saved to {output_path}")


if __name__ == "__main__":
    main()
