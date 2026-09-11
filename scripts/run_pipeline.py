"""Run Pipeline Entrypoint — Master Ingestion and Indexing Orchestrator."""

import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.ingest_real_documents import run_ingestion

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Sovereign RAG Ingestion Pipeline")
    parser.add_argument("--dataset-dir", default="datasets", help="Path to datasets folder")
    parser.add_argument("--max-per-cat", type=int, default=None, help="Max docs per category")
    parser.add_argument("--collection", default="mrpl_docs_v1", help="Collection name")
    args = parser.parse_args()

    run_ingestion(
        dataset_dir=args.dataset_dir,
        max_docs_per_category=args.max_per_cat,
        collection_name=args.collection,
    )
