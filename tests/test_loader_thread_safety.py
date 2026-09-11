"""Concurrency and thread-safety tests for document loaders."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pytest

from rag_engine.loaders.loader_factory import global_loader_factory
from rag_engine.loaders.loader_metrics import global_metrics
from rag_engine.loaders.processing_context import ProcessingContext


def test_concurrent_loader_execution(tmp_path):
    # Create 20 test files of mixed types
    files = []
    for i in range(10):
        txt_file = tmp_path / f"thread_test_{i}.txt"
        txt_file.write_text(f"Thread safe text content for worker {i}\n", encoding="utf-8")
        files.append(txt_file)

        csv_file = tmp_path / f"thread_data_{i}.csv"
        csv_file.write_text(f"id,val\n{i},{i*10}\n", encoding="utf-8")
        files.append(csv_file)

    results = []

    def load_task(path):
        ctx = ProcessingContext(category="concurrency_test", subcategory=path.suffix)
        return global_loader_factory.load(path, context=ctx)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(load_task, f) for f in files]
        for fut in futures:
            doc = fut.result()
            results.append(doc)

    assert len(results) == 20
    assert all(d.loading_status == "SUCCESS" for d in results)
    assert all(d.metadata.category == "concurrency_test" for d in results)
