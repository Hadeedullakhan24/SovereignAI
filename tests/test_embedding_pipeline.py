"""Comprehensive test suite for Milestone 6: Offline Embedding Pipeline.

Tests:
1. BaseEmbedder contract and DeterministicTestEmbedder.
2. Local model loading and registry resolution.
3. Persistent SQLite vector caching (get, put, get_batch, put_batch, stats).
4. Checkpoint & Resume manager (interrupted run recovery).
5. VectorValidator (valid, NaN/Inf rejection, dimension mismatch, L2 normalization, checksums).
6. Batch processing and adaptive batching.
7. Model switching (dynamic embedder switching).
8. End-to-end EmbeddingPipeline with real Chunk objects.
9. Corrupted model detection and error handling.
10. Thread safety across concurrent embedding invocations.
"""

from __future__ import annotations

import math
from pathlib import Path
import threading
import time
import pytest

from rag_engine.embeddings.checkpoint_manager import CheckpointManager
from rag_engine.embeddings.embedding_cache import EmbeddingCache
from rag_engine.embeddings.embedding_factory import EmbeddingFactory
from rag_engine.embeddings.embedding_pipeline import EmbeddingPipeline
from rag_engine.embeddings.embedding_registry import EmbeddingRegistry
from rag_engine.embeddings.exceptions import (
    ModelIntegrityError,
    ModelNotFoundError,
    VectorValidationError,
)
from rag_engine.embeddings.local_embedder import (
    DeterministicTestEmbedder,
    LocalHuggingFaceEmbedder,
    resolve_model_name,
)
from rag_engine.embeddings.vector_validator import VectorValidator
from rag_engine.schemas.chunk import Chunk, ChunkHierarchy, ChunkMetadata
from rag_engine.schemas.embedding import (
    EmbeddedChunk,
    compute_vector_checksum,
)


@pytest.fixture
def temp_cache_db(tmp_path: Path) -> Path:
    return tmp_path / "test_cache.db"


@pytest.fixture
def temp_checkpoint_db(tmp_path: Path) -> Path:
    return tmp_path / "test_checkpoints.db"


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    chunks = []
    texts = [
        "Centrifugal pump P-101 operates at 150 bar pressure.",
        "Refinery crude distillation column C-201 temperature is 380 deg C.",
        "Emergency shutdown valve XV-301 inspection according to OISD-105.",
        "Catalytic cracking unit fluid velocity must not exceed 25 m/s.",
        "Safety relief valve PRV-502 set pressure is calibrated to 18.5 bar.",
    ]
    for idx, text in enumerate(texts):
        ch = Chunk(
            chunk_id=f"chk_doc1_p1_{idx:04d}_hash{idx}",
            chunk_hash=f"hash_{idx}_{len(text)}",
            content=text,
            token_count=len(text.split()),
            character_count=len(text),
            hierarchy=ChunkHierarchy(document_id="doc_refinery_01", chunk_index=idx),
            metadata=ChunkMetadata(
                document_name="Refinery Operating Manual",
                equipment_tags=["P-101", "C-201", "XV-301"][idx % 3 : (idx % 3) + 1],
                operating_parameters={"pressure": "150 bar"},
                safety_standards=["OISD-105"],
            ),
        )
        chunks.append(ch)
    return chunks


# =====================================================================
# 1. BaseEmbedder & Deterministic Test Embedder
# =====================================================================

def test_deterministic_test_embedder_properties():
    embedder = DeterministicTestEmbedder(model_name="test-embedder-384", dimension=384)
    assert embedder.get_dimension() == 384
    assert embedder.get_model_name() == "test-embedder-384"
    assert embedder.is_normalized() is True

    # Determinism check
    text = "Crude distillation unit safety procedure"
    vec1 = embedder.embed(text)
    vec2 = embedder.embed(text)
    assert len(vec1) == 384
    assert vec1 == vec2

    # L2-normalization check
    norm = math.sqrt(sum(v * v for v in vec1))
    assert pytest.approx(norm, rel=1e-3) == 1.0


def test_embed_batch_deterministic():
    embedder = DeterministicTestEmbedder(model_name="test-embedder-384", dimension=384)
    texts = ["Refinery unit 1", "Safety SOP 2", "Valve maintenance 3"]
    batch_vecs = embedder.embed_batch(texts)
    assert len(batch_vecs) == 3
    for v in batch_vecs:
        assert len(v) == 384
        norm = math.sqrt(sum(x * x for x in v))
        assert pytest.approx(norm, rel=1e-3) == 1.0


def test_embed_empty_text_raises_error():
    embedder = DeterministicTestEmbedder()
    with pytest.raises(ValueError, match="must not be empty"):
        embedder.embed("")


# =====================================================================
# 2. Embedding Registry & Model Name Resolution
# =====================================================================

def test_model_resolution_and_aliases():
    assert resolve_model_name("bge-small") == "BAAI/bge-small-en-v1.5"
    assert resolve_model_name("e5-base") == "intfloat/e5-base-v2"
    assert resolve_model_name("bge-base-en-v1.5") == "BAAI/bge-base-en-v1.5"


def test_embedding_registry_lookup():
    registry = EmbeddingRegistry.get_instance()
    spec = registry.get_specification("bge-small")
    assert spec["dimension"] == 384
    assert "local_folder" in spec

    spec_e5 = registry.get_specification("intfloat/e5-base-v2")
    assert spec_e5["dimension"] == 768
    assert spec_e5["passage_prefix"] == "passage: "


def test_embedding_registry_unregistered_model():
    registry = EmbeddingRegistry.get_instance()
    with pytest.raises(ModelNotFoundError):
        registry.get_specification("non-existent-unknown-model-xyz")


# =====================================================================
# 3. Vector Validator
# =====================================================================

def test_vector_validator_success():
    validator = VectorValidator(strict=True)
    embedder = DeterministicTestEmbedder(dimension=384)
    vec = embedder.embed("P-101 vibration test")
    assert validator.validate_vector(vec, expected_dimension=384, check_normalized=True)


def test_vector_validator_dimension_mismatch():
    validator = VectorValidator(strict=True)
    vec = [0.1] * 100
    with pytest.raises(VectorValidationError, match="Dimension mismatch"):
        validator.validate_vector(vec, expected_dimension=384)


def test_vector_validator_nan_rejection():
    validator = VectorValidator(strict=True)
    vec = [0.0] * 384
    vec[10] = float("nan")
    with pytest.raises(VectorValidationError, match="contains NaN"):
        validator.validate_vector(vec, expected_dimension=384)


def test_vector_validator_inf_rejection():
    validator = VectorValidator(strict=True)
    vec = [0.0] * 384
    vec[50] = float("inf")
    with pytest.raises(VectorValidationError, match="contains Inf"):
        validator.validate_vector(vec, expected_dimension=384)


def test_vector_validator_unnormalized_rejection():
    validator = VectorValidator(strict=True)
    vec = [2.0] * 384  # Norm is significantly greater than 1.0
    with pytest.raises(VectorValidationError, match="not L2-normalized"):
        validator.validate_vector(vec, expected_dimension=384, check_normalized=True)


def test_vector_checksum_verification():
    validator = VectorValidator(strict=True)
    vec = [0.1, -0.2, 0.5, 0.8]
    checksum = compute_vector_checksum(vec)
    assert len(checksum) == 64
    assert validator.verify_checksum(vec, checksum)

    with pytest.raises(VectorValidationError, match="Checksum mismatch"):
        validator.verify_checksum(vec, "0" * 64)


# =====================================================================
# 4. Persistent Embedding Cache
# =====================================================================

def test_cache_put_get_hit_miss(temp_cache_db: Path):
    cache = EmbeddingCache(db_path=temp_cache_db)
    model = "BAAI/bge-small-en-v1.5"
    ch_hash = "abc123hash"
    vec = [0.123] * 384

    # Initial miss
    assert cache.get(model, ch_hash) is None

    # Put and hit
    cache.put(model, ch_hash, vec, dimension=384)
    cached = cache.get(model, ch_hash)
    assert cached is not None
    assert len(cached) == 384
    assert pytest.approx(cached[0], rel=1e-5) == 0.123

    stats = cache.get_stats()
    assert stats["cache_hits"] == 1
    assert stats["cache_misses"] == 1
    assert stats["cache_reuse_percentage"] == 50.0


def test_cache_batch_operations(temp_cache_db: Path):
    cache = EmbeddingCache(db_path=temp_cache_db)
    model = "BAAI/bge-small-en-v1.5"
    hash_to_vec = {
        "hash_1": [0.1] * 384,
        "hash_2": [0.2] * 384,
        "hash_3": [0.3] * 384,
    }

    cache.put_batch(model, hash_to_vec, dimension=384)
    retrieved = cache.get_batch(model, ["hash_1", "hash_2", "missing_hash"])
    assert len(retrieved) == 2
    assert "hash_1" in retrieved
    assert "hash_2" in retrieved
    assert "missing_hash" not in retrieved


# =====================================================================
# 5. Checkpoint and Resume
# =====================================================================

def test_checkpoint_resume_tracking(temp_checkpoint_db: Path, sample_chunks: list[Chunk]):
    mgr = CheckpointManager(db_path=temp_checkpoint_db)
    job_id = "job_indexing_run_01"

    # Initially all unprocessed
    unprocessed = mgr.filter_unprocessed_chunks(job_id, sample_chunks)
    assert len(unprocessed) == len(sample_chunks)

    # Mark first 2 completed
    mgr.record_completed(job_id, sample_chunks[0].chunk_id, sample_chunks[0].chunk_hash)
    mgr.record_completed(job_id, sample_chunks[1].chunk_id, sample_chunks[1].chunk_hash)

    completed_ids = mgr.get_completed_chunk_ids(job_id)
    assert len(completed_ids) == 2
    assert sample_chunks[0].chunk_id in completed_ids
    assert sample_chunks[1].chunk_id in completed_ids

    # Resuming filters out the 2 completed chunks
    remaining = mgr.filter_unprocessed_chunks(job_id, sample_chunks)
    assert len(remaining) == 3
    assert remaining[0].chunk_id == sample_chunks[2].chunk_id


# =====================================================================
# 6. End-to-End Embedding Pipeline
# =====================================================================

def test_embedding_pipeline_execution(
    temp_cache_db: Path, temp_checkpoint_db: Path, sample_chunks: list[Chunk]
):
    pipeline = EmbeddingPipeline(
        model_name="test-embedder-384",
        cache_db_path=temp_cache_db,
        checkpoint_db_path=temp_checkpoint_db,
        enable_cache=True,
        enable_checkpoints=True,
    )

    embedded = pipeline.embed_chunks(sample_chunks, job_id="test_run_1")
    assert len(embedded) == len(sample_chunks)

    for ech, orig in zip(embedded, sample_chunks):
        assert isinstance(ech, EmbeddedChunk)
        assert ech.chunk_id == orig.chunk_id
        assert ech.chunk_hash == orig.chunk_hash
        assert len(ech.embedding) == 384
        assert ech.validation_status == "VALID"
        assert ech.metadata.document_name == "Refinery Operating Manual"
        assert len(ech.vector_checksum) == 64

    # Second run should hit cache 100%
    embedded_run2 = pipeline.embed_chunks(sample_chunks, job_id="test_run_1")
    assert len(embedded_run2) == len(sample_chunks)
    for ech in embedded_run2:
        assert ech.validation_status == "CACHED"

    metrics = pipeline.get_metrics()
    assert metrics.cache_hits > 0
    assert metrics.total_chunks_processed == len(sample_chunks) * 2


def test_dynamic_model_switching(temp_cache_db: Path, temp_checkpoint_db: Path):
    pipeline = EmbeddingPipeline(
        model_name="test-embedder-384",
        cache_db_path=temp_cache_db,
        checkpoint_db_path=temp_checkpoint_db,
    )
    assert pipeline.dimension == 384

    # Switch to 768-dim test embedder
    pipeline.switch_model("test-embedder-768")
    assert pipeline.dimension == 768
    assert pipeline.model_name == "test-embedder-768"

    ch = Chunk(
        chunk_id="chk_sw_1",
        chunk_hash="sw_hash_1",
        content="Testing dynamic model switching",
        hierarchy=ChunkHierarchy(document_id="doc1", chunk_index=0),
    )
    ech = pipeline.embed_chunk(ch)
    assert len(ech.embedding) == 768


# =====================================================================
# 7. Concurrency & Thread Safety
# =====================================================================

def test_concurrent_embedding_and_cache(temp_cache_db: Path, temp_checkpoint_db: Path):
    pipeline = EmbeddingPipeline(
        model_name="test-embedder-384",
        cache_db_path=temp_cache_db,
        checkpoint_db_path=temp_checkpoint_db,
    )

    def worker(worker_id: int):
        chunks = [
            Chunk(
                chunk_id=f"chk_t_{worker_id}_{i}",
                chunk_hash=f"hash_t_{worker_id}_{i}",
                content=f"Worker {worker_id} chunk text {i}",
                hierarchy=ChunkHierarchy(document_id=f"doc_{worker_id}", chunk_index=i),
            )
            for i in range(10)
        ]
        res = pipeline.embed_chunks(chunks)
        assert len(res) == 10

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    stats = pipeline.get_cache_stats()
    assert stats["total_entries"] == 100


# =====================================================================
# 8. Real Local Model Inference & Corrupted Model Detection
# =====================================================================

def test_real_local_bge_small_model_inference(temp_cache_db: Path, temp_checkpoint_db: Path):
    model_dir = Path("models/embeddings/bge-small-en-v1.5")
    if not model_dir.exists() or not (model_dir / "config.json").exists():
        pytest.skip("Local bge-small model weights not downloaded yet.")

    embedder = LocalHuggingFaceEmbedder(
        model_name="BAAI/bge-small-en-v1.5",
        models_dir="models/embeddings",
        device="cpu",
    )
    assert embedder.get_dimension() == 384
    assert embedder.get_model_name() == "BAAI/bge-small-en-v1.5"

    vec = embedder.embed("MRPL refinery crude distillation unit safety guidelines")
    assert len(vec) == 384
    norm = math.sqrt(sum(x * x for x in vec))
    assert pytest.approx(norm, rel=1e-3) == 1.0


def test_corrupted_model_detection(tmp_path: Path):
    corrupt_dir = tmp_path / "corrupt_model"
    corrupt_dir.mkdir(parents=True, exist_ok=True)
    # Empty directory missing config.json and weights
    with pytest.raises((ModelNotFoundError, ModelIntegrityError)):
        LocalHuggingFaceEmbedder(
            model_name="unknown-model",
            models_dir=corrupt_dir,
        )

