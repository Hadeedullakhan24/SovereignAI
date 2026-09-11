"""Comprehensive Test Suite for Milestone 7 Enterprise Vector Storage Platform.

Tests all 10 enterprise components, Qdrant Local driver, metadata filtering,
multi-collection routing, versioning, transactions, snapshots, concurrency,
and backward compatibility.
"""

from __future__ import annotations

from pathlib import Path
import threading
import time
import pytest

from rag_engine.interfaces.base_vector_store import BaseVectorStore
from rag_engine.schemas.chunk import Chunk, ChunkMetadata
from rag_engine.schemas.embedding import EmbeddedChunk, EmbeddingVector, compute_vector_checksum
from rag_engine.schemas.vector_store import (
    DistanceMetric,
    FieldFilter,
    FilterOperator,
    MetadataFilter,
    PayloadSchemaType,
)
from rag_engine.vector_store.collection_config import CollectionConfig, HNSWConfig, VectorStoreConfig
from rag_engine.vector_store.collection_manager import CollectionManager
from rag_engine.vector_store.collection_router import CollectionRouter
from rag_engine.vector_store.collection_version_manager import CollectionVersionManager
from rag_engine.vector_store.exceptions import (
    CollectionAlreadyExistsError,
    CollectionNotFoundError,
    PayloadValidationError,
    SnapshotError,
    VectorDimensionMismatchError,
    VectorQualityError,
    VersioningError,
)
from rag_engine.vector_store.incremental_indexer import IncrementalIndexer
from rag_engine.vector_store.index_manager import IndexManager
from rag_engine.vector_store.metadata_serializer import MetadataSerializer
from rag_engine.vector_store.payload_index_manager import PayloadIndexManager
from rag_engine.vector_store.payload_validator import PayloadValidator
from rag_engine.vector_store.qdrant_store import QdrantVectorStore
from rag_engine.vector_store.schema_validator import SchemaValidator
from rag_engine.vector_store.snapshot_manager import SnapshotManager
from rag_engine.vector_store.storage_stats import StorageStatsCollector
from rag_engine.vector_store.transaction_manager import TransactionManager
from rag_engine.vector_store.vector_factory import VectorFactory
from rag_engine.vector_store.vector_lifecycle_manager import VectorLifecycleManager
from rag_engine.vector_store.vector_optimizer import VectorOptimizer
from rag_engine.vector_store.vector_registry import VectorRegistry
from rag_engine.vector_store.vector_repository import VectorRepository


@pytest.fixture
def temp_store(tmp_path: Path) -> QdrantVectorStore:
    """Fixture providing an isolated local filesystem Qdrant store."""
    cfg = VectorStoreConfig(
        backend="qdrant",
        storage_path=tmp_path / "qdrant_db",
        journal_path=tmp_path / "journal",
        backup_path=tmp_path / "backups",
        telemetry_path=tmp_path / "telemetry",
    )
    store = QdrantVectorStore(config=cfg)
    yield store
    store.close()


def _make_sample_chunk(
    chunk_id: str,
    doc_id: str = "DOC_001",
    cat: str = "Manual",
    plant: str = "HCU",
    equip: list[str] = None,
    safety: list[str] = None,
    vec: list[float] = None,
    prev_id: str = None,
    next_id: str = None,
    content_hash: str = "hash_123",
) -> EmbeddedChunk:
    """Helper to construct a valid EmbeddedChunk for testing."""
    vector = vec or [0.1, 0.2, 0.3, 0.4]
    meta = ChunkMetadata(
        document_id=doc_id,
        document_name=f"{doc_id}.pdf",
        category=cat,
        plant_unit=plant,
        equipment_entities=equip if equip is not None else ["P-101"],
        safety_entities=safety if safety is not None else [],
        page_number=1,
        chunk_index=0,
        section_title="Operating Procedures",
        sha256=content_hash,
    )
    return EmbeddedChunk(
        chunk_id=chunk_id,
        chunk_hash=content_hash,
        embedding=vector,
        model_name="BAAI/bge-small-en-v1.5",
        embedding_dimension=len(vector),
        vector_checksum=compute_vector_checksum(vector),
        metadata=meta,
        text_preview=f"Preview for {chunk_id}",
        prev_chunk_id=prev_id,
        next_chunk_id=next_id,
    )


# =============================================================================
# 1. Qdrant Driver Core Tests
# =============================================================================

def test_qdrant_store_collection_lifecycle(temp_store: QdrantVectorStore):
    """Test creation, existence check, stats, and deletion of collections."""
    cfg = CollectionConfig(name="test_col_01", vector_size=4, distance=DistanceMetric.COSINE)
    assert not temp_store.collection_exists("test_col_01")

    assert temp_store.create_collection(cfg) is True
    assert temp_store.collection_exists("test_col_01") is True

    # Duplicate creation raises error
    with pytest.raises(CollectionAlreadyExistsError):
        temp_store.create_collection(cfg)

    # Check stats
    stats = temp_store.get_collection_stats("test_col_01")
    assert stats.name == "test_col_01"
    assert stats.vector_count == 0

    # Delete collection
    assert temp_store.delete_collection("test_col_01") is True
    assert not temp_store.collection_exists("test_col_01")

    with pytest.raises(CollectionNotFoundError):
        temp_store.get_collection_stats("test_col_01")


def test_qdrant_store_upsert_and_retrieve(temp_store: QdrantVectorStore):
    """Test upserting chunks, retrieving by ID, and finding neighbors."""
    cfg = CollectionConfig(name="test_crud", vector_size=4)
    temp_store.create_collection(cfg)

    chk1 = _make_sample_chunk("chk_001", next_id="chk_002")
    chk2 = _make_sample_chunk("chk_002", prev_id="chk_001", next_id="chk_003")
    chk3 = _make_sample_chunk("chk_003", prev_id="chk_002")

    res = temp_store.upsert_chunks("test_crud", [chk1, chk2, chk3])
    assert res.inserted == 3

    # Retrieve single chunk
    retrieved = temp_store.get_chunk("test_crud", "chk_002")
    assert retrieved is not None
    assert retrieved.chunk_id == "chk_002"
    assert retrieved.metadata.document_id == "DOC_001"

    # Retrieve neighbors
    neighbors = temp_store.get_neighbors("test_crud", "chk_002", window=1)
    neighbor_ids = [c.chunk_id for c in neighbors]
    assert "chk_001" in neighbor_ids
    assert "chk_002" in neighbor_ids
    assert "chk_003" in neighbor_ids


def test_qdrant_store_search_with_filter(temp_store: QdrantVectorStore):
    """Test vector similarity search with payload metadata filters."""
    cfg = CollectionConfig(name="test_search", vector_size=4)
    temp_store.create_collection(cfg)

    c1 = _make_sample_chunk("chk_hcu_1", plant="HCU", equip=["P-101", "K-102"], vec=[1.0, 0.0, 0.0, 0.0])
    c2 = _make_sample_chunk("chk_dhu_2", plant="DHU", equip=["P-201"], vec=[0.9, 0.1, 0.0, 0.0])
    c3 = _make_sample_chunk("chk_hcu_3", plant="HCU", equip=["E-301"], vec=[0.8, 0.2, 0.0, 0.0])

    temp_store.upsert_chunks("test_search", [c1, c2, c3])

    # 1. Unfiltered query
    results = temp_store.search_vectors("test_search", [1.0, 0.0, 0.0, 0.0], limit=3)
    assert len(results) == 3
    assert results[0].chunk_id == "chk_hcu_1"

    # 2. Filter by plant_unit EQUALS HCU
    flt_plant = MetadataFilter(
        must=[FieldFilter(field="plant_unit", operator=FilterOperator.EQUALS, value="HCU")]
    )
    res_hcu = temp_store.search_vectors("test_search", [1.0, 0.0, 0.0, 0.0], filters=flt_plant)
    assert len(res_hcu) == 2
    for r in res_hcu:
        assert r.plant_unit == "HCU"

    # 3. Filter by equipment_entities CONTAINS P-101
    flt_equip = MetadataFilter(
        must=[FieldFilter(field="equipment_entities", operator=FilterOperator.CONTAINS, value="P-101")]
    )
    res_equip = temp_store.search_vectors("test_search", [1.0, 0.0, 0.0, 0.0], filters=flt_equip)
    assert len(res_equip) == 1
    assert res_equip[0].chunk_id == "chk_hcu_1"


def test_qdrant_store_deletion(temp_store: QdrantVectorStore):
    """Test chunk deletion and deletion by document ID."""
    cfg = CollectionConfig(name="test_delete", vector_size=4)
    temp_store.create_collection(cfg)

    c1 = _make_sample_chunk("chk_1", doc_id="DOC_A")
    c2 = _make_sample_chunk("chk_2", doc_id="DOC_A")
    c3 = _make_sample_chunk("chk_3", doc_id="DOC_B")
    temp_store.upsert_chunks("test_delete", [c1, c2, c3])

    # Delete single chunk
    temp_store.delete_chunks("test_delete", ["chk_1"])
    assert temp_store.get_chunk("test_delete", "chk_1") is None
    assert temp_store.get_chunk("test_delete", "chk_2") is not None

    # Delete by document
    temp_store.delete_by_document("test_delete", "DOC_A")
    assert temp_store.get_chunk("test_delete", "chk_2") is None
    assert temp_store.get_chunk("test_delete", "chk_3") is not None


# =============================================================================
# 2. Schema and Payload Validator Tests
# =============================================================================

def test_schema_validator():
    """Test vector dimensionality, zero norm, NaN/Inf, and checksum checks."""
    validator = SchemaValidator()

    # Valid vector passes
    validator.validate_vector([0.1, 0.2, 0.3], expected_dim=3)

    # Dimension mismatch
    with pytest.raises(VectorDimensionMismatchError):
        validator.validate_vector([0.1, 0.2], expected_dim=3)

    # Empty vector
    with pytest.raises(VectorQualityError):
        validator.validate_vector([])

    # NaN detection
    with pytest.raises(VectorQualityError):
        validator.validate_vector([0.1, float("nan"), 0.3])

    # Inf detection
    with pytest.raises(VectorQualityError):
        validator.validate_vector([0.1, float("inf"), 0.3])

    # Checksum verification
    chunk = _make_sample_chunk("test_chk", vec=[0.1, 0.2, 0.3, 0.4])
    validator.validate_chunk(chunk, expected_dim=4, verify_checksum=True)

    # Tampered checksum
    chunk.vector_checksum = "tampered_bad_checksum"
    with pytest.raises(VectorQualityError):
        validator.validate_chunk(chunk, expected_dim=4, verify_checksum=True)


def test_payload_validator():
    """Test required fields, types, string length bounds, and enums."""
    validator = PayloadValidator(strict_enums=True)

    # Valid payload passes
    valid_payload = {
        "chunk_id": "chk_001",
        "document_id": "DOC_1",
        "page_number": 2,
        "chunk_index": 0,
        "category": "Manual",
        "plant_unit": "HCU",
        "equipment_entities": ["P-101"],
    }
    validator.validate_payload(valid_payload)

    # Missing chunk_id
    with pytest.raises(PayloadValidationError):
        validator.validate_payload({"document_id": "DOC_1"})

    # Missing document_id
    with pytest.raises(PayloadValidationError):
        validator.validate_payload({"chunk_id": "chk_001"})

    # Invalid page_number type
    with pytest.raises(PayloadValidationError):
        validator.validate_payload({
            "chunk_id": "chk_001",
            "document_id": "DOC_1",
            "page_number": "two",
        })

    # Runaway section_title length (>512)
    with pytest.raises(PayloadValidationError):
        validator.validate_payload({
            "chunk_id": "chk_001",
            "document_id": "DOC_1",
            "section_title": "X" * 600,
        })

    # Invalid category enum
    with pytest.raises(PayloadValidationError):
        validator.validate_payload({
            "chunk_id": "chk_001",
            "document_id": "DOC_1",
            "category": "NonExistentRandomCategory",
        })


# =============================================================================
# 3. Multi-Collection Router Tests
# =============================================================================

def test_collection_router():
    """Test deterministic routing across refinery document types."""
    router = CollectionRouter()

    # Manual -> mrpl_manuals_v1
    c_manual = _make_sample_chunk("c1", cat="Manual")
    assert router.get_target_collection(c_manual) == "mrpl_manuals_v1"

    # Safety -> mrpl_safety_v1
    c_safety = _make_sample_chunk("c2", cat="Safety Standard", safety=["OISD-105"])
    assert router.get_target_collection(c_safety) == "mrpl_safety_v1"

    # Inspection -> mrpl_inspection_v1
    c_insp = _make_sample_chunk("c3", cat="Inspection Report")
    assert router.get_target_collection(c_insp) == "mrpl_inspection_v1"

    # Maintenance -> mrpl_maintenance_v1
    c_maint = _make_sample_chunk("c4", cat="Maintenance Record")
    assert router.get_target_collection(c_maint) == "mrpl_maintenance_v1"

    # SOP -> mrpl_sops_v1
    c_sop = _make_sample_chunk("c5", cat="SOP")
    assert router.get_target_collection(c_sop) == "mrpl_sops_v1"

    # Email -> mrpl_emails_v1
    c_email = _make_sample_chunk("c6", cat="Email")
    assert router.get_target_collection(c_email) == "mrpl_emails_v1"

    # Drawing -> mrpl_drawings_v1
    c_dwg = _make_sample_chunk("c7", cat="Engineering Drawing")
    assert router.get_target_collection(c_dwg) == "mrpl_drawings_v1"

    # P&ID -> mrpl_pids_v1
    c_pid = _make_sample_chunk("c8", cat="P&ID")
    assert router.get_target_collection(c_pid) == "mrpl_pids_v1"

    # Batch grouping
    grouped = router.route_chunks([c_manual, c_safety, c_insp])
    assert len(grouped["mrpl_manuals_v1"]) == 1
    assert len(grouped["mrpl_safety_v1"]) == 1
    assert len(grouped["mrpl_inspection_v1"]) == 1


# =============================================================================
# 4. Collection Versioning & Rollback Tests
# =============================================================================

def test_collection_version_manager(temp_store: QdrantVectorStore, tmp_path: Path):
    """Test version creation, atomic alias activation, and rollback."""
    ledger_file = tmp_path / "versions.json"
    vm = CollectionVersionManager(store=temp_store, ledger_path=ledger_file)

    # 1. Create v1
    v1_col = vm.create_version("engineering_docs", "v1", CollectionConfig(name="engineering_docs_v1", vector_size=4))
    assert v1_col == "engineering_docs_v1"
    assert vm.get_active_collection("engineering_docs") == "engineering_docs_v1"

    # 2. Create v2
    v2_col = vm.create_version("engineering_docs", "v2", CollectionConfig(name="engineering_docs_v2", vector_size=4))
    assert v2_col == "engineering_docs_v2"
    # Live remains v1 until activation
    assert vm.get_active_collection("engineering_docs") == "engineering_docs_v1"

    # 3. Activate v2
    assert vm.activate_version("engineering_docs", "v2") is True
    assert vm.get_active_collection("engineering_docs") == "engineering_docs_v2"

    # 4. List versions
    history = vm.list_versions("engineering_docs")
    assert len(history) == 2
    v2_info = next(h for h in history if h.version == "v2")
    assert v2_info.is_active is True

    # 5. Rollback to v1
    rolled_back_to = vm.rollback("engineering_docs")
    assert rolled_back_to == "v1"
    assert vm.get_active_collection("engineering_docs") == "engineering_docs_v1"


# =============================================================================
# 5. Payload Index Manager Tests
# =============================================================================

def test_payload_index_manager(temp_store: QdrantVectorStore):
    """Test automated creation and health auditing of payload indexes."""
    cfg = CollectionConfig(name="test_indexes", vector_size=4)
    temp_store.create_collection(cfg)

    pim = PayloadIndexManager(store=temp_store)
    results = pim.create_payload_indexes("test_indexes")
    assert "document_id" in results
    assert "equipment_entities" in results
    assert "section_title" in results

    health = pim.report_payload_index_health("test_indexes")
    assert health.is_healthy is True
    assert len(health.active_indexes) >= 12


# =============================================================================
# 6. Vector Optimizer Tests
# =============================================================================

def test_vector_optimizer(temp_store: QdrantVectorStore):
    """Test segment merge, vacuum, payload compaction, and HNSW optimization."""
    cfg = CollectionConfig(name="test_opt", vector_size=4)
    temp_store.create_collection(cfg)

    chk = _make_sample_chunk("chk_opt_1")
    temp_store.upsert_chunks("test_opt", [chk])

    optimizer = VectorOptimizer(temp_store)
    m_seg = optimizer.optimize_segments("test_opt")
    assert m_seg.optimization_type == "segment_merge"

    m_vac = optimizer.vacuum("test_opt")
    assert m_vac.optimization_type == "vacuum"

    m_mem = optimizer.optimize_memory()
    assert m_mem["status"] == "MEMORY_COMPACTED"

    report = optimizer.optimize_all("test_opt")
    assert len(report.tasks_executed) == 4
    assert len(optimizer.get_optimization_metrics()) >= 6


# =============================================================================
# 7. Transaction Manager & WAL Recovery Tests
# =============================================================================

def test_transaction_manager(temp_store: QdrantVectorStore, tmp_path: Path):
    """Test WAL transaction begin, commit, rollback, and recovery."""
    journal_dir = tmp_path / "journal_test"
    tm = TransactionManager(temp_store, journal_dir=journal_dir)

    cfg = CollectionConfig(name="tx_col", vector_size=4)
    temp_store.create_collection(cfg)

    # Successful transaction
    tx1 = tm.begin_transaction("tx_col", ["chk_tx1"])
    temp_store.upsert_chunks("tx_col", [_make_sample_chunk("chk_tx1")])
    assert tm.commit_transaction(tx1) is True

    # Rolled back transaction
    tx2 = tm.begin_transaction("tx_col", ["chk_tx2"])
    temp_store.upsert_chunks("tx_col", [_make_sample_chunk("chk_tx2")])
    assert tm.rollback_transaction(tx2) is True
    assert temp_store.get_chunk("tx_col", "chk_tx2") is None

    # Dirty recovery simulation: append uncommitted START record to WAL
    wal_file = journal_dir / "wal.jsonl"
    with open(wal_file, "a", encoding="utf-8") as f:
        f.write('{"tx_id": "tx_dirty_001", "action": "START", "collection_name": "tx_col", "chunk_ids": []}\n')

    # Startup recovery detects dirty transaction
    uncommitted = tm.recover_on_startup()
    assert uncommitted == 1


# =============================================================================
# 8. Incremental Indexer 3-Way Diff Tests
# =============================================================================

def test_incremental_indexer(temp_store: QdrantVectorStore):
    """Test 3-way diff reconciliation: new, unchanged, modified, deleted."""
    cfg = CollectionConfig(name="diff_col", vector_size=4)
    temp_store.create_collection(cfg)

    reconciler = IncrementalIndexer(temp_store)

    c1 = _make_sample_chunk("chk_1", doc_id="DOC_X", content_hash="hash_AAA")
    c2 = _make_sample_chunk("chk_2", doc_id="DOC_X", content_hash="hash_BBB")
    temp_store.upsert_chunks("diff_col", [c1, c2])

    # Incoming: c1 unchanged, c2 modified, c3 new, c4 absent (deleted from DOC_X)
    c1_same = _make_sample_chunk("chk_1", doc_id="DOC_X", content_hash="hash_AAA")
    c2_mod = _make_sample_chunk("chk_2", doc_id="DOC_X", content_hash="hash_BBB_NEW")
    c3_new = _make_sample_chunk("chk_3", doc_id="DOC_X", content_hash="hash_CCC")

    diff = reconciler.calculate_diff("diff_col", [c1_same, c2_mod, c3_new], document_id="DOC_X")
    assert diff.unchanged_chunk_ids == ["chk_1"]
    assert diff.modified_chunk_ids == ["chk_2"]
    assert diff.new_chunk_ids == ["chk_3"]


# =============================================================================
# 9. Snapshot Manager & Verification Tests
# =============================================================================

def test_snapshot_manager(temp_store: QdrantVectorStore, tmp_path: Path):
    """Test snapshot creation, SHA-256 digest generation, and tamper detection."""
    cfg = CollectionConfig(name="snap_col", vector_size=4)
    temp_store.create_collection(cfg)
    temp_store.upsert_chunks("snap_col", [_make_sample_chunk("chk_s1")])

    sm = SnapshotManager(temp_store, backup_dir=tmp_path / "backups")
    archive_path = sm.create_snapshot("snap_col", "test_backup")

    assert archive_path.exists()
    assert archive_path.with_suffix(".tar.gz.sha256").exists()

    # Verification passes
    assert sm.verify_snapshot(archive_path) is True

    # Tamper with archive bytes
    with open(archive_path, "ab") as f:
        f.write(b"corruption_bytes")

    # Verification fails on corrupted archive
    with pytest.raises(SnapshotError):
        sm.verify_snapshot(archive_path)


# =============================================================================
# 10. Universal VectorRepository Facade & Concurrency Tests
# =============================================================================

def test_vector_repository_end_to_end(temp_store: QdrantVectorStore):
    """Test universal repository interface: save, search, equipment lookup, and stats."""
    repo = VectorRepository(store=temp_store)

    c1 = _make_sample_chunk("repo_chk_1", cat="Manual", equip=["P-101"], vec=[1.0, 0.0, 0.0, 0.0])
    c2 = _make_sample_chunk("repo_chk_2", cat="Safety Document", equip=["V-201"], vec=[0.0, 1.0, 0.0, 0.0])

    res = repo.save_chunks([c1, c2])
    assert res.inserted == 2

    # Vector search in manual collection
    matches = repo.find_by_vector([1.0, 0.0, 0.0, 0.0], collection_name="mrpl_manuals_v1", limit=1)
    assert len(matches) == 1
    assert matches[0].chunk_id == "repo_chk_1"

    # Equipment lookup
    equip_results = repo.find_by_equipment("P-101", collection_name="mrpl_manuals_v1")
    assert len(equip_results) == 1
    assert equip_results[0].chunk_id == "repo_chk_1"

    # Health & stats
    health = repo.health()
    assert health.status in ("HEALTHY", "DEGRADED")

    stats = repo.stats()
    assert stats.total_vectors == 2


def test_vector_repository_multithreaded_concurrency(temp_store: QdrantVectorStore):
    """Test 10 concurrent threads saving and querying through VectorRepository."""
    repo = VectorRepository(store=temp_store)
    errors: list[Exception] = []

    def worker(worker_id: int):
        try:
            chunk = _make_sample_chunk(f"chk_worker_{worker_id}", doc_id=f"DOC_W_{worker_id}")
            repo.save_chunk(chunk, collection_name="mrpl_manuals_v1")
            found = repo.find_by_id(f"chk_worker_{worker_id}", collection_name="mrpl_manuals_v1")
            assert found is not None
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0, f"Concurrent repository errors: {errors}"


# =============================================================================
# 11. Milestone 1 Backward Compatibility Contract Tests
# =============================================================================

def test_base_vector_store_backward_compatibility(temp_store: QdrantVectorStore):
    """Verify that early Milestone 1 methods (add, search, count, clear) remain 100% functional."""
    chunk1 = Chunk(chunk_id="chk_legacy_1", content="Pump operating instructions", token_count=4)
    chunk2 = Chunk(chunk_id="chk_legacy_2", content="Turbine safety shutdown", token_count=4)

    emb1 = EmbeddingVector(vector=[1.0, 0.0, 0.0, 0.0], dimension=4, model_name="bge")
    emb2 = EmbeddingVector(vector=[0.0, 1.0, 0.0, 0.0], dimension=4, model_name="bge")

    # add()
    added = temp_store.add([chunk1, chunk2], [emb1, emb2])
    assert added == 2

    # count()
    assert temp_store.count() == 2

    # search()
    query_emb = EmbeddingVector(vector=[1.0, 0.0, 0.0, 0.0], dimension=4)
    scored_results = temp_store.search(query_emb, top_k=1)
    assert len(scored_results) == 1
    assert scored_results[0].chunk.chunk_id == "chk_legacy_1"

    # delete()
    deleted = temp_store.delete(["chk_legacy_1"])
    assert deleted == 1
    assert temp_store.count() == 1

    # clear()
    temp_store.clear()
    assert temp_store.count() == 0
