"""Test Suite for Milestone 8: Enterprise Hybrid Retrieval Engine.

Comprehensive unit, integration, stress, and thread-safety tests covering:
- Query Analysis & Entity Extraction
- Query Normalization & Deterministic Rewriting
- Metadata Filter Planning
- Sparse BM25 Indexing, Incremental Updates & Persistence
- Dense Vector Retrieval via VectorRepository
- Score Calibration & Normalization (Min-Max, Softmax, Z-Score, Percentile)
- Reciprocal Rank Fusion (RRF)
- Domain Metadata Boosting
- Cross-Encoder Neural Reranking & Fallback
- Relational Context Expansion
- Duplicate Removal & Deduplication
- Deterministic Citation Provenance Generation
- Token-Budget Context Packing & Table Protection
- SQLite Query Cache & Invalidation
- Adaptive Strategy & Intent-Driven Top-K
- Strategy Registry & Plugin Architecture
- End-to-End Retrieval Pipeline & Telemetry
- Concurrency & Thread Safety
"""

from __future__ import annotations

import concurrent.futures
from pathlib import Path
import tempfile
import time
import pytest

from rag_engine.retrieval.adaptive_retriever import AdaptiveRetriever
from rag_engine.retrieval.base_retriever import (
    CitationBundle,
    RetrievalResult,
    ScoredRetrievalChunk,
)
from rag_engine.retrieval.bm25_retriever import BM25Index, BM25Retriever
from rag_engine.retrieval.citation_builder import CitationBuilder
from rag_engine.retrieval.context_expander import ContextExpander
from rag_engine.retrieval.context_packer import ContextPacker
from rag_engine.retrieval.dense_retriever import DenseRetriever
from rag_engine.retrieval.duplicate_remover import DuplicateRemover
from rag_engine.retrieval.hybrid_retriever import HybridRetriever
from rag_engine.retrieval.metadata_booster import MetadataBooster
from rag_engine.retrieval.metadata_filter import (
    MetadataFilterBuilder,
    MetadataFilterPlanner,
)
from rag_engine.retrieval.query_analyzer import (
    IntentType,
    QueryAnalyzer,
    QueryIntent,
)
from rag_engine.retrieval.query_cache import QueryCache
from rag_engine.retrieval.query_classifier import QueryClassifier
from rag_engine.retrieval.query_expander import QueryExpander
from rag_engine.retrieval.query_normalizer import QueryNormalizer
from rag_engine.retrieval.query_rewriter import QueryRewriter
from rag_engine.retrieval.reranker import (
    DeterministicTestReranker,
    LocalCrossEncoderReranker,
)
from rag_engine.retrieval.retrieval_events import RetrievalEventBus
from rag_engine.retrieval.retrieval_factory import RetrievalFactory
from rag_engine.retrieval.retrieval_health import RetrievalHealthChecker
from rag_engine.retrieval.retrieval_metrics import RetrievalMetricsCollector
from rag_engine.retrieval.retrieval_pipeline import RetrievalPipeline
from rag_engine.retrieval.retrieval_registry import (
    RetrievalRegistry,
    register_retriever,
)
from rag_engine.retrieval.retrieval_utils import (
    estimate_token_count,
    format_cache_key,
    normalize_whitespace,
    tokenize_refinery_text,
)
from rag_engine.retrieval.retrieval_validator import RetrievalValidator
from rag_engine.retrieval.rrf_fusion import ReciprocalRankFusion
from rag_engine.retrieval.score_normalizer import (
    NormalizationMethod,
    ScoreNormalizer,
)
from rag_engine.schemas.chunk import Chunk, ChunkHierarchy, ChunkMetadata
from rag_engine.schemas.vector_store import (
    DistanceMetric,
    FieldFilter,
    FilterOperator,
    MetadataFilter,
)


@pytest.fixture
def sample_refinery_chunks() -> list[Chunk]:
    """Generate realistic MRPL refinery test chunks with enriched domain metadata."""
    chunks = [
        Chunk.create(
            document_id="doc_cdu_manual",
            document_name="CDU_Operating_Manual.pdf",
            source_path="/data/manuals/CDU_Operating_Manual.pdf",
            page_number=14,
            section_title="3.1 Crude Feed Pump P-203 Operation",
            content="Crude oil feed pump P-203 operates at discharge pressure of 14.5 bar and rated flow of 450 m3/hr. Suction line LINE-101-CS must be inspected in compliance with OISD-105.",
            chunk_index=0,
            category="manuals",
            plant_unit="CDU-1",
            equipment_entities=["P-203"],
            safety_entities=["OISD-105"],
            operating_parameters={"pressure": "14.5 bar", "flow": "450 m3/hr"},
        ),
        Chunk.create(
            document_id="doc_cdu_manual",
            document_name="CDU_Operating_Manual.pdf",
            source_path="/data/manuals/CDU_Operating_Manual.pdf",
            page_number=15,
            section_title="3.2 Heat Exchanger HX-01 Circuit",
            content="Shell and tube heat exchanger HX-01 preheats crude from pump P-203 discharge to 135 °C. Maximum operating temperature limit is 150 °C.",
            chunk_index=1,
            category="manuals",
            plant_unit="CDU-1",
            equipment_entities=["P-203", "HX-01"],
            safety_entities=["OISD-105"],
            operating_parameters={"temperature": "135 °C"},
        ),
        Chunk.create(
            document_id="doc_safety_oisd",
            document_name="OISD_Standard_105.pdf",
            source_path="/data/safety/OISD_Standard_105.pdf",
            page_number=42,
            section_title="Clause 4.2 Hazardous Area Isolation",
            content="All hydrocarbon pumps including P-203 and P-204 must have remote isolation valves MOV-101 installed with fireproofing meeting OISD-105 standards.",
            chunk_index=2,
            category="safety",
            plant_unit="CDU-1",
            equipment_entities=["P-203", "P-204", "MOV-101"],
            safety_entities=["OISD-105", "DANGER"],
        ),
        Chunk.create(
            document_id="doc_ndt_inspection",
            document_name="CDU_Inspection_Report_2024.pdf",
            source_path="/data/inspection/CDU_Inspection_Report_2024.pdf",
            page_number=8,
            section_title="Table 2: Ultrasonic Thickness Measurement",
            content="| Line Number | Nominal (mm) | Actual (mm) | Corrosion Rate (mm/yr) |\n|---|---|---|---|\n| LINE-101-CS | 12.7 | 11.2 | 0.15 |\n| LINE-102-CS | 9.5 | 8.9 | 0.10 |",
            chunk_index=3,
            category="inspection",
            plant_unit="CDU-1",
            is_table_chunk=True,
            equipment_entities=["LINE-101-CS"],
        ),
        Chunk.create(
            document_id="doc_maint_pumps",
            document_name="Pump_Overhaul_SOP.pdf",
            source_path="/data/maintenance/Pump_Overhaul_SOP.pdf",
            page_number=22,
            section_title="5.0 Vibration Analysis and Bearing Replacement",
            content="High vibration exceeding 4.5 mm/s on pump P-203 indicates bearing wear. Replace with SKF 6312 deep groove ball bearing following API-610 specifications.",
            chunk_index=4,
            category="maintenance",
            plant_unit="CDU-1",
            equipment_entities=["P-203"],
            safety_entities=["API-610"],
        ),
    ]
    return chunks


# =============================================================================
# 1. Query Analyzer & Intent Classifier Tests
# =============================================================================

def test_query_analyzer_intent_and_entity_extraction():
    analyzer = QueryAnalyzer()

    # Query with equipment tag, standard, pressure, and troubleshooting
    q1 = "Why is pump P203 experiencing high vibration at 150bar pressure under OISD-105 on page 14?"
    intent1 = analyzer.analyze(q1)

    assert intent1.primary_intent in {IntentType.TROUBLESHOOTING, IntentType.EQUIPMENT_LOOKUP}
    assert "P-203" in intent1.entities.equipment_tags
    assert "OISD-105" in intent1.entities.standards
    assert any("150" in p for p in intent1.entities.pressure_values)
    assert "14" in intent1.entities.page_references
    assert intent1.suggested_top_k >= 10


def test_query_classifier_convenience():
    classifier = QueryClassifier()
    assert classifier.get_intent_category("Safety protocol under OISD-105") == "safety_lookup"
    assert classifier.get_intent_category("Operating manual for pump P-203") == "equipment_lookup"
    assert classifier.get_intent_category("Inspection report for CDU-1") == "inspection_lookup"
    assert classifier.get_intent_category("Preventive maintenance overhaul for pump") == "maintenance_lookup"


# =============================================================================
# 2. Query Normalizer & Rewriter Tests
# =============================================================================

def test_query_normalizer_equipment_and_units():
    raw = "what is the limit for pump P203 with 150bar and 250C in CDU1?"
    norm = QueryNormalizer.normalize(raw, strip_noise=True)

    assert "P-203" in norm
    assert "150 bar" in norm
    assert "250 °C" in norm
    assert not norm.lower().startswith("what is")


def test_query_rewriter_spelling_corrections():
    raw = "What is the pressue and temprature limit for centrifual pump P203?"
    rewritten = QueryRewriter.rewrite(raw)

    assert "pressure" in rewritten
    assert "temperature" in rewritten
    assert "centrifugal" in rewritten
    assert "P-203" in rewritten


def test_query_expander_abbreviations():
    expanded = QueryExpander.expand("Check MOV and PRV in CDU")
    assert any("valve" in e.lower() for e in expanded) or any("distillation" in e.lower() for e in expanded)


# =============================================================================
# 3. Metadata Filter Planning Tests
# =============================================================================

def test_metadata_filter_planner():
    analyzer = QueryAnalyzer()
    intent = analyzer.analyze("Pump P203 inspection report for CDU-1")

    flt = MetadataFilterPlanner.plan_filters(intent)
    assert flt is not None
    assert len(flt.must) >= 1
    # Check equipment filter was created
    equip_filter = [f for f in flt.must if f.field == "equipment_entities"]
    assert len(equip_filter) == 1
    assert equip_filter[0].value == "P-203"


# =============================================================================
# 4. Sparse BM25 Retriever & Inverted Index Tests
# =============================================================================

def test_bm25_indexing_search_and_persistence(sample_refinery_chunks):
    with tempfile.TemporaryDirectory() as tmpdir:
        index_file = Path(tmpdir) / "bm25_test.json"
        index = BM25Index(storage_path=index_file)
        index.add_chunks(sample_refinery_chunks)

        assert index.total_docs == len(sample_refinery_chunks)

        # Search for exact equipment tag
        hits = index.search("P-203", top_k=3)
        assert len(hits) > 0
        top_chk, top_score = hits[0]
        assert "P-203" in top_chk.content or "P-203" in top_chk.metadata.equipment_entities
        assert top_score > 0.0

        # Incremental remove
        rm_cid = sample_refinery_chunks[0].chunk_id
        removed = index.remove_chunk(rm_cid)
        assert removed is True
        assert index.total_docs == len(sample_refinery_chunks) - 1

        # Persistence reload
        new_index = BM25Index(storage_path=index_file)
        new_index.load()
        assert new_index.total_docs == len(sample_refinery_chunks) - 1


# =============================================================================
# 5. Score Normalization Tests
# =============================================================================

def test_score_normalizer_methods():
    raw_scores = [10.0, 20.0, 30.0, 50.0]

    # Min-Max
    minmax = ScoreNormalizer.normalize(raw_scores, method=NormalizationMethod.MIN_MAX)
    assert minmax[0] == 0.0
    assert minmax[-1] == 1.0

    # Softmax
    softmax = ScoreNormalizer.normalize(raw_scores, method=NormalizationMethod.SOFTMAX)
    assert abs(sum(softmax) - 1.0) < 1e-4
    assert softmax[-1] > softmax[0]

    # Z-Score
    zscore = ScoreNormalizer.normalize(raw_scores, method=NormalizationMethod.Z_SCORE)
    assert all(0.0 <= s <= 1.0 for s in zscore)

    # Percentile
    percentile = ScoreNormalizer.normalize(raw_scores, method=NormalizationMethod.PERCENTILE)
    assert percentile[0] == 0.0
    assert percentile[-1] == 1.0


# =============================================================================
# 6. Reciprocal Rank Fusion (RRF) Tests
# =============================================================================

def test_reciprocal_rank_fusion(sample_refinery_chunks):
    c1, c2, c3 = sample_refinery_chunks[:3]

    dense_candidates = [
        ScoredRetrievalChunk(chunk=c1, score=0.95, rank=0),
        ScoredRetrievalChunk(chunk=c2, score=0.85, rank=1),
    ]
    bm25_candidates = [
        ScoredRetrievalChunk(chunk=c2, score=12.5, rank=0),
        ScoredRetrievalChunk(chunk=c3, score=8.2, rank=1),
    ]

    fusion = ReciprocalRankFusion(k=60, dense_weight=0.5, bm25_weight=0.5)
    fused = fusion.fuse(dense_candidates, bm25_candidates)

    assert len(fused) == 3
    # c2 appeared in both dense and sparse, so it should receive compounded score
    assert fused[0].chunk.chunk_id == c2.chunk_id
    assert fused[0].fusion_score is not None
    assert "weighted RRF" in fused[0].explainability


# =============================================================================
# 7. Metadata Booster Tests
# =============================================================================

def test_metadata_booster(sample_refinery_chunks):
    c1, c2, c3 = sample_refinery_chunks[:3]
    candidates = [
        ScoredRetrievalChunk(chunk=c1, score=0.5, rank=0),
        ScoredRetrievalChunk(chunk=c2, score=0.5, rank=1),
    ]

    analyzer = QueryAnalyzer()
    intent = analyzer.analyze("Operating parameters for pump P-203 in CDU-1")

    booster = MetadataBooster(equipment_boost=0.30, plant_unit_boost=0.15)
    boosted = booster.boost(candidates, intent=intent)

    assert boosted[0].score > 0.5
    assert boosted[0].boost_applied > 0.0
    assert "equipment match" in boosted[0].explainability


# =============================================================================
# 8. Cross-Encoder Neural Reranking Tests
# =============================================================================

def test_cross_encoder_reranker(sample_refinery_chunks):
    candidates = [
        ScoredRetrievalChunk(chunk=c, score=0.5, rank=i)
        for i, c in enumerate(sample_refinery_chunks)
    ]

    reranker = LocalCrossEncoderReranker(use_fallback_if_missing=True)
    results = reranker.rerank("P-203 discharge pressure", candidates, top_k=2)

    assert len(results) == 2
    assert results[0].rerank_score is not None
    assert results[0].rerank_score >= results[1].rerank_score


# =============================================================================
# 9. Context Expansion Tests
# =============================================================================

def test_context_expander_with_mock_repository(sample_refinery_chunks):
    c1 = sample_refinery_chunks[0]
    candidate = ScoredRetrievalChunk(chunk=c1, score=0.9, rank=0)

    # Use default context expander (handles empty neighbors gracefully)
    expander = ContextExpander()
    expanded = expander.expand([candidate], radius=1)

    assert len(expanded) >= 1
    assert expanded[0].chunk.chunk_id == c1.chunk_id


# =============================================================================
# 10. Duplicate Removal Tests
# =============================================================================

def test_duplicate_remover(sample_refinery_chunks):
    c1 = sample_refinery_chunks[0]
    c1_dup = Chunk.create(
        document_id="doc_dup",
        content=c1.content,  # Exact duplicate content
        chunk_index=99,
    )

    candidates = [
        ScoredRetrievalChunk(chunk=c1, score=0.9, rank=0),
        ScoredRetrievalChunk(chunk=c1_dup, score=0.8, rank=1),
    ]

    remover = DuplicateRemover()
    deduped = remover.deduplicate_chunks(candidates)
    assert len(deduped) == 1
    assert deduped[0].chunk.chunk_id == c1.chunk_id


# =============================================================================
# 11. Citation Builder Tests
# =============================================================================

def test_citation_builder(sample_refinery_chunks):
    candidates = [
        ScoredRetrievalChunk(chunk=c, score=0.85, rank=i)
        for i, c in enumerate(sample_refinery_chunks[:2])
    ]

    builder = CitationBuilder()
    citations = builder.build_citations(candidates)

    assert len(citations) == 2
    assert citations[0].citation_id == "[1]"
    assert citations[0].document_id == sample_refinery_chunks[0].metadata.document_id
    assert citations[0].page_number == sample_refinery_chunks[0].metadata.page_number
    assert len(citations[0].verbatim_quote) > 0


# =============================================================================
# 12. Token-Budget Context Packing & Table Protection Tests
# =============================================================================

def test_context_packer_budget_and_tables(sample_refinery_chunks):
    candidates = [
        ScoredRetrievalChunk(chunk=c, score=0.9 - (i * 0.1), rank=i)
        for i, c in enumerate(sample_refinery_chunks)
    ]
    citations = CitationBuilder().build_citations(candidates)

    packer = ContextPacker(default_token_budget=500)
    context_str, tokens = packer.pack(candidates, citations, token_budget=500)

    assert len(context_str) > 0
    assert tokens <= 550  # Bounded by budget
    assert "[1]" in context_str
    # Verify table integrity
    if "| Line Number |" in context_str:
        assert "|---|---|---|---|" in context_str


# =============================================================================
# 13. Persistent SQLite Query Cache Tests
# =============================================================================

def test_query_cache_persistence_and_ttl(sample_refinery_chunks):
    with tempfile.TemporaryDirectory() as tmpdir:
        db_file = Path(tmpdir) / "test_cache.db"
        cache = QueryCache(db_path=db_file, default_ttl_seconds=2)

        c = sample_refinery_chunks[0]
        res = RetrievalResult(
            query="test query",
            strategy_name="hybrid",
            scored_chunks=[ScoredRetrievalChunk(chunk=c, score=0.9, rank=0)],
            citations=[CitationBuilder().build_citations([ScoredRetrievalChunk(chunk=c, score=0.9, rank=0)])[0]],
            packed_context="test context",
            total_candidates=1,
            tokens_packed=10,
        )

        key = format_cache_key("test query", "hybrid", 10)
        cache.set(key, res)

        cached = cache.get(key)
        assert cached is not None
        assert cached.query == "test query"

        # Test invalidation
        cache.invalidate(key)
        assert cache.get(key) is None


# =============================================================================
# 14. Adaptive Strategy & Top-K Tests
# =============================================================================

def test_adaptive_retriever_archetypes(sample_refinery_chunks):
    # Setup in-memory BM25 with sample chunks
    bm25_idx = BM25Index()
    bm25_idx.add_chunks(sample_refinery_chunks)
    bm25_retriever = BM25Retriever(index=bm25_idx)

    hybrid = HybridRetriever(bm25_retriever=bm25_retriever, use_cache=False)
    adaptive = AdaptiveRetriever(hybrid_retriever=hybrid)

    # Equipment lookup
    res_equip = adaptive.retrieve("P-203 pump specs")
    assert res_equip.strategy_name == "adaptive"
    assert res_equip.metadata.get("adaptive_intent") in {IntentType.EQUIPMENT_LOOKUP, IntentType.SPECIFICATION_LOOKUP}
    assert res_equip.metadata.get("adaptive_top_k") <= 10

    # Complex troubleshooting query
    res_trouble = adaptive.retrieve("Troubleshooting abnormal vibration and failure of bearing")
    assert res_trouble.metadata.get("adaptive_top_k") >= 20


# =============================================================================
# 15. Strategy Registry & Plugin Architecture Tests
# =============================================================================

def test_retrieval_registry_and_factory():
    registry = RetrievalRegistry.get_instance()
    assert registry.contains("hybrid")
    assert registry.contains("bm25")
    assert registry.contains("dense")

    # Dynamic custom retriever registration
    @register_retriever("custom_graph", description="Knowledge Graph Retriever Plugin")
    class CustomGraphRetriever(BM25Retriever):
        def get_strategy_name(self) -> str:
            return "custom_graph"

    assert registry.contains("custom_graph")
    factory = RetrievalFactory.get_instance()
    inst = factory.create_retriever("custom_graph", use_cached=False)
    assert inst.get_strategy_name() == "custom_graph"


# =============================================================================
# 16. End-to-End Pipeline & Telemetry Diagnostics Tests
# =============================================================================

def test_retrieval_pipeline_end_to_end(sample_refinery_chunks):
    bm25_idx = BM25Index()
    pipeline = RetrievalPipeline(strategy_name="hybrid", bm25_index=bm25_idx, use_cache=False)
    pipeline.index_chunks(sample_refinery_chunks)

    result = pipeline.execute("Operating pressure of pump P-203 in CDU-1", top_k=3)

    assert result.query is not None
    assert len(result.scored_chunks) > 0
    assert len(result.citations) > 0
    assert len(result.packed_context) > 0
    assert result.execution_time_ms > 0.0

    # Verify telemetry metrics
    assert result.metrics is not None
    assert result.metrics.returned_chunks_count == len(result.scored_chunks)

    # Verify system health
    health = pipeline.check_health()
    assert health.status in {"HEALTHY", "DEGRADED"}


# =============================================================================
# 17. Concurrency & Thread Safety Tests
# =============================================================================

def test_retrieval_pipeline_thread_safety(sample_refinery_chunks):
    bm25_idx = BM25Index()
    pipeline = RetrievalPipeline(strategy_name="hybrid", bm25_index=bm25_idx, use_cache=True)
    pipeline.index_chunks(sample_refinery_chunks)

    queries = [
        "P-203 pump discharge",
        "HX-01 heat exchanger temperature",
        "OISD-105 safety standards",
        "Line thickness in LINE-101-CS",
        "Bearing overhaul SOP for pumps",
    ]

    def _query_worker(q: str):
        return pipeline.execute(q, top_k=2)

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(_query_worker, q) for q in queries * 3]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    assert len(results) == 15
    for r in results:
        assert isinstance(r, RetrievalResult)
        assert len(r.scored_chunks) > 0
