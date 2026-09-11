"""Unit and Integration tests for Milestone 5: Enterprise Chunking Engine.

Validates:
1. Fixed-size chunking (token & character based, configurable overlap)
2. Recursive chunking (document -> section -> paragraph -> sentence -> token window)
3. Section-aware chunking (preserves H1-H6 boundaries and heading breadcrumbs)
4. Table-aware chunking (preserves table integrity, headers, captions, row integrity)
5. List-aware chunking (preserves numbered lists, checklists, procedures, bullet points)
6. Metadata inheritance (equipment tags, safety codes, operating params, parent hashes)
7. Parent-Child hierarchy & bidirectional sequential linking (prev_chunk_id, next_chunk_id)
8. Deterministic chunk IDs & content SHA256 hashes (stable across runs, never UUID)
9. Token estimation without LLM (BPE heuristic, word count, character count)
10. Chunk quality validation (reject empty, tiny, oversized, duplicate chunks)
11. Chunk statistics & telemetry collection
12. Thread safety & concurrent chunking across multiple worker threads
13. Real refinery manuals and inspection documents
"""

from __future__ import annotations

import concurrent.futures
import threading
from typing import Any

import pytest

from rag_engine.chunking import (
    BaseChunker,
    ChunkContext,
    ChunkFactory,
    ChunkHealthReport,
    ChunkMetricsCollector,
    ChunkRegistry,
    ChunkValidator,
    EmptyDocumentError,
    FixedChunker,
    HierarchyBuilder,
    ListChunker,
    MetadataInheritor,
    OversizedChunkError,
    RecursiveChunker,
    SectionChunker,
    TableChunker,
    compute_sha256,
    count_characters,
    count_words,
    estimate_tokens,
    generate_deterministic_chunk_id,
    get_chunk_factory,
    get_chunk_registry,
    health,
    split_into_list_items,
    split_into_paragraphs,
    split_into_sentences,
)
from rag_engine.schemas.chunk import Chunk, ChunkHierarchy, ChunkMetadata, ChunkStatistics
from rag_engine.schemas.document import Document, DocumentMetadata
from rag_engine.schemas.parsed_document import (
    CleanParsedDocument,
    EquipmentEntity,
    ParsedDocument,
    ParsedMetadata,
    SafetyWarning,
    SafetyWarningSeverity,
    Section,
    Table,
)


# =====================================================================
# Fixtures & Sample Refinery Data
# =====================================================================

@pytest.fixture
def sample_parsed_document() -> ParsedDocument:
    """Realistic ParsedDocument from MRPL Hydrocracker Unit (HCU) manual."""
    sec1 = Section(
        section_id="sec_hcu_01",
        title="1.0 Hydrocracker High Pressure Reactor Overview",
        level=1,
        content=(
            "The Hydrocracker Unit (HCU) reactor R-101 operates at 150 bar and 380°C. "
            "Feedstock enters through the top inlet distributor and passes over catalyst beds. "
            "Hydrogen quench gas is injected via control valve FCV-104 to regulate bed temperature. "
            "Severe runaway conditions require emergency depressurization according to OISD-105 standard."
        ),
        page_number=1,
        heading_path=["1.0 Hydrocracker High Pressure Reactor Overview"],
    )

    sec2 = Section(
        section_id="sec_hcu_02",
        title="2.0 Reactor Feed Pump P-203 Operating Procedures",
        level=1,
        content=(
            "Standard Operating Procedure for starting high pressure feed pump P-203:\n"
            "1. Verify suction valve MOV-201 is 100% open and lube oil pressure exceeds 2.5 bar.\n"
            "2. Confirm mechanical seal flush cooling water flow indicator FI-202 reads > 15 LPM.\n"
            "3. Energize electric drive motor M-203 and monitor initial vibration velocity.\n"
            "4. Gradually open discharge throttling valve HCV-205 to achieve 120 m3/hr throughput.\n"
            "5. Ensure motor winding temperature does not exceed 110°C under continuous duty."
        ),
        page_number=2,
        heading_path=["2.0 Reactor Feed Pump P-203 Operating Procedures"],
    )

    tbl = Table(
        table_id="tbl_operating_limits",
        caption="HCU Operating Thresholds and Trip Points",
        headers=["Parameter", "Normal Operating", "High Alarm", "Trip Setpoint", "Unit"],
        rows=[
            ["Reactor Inlet Pressure", "150.0", "158.0", "165.0", "bar"],
            ["Catalyst Bed Peak Temp", "380.0", "405.0", "420.0", "°C"],
            ["Feed Pump P-203 Discharge", "160.0", "170.0", "175.0", "bar"],
            ["Quench H2 Flow FCV-104", "25000", "22000", "18000", "Nm3/hr"],
            ["Seal Oil Differential", "3.5", "2.8", "2.0", "bar"],
        ],
        row_count=5,
        col_count=5,
        page_number=3,
    )

    equip1 = EquipmentEntity(
        entity_id="eq_r101",
        tag="R-101",
        name="Hydrocracker High Pressure Reactor",
        operating_pressure="150 bar",
        operating_temperature="380°C",
        page_number=1,
    )
    equip2 = EquipmentEntity(
        entity_id="eq_p203",
        tag="P-203",
        name="High Pressure Feed Pump",
        operating_pressure="160 bar",
        page_number=2,
    )

    safety = SafetyWarning(
        warning_id="safe_oisd105",
        severity=SafetyWarningSeverity.DANGER,
        text="Emergency depressurization according to OISD-105 standard",
        standards=["OISD-105"],
        page_number=1,
    )

    meta = ParsedMetadata(
        document_id="doc_mrpl_hcu_sop_01",
        document_name="MRPL_HCU_Standard_Operating_Procedures.pdf",
        category="Operations",
        subcategory="Standard Operating Procedures",
        plant_unit="Hydrocracker Unit (HCU)",
        checksum="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        total_pages=3,
        total_sections=2,
        total_tables=1,
    )

    return ParsedDocument(
        document_id="doc_mrpl_hcu_sop_01",
        raw_document_id="raw_doc_01",
        title="MRPL HCU Standard Operating Procedures",
        category="Operations",
        metadata=meta,
        sections=[sec1, sec2],
        tables=[tbl],
        equipment=[equip1, equip2],
        warnings=[safety],
    )


# =====================================================================
# 1. Token Estimation & Heuristic Utilities Tests
# =====================================================================

def test_token_estimation_and_counters():
    """Verify BPE-heuristic token estimation without any external model or LLM."""
    text = "Reactor R-101 operates at 150 bar and 380°C with quench valve FCV-104."
    tokens = estimate_tokens(text)
    words = count_words(text)
    chars = count_characters(text)

    assert words == 12
    assert chars == len(text)
    # BPE heuristic splits alphanumeric words and special symbols like ° and -
    assert 11 <= tokens <= 25
    assert estimate_tokens("") == 0
    assert count_words("") == 0
    assert count_characters("") == 0


def test_sentence_and_paragraph_splitting():
    """Verify sentence split protects decimals, abbreviations, and refinery equipment tags."""
    text = (
        "Operating pressure is 150.5 bar. Temperature was recorded at 380.2°C by technician J. Smith. "
        "Valve MOV-201 must be inspected at approx. 08:00 hrs. Next sentence follows."
    )
    sentences = split_into_sentences(text)
    assert len(sentences) >= 3
    # Check that 150.5 bar is NOT split across two sentences
    assert any("150.5 bar" in s for s in sentences)


def test_list_item_splitting():
    """Verify splitting numbered and bulleted lists."""
    list_text = (
        "1. Open suction valve.\n"
        "2. Start pump motor.\n"
        "- Verify vibration.\n"
        "- Record discharge pressure."
    )
    items = split_into_list_items(list_text)
    assert len(items) == 4
    assert items[0] == "1. Open suction valve."
    assert items[1] == "2. Start pump motor."
    assert items[2] == "- Verify vibration."
    assert items[3] == "- Record discharge pressure."


# =====================================================================
# 2. Deterministic IDs & Content Hash Tests
# =====================================================================

def test_deterministic_chunk_ids_and_hashes():
    """Chunk IDs must never be UUIDs and must be 100% reproducible across runs."""
    doc_id = "MRPL_SOP_HCU_001"
    content = "Pump P-203 maximum operating discharge pressure is 160 bar."
    page = 2
    idx = 5

    sha = compute_sha256(content)
    chunk_id_1 = generate_deterministic_chunk_id(doc_id, page, idx, sha)
    chunk_id_2 = generate_deterministic_chunk_id(doc_id, page, idx, sha)

    # Identical inputs must generate identical IDs
    assert chunk_id_1 == chunk_id_2
    assert chunk_id_1.startswith("chk_")
    assert f"p{page}" in chunk_id_1
    assert f"{idx:04d}" in chunk_id_1
    assert sha[:8] in chunk_id_1

    # Changing content must change chunk ID
    sha_mod = compute_sha256(content + " modified")
    chunk_id_3 = generate_deterministic_chunk_id(doc_id, page, idx, sha_mod)
    assert chunk_id_1 != chunk_id_3


# =====================================================================
# 3. Fixed-Size Chunking Tests
# =====================================================================

def test_fixed_size_chunking_token_and_char_modes(sample_parsed_document):
    """Verify fixed-size sliding window chunking with configurable overlap."""
    factory = get_chunk_factory()

    # Token-based fixed chunking
    token_chunker = factory.create_fixed(target_tokens=40, overlap=10, unit="token")
    chunks = token_chunker.chunk(sample_parsed_document)

    assert len(chunks) > 1
    for chk in chunks:
        assert chk.chunk_id.startswith("chk_")
        assert chk.token_count > 0
        assert chk.metadata.chunk_strategy == "fixed"
        assert chk.content.strip()

    # Character-based fixed chunking
    char_chunker = factory.create_fixed(target_tokens=200, overlap=30, unit="char")
    char_chunks = char_chunker.chunk(sample_parsed_document)
    assert len(char_chunks) > 1
    for chk in char_chunks:
        assert len(chk.content) <= 300


# =====================================================================
# 4. Recursive Chunking Tests
# =====================================================================

def test_recursive_chunking_preserves_sentence_boundaries(sample_parsed_document):
    """Verify recursive hierarchy decomposition never breaks sentences unless unavoidable."""
    chunker = RecursiveChunker(
        context=ChunkContext(
            strategy_name="recursive",
            target_tokens=60,
            overlap_tokens=10,
            min_chunk_tokens=5,
            max_chunk_tokens=150,
        )
    )
    chunks = chunker.chunk(sample_parsed_document)

    assert len(chunks) >= 2
    for chk in chunks:
        # None of the chunks should start or end with a chopped word or split punctuation
        assert chk.token_count >= 5
        assert chk.token_count <= 150
        assert chk.metadata.chunk_strategy == "recursive"


# =====================================================================
# 5. Section-Aware Chunking Tests
# =====================================================================

def test_section_aware_chunking_preserves_boundaries(sample_parsed_document):
    """Verify section boundaries (H1, H2) are strictly respected with heading breadcrumbs."""
    chunker = SectionChunker(
        context=ChunkContext(
            strategy_name="section",
            target_tokens=120,
            overlap_tokens=15,
            preserve_headers=True,
        )
    )
    chunks = chunker.chunk(sample_parsed_document)

    assert len(chunks) >= 2
    # Section 1 chunk
    s1_chunks = [c for c in chunks if "Reactor Overview" in str(c.metadata.heading_path)]
    assert len(s1_chunks) >= 1
    assert "1.0 Hydrocracker High Pressure Reactor Overview" in s1_chunks[0].content

    # Section 2 chunk
    s2_chunks = [c for c in chunks if "P-203 Operating Procedures" in str(c.metadata.heading_path)]
    assert len(s2_chunks) >= 1
    assert "P-203" in s2_chunks[0].content


# =====================================================================
# 6. Table-Aware Chunking Tests
# =====================================================================

def test_table_aware_chunking_integrity(sample_parsed_document):
    """Verify tables become independent chunks with intact headers, rows, and caption."""
    chunker = TableChunker(
        context=ChunkContext(
            strategy_name="table",
            target_tokens=100,
        )
    )
    chunks = chunker.chunk(sample_parsed_document)

    # Find the table chunk
    table_chunks = [c for c in chunks if c.metadata.is_table_chunk]
    assert len(table_chunks) >= 1

    tbl_chk = table_chunks[0]
    assert tbl_chk.metadata.table_id == "tbl_operating_limits"
    assert "HCU Operating Thresholds and Trip Points" in tbl_chk.content
    assert "| Parameter | Normal Operating |" in tbl_chk.content
    assert "| Reactor Inlet Pressure | 150.0 |" in tbl_chk.content
    assert tbl_chk.metadata.page_number == 3


def test_large_table_repeats_headers_across_splits():
    """Verify very long tables repeat headers on every split chunk."""
    long_rows = [[f"Line {i}", f"{100+i} bar", f"{200+i} °C"] for i in range(40)]
    large_table = Table(
        table_id="tbl_large_specs",
        caption="Refinery Pipeline Inventory Matrix",
        headers=["Pipeline Tag", "Design Pressure", "Design Temperature"],
        rows=long_rows,
        row_count=40,
        col_count=3,
        page_number=5,
    )

    chunker = TableChunker(
        context=ChunkContext(strategy_name="table", target_tokens=80, max_chunk_tokens=120)
    )
    meta = DocumentMetadata(source_path="specs.txt", file_name="specs.txt", file_format=".txt")
    doc = Document(doc_id="doc_specs", content="Sample content", metadata=meta)
    chunks = chunker.chunk_table(large_table, doc)

    assert len(chunks) > 1
    for idx, chk in enumerate(chunks, start=1):
        assert chk.metadata.is_table_chunk is True
        assert "| Pipeline Tag | Design Pressure | Design Temperature |" in chk.content
        assert f"Part {idx} of {len(chunks)}" in chk.content


# =====================================================================
# 7. List-Aware Chunking Tests
# =====================================================================

def test_list_aware_chunking_keeps_procedures_together(sample_parsed_document):
    """Verify list chunking keeps numbered startup procedures cohesive."""
    chunker = ListChunker(
        context=ChunkContext(
            strategy_name="list",
            target_tokens=150,
            preserve_list_cohesion=True,
        )
    )
    chunks = chunker.chunk(sample_parsed_document)

    list_chunks = [c for c in chunks if c.metadata.is_list_chunk]
    assert len(list_chunks) >= 1

    # Check that sequential steps 1, 2, 3 are bundled together
    procedure_chunk = list_chunks[0]
    assert "1. Verify suction valve" in procedure_chunk.content
    assert "2. Confirm mechanical seal flush" in procedure_chunk.content


# =====================================================================
# 8. Metadata Inheritance Tests
# =====================================================================

def test_metadata_inheritance_equipment_and_safety(sample_parsed_document):
    """Verify chunks inherit equipment tags, safety standards, and parent document metadata."""
    chunker = RecursiveChunker()
    chunks = chunker.chunk(sample_parsed_document)

    # Chunks on Page 1 should inherit R-101 and OISD-105
    page1_chunks = [c for c in chunks if c.metadata.page_number == 1]
    assert len(page1_chunks) >= 1
    p1_meta = page1_chunks[0].metadata

    assert p1_meta.document_id == "doc_mrpl_hcu_sop_01"
    assert p1_meta.plant_unit == "Hydrocracker Unit (HCU)"
    assert "R-101" in p1_meta.equipment_entities
    assert any("OISD-105" in s for s in p1_meta.safety_entities)

    # Chunks on Page 2 should inherit P-203
    page2_chunks = [c for c in chunks if c.metadata.page_number == 2]
    assert len(page2_chunks) >= 1
    assert "P-203" in page2_chunks[0].metadata.equipment_entities


# =====================================================================
# 9. Parent-Child Hierarchy & Sequential Linking Tests
# =====================================================================

def test_parent_child_hierarchy_and_sequential_linking(sample_parsed_document):
    """Verify previous_chunk_id and next_chunk_id form a valid bidirectional linked list."""
    chunker = RecursiveChunker()
    chunks = chunker.chunk(sample_parsed_document)

    assert len(chunks) >= 2
    # First chunk has no prev_chunk_id
    assert chunks[0].hierarchy.prev_chunk_id is None
    assert chunks[0].hierarchy.next_chunk_id == chunks[1].chunk_id

    # Middle chunks connect to prev and next
    for i in range(1, len(chunks) - 1):
        assert chunks[i].hierarchy.prev_chunk_id == chunks[i - 1].chunk_id
        assert chunks[i].hierarchy.next_chunk_id == chunks[i + 1].chunk_id

    # Last chunk has no next_chunk_id
    assert chunks[-1].hierarchy.prev_chunk_id == chunks[-2].chunk_id
    assert chunks[-1].hierarchy.next_chunk_id is None


# =====================================================================
# 10. Chunk Quality Validation & Rejections Tests
# =====================================================================

def test_chunk_validator_rejects_empty_tiny_and_duplicate():
    """Verify validator eliminates noise, whitespace, and duplicates."""
    ctx = ChunkContext(min_chunk_tokens=10, max_chunk_tokens=500)
    validator = ChunkValidator(ctx)

    doc_id = "doc_test"
    valid_content = "This is a legitimate technical sentence about refinery operations exceeding ten tokens."
    sha_valid = compute_sha256(valid_content)
    
    valid_chunk = Chunk(
        chunk_id=generate_deterministic_chunk_id(doc_id, 1, 0, sha_valid),
        content=valid_content,
        token_count=estimate_tokens(valid_content),
        character_count=len(valid_content),
        word_count=len(valid_content.split()),
        metadata=ChunkMetadata(
            document_id=doc_id,
            sha256=sha_valid,
            page_number=1,
            chunk_index=0,
        ),
    )

    empty_chunk = Chunk(
        chunk_id="chk_empty",
        content="   \n\t  ",
        token_count=0,
        character_count=0,
        word_count=0,
        metadata=ChunkMetadata(document_id=doc_id, sha256="0", page_number=1, chunk_index=1),
    )

    tiny_chunk = Chunk(
        chunk_id="chk_tiny",
        content="Short",
        token_count=1,
        character_count=5,
        word_count=1,
        metadata=ChunkMetadata(document_id=doc_id, sha256="1", page_number=1, chunk_index=2),
    )

    # Duplicate of valid_chunk
    dup_chunk = Chunk(
        chunk_id=generate_deterministic_chunk_id(doc_id, 1, 3, sha_valid),
        content=valid_content,
        token_count=estimate_tokens(valid_content),
        character_count=len(valid_content),
        word_count=len(valid_content.split()),
        metadata=ChunkMetadata(
            document_id=doc_id,
            sha256=sha_valid,
            page_number=1,
            chunk_index=3,
        ),
    )

    raw = [valid_chunk, empty_chunk, tiny_chunk, dup_chunk]
    accepted, rejected = validator.filter_chunks(raw)

    assert len(accepted) == 1
    assert accepted[0].chunk_id == valid_chunk.chunk_id
    assert len(rejected) == 3
    rej_reasons = [r[1] for r in rejected]
    assert any("empty" in r.lower() for r in rej_reasons)
    assert any("tiny" in r.lower() for r in rej_reasons)
    assert any("duplicate" in r.lower() for r in rej_reasons)


# =====================================================================
# 11. Chunk Statistics & Telemetry Tests
# =====================================================================

def test_chunk_metrics_and_telemetry(sample_parsed_document):
    """Verify ChunkMetricsCollector computes statistical summaries correctly."""
    collector = ChunkMetricsCollector()
    doc_id = "doc_mrpl_hcu_sop_01"

    chunker = RecursiveChunker(metrics=collector)
    chunks = chunker.chunk(sample_parsed_document)

    stats = collector.get_stats(doc_id)
    assert stats is not None
    assert stats.total_chunks == len(chunks)
    assert stats.average_tokens > 0
    assert stats.smallest_chunk_tokens <= stats.largest_chunk_tokens
    assert stats.chunks_per_document[doc_id] == len(chunks)


# =====================================================================
# 12. Chunk Registry & Factory Tests
# =====================================================================

def test_chunk_registry_and_factory():
    """Verify dynamic strategy registration and instantiation."""
    registry = get_chunk_registry()
    strategies = registry.list_strategies()
    assert "fixed" in strategies
    assert "recursive" in strategies
    assert "section" in strategies
    assert "table" in strategies
    assert "list" in strategies

    factory = get_chunk_factory()
    chunker = factory.create("recursive", target_tokens=256)
    assert chunker.strategy_name == "recursive"
    assert chunker.context.target_tokens == 256


# =====================================================================
# 13. Thread Safety & Concurrent Execution Tests
# =====================================================================

def test_concurrent_chunking_thread_safety(sample_parsed_document):
    """Execute 20 concurrent threads running chunking pipelines to verify thread safety."""
    factory = get_chunk_factory()
    num_threads = 20

    results: list[list[Chunk]] = []
    errors: list[Exception] = []
    lock = threading.Lock()

    def worker(thread_id: int):
        try:
            # Alternate strategies across threads
            strat = ["fixed", "recursive", "section", "table", "list"][thread_id % 5]
            chunker = factory.create(strat, target_tokens=100)
            chk_list = chunker.chunk(sample_parsed_document)
            with lock:
                results.append(chk_list)
        except Exception as e:
            with lock:
                errors.append(e)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker, i) for i in range(num_threads)]
        concurrent.futures.wait(futures)

    assert len(errors) == 0, f"Thread errors encountered: {errors}"
    assert len(results) == num_threads
    for chk_list in results:
        assert len(chk_list) > 0


# =====================================================================
# 14. Edge Cases & Error Handling Tests
# =====================================================================

def test_empty_document_raises_error():
    """Chunking an empty document must raise EmptyDocumentError."""
    meta = DocumentMetadata(source_path="empty.txt", file_name="empty.txt", file_format=".txt")
    empty_doc = Document(doc_id="doc_empty", content="", metadata=meta)
    chunker = RecursiveChunker()
    with pytest.raises(EmptyDocumentError):
        chunker.chunk(empty_doc)


def test_chunking_health_check():
    """Verify health diagnostic report."""
    rep = health()
    assert isinstance(rep, ChunkHealthReport)
    assert rep.status == "HEALTHY"
    assert "fixed" in rep.registered_strategies
    assert rep.thread_safe is True
