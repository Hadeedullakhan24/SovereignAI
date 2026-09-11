# Integration Notes

**Project:** Sovereign On-Premise Agentic AI Workbench (SIH26117 / MRPL)  
**Component:** Member 1 — Knowledge Base / RAG Engine  

---

## 1. Upstream & Downstream Contracts

- **Upstream Input:**
  - Files identified and cataloged in `outputs/manifest.json` from Milestone 2.
  - Raw filesystem paths passed directly to `global_loader_factory.load(file_path, context)`.
- **Downstream Consumers:**
  - **Milestone 3B (Deep Document Parsers):** Takes unified `Document` instances from loaders and applies deep section hierarchy parsing and table extraction.
  - **Milestone 4 (Smart Chunking Engine):** Uses `Document.content` and `Document.metadata` to generate semantically bounded `Chunk` objects.
  - **Member 2 (OCR / Vision Subsystem):** Inspects `Document.metadata.image_reference` to run OCR models on drawings, P&IDs, and scanned certificates.
  - **Member 3 (Agent / LLM Subsystem):** Reads `Document.metadata.processing_history` and `Document.metadata.source_path` for authoritative citation grounding.

## 2. Loader Interface Contract

```python
from rag_engine.loaders import global_loader_factory, ProcessingContext

# Program against the factory interface
context = ProcessingContext(category="manuals", subcategory="pumps")
document = global_loader_factory.load("d:/SovereignAI/datasets/manuals/sample.pdf", context=context)

# Universal Document attributes
assert document.uuid is not None
assert document.content is not None
assert document.metadata.file_format == ".pdf"
assert document.metadata.loading_status == "SUCCESS"
```

## 3. Observability & Telemetry Contracts

- **Metrics:** `rag_engine.loaders.global_metrics.get_summary()` returns operational counts, average duration (ms), and primary vs fallback driver usage.
- **Events:** `rag_engine.loaders.global_event_bus.subscribe(DocumentLoaded, callback)` allows real-time audit logging for the workbench.

---

## 4. Parser Interface Contract (Milestone 3B)

```python
from rag_engine.loaders import global_loader_factory
from rag_engine.parsers import ParserFactory, ParsingContext, get_profile

# 1. Load document via loader framework
doc = global_loader_factory.load("d:/SovereignAI/datasets/manuals/sample.pdf")

# 2. Configure parsing context with refinery profile
context = ParsingContext(
    profile=get_profile("mrpl"),
    extract_sections=True,
    extract_tables=True,
    extract_equipment=True,
    extract_relationships=True,
    extract_safety=True,
    run_validation=True,
)

# 3. Resolve parser via DocumentClassifier + ParserFactory
parser_factory = ParserFactory()
parser, classification = parser_factory.get_parser(doc)

# 4. Execute deterministic parsing
parsed_doc = parser.parse(doc, context=context)

# 5. Access structured refinery knowledge
print("Title:", parsed_doc.title)
print("Sections count:", len(parsed_doc.sections))
print("Tables count:", len(parsed_doc.tables))
print("Equipment extracted:", [e.tag for e in parsed_doc.equipment])
print("EntityGraph edges:", len(parsed_doc.entity_graph.edges))
print("Safety notices:", len(parsed_doc.warnings))
print("Validation Report Valid:", parsed_doc.validation_report.is_valid)

# 6. Streaming support for large manuals (1000+ pages)
for section in parser.parse_stream(doc, context=context):
    print("Section heading:", section.title)
```

---

## 5. Cleaning & Normalization Interface Contract (Milestone 4)

```python
from rag_engine.preprocessing import CleaningPipeline

pipeline = CleaningPipeline(strict_token_verification=True)
clean_doc = pipeline.clean(parsed_doc)

assert clean_doc.cleaning_status == "CLEANED"
assert len(clean_doc.cleaning_statistics.stages_executed) == 12
```

---

## 6. Enterprise Chunking Engine Interface Contract (Milestone 5)

```python
from rag_engine.chunking import get_chunk_factory, ChunkContext

# 1. Instantiate or auto-select chunker based on document structure
factory = get_chunk_factory()
chunker = factory.for_document(clean_doc)

# 2. Execute chunking
chunks = chunker.chunk(clean_doc)

# 3. Retrieve retrieval-ready Chunk objects
for chk in chunks:
    # Guaranteed deterministic ID (e.g. 'chk_3835c947_p1_0000_7b4c6fc6') - NEVER UUID
    print("Chunk ID:", chk.chunk_id)
    # Stable content SHA256 for embedding caching in Milestone 6
    print("Content SHA-256:", chk.metadata.sha256)
    # Tokens without LLM
    print("Tokens:", chk.token_count)
    # Inherited refinery metadata
    print("Equipment Tags:", chk.metadata.equipment_entities)
    print("Safety Standards:", chk.metadata.safety_entities)
    # Bidirectional hierarchy
    print("Previous Chunk ID:", chk.hierarchy.prev_chunk_id)
    print("Next Chunk ID:", chk.hierarchy.next_chunk_id)
```

---

## 7. Integration Points for Milestone 6 (Embedding Pipeline)

1. **Embedding Input Contract:**
   - The embedding pipeline consumes `list[Chunk]` produced by `BaseChunker.chunk()`.
   - The embedder reads `chk.content` (and optionally `chk.metadata.heading_path` for contextual embedding prompts).
2. **Deterministic Content Hash Caching:**
   - Before executing vector model inference, the embedder checks local embedding cache using `chk.metadata.sha256`.
   - If the hash exists in the cache, retrieve the pre-computed vector with zero CPU/GPU inference.
3. **Sequential & Hierarchical Chunk Navigation:**
   - `chk.hierarchy.prev_chunk_id` and `chk.hierarchy.next_chunk_id` allow vector search results to dynamically expand into adjacent chunks during generation without re-retrieval.
4. **Metadata Filtering in Vector Store (Milestone 7):**
   - Chunks contain structured query filter attributes (`plant_unit`, `category`, `equipment_entities`, `safety_entities`, `is_table_chunk`, `page_number`) ready for ChromaDB/Qdrant metadata payloads.

---

## 8. Offline Embedding Pipeline Interface Contract (Milestone 6)

```python
from rag_engine.embeddings import EmbeddingPipeline

# 1. Initialize pipeline with local model and SQLite cache
pipeline = EmbeddingPipeline(
    model_name="BAAI/bge-small-en-v1.5",
    models_dir="models/embeddings",
    cache_db_path="cache/embeddings/embedding_cache.db",
    enable_cache=True,
    enable_checkpoints=True,
)

# 2. Embed chunks into retrieval-optimized dense vectors
embedded_chunks = pipeline.embed_chunks(chunks, job_id="job_001")

# 3. Access EmbeddedChunk attributes for Milestone 7 Vector Database
for ech in embedded_chunks:
    print("Chunk ID:", ech.chunk_id)
    print("Vector length:", len(ech.embedding))  # 384 or 768
    print("Vector checksum:", ech.vector_checksum)
    print("Validation status:", ech.validation_status)  # "VALID" or "CACHED"
    print("Inherited equipment:", ech.metadata.equipment_entities)

# 4. Check cache efficiency
metrics = pipeline.get_metrics()
print("Cache reuse percentage:", metrics.cache_reuse_percentage)
```

---

## 9. Integration Points for Milestone 7 (Enterprise Vector Storage Platform)

1. **Vector Ingestion Input Contract:**
   - Vector Store consumes `list[EmbeddedChunk]` produced by Milestone 6.
   - Primary key mapping: `ech.chunk_id` (deterministic string ID, e.g. `chk_3835c947_p1_0000_7b4c6fc6`) converted to RFC 4122 UUIDv5 for Qdrant storage.
   - Dense vector: `ech.embedding` (384-dim or 768-dim float32 unit-normalized vector).
   - Document text payload: `ech.text_preview` and serialized chunk metadata.
2. **Universal Repository Layer (`VectorRepository`):**
   - High-level pipelines and agent tools interact exclusively with `VectorRepository`. Direct calls to `QdrantClient` are prohibited.
   - Decoupled backend architecture enables pluggability of Qdrant, Milvus, FAISS, or pgvector.
3. **Multi-Collection Routing Contract (`CollectionRouter`):**
   - Automatically routes chunks based on document category and metadata into dedicated collections:
     - `mrpl_manuals_v1`, `mrpl_inspection_v1`, `mrpl_maintenance_v1`, `mrpl_safety_v1`
     - `mrpl_emails_v1`, `mrpl_drawings_v1`, `mrpl_pids_v1`, `mrpl_sops_v1`, `mrpl_general_v1`
4. **Collection Versioning Contract (`CollectionVersionManager`):**
   - Supports atomic version switches (`v1` -> `v2`), rollback, and data migration without data loss.
5. **Automated Payload Indexing Contract (`PayloadIndexManager`):**
   - Automatically indexes high-frequency search attributes:
     - `document_id`, `document_type`, `category`, `plant_unit`
     - `equipment_entities`, `safety_entities`, `page_number`, `section_title`
     - `revision`, `version`, `source_file`, `language`
6. **Maintenance & Optimization (`VectorOptimizer`):**
   - Provides background segment optimization, payload compaction, and disk vacuuming.

---

## 10. Vector Repository Interface Contract (Milestone 7)

```python
from rag_engine.vector_store import VectorRepository, get_vector_repository

# 1. Access the enterprise repository
repo = get_vector_repository()

# 2. Ingest embedded chunks (automatic routing, validation, transaction logging)
result = repo.save_chunks(embedded_chunks, document_id="DOC_MANUAL_HCU_001")
print(f"Indexed: {result.inserted}, Skipped: {result.skipped}, Duration: {result.duration_ms}ms")

# 3. Dense vector search with metadata filtering
query_vec = [0.05] * 384
matches = repo.find_by_vector(
    query_vector=query_vec,
    category="Manual",
    filters={"plant_unit": "HCU", "equipment_tags": "P-101"},
    limit=5,
)
for match in matches:
    print(f"Chunk ID: {match.chunk_id}, Score: {match.score:.4f}, Preview: {match.text_preview[:80]}")

# 4. Filtered lookup by equipment tag or document ID
equip_chunks = repo.find_by_equipment("P-101", limit=10)
doc_chunks = repo.find_by_document("DOC_MANUAL_HCU_001")

# 5. Adjacent context expansion without re-querying
neighbors = repo.find_neighbors(chunk_id=matches[0].chunk_id, window=2)

# 6. Enterprise telemetry & health
health = repo.health()
stats = repo.stats()
print(f"Health: {health.status}, Active Collections: {stats.active_collections}, Total Vectors: {stats.total_vectors}")
```

---

## 11. Downstream Contracts for Milestone 8 (Retriever Engine)

1. **Hybrid Dense + Sparse Integration:**
   - Milestone 8 queries `VectorRepository.find_by_vector(...)` for dense semantic recall.
   - Sparse BM25 inverted index results are merged via Reciprocal Rank Fusion (RRF).
2. **Context Window Expansion:**
   - Retrieval results call `VectorRepository.find_neighbors(chunk_id, window=2)` to expand surrounding context before sending to the reranker.

---

## 12. Retrieval Engine Interface Contract (Milestone 8)

```python
from rag_engine.retrieval import RetrievalPipeline, get_retriever

# 1. Initialize master production retrieval pipeline
pipeline = RetrievalPipeline(strategy_name="adaptive", use_cache=True)

# 2. Execute natural language query with automatic intent analysis, dual-channel search,
# RRF fusion, local CrossEncoder reranking, neighbor expansion, and token packing
result = pipeline.execute("What is the discharge pressure limit of pump P-203 in CDU-1?", top_k=5)

# 3. Consume retrieved chunks and explainability
for chunk in result.scored_chunks:
    print(f"[{chunk.rank}] Chunk ID: {chunk.chunk.chunk_id} | Score: {chunk.score:.4f}")
    print(f"Explainability: {chunk.explainability}")

# 4. Consume verified citations for grounding
for cit in result.citations:
    print(f"{cit.citation_id} Source: {cit.document_name} | Page {cit.page_number} | Equipment: {cit.equipment_tags}")
    print(f"Quote: \"{cit.verbatim_quote}\"")

# 5. Inject token-budgeted packed context into downstream local LLM prompt
llm_prompt = f"{result.packed_context}\n\nUser Question: {result.query}\nAnswer:"

# 6. Check retrieval health & metrics
health = pipeline.check_health()
metrics = pipeline.get_metrics_summary()
print(f"Status: {health.status} | Cache Hit Ratio: {metrics['cache_hit_ratio']}")
```
