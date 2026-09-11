# Engineering Decisions Record (EDR)

**Project:** Sovereign On-Premise Agentic AI Workbench (SIH26117 / MRPL)  
**Lead Component:** Member 1 — Knowledge Base / RAG Engine  

---

## EDR-001: Clean Architecture and Inversion of Control
- **Date:** 2026-09-06
- **Status:** APPROVED
- **Context:** The system must run 100% offline, on-premise at MRPL without external API calls or telemetry. Multiple engineers will implement separate subsystems.
- **Decision:** Establish strict layer boundaries. Domain entities and schemas are defined as immutable Pydantic v2 DTOs (`ConfigDict(frozen=True)`). High-level pipelines communicate exclusively through Abstract Base Classes (ABCs) in `interfaces/`. Concrete implementations are injected at runtime via configuration.

---

## EDR-002: Modular Dataset Engine (`dataset_engine`) for Milestone 2
- **Date:** 2026-09-06
- **Status:** APPROVED
- **Context:** Ingesting 29,000+ files of heterogeneous types (refinery manuals, P&IDs, safety protocols, inspection reports, images) requires a deterministic, fault-tolerant validation engine before any chunking or embedding occurs.
- **Decision:** Implement a dedicated, decoupled `dataset_engine` comprising 7 focused single-responsibility modules:
  1. `scanner.py`: Traverses filesystem, respects ignored patterns, categorizes files.
  2. `validator.py`: Verifies existence, read permissions, non-empty content, size limits, and format-specific magic bytes/headers.
  3. `hash_generator.py`: Generates streaming SHA-256 digests using 64KB buffers to avoid memory spikes.
  4. `duplicate_detector.py`: Detects content (hash) duplicates, filename clashes, and path collisions without deleting files.
  5. `manifest_generator.py`: Emits a standardized `manifest.json` with UUIDs, timestamps, and file metadata.
  6. `statistics.py`: Produces `dataset_statistics.json` and human-readable `dataset_health_report.md`.
  7. `orchestrator.py`: Coordinates the full pipeline sequentially with comprehensive structured logging.

---

## EDR-003: Externalized Validation Configuration
- **Date:** 2026-09-06
- **Status:** APPROVED
- **Context:** Hardcoding file extensions, size limits, or category mappings creates tight coupling and breaks adaptability across refinery sites.
- **Decision:** All validation thresholds, supported extensions, ignored patterns, and category-to-doctype mappings are loaded dynamically from `config/validation_rules.yaml`.

---

## EDR-004: Universal Document Loading Framework & Dual-Driver Fallback
- **Date:** 2026-09-06
- **Status:** APPROVED
- **Context:** In high-security, air-gapped refinery machines, third-party libraries (e.g. `pypdf`, `python-docx`, `openpyxl`, `Pillow`) may not always be pre-installed or updated. Additionally, hardcoded `if/elif/else` dispatch chains violate the Open/Closed Principle.
- **Decision:**
  1. Implement a thread-safe `LoaderRegistry` that maps normalized file extensions and MIME types to loader classes using dynamic registration and the `@register_loader` decorator.
  2. Implement a `LoaderFactory` that resolves MIME type first, then extension fallback.
  3. Every concrete loader inherits from `BaseLoader` and implements a **Dual-Driver Pattern**:
     - **Primary Driver:** Uses specialized third-party packages if available.
     - **Fallback Driver:** Pure Python standard library parsing (`zipfile` + `xml.etree.ElementTree`, `csv`, `struct`, raw byte/text stream decoders).
  4. If the primary driver fails or is absent, the loader automatically engages the fallback driver and emits a `FallbackActivated` audit event without crashing the ingestion pipeline.
  5. The unified `Document` schema is extended with all downstream attributes (UUID, checksum, lifecycle state, token estimates, page/word/char counts, image references, and processing history) to ensure zero future schema redesigns.

---

## EDR-005: Deep Document Parsing Engine, Profiles & Deterministic Entity Graphs
- **Date:** 2026-09-06
- **Status:** APPROVED
- **Context:** Raw loaded documents from Milestone 3A must be decomposed into rich semantic structures (sections, tables, equipment, warnings, operational metadata, cross-references) without nondeterministic AI/LLM calls, and without hardcoding MRPL-specific conventions into core parser logic.
- **Decision:**
  1. **Pre-parsing Deterministic Classification:** `ParserFactory` first invokes `DocumentClassifier` to classify the incoming document into domain categories (`Manual`, `Inspection Report`, `Safety Document`, `Maintenance Record`, `Email`, `Engineering Drawing`, `Template`, `Unknown`) using folder hierarchy, file naming patterns, metadata tags, and header signatures.
  2. **Externalized Profile Architecture (`profiles/`):** All refinery equipment tag patterns, plant units, equipment type prefixes, and standard names are encapsulated in `RefineryProfile` subclasses (`GenericRefineryProfile`, `MRPLProfile`). Future refineries (HPCL, IOCL, ONGC, BPCL, Reliance) are supported purely by declaring a new profile without parser code modification.
  3. **Rich `ParsedDocument` Representation:** Every extracted entity preserves confidence scores, raw text, normalized text, page numbers, section headers, and character offsets (`CitationCoordinates`) for full auditability and future citation indexing.
  4. **EntityGraph Construction:** Discovered equipment entities within sentence contexts are mapped into directed graph structures (`EntityGraph` containing nodes and directed edges like `connected_to`, `feeds`, `regulates`, `isolates`, `monitors`) for direct agent navigation in future milestones.
  5. **Validation Layer (`ParserValidator`):** Explicitly verifies heading hierarchies (detecting level jumps and duplicate headings), table structural parity (row vs column counts), entity type consistency, and cross-reference integrity, producing an auditable `ValidationReport`.
  6. **Streaming Generator Interface:** Implemented `parse_stream` yielding `Section` instances sequentially, allowing lazy processing of 1,000+ page manuals without unbounded memory allocation.

---

## EDR-006: Cleaning & Normalization Engine, 12-Stage Pipeline & Zero-Corruption Token Masking
- **Date:** 2026-09-06
- **Status:** APPROVED
- **Context:** Text extracted from industrial engineering documents (PDFs, DOCX, CSVs, scans) contains OCR artifacts, broken line wraps, hyphenated splits, repeated headers/footers, and irregular whitespace. Downstream chunking and embedding pipelines fail or hallucinate when text is corrupt. However, naive cleaning or standard NFKC normalization corrupts critical refinery tokens (`kg/cm²` -> `kg/cm2`, `m³/hr` -> `m3/hr`, `Pump P-203` -> `P203`).
- **Decision:**
  1. **Deterministic 12-Stage Pipeline:** Execute cleaning in a strict, deterministic sequence:
     - Stage 1: Unicode Normalization (NFKC)
     - Stage 2: Encoding Normalization (control characters, zero-width spaces, smart quotes)
     - Stage 3: Whitespace Normalization (horizontal space collapse, line end trimming)
     - Stage 4: Line Ending Normalization (CRLF/CR -> LF, 3+ newlines -> 2)
     - Stage 5: Broken Paragraph Reconstruction (soft wraps reassembled without merging headings/lists)
     - Stage 6: Hyphenated Word Reconstruction (`Oper-\nating` -> `Operating`, compound prefixes preserved)
     - Stage 7: Header/Footer Detection & Safe Removal (frequency thresholding across sections, never removing unique content)
     - Stage 8: Page Number Preservation & Citation Mapping (`PageMapper` generating `page_map`)
     - Stage 9: Table Whitespace Cleanup (cell text cleaned while preserving row/col grid and markdown representation)
     - Stage 10: Bullet Normalization (diverse bullets standardized to Markdown `- `)
     - Stage 11: List Normalization (numbered lists standardized to Markdown `1. `)
     - Stage 12: Engineering Token Protection & Verification
  2. **Sentinel Token Masking & Unmasking:** To protect refinery equipment tags, valve IDs, regulatory standards (API, OISD, ASME, PNGRB, ISO), and physical units (`kg/cm²`, `m³/hr`, `bar`, `°C`), tokens are identified and masked with non-colliding sentinels (`__ENG_TOKEN_{idx}__`) prior to Stage 1, and unmasked in Stage 12.
  3. **Strict Preservation Verification:** Stage 12 includes automated verification ensuring 100% of detected original tokens are present in the final cleaned text, raising `TokenProtectionError` or warnings if any degradation occurs.
  4. **Thread-Safe Architecture:** Shared components (`CleanerRegistry`, `CleanerFactory`, `CleaningMetricsCollector`, `CleaningEventBus`) utilize re-entrant locks (`threading.RLock`) to enable parallel cleaning across worker threads.
  5. **Auditability & Observability:** Every transformation is recorded in `document.processing_history` with before/after character counts and timestamps. The engine emits lifecycle events and tracks comprehensive telemetry via `CleaningMetrics`.

---

## EDR-007: Enterprise Chunking Engine, Strategy Matrix & Hierarchical Chaining
- **Date:** 2026-09-06
- **Status:** APPROVED
- **Context:** Downstream dense vector retrieval, hybrid BM25 search, and cross-encoder re-ranking depend fundamentally on chunk semantic cohesion. Arbitrary character slicing destroys tables mid-cell, fractures multi-step refinery operating procedures, strips heading context, and introduces non-deterministic chunk IDs (e.g. UUIDs) that invalidate vector database caches on subsequent ingestion runs.
- **Decision:**
  1. **Multi-Strategy Pluggable Architecture:**
     - `FixedChunker`: Sliding window in token or character mode with configurable overlap.
     - `RecursiveChunker`: Hierarchical semantic decomposition (`Document -> Section -> Subsection -> Paragraph -> Sentence -> Token window`), guaranteeing sentence integrity.
     - `SectionChunker`: Strictly respects H1-H6 boundaries in formal manuals and inspection reports with hierarchical breadcrumbs (`[1.0 HCU > 1.1 Reactor]`).
     - `TableChunker`: Isolates tabular data into dedicated chunks, preserving captions, column headers, and row integrity; multi-part splits repeat headers across all chunks.
     - `ListChunker`: Bundles numbered checklists, SOP procedures, and bullet items together to preserve operational logic for refinery technicians.
  2. **Deterministic Chunk Identifiers (Zero UUIDs):**
     - Chunk IDs are strictly derived via SHA-256 digests over `(document_id, page_number, chunk_index, content_hash)` formatted as `chk_{doc_hash8}_{page_str}_{chunk_index:04d}_{content_hash8}`.
     - Stable across pipeline re-runs, eliminating duplicate re-embedding in Milestone 6.
  3. **Metadata Inheritance & Domain Tagging:**
     - Every chunk automatically cascades parent document metadata, source coordinates, page numbers, heading paths, equipment tags (`R-101`, `P-203`), operating parameters (`150 bar`, `380°C`), and safety standards (`OISD-105`, `API-610`).
  4. **Parent-Child Hierarchy & Bidirectional Relational Chaining:**
     - Each chunk maintains bidirectional sequential pointers (`prev_chunk_id`, `next_chunk_id`) and parent tree depth (`parent_section_id`, `hierarchy_depth`), enabling downstream agents to expand retrieval context window dynamically.
  5. **Heuristic Token Estimation Without External LLMs:**
     - Deterministic BPE-heuristic token estimation using regex tokenization and character bounds, operating 100% offline with zero external model dependencies.
  6. **Quality Validation & Statistical Telemetry:**
     - `ChunkValidator` eliminates empty, tiny (<10 tokens), oversized, and duplicate chunks before downstream ingestion.
     - `ChunkMetricsCollector` tracks document- and dataset-level statistics (average size, min/max tokens, category/section distributions, throughput).

---

## EDR-008: Offline Embedding Pipeline and Persistent Vector Caching
- **Date:** 2026-09-06
- **Status:** APPROVED & IMPLEMENTED
- **Context:** Milestone 6 requires converting atomic `Chunk` objects from Milestone 5 into dense embedding vectors for downstream vector storage (Milestone 7) in an air-gapped refinery environment without cloud APIs.
- **Decisions:**
  1. **Local Open-Weight Model Architecture:**
     - Supported models: `BAAI/bge-small-en-v1.5` (default, 384 dimensions), `BAAI/bge-base-en-v1.5` (768 dimensions), `intfloat/e5-small-v2` (384 dimensions), and `intfloat/e5-base-v2` (768 dimensions).
     - Models downloaded once to `models/embeddings/` via `scripts/download_embedding_models.py` with cryptographic integrity verification (`config.json`, weights, tokenizer, dimensions).
     - Zero external network calls after download (`local_files_only=True`).
  2. **Deterministic Vector Caching (SQLite WAL):**
     - Cache key: `SHA-256(model_name + ":" + chunk_hash)`.
     - Stored as compact IEEE 754 float32 binary blobs in SQLite with Write-Ahead Logging (WAL).
     - Achieves over 1,300x speedup on cache hits, avoiding re-embedding unchanged documents.
  3. **Checkpoint & Resume State Machine:**
     - `CheckpointManager` records completed chunk IDs per indexing job, enabling interrupted batch jobs to resume without zeroing out progress.
  4. **Strict Vector Numerical Quality Validation:**
     - `VectorValidator` verifies vector dimensionality, ensures float32 precision, confirms unit L2 normalization ($||v||_2 \approx 1.0$), and enforces absolute zero NaN/Inf tolerance.
  5. **Dynamic Model Switching:**
     - `EmbeddingFactory` and `EmbeddingRegistry` allow hot-swapping between model sizes (e.g. 384-dim vs 768-dim) with instance pooling and telemetry tracking.

---

## EDR-009: Sovereign Enterprise Vector Storage Platform (Decoupled Repository, Qdrant Local Engine, Multi-Collection Routing & Versioning)
- **Date:** 2026-09-06
- **Status:** APPROVED (Architecture Design Phase)
- **Context:** Storing millions of vectors across 30,000+ heterogeneous refinery documents requires upgrading the vector storage layer into an enterprise-grade Vector Storage Platform. The system must operate 100% air-gapped and offline, support sub-15ms filtered vector search, maintain memory footprint < 1 GB RAM, avoid coupling business logic to any concrete vector database vendor, and provide long-term production governance inside MRPL.
- **Decisions:**
  1. **Qdrant Local (Embedded Mode) as Default Storage Driver:**
     - Deploy Qdrant in embedded mode using `qdrant-client` targeting local filesystem storage (`vector_db/qdrant/`). Zero Docker daemon, zero network ports, zero external SaaS.
     - Vectors and payloads configured with on-disk memory mapping (`on_disk=True`), bounding RAM consumption under 1 GB even for millions of vectors.
     - Deterministic UUID mapping: Translate Milestone 5 deterministic chunk string IDs (`chk_...`) into RFC 4122 UUIDv5 via `uuid.uuid5(QDRANT_CHUNK_NAMESPACE, chunk_id)` using a dedicated namespace (`3c87e382-7e04-4f24-913a-a10c7bf7ea88`) while preserving original IDs in payloads.
  2. **Universal Repository Layer (`VectorRepository`):**
     - All indexing, search, deletion, neighbor expansion, and payload filtering operations pass through `VectorRepository`.
     - Zero pipeline components interact directly with Qdrant. All storage operations abstract through `BaseVectorStore`.
     - Enables seamless future pluggability for Milvus, FAISS, pgvector, or ChromaDB via `VectorRegistry` and `VectorFactory`.
  3. **Deterministic Multi-Collection Routing (`CollectionRouter`):**
     - Rather than dumping all chunks into a monolithic collection, incoming chunks are deterministically routed to specialized collections:
       - `mrpl_manuals_v1`: Engineering manuals, vendor handbooks, design guides ($M=24, ef=200$).
       - `mrpl_inspection_v1`: Inspection reports, NDT sheets, corrosion surveys ($M=16, ef=100$).
       - `mrpl_maintenance_v1`: Work orders, corrective maintenance logs ($M=16, ef=100$).
       - `mrpl_safety_v1`: Safety standards (OISD/API), work permits, HAZOP sheets ($M=16, ef=100$).
       - `mrpl_emails_v1`: Technical emails and correspondence ($M=16, ef=100$).
       - `mrpl_drawings_v1`: Engineering drawings, schematics, GA drawings ($M=32, ef=250$).
       - `mrpl_pids_v1`: Piping & Instrumentation Diagrams ($M=32, ef=250$).
       - `mrpl_sops_v1`: Standard operating procedures and checklists ($M=16, ef=100$).
       - `mrpl_general_v1`: General bulletins and circulars fallback ($M=16, ef=100$).
  4. **Collection Versioning & Migration (`CollectionVersionManager`):**
     - Full support for versioned collections (`engineering_docs_v1`, `v2`, `v3`).
     - Atomic operational alias switching (`activate_version`), instant zero-data-loss rollback (`rollback`), version comparison (`compare_versions`), and batch data migration.
  5. **Automated Payload Indexing (`PayloadIndexManager`):**
     - Automatically creates, validates, rebuilds, and optimizes payload indexes for high-cardinality metadata: `document_id`, `document_type`, `category`, `plant_unit`, `equipment_entities`, `safety_entities`, `page_number`, `section_title`, `revision`, `version`, `source_file`, `language`.
     - Prevents costly sequential segment scans, preserving sub-15ms filtered search latency.
  6. **Vector & Segment Optimizer (`VectorOptimizer`):**
     - Background maintenance engine supporting segment merging, payload dictionary compaction, soft-deleted point purging, vacuum disk reclamation, HNSW graph rebalancing, and OS cache compaction. Exposes auditable optimization metrics.
  7. **Full Lifecycle Governance (`VectorLifecycleManager`):**
     - Governs cold start initialization, metadata validation, schema upgrades, index pre-warming, graceful shutdown, and automatic crash recovery via transaction journals.
  8. **Strict Payload & Schema Validation (`PayloadValidator` & `SchemaValidator`):**
     - Validates payload consistency before insertion: required fields, types, field lengths, nested metadata, missing fields, enum validity, checksum consistency, float32 precision, unit normalization, and zero NaN/Inf tolerance.
  9. **Comprehensive Storage Telemetry (`StorageStatsCollector`):**
     - Tracks collection counts, vector counts, payload index statistics, average payload size, disk footprint, HNSW topology, tombstone counts, optimization history, and snapshot history.
  10. **Future Scalability & Cluster Readiness:**
      - Architected without single-process locking. `QdrantVectorStore` switches dynamically between embedded path dispatch and URL dispatch (`http://qdrant-cluster:6333`) without pipeline code changes.

---

## EDR-010: Enterprise Hybrid Retrieval Engine (Dual-Channel Candidate Search, BM25 Indexing, RRF Fusion, Local Neural Cross-Encoder Reranking & Deterministic Citations)
- **Date:** 2026-09-09
- **Status:** APPROVED & IMPLEMENTED (Milestone 8)
- **Context:** Downstream reasoning and generation by local LLMs at MRPL requires high-precision, low-latency, and 100% air-gapped retrieval. Relying purely on dense vector similarity causes severe false-positive clustering on alphanumeric refinery tags (`P-203` vs `P-204`), line codes (`LINE-101-CS`), and regulatory standards (`OISD-105`), while pure lexical BM25 misses semantic paraphrases. The engine must retrieve, rerank, expand, and pack context strictly within token budgets without hallucinations or internet dependencies.
- **Decisions:**
  1. **Multi-Stage Decoupled Pipeline:**
     - Stage 1: Query Analysis & Normalization (`QueryAnalyzer`, `QueryNormalizer`, `QueryRewriter`). Extracts equipment tags, standards, units, and classifies intent into 6 operational archetypes.
     - Stage 2: Metadata Filter Planning (`MetadataFilterPlanner`). Synthesizes payload filters automatically before search.
     - Stage 3: Dual-Channel Candidate Generation. Runs parallel Dense search via `VectorRepository.find_by_vector` and Sparse search via pure-Python `BM25Index`.
     - Stage 4: Score Calibration & Normalization (`ScoreNormalizer`). Provides Min-Max, Softmax, Z-score, and Percentile scaling to reconcile disparate score distributions.
     - Stage 5: Weighted Reciprocal Rank Fusion (`ReciprocalRankFusion`). Merges ranked lists using $RRF(d) = w_{dense} \cdot \frac{1}{k + r_{dense}} + w_{sparse} \cdot \frac{1}{k + r_{sparse}}$ with configurable smoothing $k=60$.
     - Stage 6: Contextual Metadata Boosting (`MetadataBooster`). Awards score increments for exact equipment tag matches, plant unit matches, safety standards, and page proximity.
     - Stage 7: Local Neural Cross-Encoder Reranking (`LocalCrossEncoderReranker`). Evaluates full cross-attention between query and passage using local transformer models (`BAAI/bge-reranker-base`, `bge-reranker-large`, `ms-marco`) with deterministic offline fallback for air-gapped test environments.
     - Stage 8: Relational Context Expansion (`ContextExpander`). Traverses `ChunkHierarchy` (`prev_chunk_id`, `next_chunk_id`) to expand adjacent sequential chunks without duplicates.
     - Stage 9: Semantic & Coordinate Deduplication (`DuplicateRemover`). Eliminates near-duplicate passages, duplicate tables, and duplicate citation anchors.
     - Stage 10: Deterministic Citation Construction (`CitationBuilder`). Maps every chunk to auditable `CitationBundle` anchors (`[1]`, `[2]`) with verbatim quotes and document geometry.
     - Stage 11: Token-Aware Context Packing (`ContextPacker`). Respects configured token budgets and strictly protects table rows from mid-row truncation.
     - Stage 12: Persistent SQLite WAL Query Caching (`QueryCache`). Keys retrieval results by `SHA256(query + config)` with TTL expiration.
  2. **Strict Offline Air-Gapped Enforcement:**
     - Zero external network requests, zero cloud endpoints. Local weights only.
  3. **Adaptive Query Scaling:**
     - Dynamically adjusts Top-K (5 for simple lookup, 30 for troubleshooting), fusion weights, neighbor expansion, and token budgets based on query archetype.
  4. **Full Observability & Diagnostics:**
     - Emits domain lifecycle events via `RetrievalEventBus` and records detailed stage-by-stage latencies, throughput, and cache hit ratios via `RetrievalMetricsCollector`.

---

## EDR-011: Enterprise In-Process Generation Engine & Context Assembly Gateway (Hugging Face Transformers, Cryptographic Prompt Stamping, Anti-Hallucination Guardrails & Bibliographic Provenance)
- **Date:** 2026-09-09
- **Status:** APPROVED & IMPLEMENTED (Milestone 9)
- **Context:** The final stage of Member 1 connects the Milestone 8 Hybrid Retrieval Engine with downstream local Large Language Models to generate technically accurate, safe, and verifiable answers. Relying on external inference servers (e.g. Ollama, vLLM daemons) introduces unmanaged background processes, port collisions, and deployment friction on isolated refinery edge workstations. Furthermore, language models risk hallucinating nonexistent citations (`[7]` when only 2 chunks exist) or asserting ungrounded operating limits.
- **Decisions:**
  1. **Elimination of External Inference Daemons:**
     - Generation executes strictly in-process using Hugging Face `transformers` (`AutoTokenizer`, `AutoModelForCausalLM`, `local_files_only=True`). Open-weight models are downloaded once via `scripts/download_llm_models.py` into `models/llms/<model_name>/` and loaded without network calls.
     - A high-fidelity `DeterministicTestLLM` provides reproducible testing and benchmarking without requiring multi-gigabyte GPU allocations in automated CI.
  2. **Deterministic Cryptographic Lineage:**
     - Every prompt is hashed: $\text{prompt\_hash} = \text{SHA256}(\text{prompt\_version} + \text{system\_prompt} + \text{context} + \text{query})$.
     - Responses are cached in a persistent SQLite WAL database keyed by $\text{SHA256}(\text{prompt\_hash} + \text{chunk\_hashes} + \text{model\_name} + \text{parameters})$.
  3. **Structured Conversation Memory:**
     - Replaces unstructured string appending with `ConversationMemory` tracking typed `TurnRecord` objects (`session_id`, `user_query`, `retrieved_chunk_ids`, `citations`, `response`, `metadata`) with sliding-window capacity control.
  4. **Strict Guardrails & Anti-Hallucination:**
     - `CitationValidator` cross-checks every bracketed anchor `[n]` against the authentic chunk map, stripping phantom citations.
     - `HallucinationGuard` extracts numerical parameters, units, and equipment tags from generated text and verifies their presence in retrieved source text.
     - `SafetyValidator` detects prompt injection attacks and redacts sensitive credentials.
     - `ConfidenceScorer` produces a composite score from retrieval relevance, citation precision, and grounding overlap.
  5. **Master RAG Pipeline Facade:**
     - `RAGPipeline` provides a unified entry point combining `RetrievalPipeline.execute()` and `GenerationPipeline.generate()` returning `RAGResponse`.
