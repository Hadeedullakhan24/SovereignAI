# Changelog

All notable changes to the Sovereign On-Premise Agentic AI Workbench (SIH26117 / MRPL) will be documented in this file.

---

## [Milestone 9 - Final Gate Review & Architectural Completion] - 2026-09-10

### Added
- **Formal Base Interfaces (`rag_engine/interfaces/base_prompt.py`):**
  - Standardized abstract contracts: `BasePromptBuilder`, `BasePromptTemplate`, `BasePromptContextBuilder`, `BaseTokenBudgetManager`, `BaseContextCompressor`, `BaseConversationFormatter`, `BaseCitationFormatter`, `BasePromptValidator`, `BaseSystemPromptManager`. Exported directly in `rag_engine/interfaces`.
- **Milestone 9 Prompt Engineering Schemas (`rag_engine/schemas/prompt.py`):**
  - Typed Pydantic v2 schemas: `RetrievedPrompt`, `PromptPayload`, `SystemPrompt`, `ContextWindow`, `ConversationTurn`, `PromptValidationResult`. Exported in `rag_engine/schemas`.
- **Modular Prompt Architecture Modules (`rag_engine/generation/prompt/`):**
  - `prompt_pipeline.py`: Orchestrated prompt assembly pipeline (`PromptPipeline`) implementing Retrieval evidence -> Context Packing -> Compression -> Citation Anchoring -> Conversation Formatting -> System Prompt Formatting -> Assembly -> Validation -> Cryptographic Lineage -> `RetrievedPrompt`.
  - `prompt_factory.py`: Central thread-safe factory (`PromptFactory`) with singleton caching.
  - `system_prompt_manager.py`: Refinery domain persona and safety directive manager (`SystemPromptManager`).
  - `context_compressor.py`: Multi-strategy context compressor (`ContextCompressor` with `SELECTIVE_EXTRACTION`, `TRUNCATE`, `COMPACT`, `REDUNDANCY_PRUNING`).
  - `conversation_formatter.py`: Multi-turn chat memory formatter (`ConversationFormatter` supporting Markdown, ChatML, and Plain).
  - `citation_formatter.py`: Flexible bibliographic reference formatter (`CitationFormatter` supporting inline anchors, footnotes, tabular layout).
  - `prompt_validator.py`: Pre-generation safety, injection attack detection, token limits, and anchor audit validator (`PromptValidator`).
  - `prompt_metrics.py`: Prompt assembly telemetry tracker (`PromptMetrics`, `PromptMetricsCollector`).
  - `prompt_health.py`: Active canary diagnostics monitor (`PromptHealthMonitor`, `PromptHealthReport`).
  - `prompt_events.py`: Pub/Sub lifecycle event bus (`PromptEventBus`, `PromptEvent`, `PromptEventType`).
  - `prompt_exceptions.py`: Domain-specific exceptions (`PromptValidationError`, `PromptBudgetExceededError`, `PromptTemplateNotFoundError`, etc.).
  - `prompt_config.py`: Declarative Pydantic configuration (`PromptConfig`).
  - Explicit architectural aliases: `PromptRegistry = PromptTemplateRegistry`, `PromptContextBuilder = ContextWindowBuilder`.
- **Master Operational Pipeline Integration (`rag_engine/pipeline/rag_pipeline.py`):**
  - Added `build_retrieved_prompt()` method to `RAGPipeline` uniting Retrieval -> Context Packing -> Prompt Builder into verified `RetrievedPrompt`.
  - Complete chain verified operational: Dataset -> Loader -> Parser -> Cleaning -> Chunking -> Embedding -> Qdrant Vector Store -> Hybrid Retrieval -> Context Packing -> Prompt Builder -> RetrievedPrompt -> Generation -> Verified RAGResponse.
- **Air-Gapped Security Hardening (`rag_engine/embeddings/local_embedder.py`):**
  - Completely eliminated online fallback attempt in `LocalHuggingFaceEmbedder`; enforced strict `local_files_only=True` and immediate local path verification to eliminate any external network requests or timeouts in air-gapped environments.
- **Verification Suites:**
  - `tests/test_prompt_architecture.py`: 13 comprehensive tests validating all 19 prompt subsystem components.
  - `tests/test_rag_pipeline_end_to_end.py`: End-to-end integration test validating the entire 11-stage pipeline.
  - **187 total repository tests passing with zero failures (100% pass rate).**

---

## [Milestone 9] - 2026-09-09

### Added
- **Enterprise In-Process Generation Engine & Context Assembly Gateway (`rag_engine/generation/`):**
  - `models/hf_causal_lm.py` & `models/base_model.py`: 100% in-process offline execution using Hugging Face `transformers` (`AutoModelForCausalLM`, `AutoTokenizer`, `local_files_only=True`). Fully eliminates external inference servers (no Ollama, no vLLM).
  - `models/deterministic_test_llm.py`: High-fidelity in-memory deterministic model for repeatable CI testing and benchmarking without GPU memory overhead.
  - `models/model_registry.py` & `models/model_factory.py`: Dynamic registry and thread-safe factory with instance caching.
  - `prompt/prompt_templates.py`: 7 domain-specific prompt templates for MRPL operations (Equipment Lookup, SOP Retrieval, Maintenance, Safety Compliance, Troubleshooting, Comparison, General QA).
  - `prompt/prompt_builder.py`: Cryptographically versioned deterministic prompt synthesis ($\text{prompt\_hash} = \text{SHA256}(\dots)$).
  - `prompt/token_budget_manager.py`: Dynamic partition allocation (10% system, 20% history, 50% context, 20% generation).
  - `prompt/context_window_builder.py`: Table-preserving context window builder protecting markdown tables from mid-row truncation.
  - `memory/conversation_memory.py`: Structured multi-turn session tracking with sliding-window capacity control.
  - `memory/generation_cache.py`: Persistent SQLite WAL cache keyed by $\text{SHA256}(\text{prompt} + \text{chunks} + \text{model} + \text{params})$.
  - `guardrails/citation_validator.py`: Strict citation provenance verification and automatic phantom citation pruning.
  - `guardrails/hallucination_guard.py`: Technical entity cross-verification checking numerical parameters, units, and equipment tags against context.
  - `guardrails/safety_validator.py`: Prompt injection detection and sensitive credential redaction.
  - `guardrails/confidence_scorer.py`: Weighted composite confidence calculation (retrieval score + citation precision + grounding overlap).
  - `streaming_manager.py`: Token stream wrapper measuring Time-To-First-Token (TTFT) and tokens/sec throughput.
  - `response_formatter.py`: Verifiable bibliographic provenance references formatter with verbatim excerpts.
  - `generation_pipeline.py`: Master 11-stage generation orchestrator.
- **End-to-End Master RAG Pipeline (`rag_engine/pipeline/`):**
  - `rag_pipeline.py`: Master `RAGPipeline` coordinating Retrieval (M8) and Generation (M9) into unified `query()` and `stream_query()` returning `RAGResponse`.
- **Testing & Benchmarking:**
  - `tests/test_generation_engine.py`: 24 comprehensive tests passing in 16.22s.
  - `scripts/benchmark_generation.py`: Multi-scale benchmark suite achieving 351.12–383.91 QPS and sub-millisecond prompt construction (0.155 ms).
  - `scripts/download_llm_models.py`: One-time open-weight downloader utility for air-gapped deployment with SHA-256 manifest verification and offline loading test checks.
  - **173 total repository tests passing with zero failures and zero regressions (100% pass rate).**

---

## [Milestone 8] - 2026-09-09

### Added
- **Enterprise Hybrid Retrieval Engine (`rag_engine/retrieval/`):**
  - `query_analyzer.py` & `query_classifier.py`: Deep semantic intent classification (Equipment, Safety, Inspection, Maintenance, Procedure, Specification, Table, Numerical, Comparison, Troubleshooting) and domain entity extractor (equipment tags, plant units, standards, pressures, temperatures, line numbers, dates, revisions).
  - `query_normalizer.py`, `query_rewriter.py`, `query_expander.py`: Deterministic NFKC unicode normalization, equipment tag standard hyphenation (`P203` -> `P-203`), unit spacing (`150bar` -> `150 bar`), spelling corrections, and refinery acronym expansion.
  - `metadata_filter.py`: Automatic Qdrant `MetadataFilter` synthesis and fluent filter builder.
  - `bm25_retriever.py`: Persistent, incremental pure-Python Okapi BM25 index with disk storage, inverted term frequency mapping, and document removal.
  - `dense_retriever.py`: Dedicated vector retriever communicating exclusively with Milestone 7 `VectorRepository`.
  - `score_normalizer.py`: Configurable score calibration supporting Min-Max, Softmax, Z-score, and Percentile normalization.
  - `rrf_fusion.py`: Weighted Reciprocal Rank Fusion (RRF) with configurable smoothing factor $k=60$ and channel weights.
  - `metadata_booster.py`: Contextual score booster boosting scores for equipment, plant unit, safety standards, and page proximity.
  - `reranker.py`: Local offline Cross-Encoder reranker supporting `BAAI/bge-reranker-base`, `bge-reranker-large`, and `ms-marco` with deterministic air-gapped test fallback.
  - `context_expander.py`: Relational neighbor expansion navigating `ChunkHierarchy` (`prev_chunk_id`, `next_chunk_id`).
  - `duplicate_remover.py`: Semantic and coordinate deduplication for chunks and citation anchors.
  - `citation_builder.py`: Deterministic citation compiler creating `CitationBundle` anchors with verbatim quotes, document IDs, page numbers, and equipment tags.
  - `context_packer.py`: Token-aware prompt context assembler preserving table rows without mid-row cuts and enforcing strict token budgets.
  - `query_cache.py`: Persistent SQLite WAL query cache keyed by `SHA256(query + config)` with TTL expiration.
  - `hybrid_retriever.py`: Multi-channel candidate search combining dense and BM25 channels.
  - `adaptive_retriever.py`: Query-adaptive strategy dynamically tuning Top-K, fusion weights, neighbor expansion, and token budget.
  - `retrieval_pipeline.py`: Master 12-stage retrieval pipeline orchestrator.
  - `retrieval_factory.py` & `retrieval_registry.py`: Extensible plugin architecture supporting custom retrievers via `@register_retriever`.
  - `retrieval_validator.py`: Strict post-retrieval invariant validator.
  - `retrieval_health.py` & `retrieval_metrics.py`: Canary diagnostics monitor and stage latency telemetry collector.
- **Testing & Benchmarking:**
  - `tests/test_retrieval_engine.py`: 20 comprehensive unit, integration, and thread-safety tests (100% pass rate in 18.42s).
  - `scripts/benchmark_retrieval.py`: Stress benchmark evaluating 100, 500, 1000 queries. BM25 sub-2ms latency (651 QPS), Hybrid search ~80ms on CPU, End-to-end pipeline ~81ms on CPU.
  - 149 total repository tests passing with zero regressions and zero failures.

---

## [Milestone 7] - 2026-09-06

### Added
- **Enterprise Vector Storage Platform (`rag_engine/vector_store/`):**
  - `qdrant_store.py`: Production-grade `QdrantVectorStore` driver operating 100% offline in embedded local mode via `qdrant-client` 1.19.0. Implements HNSW graph indexing, memory mapping (`mmap`), transactional upserts, cosine similarity search, filtered queries, neighbor context traversal, vacuum, snapshotting, and canary health diagnostics.
  - `vector_repository.py`: Universal decoupled repository boundary (`VectorRepository`) exposing clean business methods (`save_chunks`, `find_by_vector`, `find_by_equipment`, `find_by_safety_standard`, `find_by_page_range`, `find_by_text`, `health`, `stats`), isolating retrieval pipelines from underlying database implementations.
  - `collection_router.py`: Rule-based deterministic routing engine dispatching documents and chunks across 8 refinery domain collections (`mrpl_manuals_v1`, `mrpl_safety_v1`, `mrpl_inspection_v1`, `mrpl_maintenance_v1`, `mrpl_emails_v1`, `mrpl_drawings_v1`, `mrpl_pids_v1`, `mrpl_sops_v1`, `mrpl_general_v1`).
  - `collection_version_manager.py`: Immutable collection versioning engine (`CollectionVersionManager`) supporting semantic versioning (`v1`, `v2`), atomic version switching, instant rollback, diff reporting, zero-downtime ledger tracking, and background migration.
  - `payload_index_manager.py`: Automated payload indexing manager (`PayloadIndexManager`) managing schema registration, automated creation, health audits, and defragmentation rebuilding for 16+ refinery metadata fields.
  - `vector_optimizer.py`: Segment compaction and disk vacuum engine (`VectorOptimizer`) consolidating memory-mapped segments, compacting payloads, purging soft-deleted points, and rebalancing HNSW graph topologies.
  - `vector_lifecycle_manager.py`: End-to-end lifecycle coordinator (`VectorLifecycleManager`) orchestrating cold-start initialization, schema verification, memory warming, dirty startup WAL recovery, and graceful shutdown.
  - `schema_validator.py` & `payload_validator.py`: Strict pre-ingestion validation engines verifying vector dimensionality, zero NaN/Inf tolerance, float precision, vector checksum integrity, mandatory refinery metadata fields, field string limits, and categorical enum conformity.
  - `storage_stats.py`: Telemetry and telemetry audit engine (`StorageStatsCollector`) collecting collection vector counts, segment counts, disk consumption, average payload sizes, and persisting historical audits to `vector_db/telemetry/storage_stats.json`.
  - `collection_config.py`: Declarative, backend-independent configuration models (`CollectionConfig`, `HNSWConfig`, `OptimizerConfig`, `ReplicationConfig`, `QuantizationConfig`, `VectorStoreConfig`) supporting on-disk storage, memory-mapped vectors, and seamless single-line configuration switching to distributed Qdrant Server/Cluster.
  - `transaction_manager.py`: Write-Ahead Logging (WAL) journal manager (`TransactionManager`) maintaining append-only `wal.jsonl` log, supporting atomic transactions, rollbacks, and automatic dirty shutdown recovery.
  - `incremental_indexer.py`: 3-way reconciliation engine (`IncrementalIndexer`) detecting New, Modified, Unchanged, and Deleted chunks using SHA-256 hashes, eliminating redundant vector writes.
  - `index_manager.py`: Master 8-stage ingestion orchestrator (`IndexManager`) coordinating validation, routing, diffing, transactional WAL logging, batch slicing, upserts, and manifest tracking.
  - `snapshot_manager.py`: Disaster recovery snapshot engine (`SnapshotManager`) producing verified `tar.gz` point-in-time archives with cryptographic SHA-256 checksums and automated restoration.
  - `vector_health.py`: Canary diagnostics monitor (`VectorHealthMonitor`) executing active write/search/delete probes.
  - `vector_metrics.py`: Telemetry collector (`VectorMetricsCollector`) tracking ingestion throughput and query latency distributions.
  - `vector_events.py`: Thread-safe publish-subscribe event bus (`VectorEventBus`) emitting 10 lifecycle events.
  - `vector_registry.py` & `vector_factory.py`: Extensible backend registry and dynamic driver factory with connection pooling.
  - `vector_utils.py`: Deterministic RFC 4122 UUIDv5 generator and cryptographic SHA-256 helpers.
  - `exceptions.py`: Comprehensive exception hierarchy with 14 specialized vector store exceptions.
- **Schemas (`rag_engine/schemas/vector_store.py`):**
  - `DistanceMetric`, `PayloadSchemaType`, `FilterOperator`, `FieldFilter`, `MetadataFilter`, `ScoredVectorChunk`, `IndexingResult`, `DiffPlan`, `CollectionStats`, `CollectionVersionInfo`, `VersionDiffReport`, `MigrationReport`, `PayloadIndexHealthReport`, `OptimizationMetrics`, `VectorDBHealthReport`, `StorageStatsReport`.
- **Scripts & Production Management Tools:**
  - `scripts/manage_vector_db.py`: Production CLI administration utility supporting `status`, `health`, `list-collections`, `stats`, `optimize`, `vacuum`, `backup`, and `restore`.
  - `scripts/benchmark_vector_db.py`: Stress and scale benchmark suite evaluating batch ingestion throughput, query latency (P50/P95/P99), filtered retrieval, neighbor context expansion, incremental reconciliation, compaction, and disaster recovery.
- **Testing & Benchmark Results:**
  - 16 new comprehensive tests in `tests/test_vector_database.py` (100% pass rate in 3.58s).
  - 129 total repository tests passing in 21.57s with zero failures, zero regressions, and full backward compatibility.
  - Stress benchmark verified: 520 chunks indexed, dense search P50 latency of **3.60 ms** (mean 3.78 ms), filtered retrieval latency of **16.54 ms**, neighbor traversal in **0.63 ms**, incremental diff reconciliation in **0.10s** (490 unchanged chunks skipped), and canary diagnostics latency of **66.38 ms**.

---

## [Milestone 6] - 2026-09-06

### Added
- **Offline Embedding Pipeline (`rag_engine/embeddings/`):**
  - `base_embedder.py`: Enhanced `BaseEmbedder` abstract interface supporting single text embedding, batch embedding, model prefixing (`embed_passage`, `embed_query`), dimension inspection, normalization checks, and device awareness.
  - `local_embedder.py`: Implemented `LocalHuggingFaceEmbedder` wrapping local sentence-transformers models from disk with `local_files_only=True`, plus `DeterministicTestEmbedder` providing offline, deterministic pseudo-random unit vectors for air-gapped test suites.
  - `embedding_registry.py`: Thread-safe model registry and catalog supporting canonical names, aliases, dimensions, and custom class bindings.
  - `embedding_factory.py`: Enterprise factory providing instance pooling, dynamic model switching, and graceful air-gapped test fallbacks.
  - `embedding_cache.py`: High-performance SQLite vector cache with Write-Ahead Logging (WAL) and compact IEEE 754 float32 binary packing; keyed by `SHA-256(model_name + ":" + chunk_hash)`.
  - `checkpoint_manager.py`: Interrupted job tracking engine enabling long-running batch embedding runs to resume from the last completed chunk without re-embedding.
  - `vector_validator.py`: Strict vector quality validator verifying dimensionality, float32 precision, zero NaN/Inf tolerance, unit L2 normalization ($||v||_2 \approx 1.0$), and cryptographic vector checksums.
  - `embedding_pipeline.py`: Master orchestrator coordinating batch inference, persistent vector caching, checkpointing, vector validation, and telemetry collection.
  - `embedding_metrics.py`: Telemetry collector tracking throughput (chunks/sec), latency, CPU & RAM utilization, cache hit/miss counts, and reuse percentage.
  - `exceptions.py`: Custom exception hierarchy (`EmbeddingError`, `ModelNotFoundError`, `ModelIntegrityError`, `VectorValidationError`, `CacheError`, `CheckpointError`, `DeviceError`).
- **Schemas (`rag_engine/schemas/embedding.py`):**
  - `EmbeddedChunk`: Production ingestion schema containing `chunk_id`, `chunk_hash`, `embedding`, `model_name`, `model_version`, `embedding_dimension`, `embedding_timestamp`, `vector_checksum`, `validation_status`, and inherited `metadata`.
  - `EmbeddingBatchRequest`, `EmbeddingBatchResponse`, and `EmbeddingMetrics` models.
- **Scripts & Tools:**
  - `scripts/download_embedding_models.py`: CLI tool to download BGE and E5 models once to `models/embeddings/` and verify weights, tokenizers, config, and dimensions.
  - `scripts/validate_datasets_embedding.py`: Real refinery dataset benchmark runner executing `Loader -> Parser -> Cleaner -> Chunker -> EmbeddingPipeline` on real MRPL documents.
- **Testing & Benchmarks:**
  - 20 dedicated unit and integration tests in `tests/test_embedding_pipeline.py` (100% pass rate).
  - 113 total repository tests passing in 18.30s with zero failures and zero regressions.
  - Real refinery benchmark verified on Emerson Handbook, OISD standards, and inspection reports: 135 chunks embedded in 170.99s (cold) and loaded in 0.1294s (warm) with a **1,321.6x cache speedup** and 100% vector validation pass rate.

---

## [Milestone 5] - 2026-09-06

### Added
- **Enterprise Chunking Engine (`rag_engine/chunking/`):**
  - `base_chunker.py`: Abstract Base Class implementing the template method pattern (validation, extraction, quality filtering, sequential linking, metrics collection, and lifecycle event publishing).
  - `fixed_chunker.py`: Fixed-size sliding window chunking in token-based and character-based modes with configurable overlap.
  - `recursive_chunker.py`: Hierarchical semantic decomposition (`Document -> Section -> Subsection -> Paragraph -> Sentence -> Token window`), guaranteeing sentence integrity without arbitrary cuts.
  - `section_chunker.py`: Section-aware chunking strictly preserving H1–H6 boundaries and injecting hierarchical heading breadcrumbs (`[1.0 HCU > 1.1 Reactor]`).
  - `table_chunker.py`: Table-aware chunking isolating tables into dedicated chunks, rendering clean Markdown, and repeating column headers and captions across multi-part table splits.
  - `list_chunker.py`: List-aware chunking keeping numbered procedures, startup/shutdown sequences, and checklists cohesive without breaking individual list items.
  - `metadata_inheritance.py`: Metadata inheritance engine cascading parent document metadata, page coordinates, plant units, equipment tags (`R-101`, `P-203`), operating limits (`150 bar`, `380°C`), and safety standards (`OISD-105`, `API-610`).
  - `hierarchy_builder.py`: Relational linking engine constructing bidirectional sequential chains (`prev_chunk_id`, `next_chunk_id`) and calculating document tree depth.
  - `chunk_validator.py`: Quality control validator rejecting empty/whitespace-only, non-alphanumeric noise, tiny (<10 tokens), oversized, and content-duplicate chunks.
  - `chunk_utils.py`: Heuristic token estimation (BPE/WordPiece correlation without LLMs), word/character counting, streaming SHA-256 calculation, and deterministic chunk ID generation (`chk_{doc_hash8}_{page_str}_{chunk_index:04d}_{content_hash8}`).
  - `chunk_factory.py`: Enterprise factory supporting explicit strategy creation and document-structure adaptive auto-selection.
  - `chunk_registry.py`: Thread-safe registry mapping strategy identifiers to chunker classes with `@register_chunker`.
  - `chunk_metrics.py`: Telemetry engine collecting document-level and global statistics (`ChunkStatistics`).
  - `chunk_events.py`: Thread-safe publish-subscribe event bus emitting `ChunkingStarted`, `ChunkCreated`, `ChunkValidationFailed`, `ChunkingFinished`, `ChunkingFailed`.
  - `chunk_health.py`: Diagnostics reporting engine exposing `health()`, `version()`, `dependencies()`, and `supported_features()`.
  - `exceptions.py`: Custom exception hierarchy (`ChunkingError`, `EmptyDocumentError`, `ValidationRejectionError`, `StrategyNotFoundError`, `OversizedChunkError`).
- **Enriched Chunk Schema Suite (`rag_engine/schemas/chunk.py`):**
  - `Chunk`: Retrieval-optimized atomic unit with deterministic ID, content SHA-256, token/word/character counts, rich metadata, and hierarchy.
  - `ChunkMetadata`: Enriched provenance, coordinates, equipment tags, safety entities, operating parameters, and strategy tags.
  - `ChunkHierarchy`: Parent document/section IDs, heading breadcrumbs, bidirectional prev/next links, and hierarchy depth.
  - `ChunkStatistics`: Statistical summaries including average/min/max tokens and characters, category/section distributions, and table/list counts.
- **Testing & Real Refinery Benchmarking:**
  - 18 dedicated unit, integration, and thread-safety tests (`tests/test_chunking_engine.py`) covering all 5 strategies, deduplication, deterministic hashing, sequential linking, and concurrent execution under 20 threads.
  - 93 total repository tests passing in 1.80s (100% pass rate).
  - Real refinery benchmark (`scripts/validate_datasets_chunking.py`) executed across 10 documents (including Emerson Control Valve Handbook, Fisher Manuals, OISD standards, and 10,000-row AI4I 2020 maintenance matrix) yielding 1,512 retrieval-ready chunks in 79.7s with zero errors.

---

## [Milestone 4] - 2026-09-06

### Added
- **Production Cleaning & Normalization Engine (`rag_engine/preprocessing/`):**
  - `cleaning_pipeline.py`: Production 12-stage sequential deterministic pipeline transforming `ParsedDocument` to `CleanParsedDocument`.
  - `engineering_token_protector.py`: Pre-stage-1 regex masking (`__ENG_TOKEN_{idx}__`), unmasking, and Stage-12 verification engine ensuring 100% preservation of refinery equipment tags (`Pump P-203`, `MOV-101`), standards (`API-610`, `OISD-105`, `ASME`, `PNGRB`, `ISO`), physical limits (`10 bar`, `250°C`, `MPa`, `psi`), and superscript units (`kg/cm²`, `m³/hr`).
  - `normalizers.py`:
    - `UnicodeNormalizer`: Stage 1 Unicode NFKC normalization.
    - `EncodingNormalizer`: Stage 2 control character removal, zero-width space elimination, and typographic smart-quote normalization.
    - `BulletNormalizer`: Stage 10 normalization converting heterogeneous bullets (`•`, `*`, `▪`, `►`) to Markdown `- `.
    - `ListNormalizer`: Stage 11 normalization converting numbered lists (`1)`, `(1)`, `1 -`) to Markdown `1. `.
  - `whitespace_cleaner.py`:
    - Stage 3: Horizontal whitespace collapse and line-end trimming.
    - Stage 4: Line ending normalization (`\r\n` and `\r` -> `\n`, max 2 consecutive newlines).
    - Stage 5: Broken paragraph reconstruction rejoining soft-wrapped sentences while preserving headings, lists, and tables.
    - Stage 6: Hyphenated word reconstruction (`Oper-\nating` -> `Operating`, compound prefixes preserved).
  - `header_footer_detector.py`: Stage 7 high-confidence detection and safe removal of repeated headers and footers across sections/pages without removing unique content.
  - `page_mapper.py`: Stage 8 page number preservation and citation coordinate mapping building `page_map: dict[int, PageMapEntry]`.
  - `table_cleaner.py`: Stage 9 table whitespace and cell cleanup with fast C-level matrix transpose (`zip(*rows)`) preserving grid dimensions, column parity, and coordinate integrity.
  - `text_cleaner.py`: Section and string level coordinator for composite text cleaning.
  - `base_cleaner.py`: Abstract Base Class for all cleaners and pipeline stages.
  - `registry.py` & `factory.py`: Thread-safe registry and factory utilizing `threading.RLock`.
  - `plugin_cleaner.py`: Runtime plugin discovery and loading system for external cleaner classes without framework modification.
  - `cleaning_metrics.py`: Observability telemetry tracking latency, characters removed, characters normalized, whitespace reductions, headers/footers removed, and protected tokens (`CleaningMetrics`, `CleaningMetricsCollector`).
  - `cleaning_events.py`: Thread-safe event bus publishing `CleaningStarted`, `CleaningFinished`, `CleaningFailed`, `HeaderRemoved`, `FooterRemoved`, `ProtectedTokenDetected`, `NormalizationApplied`.
  - `cleaning_health.py`: Diagnostic health reporting exposing `health()`, `version()`, `dependencies()`, and `supported_features()`.
  - `processing_history.py`: Detailed audit logging for each transformation step appended to document history.
  - `exceptions.py`: Custom exception hierarchy (`CleaningError`, `PipelineStageError`, `TokenProtectionError`, `PluginError`, `CorruptedDocumentError`).
- **Extended ParsedDocument Schema Suite (`rag_engine/schemas/`):**
  - `CleanParsedDocument`: Cleaned and normalized document representation ready for Chunking Engine (Milestone 5).
  - `CleaningStatistics`: Observable metrics and performance telemetry.
  - `PageMapEntry`: Citation and coordinate index mapping page numbers to character offsets and section/table IDs.
  - Extended `ParsedDocument` with `cleaning_status`, `cleaning_statistics`, `normalization_version`, `page_map`, `removed_headers`, `removed_footers`, `protected_tokens`, `cleaning_warnings`.
  - Added `CLEANED` to `DocumentLifecycleState`.
- **Testing & Verification:**
  - 17 dedicated tests covering all 12 pipeline stages, thread safety, failure handling, and plugins.
  - 75 total tests across all milestones passing in 1.7s (100% pass rate).
  - Real dataset validation against Emerson Control Valve Handbook (1.89M chars), Fisher Valve Handbooks, OISD standards (105, 116, 117), inspection reports, and maintenance records with zero errors and 100% token preservation.

---

## [Milestone 3B] - 2026-09-06

### Added
- **Deep Document Parsing Engine (`rag_engine/parsers/`):**
  - `document_classifier.py`: Deterministic classifier detecting document categories (`Manual`, `Inspection Report`, `Safety Document`, `Maintenance Record`, `Email`, `Engineering Drawing`, `Template`, `Unknown`) without AI/LLMs.
  - `profiles/`: Externalized refinery profiles (`base_profile.py`, `generic_refinery_profile.py`, `mrpl_profile.py`) enabling multi-refinery deployment (HPCL, IOCL, ONGC, BPCL, Reliance).
  - `parsing_context.py`: Configurable execution context for extraction features, profiles, and streaming batch control.
  - `parser_validator.py`: Structural validator checking heading hierarchy jumps, duplicate titles, malformed table rows, conflicting equipment types, and broken cross-references (`ValidationReport`).
  - `parser_utils.py`: Deterministic extraction utilities for regex tokenization, table conversion, equipment tag extraction with operating limits, directed `EntityGraph` builder, safety warnings, and citation coordinate generation.
  - `base_parser.py`: Template method pattern executing the parsing lifecycle with thread-safe metrics, event emission, validation, and streaming generator support (`parse_stream`).
  - `parser_registry.py` & `parser_factory.py`: Registry and factory with mandatory `DocumentClassifier` pre-invocation and fallback resolution.
  - `plugin_parser.py`: Runtime dynamic plugin parser discovery and loading.
  - `parser_metrics.py` & `parser_events.py`: Thread-safe observability telemetry and event bus.
  - `parser_health.py`: Health reporting and diagnostic models.
  - `exceptions.py`: Custom exception hierarchy (`ParserError`, `UnsupportedDocumentError`, `ParsingFailedError`, `ValidationError`, `ProfileError`).
- **11 Concrete Specialized Parsers:**
  - `generic_text_parser.py`: Universal text parser.
  - `pdf_parser.py`: Page-aware PDF parser with page coordinate mapping.
  - `docx_parser.py`: Word parser with native OpenXML table extraction.
  - `pptx_parser.py`: PowerPoint presentation parser extracting slides and tables.
  - `csv_parser.py`: Tabular dataset parser with column typing and cell coordinates.
  - `markdown_parser.py`: GFM parser extracting frontmatter, hierarchy, and markdown tables.
  - `image_metadata_parser.py`: Drawing parser extracting title blocks, dimensions, and drawing numbers without OCR/CV.
  - `email_parser.py`: RFC 822 email parser extracting headers, body, and action items.
  - `engineering_parser.py`: Refinery manual parser extracting operating limits, plant units, and equipment tags.
  - `inspection_parser.py`: Inspection report parser extracting test findings, wall thickness, and inspection dates.
  - `safety_parser.py`: Safety standard parser extracting DANGER/WARNING notices, PPE requirements, and OISD/PNGRB compliance clauses.
- **ParsedDocument Schema Suite (`rag_engine/schemas/parsed_document.py`):**
  - `ParsedDocument`, `Section`, `Table`, `TableCell`, `EquipmentEntity`, `EquipmentType`, `EntityRelationship`, `RelationType`, `EntityGraph`, `SafetyWarning`, `SafetyWarningSeverity`, `CrossReference`, `DrawingMetadata`, `ParsedMetadata`, `DocumentStatistics`, `ValidationIssue`, `ValidationReport`, `CitationCoordinates`.
- **Testing & Verification:**
  - 58 automated unit, integration, stress, and concurrency tests passing with a 100% pass rate.
  - Zero AI, zero OCR, zero LLM dependencies; 100% air-gapped and deterministic.

---

## [Milestone 3A] - 2026-09-06


### Added
- **Universal Document Loading Framework (`rag_engine/loaders/`):**
  - `base_loader.py`: Abstract Base Class with execution template, multi-stage validation, telemetry, and dual-driver fallback pattern.
  - `loader_registry.py`: Thread-safe registry mapping extensions and MIME types with `@register_loader` decorator.
  - `loader_factory.py`: Master factory resolving MIME types first, with extension fallback.
  - `exceptions.py`: Custom exception hierarchy (`LoaderError`, `UnsupportedFormatError`, `CorruptedDocumentError`, `LoaderInitializationError`, `ValidationFailedError`).
  - `processing_context.py`: Immutable context object accompanying documents throughout the pipeline.
  - `loader_metrics.py`: Operational telemetry tracking latency, memory delta, and driver usage (`LoadMetrics`, `MetricsCollector`).
  - `loader_events.py`: Structured audit events (`DocumentLoaded`, `FallbackActivated`, `ValidationFailed`) and thread-safe `EventBus`.
  - `loader_health.py`: Operational health monitoring and dependency diagnostics (`LoaderHealthReport`, `LoaderHealthStatus`).
  - `loader_utils.py`: MIME type detection, heuristic token estimation, and file metadata utilities.
  - `plugin_loader.py`: Dynamic runtime plugin loader manager.
  - `pdf_loader.py`: PDFLoader with primary (`pypdf`) and air-gapped fallback (`zlib` + regex).
  - `docx_loader.py`: DOCXLoader with primary (`python-docx`) and air-gapped fallback (`zipfile` + `xml.etree.ElementTree`).
  - `pptx_loader.py`: PPTXLoader with primary (`python-pptx`) and air-gapped fallback (`zipfile` + `xml.etree.ElementTree`).
  - `xlsx_loader.py`: XLSXLoader with primary (`openpyxl`) and air-gapped fallback (`zipfile` + `xml.etree.ElementTree`).
  - `csv_loader.py`: CSVLoader with dialect sniffing and multi-encoding decoders.
  - `txt_loader.py`: TXTLoader with multi-encoding fallback decoders.
  - `markdown_loader.py`: MarkdownLoader with frontmatter YAML, heading count, and code block extraction.
  - `image_loader.py`: ImageLoader indexing visual assets with binary struct header dimensions.
  - `loaders/__init__.py`: Compatibility package alias.
- **Document Schema Enhancements:**
  - `Document` & `DocumentMetadata`: Added UUID, checksum, lifecycle states (`LOADED`, `PARSED`, `CHUNKED`, `EMBEDDED`, `INDEXED`), token estimates, word/char counts, image references, and processing history.
- **Engineering Decisions & Reporting:**
  - `EDR-004`: Universal Document Loading Framework & Dual-Driver Fallback.
  - `project_management/milestone_reports/milestone_03A.md`: Full sign-off report with sequence & class diagrams.
- **Testing:**
  - 36 automated unit, integration, and concurrency tests with a 100% pass rate.

---

## [Milestone 2] - 2026-09-06

### Added
- **`dataset_engine` Package:**
  - `scanner.py`: Recursive filesystem crawler categorizing files into refinery domains (`manuals`, `safety_docs`, `maintenance`, `engineering_drawings`, `ocr`, `pidqa`, `Vision`, etc.).
  - `validator.py`: Deep file integrity verification, magic byte checks (PDF, OpenXML, JSON, XML, PNG, JPG), size threshold enforcement (max 500MB).
  - `hash_generator.py`: Streaming SHA-256 calculation using 64KB buffers for memory-safe hashing of 29,000+ files.
  - `duplicate_detector.py`: Non-destructive duplicate detection identifying exact content duplicates, filename collisions, and path redundancy.
  - `manifest_generator.py`: Atomic generation of `outputs/manifest.json` with deterministic UUIDs and document metadata.
  - `statistics.py`: Aggregate metrics calculation, JSON statistics output, and executive Markdown health report generation.
  - `orchestrator.py`: Master coordination pipeline orchestrating Scanner -> Validator -> Hash Generator -> Duplicate Detector -> Manifest Generator -> Statistics Generator.
- **Configuration:**
  - `config/validation_rules.yaml`: Externalized validation rules, allowed extensions, ignored directories, and category mappings.
- **Project Management System:**
  - `project_management/progress.json`: Milestone progress and status tracking.
  - `project_management/engineering_decisions.md`: Engineering Decision Records (EDR-001 to EDR-003).
  - `project_management/pending_tasks.md`: Active and pending milestone task board.
  - `project_management/known_issues.md`: Tracked edge cases and mitigations.
  - `project_management/integration_notes.md`: Upstream/downstream integration contracts.
  - `project_management/milestone_reports/milestone_02.md`: Comprehensive Milestone 2 sign-off report.
- **Automated Tests:**
  - 18 unit and integration tests covering all `dataset_engine` components and `rag_engine.utils`.
  - 100% test pass rate.

---

## [Milestone 1] - 2026-09-06

### Added
- `rag_engine` package scaffolding with Clean Architecture structure.
- Abstract Base Classes in `rag_engine/interfaces/` (`BaseLoader`, `BaseChunker`, `BaseEmbedder`, `BaseVectorStore`, `BaseRetriever`).
- Immutable Pydantic v2 schemas in `rag_engine/schemas/` (`Document`, `Chunk`, `EmbeddingVector`, `Citation`, `RetrievedDocument`, `Manifest`).
- Centralized configuration via `pydantic-settings` in `rag_engine/config/settings.py`.
- Path registry in `rag_engine/config/paths.py`.
- Application constants in `rag_engine/config/constants.py`.
- Structured rotating file logging in `rag_engine/config/logging_config.py`.
