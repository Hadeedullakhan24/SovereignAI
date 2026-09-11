# Complete Project Tree After Milestone 9
## Sovereign On-Premise Agentic AI Workbench (SIH26117 / MRPL)
### Final Member 1 (Knowledge Base / RAG Engine) Repository Architecture

---

```text
d:\SovereignAI\
│
├── CHANGELOG.md                                   # Chronological changelog tracking all milestones and engineering changes.
├── MASTER_RAG_ARCHITECTURE.md                     # Master architectural blueprint and design invariants for the RAG engine.
├── RAG_DATA_FLOW.md                               # End-to-end data flow specifications from dataset to verified LLM answer.
├── CHUNK_SCHEMA.md                                # Authoritative chunk object and enriched metadata schema definitions.
├── EMBEDDING_ARCHITECTURE.md                      # Local offline embedding pipeline, caching, and model switching specifications.
├── VECTOR_DB_ARCHITECTURE.md                      # ChromaDB and BM25 vector store persistence, backup, and restore architecture.
├── RETRIEVAL_ARCHITECTURE.md                      # Multi-stage hybrid retrieval, cross-encoder reranking, and citation engine design.
├── PROJECT_TREE_AFTER_MILESTONE9.md               # Definitive post-Milestone 9 repository file layout with single-line responsibilities.
├── IMPLEMENTATION_ORDER.md                        # Phased implementation plan, dependencies, tests, and risks for remaining milestones.
├── ARCHITECTURE_REVIEW.md                         # Self-critical architectural review, bottleneck analysis, and mitigations.
├── MASTER_PROGRESS_REPORT.md                      # High-level executive milestone status, metrics, and readiness audit.
├── MILESTONE_03A_COMPLETION_REPORT.md             # Formal milestone sign-off report for Universal Document Loading Framework.
├── MILESTONE_03B_COMPLETION_REPORT.md             # Formal milestone sign-off report for Deep Document Parsing Engine.
├── README.md                                      # Repository overview, architectural overview, setup, and test execution commands.
├── pyproject.toml                                 # Python package specification, dependencies, build system, and pytest configuration.
├── requirements.txt                               # Explicit pip dependency lock for air-gapped on-premise installation.
│
├── config/                                        # Externalized YAML configurations for validation and RAG parameters.
│   ├── validation_rules.yaml                      # Dataset validation thresholds, allowed extensions, and category mappings.
│   └── rag_settings.yaml                          # Configurable model paths, chunk sizes, overlaps, and retrieval thresholds.
│
├── datasets/                                      # Raw refinery operational datasets (immutable, read-only).
│   ├── coding/                                    # Industrial automation and instrumentation code samples.
│   ├── emails/                                    # Operational emails, shift handover memos, and management communications.
│   ├── engineering_drawings/                      # P&IDs, electrical schematics, mechanical blueprints, and CAD files.
│   ├── handwritten_notes/                         # Scanned technician maintenance logs and field inspection notes.
│   ├── inspection_reports/                        # Ultrasonic, dye penetrant, radiography, and visual NDT inspection reports.
│   ├── maintenance/                               # Preventive maintenance work orders, overhaul logs, and vibration surveys.
│   ├── manuals/                                   # Operating manuals, vendor documentation, and equipment data sheets.
│   ├── ocr/                                       # Scanned legacy paper documents, certificates, and reports.
│   ├── pidqa/                                     # Question-answering benchmark pairs on piping and instrumentation diagrams.
│   ├── safety_docs/                               # OISD standards, PNGRB regulations, JSA sheets, and safety SOPs.
│   ├── templates/                                 # Standardized blank inspection forms, checklists, and permit templates.
│   └── Vision/                                    # Optical inspection photos, thermal scans, and visual asset images.
│
├── dataset_engine/                                # Milestone 2: Dataset discovery, validation, hashing, and indexing engine.
│   ├── __init__.py                                # Exports public dataset validation engine interfaces and orchestrator.
│   ├── duplicate_detector.py                      # Identifies exact content SHA-256 duplicates and path collisions across datasets.
│   ├── hash_generator.py                          # Memory-safe chunked streaming SHA-256 digest generator for large datasets.
│   ├── manifest_generator.py                      # Generates standardized outputs/manifest.json containing unique document UUIDs.
│   ├── orchestrator.py                            # Coordinates scanner, validator, hash generator, and statistics CLI workflow.
│   ├── scanner.py                                 # Crawls datasets directory and deterministically categorizes discovered files.
│   ├── statistics.py                              # Aggregates dataset health metrics and outputs dataset_health_report.md.
│   └── validator.py                               # Validates magic bytes, file sizes, readability, and structural file integrity.
│
├── models/                                        # Local offline open-weight model directory (air-gapped storage).
│   ├── embeddings/                                # Local embedding model weights and tokenizers.
│   │   ├── bge-small-en-v1.5/                     # Local files for BAAI/bge-small-en-v1.5 (384 dimensions).
│   │   ├── bge-base-en-v1.5/                      # Local files for BAAI/bge-base-en-v1.5 (768 dimensions).
│   │   ├── e5-base-v2/                            # Local files for intfloat/e5-base-v2 (768 dimensions).
│   │   └── all-MiniLM-L6-v2/                      # Local files for sentence-transformers/all-MiniLM-L6-v2 (384 dimensions).
│   └── reranker/                                  # Local cross-encoder reranker model weights.
│       └── bge-reranker-base/                     # Local files for BAAI/bge-reranker-base cross-encoder model.
│
├── outputs/                                       # Generated artifacts, manifests, and system health reports.
│   ├── dataset_health_report.md                   # Human-readable markdown audit of dataset integrity and categorization.
│   ├── dataset_statistics.json                    # Aggregated file counts, byte volumes, and format distributions.
│   └── manifest.json                              # Master index mapping every file to its SHA-256, UUID, and operational bucket.
│
├── project_management/                            # Project coordination, decision tracking, and milestone audit reports.
│   ├── engineering_decisions.md                   # Architectural Decision Records (EDR-001 through EDR-005).
│   ├── integration_notes.md                       # Explicit interface contracts between RAG modules and downstream team members.
│   ├── known_issues.md                            # Comprehensive register of tracked edge cases, limits, and verified resolutions.
│   ├── pending_tasks.md                           # Active milestone task boards and backlog for Member 1 deliverables.
│   ├── progress.json                              # Machine-readable milestone states, timestamps, and deliverable catalogs.
│   └── milestone_reports/                         # Formal milestone completion audit documents.
│       ├── milestone_01.md                        # Milestone 1 completion sign-off: Scaffolding and architecture foundation.
│       ├── milestone_02.md                        # Milestone 2 completion sign-off: Dataset validation and manifest generation.
│       ├── milestone_03A.md                       # Milestone 3A completion sign-off: Universal document loading framework.
│       └── milestone_03B.md                       # Milestone 3B completion sign-off: Deep document parsing engine.
│
├── rag_engine/                                    # Core Member 1 package containing RAG pipeline logic.
│   ├── __init__.py                                # Package initialization exposing top-level version and primary exports.
│   │
│   ├── config/                                    # Centralized Pydantic BaseSettings and runtime configuration.
│   │   ├── __init__.py                            # Exports global settings instance and path constants.
│   │   ├── constants.py                           # System-wide static constants, MIME types, and default buffer thresholds.
│   │   ├── logging_config.py                      # Thread-safe rotating structured loggers writing to logs/ directory.
│   │   ├── paths.py                               # Absolute path resolution ensuring POSIX portability across operating systems.
│   │   └── settings.py                            # Centralized Pydantic v2 BaseSettings loading YAML configs and env overrides.
│   │
│   ├── interfaces/                                # Clean Architecture Abstract Base Classes (contracts).
│   │   ├── __init__.py                            # Exports abstract interfaces for all pipeline stages.
│   │   ├── base_chunker.py                        # Abstract Base Class defining chunking strategy interface.
│   │   ├── base_cleaner.py                        # Abstract Base Class defining text and table cleaning interface (Milestone 4).
│   │   ├── base_embedder.py                       # Abstract Base Class defining single and batch text embedding interface.
│   │   ├── base_loader.py                         # Abstract Base Class defining document loading and dual-driver fallback.
│   │   ├── base_retriever.py                      # Abstract Base Class defining query retrieval and filtering interface.
│   │   └── base_vector_store.py                   # Abstract Base Class defining vector storage, search, and delete operations.
│   │
│   ├── schemas/                                   # Immutable Pydantic v2 Data Transfer Objects.
│   │   ├── __init__.py                            # Exports all schema models across the RAG package.
│   │   ├── chunk.py                               # Schema defining atomic Chunk and enriched ChunkMetadata models (Milestone 5).
│   │   ├── citation.py                            # Schema defining Citation and Source attribution structures (Milestone 8).
│   │   ├── document.py                            # Schema defining raw loaded Document and DocumentMetadata models (Milestone 3A).
│   │   ├── embedding.py                           # Schema defining EmbeddingVector and EmbeddingRequest models (Milestone 6).
│   │   ├── manifest.py                            # Schema defining master document catalog and entry models (Milestone 2).
│   │   ├── parsed_document.py                     # Schema defining ParsedDocument, Section, Table, EquipmentEntity (Milestone 3B).
│   │   └── retrieved_document.py                  # Schema defining ScoredChunk, RetrievedDocument, and query result models.
│   │
│   ├── loaders/                                   # Milestone 3A: Universal document loading framework.
│   │   ├── __init__.py                            # Exports loaders, factory, registry, and context.
│   │   ├── base_loader.py                         # Abstract template class executing file loading and metrics collection.
│   │   ├── csv_loader.py                          # Delimited tabular loader with dialect sniffing and encoding fallback.
│   │   ├── docx_loader.py                         # Microsoft Word loader with primary python-docx and fallback XML parser.
│   │   ├── exceptions.py                          # Custom loader exception hierarchy (LoaderError, UnsupportedFormatError).
│   │   ├── image_loader.py                        # Visual asset indexer extracting dimensions via binary header probes.
│   │   ├── loader_events.py                       # Thread-safe event bus and audit events (DocumentLoaded, FallbackActivated).
│   │   ├── loader_factory.py                      # Master factory resolving loaders via MIME type and extension dispatch.
│   │   ├── loader_health.py                       # Health monitoring diagnostic models and driver dependency checkers.
│   │   ├── loader_metrics.py                      # Thread-safe operational telemetry tracking latency, memory, and drivers.
│   │   ├── loader_registry.py                     # Thread-safe registry mapping extensions and MIME types to loader classes.
│   │   ├── loader_utils.py                        # MIME detection, heuristic token estimation, and path normalization utils.
│   │   ├── markdown_loader.py                     # Markdown document loader extracting YAML frontmatter and raw text.
│   │   ├── pdf_loader.py                          # PDF document loader with primary pypdf and fallback stream decoder.
│   │   ├── plugin_loader.py                       # Dynamic runtime loader discovering external plugin classes.
│   │   ├── pptx_loader.py                         # PowerPoint presentation loader extracting slide bodies and bullet lists.
│   │   ├── processing_context.py                  # Immutable configuration context accompanying document loading.
│   │   ├── txt_loader.py                          # Plain text loader with multi-encoding fallback decoders (UTF-8, Latin-1).
│   │   └── xlsx_loader.py                         # Excel spreadsheet loader with primary openpyxl and fallback XML parser.
│   │
│   ├── parsers/                                   # Milestone 3B: Deep document parsing engine.
│   │   ├── __init__.py                            # Exports parsers, parser factory, classifier, and profiles.
│   │   ├── base_parser.py                         # Abstract template class executing parsing lifecycle and parse_stream.
│   │   ├── csv_parser.py                          # Tabular data parser converting rows/columns into structured Table objects.
│   │   ├── document_classifier.py                 # Deterministic classifier detecting document categories without AI/LLMs.
│   │   ├── docx_parser.py                         # Word document parser extracting heading levels and native OpenXML tables.
│   │   ├── email_parser.py                        # Operational email parser extracting RFC 822 headers, body, and action items.
│   │   ├── engineering_parser.py                  # Engineering manual parser extracting operating limits and plant units.
│   │   ├── exceptions.py                          # Parser exception hierarchy (ParserError, ValidationError, ProfileError).
│   │   ├── generic_text_parser.py                 # Universal text parser for standard text files, logs, and configurations.
│   │   ├── image_metadata_parser.py               # Technical drawing parser extracting dimensions, DPI, and title blocks.
│   │   ├── inspection_parser.py                   # Inspection report parser extracting test findings and wall thickness data.
│   │   ├── markdown_parser.py                     # CommonMark parser extracting H1-H6 hierarchy, frontmatter, and tables.
│   │   ├── parser_events.py                       # Thread-safe publish-subscribe event bus emitting parsing lifecycle events.
│   │   ├── parser_factory.py                      # Master factory invoking DocumentClassifier before resolving concrete parser.
│   │   ├── parser_health.py                       # Diagnostic health reporter evaluating parser operational readiness.
│   │   ├── parser_metrics.py                      # Thread-safe telemetry collector tracking parsing duration and entity counts.
│   │   ├── parser_registry.py                     # Thread-safe registry mapping extensions and categories to parser classes.
│   │   ├── parser_utils.py                        # Deterministic regex tokenizers, table converters, and EntityGraph builders.
│   │   ├── parser_validator.py                    # Structural hierarchy and table column parity validator (ValidationReport).
│   │   ├── parsing_context.py                     # Immutable configuration context controlling extraction toggles.
│   │   ├── pdf_parser.py                          # Page-aware PDF parser preserving page numbers and coordinate mappings.
│   │   ├── plugin_parser.py                       # Dynamic runtime plugin parser discovery and loading manager.
│   │   ├── pptx_parser.py                         # Presentation parser extracting slide headings, body text, and tables.
│   │   ├── safety_parser.py                       # Safety manual parser extracting DANGER/WARNING notices and PPE rules.
│   │   └── profiles/                              # Multi-refinery nomenclature and equipment tag configuration profiles.
│   │       ├── __init__.py                        # Exports profile registry and get_profile factory function.
│   │       ├── base_profile.py                    # Abstract RefineryProfile defining tag patterns and equipment prefixes.
│   │       ├── generic_refinery_profile.py        # Generic hydrocarbon refinery profile matching international standards.
│   │       └── mrpl_profile.py                    # MRPL profile with Mangalore units (CDU, VDU), tags, and OISD standards.
│   │
│   ├── preprocessing/                             # Milestone 4: Cleaning & normalization engine.
│   │   ├── __init__.py                            # Exports cleaning engine, normalizers, and sanitization pipelines.
│   │   ├── boilerplate_stripper.py                # Removes running headers, footers, pagination strings, and repeated legal text.
│   │   ├── cleaner_factory.py                     # Factory resolving format and category-specific cleaning pipelines.
│   │   ├── cleaning_engine.py                     # Master orchestrator executing all cleaning stages on ParsedDocuments.
│   │   ├── exceptions.py                          # Preprocessing exception hierarchy (CleaningError, SanitizationError).
│   │   ├── table_normalizer.py                    # Cleans whitespace in table cells, fills empty cells, aligns ragged columns.
│   │   ├── terminology_normalizer.py              # Expands refinery acronyms (CDU, LOTO) and standardizes physical units (bar, °C).
│   │   ├── text_sanitizer.py                      # Strips control bytes, normalizes smart quotes, em-dashes, and excessive spaces.
│   │   └── unicode_normalizer.py                  # Normalizes unicode character sets using NFKC standardization.
│   │
│   ├── chunking/                                  # Milestone 5: Multi-strategy chunking engine.
│   │   ├── __init__.py                            # Exports chunking engine, strategy implementations, and metadata enricher.
│   │   ├── chunk_enricher.py                      # Injects parent document lineage, equipment tags, and safety notices into chunks.
│   │   ├── chunk_factory.py                       # Factory resolving appropriate chunker based on document category and type.
│   │   ├── chunking_engine.py                     # Master coordinator processing CleanedParsedDocuments into final Chunk lists.
│   │   ├── exceptions.py                          # Chunking exception hierarchy (ChunkingError, TokenLimitExceededError).
│   │   ├── fixed_chunker.py                       # Simple fixed-size sliding token window chunker for unstructured text.
│   │   ├── recursive_chunker.py                   # Recursive character and sentence-boundary chunker preserving paragraph flow.
│   │   ├── section_aware_chunker.py               # Hierarchical chunker respecting H1-H6 boundaries without splitting headings.
│   │   ├── table_aware_chunker.py                 # Tabular chunker serializing row groups with persistent header repetitions.
│   │   └── token_counter.py                       # Model-specific and heuristic token counting utility.
│   │
│   ├── embeddings/                                # Milestone 6: Offline embedding pipeline.
│   │   ├── __init__.py                            # Exports embedding manager, cache, and concrete embedder classes.
│   │   ├── base_embedder.py                       # Abstract base class defining single/batch embedding and dimension contracts.
│   │   ├── embedding_cache.py                     # Persistent two-tier cache (L1 Memory LRU + L2 SQLite Disk Database).
│   │   ├── embedding_manager.py                   # Thread-safe orchestrator handling batching, cache lookups, and model switching.
│   │   ├── embedding_registry.py                  # Registry of supported open-weight models (BGE, E5, MiniLM) and dimensions.
│   │   ├── exceptions.py                          # Embedding exception hierarchy (EmbeddingError, ModelLoadError).
│   │   ├── model_loader.py                        # Air-gapped local model loader enforcing local_files_only=True.
│   │   ├── onnx_embedder.py                       # High-speed ONNX runtime embedder for lightweight CPU-only edge deployments.
│   │   └── sentence_transformer_embedder.py       # PyTorch/SentenceTransformers local model wrapper supporting CPU and CUDA.
│   │
│   ├── vector_store/                              # Milestone 7: Vector database and index backend.
│   │   ├── __init__.py                            # Exports vector store factory, ChromaDB store, and BM25 index.
│   │   ├── base_vector_store.py                   # Abstract Base Class defining add, search, delete, and count operations.
│   │   ├── bm25_index.py                          # Inverted term-frequency BM25 lexical search index for exact equipment tags.
│   │   ├── chroma_store.py                        # Persistent ChromaDB vector store with HNSW index and SQLite metadata storage.
│   │   ├── exceptions.py                          # Vector store exception hierarchy (VectorStoreError, CollectionMismatchError).
│   │   ├── faiss_store.py                         # In-memory FAISS vector store for ultra-high-throughput dense search benchmarks.
│   │   ├── index_manager.py                       # Coordinates atomic updates, collection purging, and persistence flushing.
│   │   ├── metadata_filter.py                     # Translates query filter dictionaries into native ChromaDB where-clauses.
│   │   └── vector_store_factory.py                # Factory resolving configured vector database backend (Chroma vs FAISS).
│   │
│   ├── retrieval/                                 # Milestone 8: Hybrid retrieval engine.
│   │   ├── __init__.py                            # Exports hybrid retriever, dense retriever, and sparse retriever.
│   │   ├── base_retriever.py                      # Abstract Base Class defining query retrieval and top-k filtering contracts.
│   │   ├── dense_retriever.py                     # Dense vector similarity retriever querying ChromaDB vector store.
│   │   ├── exceptions.py                          # Retrieval exception hierarchy (RetrievalError, EmptyQueryError).
│   │   ├── hybrid_retriever.py                    # Master retriever combining Dense + BM25 via Reciprocal Rank Fusion (RRF).
│   │   ├── query_preprocessor.py                  # Extracts equipment tags and technical acronyms from user query strings.
│   │   ├── reciprocal_rank_fusion.py              # Mathematical implementation of Reciprocal Rank Fusion (RRF) algorithm.
│   │   └── sparse_retriever.py                    # Sparse lexical retriever querying local BM25 inverted index.
│   │
│   ├── reranking/                                 # Milestone 8: Neural cross-encoder reranking.
│   │   ├── __init__.py                            # Exports cross-encoder reranker wrapper and scoring utilities.
│   │   ├── cross_encoder_reranker.py              # Local BAAI/bge-reranker-base cross-attention model for candidate re-scoring.
│   │   ├── exceptions.py                          # Reranking exception hierarchy (RerankerError, ModelLoadError).
│   │   └── null_reranker.py                       # Pass-through identity reranker for low-latency or compute-constrained edge modes.
│   │
│   ├── citations/                                 # Milestone 8: Source attribution and citation engine.
│   │   ├── __init__.py                            # Exports citation builder, coordinate resolver, and prompt context formatter.
│   │   ├── citation_builder.py                    # Constructs auditable Citation models with exact page numbers and quotes.
│   │   ├── context_formatter.py                   # Synthesizes retrieved chunks into structured system prompt context blocks.
│   │   └── provenance_tracker.py                  # Traces chunk lineage back through manifest.json to original source files.
│   │
│   ├── cache/                                     # System-wide general caching infrastructure.
│   │   ├── __init__.py                            # Exports disk and memory cache utilities.
│   │   └── disk_cache.py                          # Thread-safe SQLite key-value store for intermediate pipeline artifacts.
│   │
│   ├── metadata/                                  # Metadata extraction and normalization utilities.
│   │   ├── __init__.py                            # Exports operational metadata extraction utilities.
│   │   └── tag_extractor.py                       # Secondary equipment tag and P&ID loop number extractor.
│   │
│   ├── pipeline/                                  # Milestone 9: Complete RAG orchestration.
│   │   ├── __init__.py                            # Exports end-to-end ingestion and query pipeline orchestrators.
│   │   ├── exceptions.py                          # Pipeline exception hierarchy (PipelineError, IngestionFailedError).
│   │   ├── ingestion_orchestrator.py              # Batch ingestion pipeline executing Loader -> Parser -> Cleaner -> Chunker -> Embedder -> VectorDB.
│   │   ├── incremental_updater.py                 # Compares source SHA-256 hashes against manifest to run delta-only updates.
│   │   ├── pipeline_events.py                     # Lifecycle event definitions (IngestionStarted, ChunksIndexed, QueryCompleted).
│   │   ├── pipeline_metrics.py                    # End-to-end latency and throughput telemetry collector.
│   │   └── query_pipeline.py                      # End-to-end query orchestrator executing Query -> Retriever -> Reranker -> Citations -> Context.
│   │
│   ├── evaluation/                                # Milestone 9: Evaluation and benchmarking engine.
│   │   ├── __init__.py                            # Exports RAG benchmark runner and evaluation metrics.
│   │   ├── benchmark_dataset.py                   # Test dataset loader of golden refinery queries, relevant chunks, and ground truths.
│   │   ├── benchmark_runner.py                    # Automated test runner executing batch queries and computing IR metrics.
│   │   └── metrics.py                             # Mathematical calculation of Recall@K, Mean Reciprocal Rank (MRR), and NDCG@K.
│   │
│   └── utils/                                     # Universal helper functions and common utilities.
│       ├── __init__.py                            # Exports utility functions.
│       ├── file_utils.py                          # File path manipulation, directory creation, and POSIX path sanitization.
│       ├── hash_utils.py                          # Streaming SHA-256 calculation utilities.
│       └── validators.py                          # Universal runtime type, boundary, and existence validators.
│
├── vector_db/                                     # Persistent on-premise vector store data directory.
│   ├── chroma/                                    # ChromaDB SQLite metadata and binary HNSW index graph files.
│   │   ├── chroma.sqlite3                         # SQLite storage for vector collections, documents, and metadata.
│   │   └── <collection_uuid>/                     # Binary HNSW index graph files (data_level0.bin, header.bin, link_list.bin).
│   ├── bm25/                                      # Serialized inverted keyword index directory.
│   │   ├── bm25_index.pkl                         # Pickled BM25 search index.
│   │   └── doc_id_map.json                        # Mapping of BM25 integer doc IDs to chunk UUIDs.
│   └── embedding_cache.sqlite                     # Persistent SQLite database storing cached float32 vector blobs.
│
└── tests/                                         # Automated Pytest test suite covering all 9 milestones.
    ├── test_concrete_loaders.py                   # Unit tests for all 8 concrete file loaders and fallback decoders (M3A).
    ├── test_concrete_parsers.py                   # Unit tests for all 11 concrete document parsers and extraction logic (M3B).
    ├── test_duplicate_detector.py                 # Unit tests for exact content and filename duplicate detection (M2).
    ├── test_hash_generator.py                     # Unit tests for streaming chunked SHA-256 hashing (M2).
    ├── test_loader_framework.py                   # Framework unit tests for loader registry, factory, and events (M3A).
    ├── test_loader_thread_safety.py               # Concurrent stress test verifying loader thread safety (M3A).
    ├── test_manifest_generator.py                 # Unit tests for atomic manifest.json generation (M2).
    ├── test_orchestrator.py                       # Integration tests for dataset engine orchestrator (M2).
    ├── test_parser_framework.py                   # Framework unit tests for parser registry, classifier, and profiles (M3B).
    ├── test_parser_thread_safety.py               # 20-thread concurrent stress test verifying parser thread safety (M3B).
    ├── test_parser_validator.py                   # Unit tests for structural hierarchy, table, and entity validation (M3B).
    ├── test_scanner.py                            # Unit tests for recursive directory crawling and categorization (M2).
    ├── test_statistics.py                         # Unit tests for dataset metrics and health report generator (M2).
    ├── test_utils.py                              # Unit tests for core hash, file, and validation utilities (M1).
    ├── test_validator.py                          # Unit tests for file readability, magic bytes, and size checks (M2).
    ├── test_cleaning_engine.py                    # Unit tests for unicode, whitespace, boilerplate, and unit normalizers (M4).
    ├── test_chunking_engine.py                    # Unit tests for section-aware, recursive, and table-aware chunkers (M5).
    ├── test_chunk_enricher.py                     # Unit tests for equipment tag and safety notice chunk enrichment (M5).
    ├── test_embedding_pipeline.py                 # Unit tests for offline model loading, batching, and embedding cache (M6).
    ├── test_model_switching.py                    # Unit tests verifying collection namespacing upon embedding model change (M6).
    ├── test_chroma_store.py                       # Unit tests for ChromaDB vector insertion, search, metadata filtering (M7).
    ├── test_bm25_index.py                         # Unit tests for sparse lexical index construction and keyword search (M7).
    ├── test_hybrid_retriever.py                   # Unit tests for parallel dense+sparse search and RRF fusion (M8).
    ├── test_cross_encoder_reranker.py             # Unit tests for local cross-encoder candidate scoring (M8).
    ├── test_citation_builder.py                   # Unit tests for citation anchor generation and verbatim quote verification (M8).
    ├── test_ingestion_orchestrator.py             # End-to-end integration test of full batch ingestion pipeline (M9).
    ├── test_incremental_updater.py                # Unit tests verifying hash-based delta-only re-indexing (M9).
    ├── test_query_pipeline.py                     # End-to-end test of query-to-grounded-context pipeline (M9).
    └── test_evaluation_metrics.py                 # Unit tests for automated Recall@K, MRR, and NDCG metric calculations (M9).
```
