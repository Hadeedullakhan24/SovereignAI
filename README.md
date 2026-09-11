# Sovereign On-Premise Agentic AI Workbench

**Problem Statement:** SIH26117 (Smart India Hackathon 2026)  
**Organization:** Mangalore Refinery and Petrochemicals Limited (MRPL)  
**Architecture:** 100% Offline-First / Air-Gapped Clean Architecture  
**Active Milestone:** Milestone 9 — Enterprise In-Process Generation Engine & Context Assembly Gateway (COMPLETED)  

---

## Overview

The **Sovereign On-Premise Agentic AI Workbench** is an enterprise, air-gapped AI platform designed for high-security critical infrastructure at MRPL refineries. The system ingests technical manuals, piping and instrumentation diagrams (P&IDs), standard operating procedures (SOPs), safety documentation, lab inspection reports, and maintenance records to provide localized semantic retrieval, multimodal OCR verification, and deterministic agent workflows without WAN connectivity or external SaaS dependencies.

---

## Repository Structure

```text
d:\SovereignAI\
├── cache/                         # Persistent caches (SQLite vector cache, checkpoints)
├── config/
│   └── validation_rules.yaml      # Externalized validation constraints
├── datasets/                      # Raw operational datasets (Read-Only)
├── models/                        # Local offline open-weight model weights
│   └── embeddings/                # Local BGE and E5 embedding models
├── outputs/                       # Generated indexes & reports
├── project_management/            # Engineering decisions, progress & logs
│   ├── engineering_decisions.md   # Architectural Decision Records (EDR-001 to EDR-008)
│   ├── integration_notes.md       # Upstream/Downstream contracts
│   ├── known_issues.md            # Tracked edge cases & mitigations
│   ├── pending_tasks.md           # Task progress tracker
│   ├── progress.json              # Machine-readable milestone states
│   └── milestone_reports/         # Formal milestone completion reports
├── rag_engine/                    # Member 1 RAG Engine Package
│   ├── chunking/                  # Milestone 5: Enterprise Chunking Engine
│   ├── embeddings/                # Milestone 6: Offline Embedding Pipeline
│   │   ├── base_embedder.py       # BaseEmbedder abstract contract
│   │   ├── local_embedder.py      # LocalHuggingFaceEmbedder & DeterministicTestEmbedder
│   │   ├── embedding_registry.py  # Thread-safe model registry & catalog
│   │   ├── embedding_factory.py   # Factory with instance pooling & dynamic switching
│   │   ├── embedding_cache.py     # SQLite WAL persistent vector cache (binary packing)
│   │   ├── checkpoint_manager.py  # Interrupted run checkpoint & resume manager
│   │   ├── vector_validator.py    # Strict numerical, NaN/Inf, & unit norm validator
│   │   ├── embedding_pipeline.py  # Master batch embedding orchestrator
│   │   └── embedding_metrics.py   # Telemetry, throughput, & resource monitoring
│   ├── interfaces/                # Clean Architecture Abstract Base Classes
│   ├── loaders/                   # Milestone 3A: Universal Document Loading Framework
│   ├── parsers/                   # Milestone 3B: Deep Document Parsing Engine
│   ├── preprocessing/             # Milestone 4: Cleaning & Normalization Engine
│   ├── generation/                # Milestone 9: In-Process Generation Gateway & Guardrails
│   ├── retrieval/                 # Milestone 8: Enterprise Hybrid Retrieval Engine
│   ├── vector_store/              # Milestone 7: Vector Database Platform (Qdrant)
│   └── pipeline/                  # End-to-End Master RAG Pipeline
└── tests/                         # Pytest automated test suite (173 tests passing, 100%)
```

---

## Milestone 5: Enterprise Chunking Engine

The Enterprise Chunking Engine transforms `CleanParsedDocument` / `ParsedDocument` into retrieval-optimized, semantically intact `Chunk` objects:

1. **Multi-Strategy Architecture:**
   - **Fixed-Size Chunking (`FixedChunker`):** Token-based and character-based sliding windows with configurable overlap.
   - **Recursive Chunking (`RecursiveChunker`):** Hierarchical semantic decomposition (`Document -> Section -> Subsection -> Paragraph -> Sentence -> Token window`), guaranteeing sentence integrity.
   - **Section-Aware Chunking (`SectionChunker`):** Preserves H1–H6 section boundaries and prepends hierarchical heading breadcrumbs (`[1.0 HCU > 1.1 Reactor]`).
   - **Table-Aware Chunking (`TableChunker`):** Never cuts across table cells; isolates tables into dedicated chunks, repeating column headers and table captions on every split part.
   - **List-Aware Chunking (`ListChunker`):** Keeps numbered procedures, startup/shutdown sequences, and checklists together without splitting individual list items.
2. **Deterministic Stable Chunk IDs (Zero UUIDs):**
   - Generates chunk IDs via SHA-256: `chk_{doc_hash8}_{page_str}_{chunk_index:04d}_{content_hash8}`.
   - Guarantees 100% stable chunk IDs across identical runs, preventing unnecessary re-embedding in Milestone 6.
3. **Metadata Inheritance Cascading:**
   - Every chunk automatically inherits document metadata, page numbers, heading paths, equipment tags (`R-101`, `P-203`), operating limits (`150 bar`, `380°C`), and safety standards (`OISD-105`, `API-610`).
4. **Parent-Child Hierarchy & Sequential Relational Linking:**
   - Maintains bidirectional sequential links (`prev_chunk_id`, `next_chunk_id`) and parent section tree depth (`parent_section_id`, `hierarchy_depth`) for downstream agentic context expansion.

---

## Milestone 6: Offline Embedding Pipeline

The Offline Embedding Pipeline converts atomic `Chunk` objects into dense vector `EmbeddedChunk` entities using completely local open-weight embedding models:

1. **Local Model Management:**
   - Supports `BAAI/bge-small-en-v1.5` (default: 384 dimensions), `BAAI/bge-base-en-v1.5` (768 dimensions), `intfloat/e5-small-v2` (384 dimensions), and `intfloat/e5-base-v2` (768 dimensions).
   - CLI utility `scripts/download_embedding_models.py` downloads and verifies weights once; runtime execution is 100% air-gapped with zero internet calls.
2. **Persistent SQLite Vector Caching (WAL Mode):**
The Offline Embedding Pipeline provides air-gapped vectorization:

1. **100% Local Inference:** Supports `BAAI/bge-small-en-v1.5`, `bge-base-en-v1.5`, and E5 models loaded strictly from disk (`local_files_only=True`).
2. **Persistent Vector Caching:** SQLite WAL vector cache preventing redundant recomputations.
3. **Checkpoint & Resume:** Fault-tolerant batch processing with incremental checkpointing.
4. **Strict Vector Validation:** Guarantees L2 normalization, dimension conformity, and zero NaN/Inf vectors.

---

## Milestone 7: Vector Database Platform (Qdrant)

The Enterprise Vector Storage Platform provides a persistent, air-gapped retrieval index powered by **Qdrant Local (Embedded Mode)**:

1. **Universal Repository Layer (`VectorRepository`):**
   - High-level pipelines and agent tools interact exclusively with `VectorRepository`.
2. **Multi-Collection Routing & Versioning (`CollectionRouter`, `CollectionVersionManager`):**
   - Categorical collection routing and zero-downtime alias switching.
3. **Automated Payload Indexing & Optimization (`PayloadIndexManager`, `VectorOptimizer`):**
   - Sub-15ms filtered searches, background segment compaction, and snapshot creation.

---

## Milestone 8: Enterprise Hybrid Retrieval Engine

Authoritative 100% offline retrieval layer for downstream prompts:

1. **Dual-Channel Search:** Parallel Dense retrieval (Qdrant) and Sparse lexical search (pure-Python Okapi BM25).
2. **Reciprocal Rank Fusion (RRF):** Mathematical fusion with $k=60$ smoothing and score calibration.
3. **Local Cross-Encoder Reranking:** Offline reranking via `BAAI/bge-reranker-base` with deterministic fallback.
4. **Context Expansion & Packing:** Neighbor traversal (`ChunkHierarchy`) and token-bounded table-preserving packing.

---

## Milestone 9: In-Process Generation Engine & Context Assembly Gateway

Connects the retrieval layer with downstream local LLMs under strict 100% air-gapped guarantees:

1. **Provider-Agnostic Model Layer (`BaseLLM`, `HFLocalLLM`):**
   - In-process execution via Hugging Face `transformers` with `local_files_only=True`. Zero cloud APIs, zero Ollama dependencies.
2. **7 Refinery Domain Archetypes & Templates:**
   - Equipment Lookup, SOP Retrieval, Maintenance, Safety Compliance, Troubleshooting, Comparison, and General QA.
3. **Modular Prompt Subsystem (`rag_engine/generation/prompt/`):**
   - `PromptBuilder` & `PromptPipeline`: Deterministic prompt assembly with cryptographic lineage ($\text{SHA256}$).
   - `SystemPromptManager`: Manages refinery engineering personas and statutory OISD/API safety directives.
   - `ContextCompressor`: Selective parameter extraction preserving critical operating limits (bar, °C, RPM, tags).
   - `ConversationFormatter`: Multi-turn chat memory formatting (Markdown, ChatML, Plain).
   - `CitationFormatter`: Bibliographic references (inline anchors, footnotes, tabular layout).
   - `PromptValidator`: Gating against token overflow, prompt injection, and missing citation anchors.
   - `PromptFactory`: Central thread-safe factory with instance pooling.
4. **Guardrails & Anti-Hallucination:**
   - `CitationValidator` (prunes phantom citations), `HallucinationGuard` (cross-verifies technical parameters against context), and `SafetyValidator` (detects injection attacks and redact credentials).
5. **Multi-Turn Memory & WAL Cache:**
   - Sliding-window `ConversationMemory` and persistent SQLite WAL `GenerationCache`.
6. **True Token Streaming:**
   - `StreamingManager` measuring Time-To-First-Token (TTFT) and throughput (tokens/sec).
7. **Unified Master RAG Pipeline (`RAGPipeline`):**
   - Full end-to-end chain: $\text{Dataset} \to \text{Loader} \to \text{Parser} \to \text{Cleaning} \to \text{Chunking} \to \text{Embedding} \to \text{Qdrant} \to \text{Hybrid Retrieval} \to \text{Context Packing} \to \text{Prompt Builder} \to \text{RetrievedPrompt} \to \text{Generation} \to \text{RAGResponse}$.

---

## Running Tests & Benchmarks

```powershell
# Run the automated pytest suite (187 unit, integration, and prompt architecture tests)
& "d:\SovereignAI\myenv\Scripts\python.exe" -m pytest tests/ -v

# Run generation engine micro-benchmark and throughput suite
& "d:\SovereignAI\myenv\Scripts\python.exe" scripts/benchmark_generation.py

# Run retrieval benchmark suite
& "d:\SovereignAI\myenv\Scripts\python.exe" scripts/benchmark_retrieval.py --scales 10 50 100

# Run vector database platform benchmark
& "d:\SovereignAI\myenv\Scripts\python.exe" scripts/benchmark_vector_db.py
```
