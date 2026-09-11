# Master RAG Architecture Blueprint
## Sovereign On-Premise Agentic AI Workbench (SIH26117 / MRPL)
### Member 1: Knowledge Base / RAG Engine

---

## 1. Executive Summary & Architectural Invariants

This document constitutes the final, authoritative architecture specification for the remaining milestones of **Member 1 (Knowledge Base & RAG Engine)** for the Smart India Hackathon 2026 project **Sovereign On-Premise Agentic AI Workbench** for **Mangalore Refinery and Petrochemicals Limited (MRPL)**.

Milestone 3A (Universal Document Loading Framework) and Milestone 3B (Deep Document Parsing Engine) are **complete and verified**. This blueprint governs everything downstream of parsing:
- **Milestone 4:** Cleaning & Normalization Engine
- **Milestone 5:** Chunking Engine
- **Milestone 6:** Embedding Pipeline
- **Milestone 7:** Vector Database Backend
- **Milestone 8:** Retrieval & Reranking Engine
- **Milestone 9:** Complete End-to-End RAG Pipeline & Evaluation

### 1.1 Non-Negotiable Architectural Invariants
1. **100% Offline & Air-Gapped:** Zero external API calls, zero web requests, zero telemetric phoning home. All model weights, configurations, vector stores, and caches must execute strictly from local on-premise hardware.
2. **Deterministic Provenance & Source Citations:** Every retrieved chunk must retain full lineage back to its source document: `document_id`, `source_path`, `page_number`, `section_title`, `equipment_entities`, `safety_standards`, `sha256`, and character offsets.
3. **Multi-Model & Multi-Backend Abstraction:** No hardcoded model weights, vector database backends, or chunking parameters. Everything is decoupled via Abstract Base Classes (ABCs), Factory patterns, and dynamic Registries.
4. **Resiliency & Incremental State Management:** Ingestion must be idempotent and hash-based (`sha256`). If a document has already been processed, it must not be re-parsed or re-embedded needlessly.
5. **Zero Modification of Completed Milestones:** Milestone 1, Milestone 2, Milestone 3A, and Milestone 3B remain completely untouched. Downstream modules interface cleanly with existing outputs (`Document` and `ParsedDocument`).

---

## 2. End-to-End Pipeline Architecture

The end-to-end system separates cleanly into two operational lifecycles:
1. **Offline Ingestion & Indexing Pipeline (Batch / Incremental):** Transforms raw refinery documents into indexed vector stores and lexical keyword indexes.
2. **Online Query & Retrieval Pipeline (Real-Time / Low-Latency):** Accepts user or agent queries, performs hybrid search, cross-encoder reranking, and constructs citation-grounded context blocks for downstream LLM generation.

```mermaid
flowchart TD
    subgraph IngestionPipeline [Ingestion & Indexing Lifecycle]
        D_RAW[Raw Files: PDF, DOCX, CSV, MD, EML, DWG] --> L_3A[Milestone 3A: Universal Loaders]
        L_3A --> DOC_OBJ[Document Object]
        DOC_OBJ --> P_3B[Milestone 3B: Deep Parsers & Classifier]
        P_3B --> PARSED_DOC[ParsedDocument: Sections, Tables, Equipment, Graph, Warnings]
        
        PARSED_DOC --> M4_CLEAN[Milestone 4: Cleaning & Normalization Engine]
        M4_CLEAN --> CLEAN_DOC[CleanedParsedDocument]
        
        CLEAN_DOC --> M5_CHUNK[Milestone 5: Multi-Strategy Chunking Engine]
        M5_CHUNK --> CHUNKS[Rich Chunk Objects with Refinery Metadata]
        
        CHUNKS --> M6_EMB[Milestone 6: Embedding Pipeline & Cache]
        M6_EMB --> EMB_VECS[Embedding Vectors 384d / 768d / 1024d]
        
        EMB_VECS --> M7_VDB[Milestone 7: Vector Database ChromaDB / FAISS]
        CHUNKS --> M7_BM25[Milestone 7: Inverted Lexical Index BM25]
    end

    subgraph RetrievalPipeline [Query & Retrieval Lifecycle]
        USER_Q[Engineer Query: e.g. Operating limit for Pump P-203 in CDU-1] --> M8_RET[Milestone 8: Hybrid Retriever]
        M8_RET -->|Dense Embedding| M6_EMB_Q[Query Embedding]
        M6_EMB_Q -->|Vector Search Top-K| M7_VDB
        M8_RET -->|Keyword Search Top-K| M7_BM25
        
        M7_VDB --> DENSE_RES[Dense Matches]
        M7_BM25 --> SPARSE_RES[BM25 Matches]
        
        DENSE_RES & SPARSE_RES --> M8_FUSION[Reciprocal Rank Fusion RRF]
        M8_FUSION --> FUSED_CANDIDATES[Top-N Candidates]
        
        FUSED_CANDIDATES --> M8_RERANK[Cross-Encoder Reranker BGE-Reranker]
        M8_RERANK --> RERANKED_CHUNKS[Top-K High-Precision Chunks]
        
        RERANKED_CHUNKS --> M8_CIT[Citation & Source Attribution Builder]
        M8_CIT --> RET_DOC[RetrievedDocument with Exact Quotes & Metadata]
        
        RET_DOC --> M9_RAG[Milestone 9: End-to-End RAG Orchestrator]
        M9_RAG --> PROMPT[Structured Context Prompt]
        PROMPT --> AGENT_LLM[Member 3 / LLM Inference Engine]
        AGENT_LLM --> FINAL_ANS[Verifiable Answer with Inline Citations]
    end
```

---

## 3. Component Interaction & Module Boundaries

Each milestone occupies an isolated, single-responsibility module within the `rag_engine` package:

```text
rag_engine/
├── preprocessing/         # Milestone 4: Cleaning, sanitization, boilerplate removal
├── chunking/              # Milestone 5: Section-aware, recursive, table-aware chunkers
├── embeddings/            # Milestone 6: Model loader, registry, cache, batch embedder
├── vector_store/          # Milestone 7: ChromaDB, FAISS, BM25 indices, persistence
├── retrieval/             # Milestone 8: Hybrid retriever, RRF, filtering, citations
├── reranking/             # Milestone 8: Local Cross-Encoder model integration
├── pipeline/              # Milestone 9: Ingestion orchestrator, query pipeline
└── evaluation/            # Milestone 9: Benchmark metrics (Recall@K, MRR, NDCG)
```

### 3.1 Module Responsibilities & Contracts

| Module | Input Contract | Output Contract | Primary Responsibilities |
|:---|:---|:---|:---|
| **`preprocessing` (M4)** | `ParsedDocument` from Milestone 3B | `CleanedParsedDocument` | Unicode normalization, whitespace control, header/footer noise stripping, refinery acronym expansion, unit standardization. |
| **`chunking` (M5)** | `CleanedParsedDocument` | `list[Chunk]` | Hierarchical section-aware chunking, table row serialization, sentence-preserving recursive chunking, chunk metadata enrichment (equipment tags, safety notices). |
| **`embeddings` (M6)** | `list[Chunk]` or `list[str]` | `list[EmbeddingVector]` | Offline local model execution (`bge-small`, `bge-base`, `e5`, `minilm`), multi-worker batching, disk-backed embedding cache. |
| **`vector_store` (M7)** | `list[Chunk]` + `list[EmbeddingVector]` | `int` count / IDs | ChromaDB persistence, collection management, dimension enforcement, atomic updates, duplicate prevention via SHA-256. |
| **`retrieval` & `reranking` (M8)** | Query string `str` + filter dict | `RetrievedDocument` | Hybrid Dense+Sparse search, Reciprocal Rank Fusion, Cross-Encoder reranking, Citation coordinate formatting. |
| **`pipeline` & `evaluation` (M9)** | Filesystem paths / Query strings | `RAGResult` & Benchmark scores | Complete batch ingestion coordinator, real-time RAG context synthesizer, automated IR evaluation. |

---

## 4. Public Interfaces & Abstract Base Classes

All downstream modules inherit from strict Abstract Base Classes located in `rag_engine/interfaces/`:

### 4.1 Cleaner Interface (`BaseCleaner`)
```python
class BaseCleaner(ABC):
    @abstractmethod
    def clean(self, parsed_doc: ParsedDocument) -> CleanedParsedDocument:
        """Sanitize text streams, clean tables, and normalize terminology."""
        ...
```

### 4.2 Chunker Interface (`BaseChunker`)
```python
class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        """Decompose a parsed document into contextual chunks."""
        ...
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return strategy identifier (e.g. 'section_aware', 'recursive', 'table_aware')."""
        ...
```

### 4.3 Embedder Interface (`BaseEmbedder`)
```python
class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, text: str) -> EmbeddingVector: ...
    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[EmbeddingVector]: ...
    @abstractmethod
    def get_dimension(self) -> int: ...
    @abstractmethod
    def get_model_name(self) -> str: ...
```

### 4.4 Vector Store Interface (`BaseVectorStore`)
```python
class BaseVectorStore(ABC):
    @abstractmethod
    def add(self, chunks: list[Chunk], embeddings: list[EmbeddingVector]) -> int: ...
    @abstractmethod
    def search(self, query_embedding: EmbeddingVector, top_k: int = 5, filters: Optional[dict] = None) -> list[ScoredChunk]: ...
    @abstractmethod
    def delete(self, chunk_ids: list[str]) -> int: ...
    @abstractmethod
    def count(self) -> int: ...
    @abstractmethod
    def clear(self) -> None: ...
```

### 4.5 Retriever Interface (`BaseRetriever`)
```python
class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5, filters: Optional[dict] = None) -> RetrievedDocument: ...
    @abstractmethod
    def get_strategy_name(self) -> str: ...
```

---

## 5. Architectural Sequence Diagrams

### 5.1 End-to-End Document Ingestion Sequence

```mermaid
sequenceDiagram
    autonumber
    actor CLI as Ingestion Orchestrator (M9)
    participant Loader as LoaderFactory (M3A)
    participant Parser as ParserFactory (M3B)
    participant Cleaner as CleaningEngine (M4)
    participant Chunker as ChunkingEngine (M5)
    participant Cache as EmbeddingCache (M6)
    participant Embedder as EmbeddingManager (M6)
    participant VectorDB as ChromaVectorStore (M7)
    participant BM25 as BM25Index (M7)

    CLI->>Loader: load(file_path)
    Loader-->>CLI: Document (raw text & metadata)
    CLI->>Parser: get_parser(doc) -> parse(doc)
    Parser-->>CLI: ParsedDocument (sections, tables, equipment, graph)
    CLI->>Cleaner: clean(parsed_doc)
    Cleaner-->>CLI: CleanedParsedDocument
    CLI->>Chunker: chunk(cleaned_doc)
    Chunker-->>CLI: list[Chunk] (with enriched metadata)
    
    CLI->>Cache: check_cached(chunks)
    Cache-->>CLI: (cached_embeddings, uncached_chunks)
    
    alt Uncached Chunks Exist
        CLI->>Embedder: embed_batch(uncached_texts)
        Embedder-->>CLI: new_embeddings
        CLI->>Cache: store(uncached_chunks, new_embeddings)
    end
    
    CLI->>VectorDB: add(chunks, all_embeddings)
    VectorDB-->>CLI: inserted_count
    CLI->>BM25: add_documents(chunks)
    BM25-->>CLI: indexed_count
    CLI-->>CLI: Emit IngestionCompleted event
```

### 5.2 Real-Time Query & Retrieval Sequence

```mermaid
sequenceDiagram
    autonumber
    actor Agent as User / Agentic LLM (Member 3)
    participant Retriever as HybridRetriever (M8)
    participant Embedder as EmbeddingManager (M6)
    participant Chroma as ChromaVectorStore (M7)
    participant BM25 as BM25Index (M7)
    participant Reranker as CrossEncoderReranker (M8)
    participant Citations as CitationBuilder (M8)

    Agent->>Retriever: retrieve(query="P-203 max pressure in CDU-1", top_k=5)
    
    par Dense Vector Search
        Retriever->>Embedder: embed(query)
        Embedder-->>Retriever: query_vector (384d / 768d)
        Retriever->>Chroma: search(query_vector, top_k=20, filters={"category": "Manual"})
        Chroma-->>Retriever: dense_results: list[ScoredChunk]
    and Sparse BM25 Search
        Retriever->>BM25: search(query, top_k=20)
        BM25-->>Retriever: sparse_results: list[ScoredChunk]
    end

    Retriever->>Retriever: Reciprocal Rank Fusion (RRF) -> Top 15 candidates
    Retriever->>Reranker: rerank(query, candidates, top_k=5)
    Reranker-->>Retriever: top_5_reranked: list[ScoredChunk]
    
    Retriever->>Citations: build_citations(top_5_reranked)
    Citations-->>Retriever: citations: list[Citation]
    
    Retriever-->>Agent: RetrievedDocument(query, chunks, citations)
    Note over Agent: Agent grounds LLM prompt<br/>using exact page numbers and quotes
```

---

## 6. Future Integration Points

1. **Member 2 (Multimodal OCR & Vision):**
   - For engineering drawings (`.dwg`, `.png`) or scanned documents, Member 2 will populate `Chunk.content` with OCR transcripts and bounding boxes.
   - The `ChunkMetadata.image_reference` field directly maps visual bounding boxes back to the original raster file.
2. **Member 3 (Agentic LLM Subsystem):**
   - Receives `RetrievedDocument` containing clean, deduplicated chunks with confidence scores and `EntityGraph` adjacency edges.
   - The agent can navigate connected equipment (e.g. `P-203` -> `connected_to` -> `V-12`) using pre-built graph nodes without running LLM extraction at query time.
3. **Incremental Ingestion & File Watcher:**
   - Manifest-level checksum comparison ensures that dropping a new manual into `datasets/manuals/` only processes the new file, skipping the other 29,000 unmodified files.
