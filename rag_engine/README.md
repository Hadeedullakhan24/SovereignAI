# RAG Engine

**Sovereign On-Premise Agentic AI Workbench — Knowledge Base / RAG Pipeline**

Problem Statement: SIH26117 · Organization: MRPL · Member 1: Knowledge Base / RAG Engine

---

## Purpose

`rag_engine` is a production-grade, fully on-premise Retrieval-Augmented Generation (RAG) pipeline designed for MRPL's petroleum refinery domain. It ingests technical manuals, safety standards (OISD/PNGRB), inspection reports, engineering drawings, and maintenance records into a searchable vector knowledge base, enabling accurate, cited answers to domain-specific queries.

**Key constraints:**
- 100% offline — zero external API calls after deployment
- Air-gapped capable — all models pre-downloaded
- Interface-driven — every component is swappable via configuration
- Independently testable — no coupling to other system modules

---

## Folder Structure

```
rag_engine/
├── __init__.py                  # Package root
├── pyproject.toml               # Build, lint, and test configuration
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git exclusions
│
├── config/                      # Centralized configuration
│   ├── settings.py              # Pydantic-based settings loader
│   ├── paths.py                 # Filesystem path registry
│   ├── constants.py             # Application constants
│   └── logging_config.py        # Structured logging setup
│
├── interfaces/                  # Abstract base classes (contracts)
│   ├── base_loader.py           # Document loading interface
│   ├── base_chunker.py          # Text chunking interface
│   ├── base_embedder.py         # Embedding generation interface
│   ├── base_vector_store.py     # Vector storage interface
│   └── base_retriever.py        # Document retrieval interface
│
├── schemas/                     # Data transfer objects (Pydantic models)
│   ├── document.py              # Document, DocumentMetadata
│   ├── chunk.py                 # Chunk, ChunkMetadata
│   ├── embedding.py             # EmbeddingVector, EmbeddingRequest
│   ├── citation.py              # Citation, Source
│   └── retrieved_document.py    # RetrievedDocument, ScoredChunk
│
├── preprocessing/               # Text normalization, cleaning, dedup
├── loaders/                     # Format-specific document loaders
├── parsers/                     # Table, section, metadata extraction
├── chunking/                    # Chunking strategies (fixed, semantic, recursive)
├── metadata/                    # Chunk metadata enrichment
├── embeddings/                  # Embedding model wrappers
├── vector_store/                # Vector DB backends (ChromaDB, FAISS)
├── retrieval/                   # Dense, hybrid, contextual retrieval
├── reranking/                   # Cross-encoder reranking
├── citations/                   # Source attribution and tracking
├── evaluation/                  # Retrieval quality metrics (MRR, NDCG)
├── pipeline/                    # End-to-end ingest and query orchestration
├── cache/                       # Query and embedding caching
├── utils/                       # Shared utilities (logging, file ops, hashing)
│
├── tests/                       # Module-wise test suite
│   ├── test_loaders.py
│   ├── test_chunking.py
│   ├── test_embeddings.py
│   ├── test_vector_store.py
│   └── test_retrieval.py
│
├── scripts/                     # Executable utility scripts
│   ├── validate_dataset.py
│   ├── build_vector_db.py
│   └── benchmark.py
│
└── docs/                        # Package documentation
    ├── architecture.md
    ├── development_plan.md
    └── module_responsibilities.md
```

---

## Pipeline Overview

### Ingestion Pipeline

```
Raw Documents (PDF, DOCX, MD, CSV, TXT, Images)
       │
       ▼
  [loaders/]         Load raw bytes, detect format
       │
       ▼
  [parsers/]         Extract tables, sections, metadata
       │
       ▼
  [preprocessing/]   Clean, normalize, deduplicate
       │
       ▼
  [chunking/]        Split into sized chunks
       │
       ▼
  [metadata/]        Enrich with source, page, equipment IDs
       │
       ▼
  [embeddings/]      Generate dense vectors
       │
       ▼
  [vector_store/]    Persist to ChromaDB / FAISS
```

### Query Pipeline

```
User Query (natural language)
       │
       ▼
  [embeddings/]      Embed the query
       │
       ▼
  [retrieval/]       Dense / hybrid retrieval
       │
       ▼
  [reranking/]       Cross-encoder reranking
       │
       ▼
  [citations/]       Attach source references
       │
       ▼
  Retrieved chunks + citations → returned to caller
```

---

## Dependencies

| Package | Purpose | Required |
|---------|---------|----------|
| `pydantic>=2.0` | Schema validation and settings | ✅ |
| `pyyaml>=6.0` | YAML configuration loading | ✅ |
| `chromadb>=0.4` | Vector database (primary backend) | ✅ |
| `sentence-transformers>=2.2` | Embedding model loading | ✅ |
| `pypdf>=3.0` | PDF document loading | ✅ |
| `python-docx>=0.8` | DOCX document loading | ✅ |
| `pandas>=2.0` | CSV/tabular data loading | ✅ |
| `Pillow>=10.0` | Image preprocessing | ✅ |
| `faiss-cpu>=1.7` | Vector database (fallback backend) | Optional |
| `pytest>=7.0` | Testing framework | Dev |
| `black>=23.0` | Code formatter | Dev |
| `ruff>=0.1` | Linter | Dev |
| `mypy>=1.0` | Type checker | Dev |

---

## Getting Started

```bash
# Activate the project virtualenv
cd d:\SovereignAI
myenv\Scripts\Activate.ps1

# Install dependencies
pip install -r rag_engine/requirements.txt

# Run tests
pytest rag_engine/tests/ -v

# Validate datasets
python rag_engine/scripts/validate_dataset.py
```

---

## Milestones Roadmap

| Milestone | Description | Status |
|-----------|-------------|--------|
| M1 | Project Scaffolding & Architecture Foundation | ✅ Complete |
| M2 | Dataset Validation Engine (Scanning, Hashing, Deduplication, Manifest) | ✅ Complete |
| M3 | Document Loaders & Deep Parsers (PDF, DOCX, XLSX, CSV, TXT, MD) | 🔲 Next |
| M4 | Chunking Strategies & Metadata Enrichment | 🔲 Planned |
| M5 | Offline Embeddings & Vector Store Backend | 🔲 Planned |
| M6 | Hybrid Retrieval + Local Cross-Encoder Reranking | 🔲 Planned |
| M7 | Citation Generator & Source Attribution | 🔲 Planned |
| M8 | End-to-End Orchestration Pipeline | 🔲 Planned |
| M9 | Evaluation & Retrieval Quality Benchmarks | 🔲 Planned |
| M10 | Integration with Agentic Workbench & Sandbox | 🔲 Planned |
