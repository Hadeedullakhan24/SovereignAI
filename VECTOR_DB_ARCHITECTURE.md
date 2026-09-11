# Master Vector Database Architecture Specification
## Sovereign On-Premise Agentic AI Workbench (SIH26117 / MRPL)
### Milestone 7: Vector Database & Indexing Backend

---

## 1. Executive Summary & Technology Decision

The Vector Database subsystem is responsible for persisting dense numerical embeddings, storing rich chunk metadata, and providing ultra-low-latency similarity search for the Sovereign AI Knowledge Base. In an on-premise refinery environment, simplicity, zero-daemon operation, robustness, and ease of backup/restore take precedence over multi-node distributed complexity.

---

## 2. Comparative Evaluation: ChromaDB vs. FAISS vs. Milvus

| Architectural Dimension | **ChromaDB (Selected Primary)** | **FAISS (Secondary / In-Memory)** | **Milvus (Enterprise Distributed)** |
|:---|:---|:---|:---|
| **Deployment Mode** | **Embedded in-process** (Serverless library). | Embedded C++ library with Python bindings. | Distributed client-server (Docker / K8s required). |
| **Air-Gapped Suitability** | **Exceptional** (Pure Python/C++ binary wheel, zero daemons). | **High** (Pure library, requires BLAS/OpenMP). | **Low / Medium** (Requires etcd, MinIO, Pulsar/Kafka). |
| **Metadata Filtering** | **Native SQL-like filtering** (`where={"category": "Manual"}`). | **Extremely primitive** (ID mapping only; requires custom index). | **Advanced** (Rich boolean expression engine). |
| **Persistence Engine** | SQLite + HNSW binary graph files. | Binary flat files (`.index` files written manually). | Distributed storage (S3 / MinIO / local disk). |
| **Incremental Updates** | **Built-in `add()`, `update()`, `upsert()`, `delete()`**. | Requires full index rebuild for many index types. | Native write-ahead logging (WAL) and compaction. |
| **Hardware Footprint** | Lightweight (~100MB RAM base). | Ultra-lightweight (~50MB RAM base). | Heavy (>4GB RAM minimum for standalone daemon). |
| **Complexity Score** | **Lowest** (Instantiate via `chromadb.PersistentClient()`). | Moderate (Manual ID tracking & synchronization). | High (Network latency, service health monitoring). |

### 2.1 Why ChromaDB is Selected as Primary Backend
1. **Zero Daemon Overhead:** Runs completely inside the Python process. No background Docker containers, no extra network ports to open in strict refinery network zones, and zero firewall approval delays.
2. **First-Class Metadata Filtering:** Crucial for refinery safety queries. Allows queries to filter strictly by `plant_unit == "CDU-1"`, `category == "Safety Document"`, or `has_warnings == True` before running vector distance computation.
3. **Atomic File-Based Storage:** The entire database resides in a single folder (`./vector_db/chroma/`). Backing up the database is as simple as copying the directory.
4. **HNSW High-Recall Indexing:** Uses Hierarchical Navigable Small World (HNSW) graphs, delivering sub-10ms nearest neighbor queries with >98% recall.

---

## 3. Storage Hierarchy & Directory Layout

The on-premise vector store structure is organized under `./vector_db/`:

```text
d:\SovereignAI\vector_db/
├── chroma/                                # ChromaDB Persistent Storage
│   ├── chroma.sqlite3                     # SQLite database: collections, documents, metadata
│   └── 3a8b2d1c-8e4f-4a6b-9c2d-0e1f2a3b4c5d/ # HNSW graph files for collection
│       ├── data_level0.bin                # Raw vector data arrays
│       ├── header.bin                     # Index dimension & configuration
│       ├── length.bin                     # Vector counts
│       └── link_list.bin                  # Graph edge connections
├── bm25/                                  # Sparse Lexical Keyword Index
│   ├── bm25_index.pkl                     # Serialized inverted index
│   └── doc_id_map.json                    # Chunk ID mapping
└── embedding_cache.sqlite                 # Milestone 6: L2 persistent embedding cache
```

---

## 4. Metadata Storage & Filtering Strategy

ChromaDB enforces strict metadata typing: values must be scalars (`str`, `int`, `float`, `bool`). Structured lists from `ChunkMetadata` (such as `equipment_entities` and `safety_entities`) must be flattened during ingestion to enable fast filtering:

### 4.1 Flattening Specification
```python
def flatten_metadata_for_chromadb(meta: ChunkMetadata) -> dict[str, str | int | float | bool]:
    return {
        "document_id": meta.document_id,
        "document_name": meta.document_name,
        "source_path": meta.source_path,
        "sha256": meta.sha256,
        "page_number": meta.page_number if meta.page_number is not None else -1,
        "section_title": meta.section_title or "",
        "chunk_index": meta.chunk_index,
        "category": meta.category,
        "subcategory": meta.subcategory,
        "plant_unit": meta.plant_unit or "GENERAL",
        "has_equipment": len(meta.equipment_entities) > 0,
        "equipment_tags_csv": ",".join(meta.equipment_entities),  # For substring / token filtering
        "has_safety_notices": len(meta.safety_entities) > 0,
        "safety_tags_csv": ",".join(meta.safety_entities),
        "is_table": meta.is_table_chunk,
        "language": meta.language,
        "embedding_model": meta.embedding_model,
        "version": meta.version,
    }
```

### 4.2 Query Filter Examples
- Filter by Category and Plant Unit:
  ```python
  where = {
      "$and": [
          {"category": {"$eq": "Manual"}},
          {"plant_unit": {"$eq": "CDU-1"}}
      ]
  }
  ```
- Filter for High-Risk Safety Notices:
  ```python
  where = {
      "$and": [
          {"category": {"$eq": "Safety Document"}},
          {"has_safety_notices": {"$eq": True}}
      ]
  }
  ```

---

## 5. Incremental Updates & Deduplication Architecture

When processing periodic updates to refinery manuals (e.g. revisions, new inspection batches):
1. **Manifest Comparison:** The orchestrator checks the document's SHA-256 against `outputs/manifest.json`.
2. **If Document SHA-256 is Unchanged:** Entire document is skipped. Zero chunking, zero embedding, zero vector DB write.
3. **If Document SHA-256 Has Changed (Revision):**
   - The orchestrator queries ChromaDB for all chunks where `document_id == target_doc_id`.
   - Executes atomic purge: `collection.delete(where={"document_id": target_doc_id})`.
   - Ingests new chunks and vectors.
4. **Chunk-Level Hash Deduplication:** If an identical paragraph exists across multiple revisions, `EmbeddingCache` provides instant vector retrieval via chunk text hash, saving 100% of compute time.

---

## 6. Multi-Process Concurrency & Team Sharing

In a multi-user workbench where 6 team members or autonomous agent processes access the knowledge base:
- **Reader Processes (Query / Retrieval):** Any number of processes can open ChromaDB in read-only mode concurrently.
- **Writer Processes (Ingestion / Indexing):** Uses a file-based lock (`filelock.FileLock("./vector_db/chroma.lock")`) ensuring that batch insertions are atomic and never corrupt HNSW graph pointers.

---

## 7. Backup and Disaster Recovery Procedure

### 7.1 Online Hot Backup
```powershell
# 1. Acquire read lock on vector database
# 2. Execute SQLite online backup for metadata
& "d:\SovereignAI\myenv\Scripts\python.exe" -c "
import sqlite3, shutil, os
src_conn = sqlite3.connect('./vector_db/chroma/chroma.sqlite3')
dst_conn = sqlite3.connect('./backups/chroma_backup.sqlite3')
src_conn.backup(dst_conn)
dst_conn.close()
src_conn.close()
# Copy HNSW vector binary folders
shutil.copytree('./vector_db/chroma', './backups/chroma_hot_backup', dirs_exist_ok=True)
print('Hot backup completed successfully.')
"
```

### 7.2 Disaster Recovery / Restore
To restore on a clean air-gapped machine:
1. Copy backup archive to `./vector_db/chroma/`.
2. Instantiate `chromadb.PersistentClient(path="./vector_db/chroma")`.
3. Verification script runs `collection.count()` and a test probe query to ensure HNSW indices are intact.
