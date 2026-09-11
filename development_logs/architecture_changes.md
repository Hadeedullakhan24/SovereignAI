# Architecture Changes — Milestone 2

**Project:** Sovereign On-Premise Agentic AI Workbench (SIH26117 / MRPL)  
**Milestone:** Milestone 2 (Dataset Validation Engine)  

---

## Changes Summary

1. **New Subsystem `dataset_engine/`:**
   - Established a dedicated, modular dataset engine to validate and index raw refinery data prior to chunking and vector storage.
   - Built with Clean Architecture principles: single-responsibility modules for scanning, validating, hashing, duplicate detection, manifest building, and statistics calculation.
2. **Externalized Validation Rules (`config/validation_rules.yaml`):**
   - Configurable limits for file size, permitted file extensions, ignored directory names, and category-to-doctype mappings.
3. **Pydantic v2 Manifest Schemas (`rag_engine/schemas/manifest.py`):**
   - Implemented immutable DTOs for `DocumentRecord`, `DatasetStatistics`, and `Manifest`.
4. **Concrete Utilities (`rag_engine/utils/`):**
   - Replaced Milestone 1 placeholders in `file_utils.py`, `hash_utils.py`, and `validators.py` with fully implemented, tested functions.
5. **Zero Redesign Invariant:**
   - No existing folders were renamed or removed.
   - Downstream components (`loaders`, `chunking`, `embeddings`, `vector_store`) remain untouched and ready for subsequent milestones.
