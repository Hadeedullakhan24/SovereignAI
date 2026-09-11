# Next Milestone Handoff Report: Milestone 4 → Milestone 5 (Chunking Engine)

**Project:** Sovereign On-Premise Agentic AI Workbench (SIH26117 – MRPL)  
**Lead Component:** Member 1 — Knowledge Base / RAG Engine & Data Engineering  
**Current Milestone Completed:** Milestone 4 — Cleaning & Normalization Engine  
**Next Milestone Ready:** Milestone 5 — Chunking Engine  
**Date:** 2026-09-06  

---

## 1. What Was Implemented

Milestone 4 implemented a production-grade, 100% offline, deterministic **Cleaning & Normalization Engine** that transforms raw `ParsedDocument` objects from Milestone 3B into `CleanParsedDocument` instances ready for the Chunking Engine (Milestone 5).

The engine executes an immutable, strictly ordered 12-stage sequential preprocessing pipeline:
- **Stage 1 (Unicode Normalization):** NFKC standardization (`unicodedata.normalize("NFKC", ...)`).
- **Stage 2 (Encoding Normalization):** Invisible control character stripping (`\x00-\x08`, `\x0b`, `\x0c`, `\x0e-\x1f`), zero-width space removal (`\u200b`, `\ufeff`), smart quote normalization.
- **Stage 3 (Whitespace Normalization):** Horizontal space and tab collapse, line-end whitespace stripping.
- **Stage 4 (Line Ending Normalization):** CRLF/CR to LF normalization, collapsing 3+ newlines to max 2 (`\n\n`).
- **Stage 5 (Broken Paragraph Reconstruction):** Reassembles lines broken mid-sentence by layout soft wraps while strictly isolating headings, bullets, and table rows.
- **Stage 6 (Hyphenated Word Reconstruction):** Reassembles words split across line breaks (`Oper-\nating` -> `Operating`), preserving compound technical terms (`cross-check`, `self-contained`).
- **Stage 7 (Header/Footer Detection & Removal):** Frequency-based detection across sections/pages (>= 2 occurrences); safely strips repeated boilerplate while preserving all unique content.
- **Stage 8 (Page Number Preservation & Citation Mapping):** Builds `page_map: dict[int, PageMapEntry]` storing character offsets and section/table IDs, ensuring source citation mapping is never lost.
- **Stage 9 (Table Whitespace Cleanup):** Cleans cell values, headers, and markdown representation using fast C-level matrix transposition (`zip(*rows)`) without disrupting row/column counts or coordinates.
- **Stage 10 (Bullet Normalization):** Standardizes disparate bullet markers (`•`, `*`, `▪`, `►`) to Markdown `- `.
- **Stage 11 (List Normalization):** Standardizes varied numbered lists (`1)`, `(1)`, `1 -`) to Markdown `1. `.
- **Stage 12 (Engineering Token Protection & Verification):** Pre-cleaning sentinel masking (`__ENG_TOKEN_{idx}__`) and post-cleaning unmasking with automated verification ensuring 100% zero-corruption preservation of all refinery tags (`Pump P-203`, `MOV-101`), standards (`API-610`, `OISD-105`, `ASME`, `PNGRB`, `ISO`), and physical units (`10 bar`, `250°C`, `kg/cm²`, `MPa`, `psi`, `m³/hr`).

The architecture is thread-safe using `threading.RLock`, includes a dynamic runtime plugin system, and provides comprehensive observability (`CleaningMetrics`, `CleaningEventBus`, `CleaningHealthReport`).

---

## 2. Files Created

### Preprocessing Modules (`rag_engine/preprocessing/`)
1. `rag_engine/preprocessing/base_cleaner.py`: Abstract Base Class `BaseCleaner` defining cleaning lifecycle and health methods.
2. `rag_engine/preprocessing/cleaning_pipeline.py`: Master 12-stage sequential deterministic pipeline emitting `CleanParsedDocument`.
3. `rag_engine/preprocessing/normalizers.py`: `UnicodeNormalizer`, `EncodingNormalizer`, `BulletNormalizer`, `ListNormalizer`.
4. `rag_engine/preprocessing/whitespace_cleaner.py`: `WhitespaceCleaner` handling whitespace, line endings, soft wrap reconstruction, and hyphen repair.
5. `rag_engine/preprocessing/header_footer_detector.py`: `HeaderFooterDetector` identifying and pruning repeated headers/footers.
6. `rag_engine/preprocessing/engineering_token_protector.py`: `EngineeringTokenProtector` masking sentinels, unmasking, and validating token preservation.
7. `rag_engine/preprocessing/page_mapper.py`: `PageMapper` building `page_map` index and updating citation coordinates.
8. `rag_engine/preprocessing/table_cleaner.py`: `TableCleaner` cleaning tabular data with matrix transposition.
9. `rag_engine/preprocessing/text_cleaner.py`: `TextCleaner` coordinating string-level composite cleaning.
10. `rag_engine/preprocessing/processing_history.py`: `ProcessingHistoryRecorder` logging transformation audit trails.
11. `rag_engine/preprocessing/cleaning_metrics.py`: `CleaningMetrics` and thread-safe `CleaningMetricsCollector`.
12. `rag_engine/preprocessing/cleaning_events.py`: `CleaningEventBus` and domain event classes (`CleaningStarted`, `CleaningFinished`, `CleaningFailed`, `HeaderRemoved`, `FooterRemoved`, `ProtectedTokenDetected`, `NormalizationApplied`).
13. `rag_engine/preprocessing/cleaning_health.py`: Diagnostics exposing `health()`, `version()`, `dependencies()`, `supported_features()`.
14. `rag_engine/preprocessing/exceptions.py`: `CleaningError`, `PipelineStageError`, `TokenProtectionError`, `PluginError`, `CorruptedDocumentError`.
15. `rag_engine/preprocessing/utils.py`: High-throughput string diff, Levenshtein similarity, character metrics.
16. `rag_engine/preprocessing/plugin_cleaner.py`: `PluginCleanerManager` dynamic runtime plugin loader.
17. `rag_engine/preprocessing/factory.py`: `CleanerFactory` thread-safe factory with category resolution.
18. `rag_engine/preprocessing/registry.py`: `CleanerRegistry` thread-safe registry with `@register_cleaner`.
19. `rag_engine/preprocessing/__init__.py`: Public interface exports.

### Tests & Validation Scripts
20. `tests/test_cleaning_engine.py`: 17 comprehensive unit, integration, concurrency, failure, and plugin tests.
21. `scripts/validate_datasets_cleaning.py`: Full dataset validation runner executing against real MRPL manuals, SOPs, inspection reports, and maintenance records.
22. `project_management/milestone_reports/cleaning_dataset_validation.json`: Structured validation results across datasets.
23. `project_management/milestone_reports/MILESTONE_04_COMPLETION_REPORT.md`: Comprehensive formal milestone sign-off report.

---

## 3. Files Modified

1. `rag_engine/schemas/document.py`: Added `CLEANED = "CLEANED"` to `DocumentLifecycleState`.
2. `rag_engine/schemas/parsed_document.py`:
   - Added `CleaningStatistics` schema.
   - Added `PageMapEntry` schema.
   - Extended `ParsedDocument` with `cleaning_status`, `cleaning_statistics`, `normalization_version`, `page_map`, `removed_headers`, `removed_footers`, `protected_tokens`, `cleaning_warnings`.
   - Added `CleanParsedDocument(ParsedDocument)` subclass.
3. `rag_engine/schemas/__init__.py`: Exported `CleanParsedDocument`, `CleaningStatistics`, and `PageMapEntry`.
4. `README.md`: Updated active milestone to Milestone 4 (COMPLETED), added repository tree paths, added Milestone 4 documentation, updated test suite count to 75.
5. `CHANGELOG.md`: Added Milestone 4 release notes.
6. `project_management/progress.json`: Marked `milestone_04` as `COMPLETED`, listed deliverables, set `current_milestone` to `milestone_05`.
7. `project_management/pending_tasks.md`: Marked Milestone 4 tasks completed, updated active task board for Milestone 5 (Chunking Engine).
8. `project_management/known_issues.md`: Added `ISSUE-007` documenting Unicode NFKC superscript decomposition and token sentinel masking mitigation.
9. `project_management/engineering_decisions.md`: Added `EDR-006: Cleaning & Normalization Engine, 12-Stage Pipeline & Zero-Corruption Token Masking`.

---

## 4. Public Interfaces

```python
# Master pipeline invocation
from rag_engine.preprocessing import CleaningPipeline, CleanerFactory, get_cleaner_factory

pipeline = CleaningPipeline(strict_token_verification=True)
clean_doc: CleanParsedDocument = pipeline.clean(parsed_document)

# Factory dispatch by category
factory = get_cleaner_factory()
cleaner = factory.create_cleaner(category="manuals")
clean_doc = cleaner.clean(parsed_document)

# Inspect clean attributes
clean_doc.cleaning_status       # "CLEANED"
clean_doc.normalization_version # "1.0.0"
clean_doc.protected_tokens      # ['Pump P-203', 'MOV-101', 'API-610', '10 bar', '250°C', ...]
clean_doc.page_map              # {1: PageMapEntry(char_start=0, char_end=1542, section_ids=[...])}
clean_doc.cleaning_statistics   # CleaningStatistics(execution_time_ms=..., characters_removed=...)

# Health and Diagnostics
from rag_engine.preprocessing import health, version, dependencies, supported_features

report = health()
ver = version()
deps = dependencies()
feats = supported_features()

# Event Bus subscription
from rag_engine.preprocessing import get_cleaning_event_bus, CleaningFinished, HeaderRemoved

bus = get_cleaning_event_bus()
bus.subscribe(CleaningFinished, lambda evt: print(f"Cleaned {evt.document_id}"))
```

---

## 5. Test Summary

```powershell
& "d:\SovereignAI\myenv\Scripts\python.exe" -m pytest tests/ -v
```

- **Total Test Cases:** 75
- **Passed:** 75 (100% pass rate)
- **Execution Time:** ~1.70 seconds
- **Coverage Highlights:**
  - `test_unicode_nfkc_normalization`: Ligatures (`ﬁ` -> `fi`), full-width numerals.
  - `test_encoding_normalization_control_chars_and_quotes`: Control chars stripped, smart quotes normalized.
  - `test_whitespace_cleanup`: Whitespace collapsed, line ends trimmed.
  - `test_line_ending_normalization`: CRLF/CR -> LF, 3+ newlines collapsed to 2.
  - `test_broken_paragraph_reconstruction`: Soft wraps rejoined without merging headings/lists.
  - `test_hyphen_repair`: Hyphenated words across line breaks repaired (`Oper-\nating` -> `Operating`).
  - `test_header_footer_removal`: Repeated headers/footers across pages pruned with zero content loss.
  - `test_page_mapping_preservation`: Page coordinate offsets mapped and citations updated.
  - `test_table_whitespace_cleanup`: Cells and markdown cleaned with grid structure preserved.
  - `test_bullet_normalization`: `•`, `*`, `▪` standardized to `- `.
  - `test_list_normalization`: `1)`, `(1)`, `1 -` standardized to `1. `.
  - `test_engineering_token_protection`: Guaranteed zero degradation of all tags and limits.
  - `test_complete_cleaning_pipeline_e2e`: End-to-end transformation verifying all schema outputs.
  - `test_cleaning_pipeline_thread_safety`: 20 concurrent threads running `CleaningPipeline.clean()` simultaneously with `threading.RLock`.
  - `test_cleaning_failure_handling`: Graceful handling of corrupted/null inputs.
  - `test_plugin_cleaner_loading`: Dynamic plugin registration and execution.
  - `test_cleaning_health_and_diagnostics`: Comprehensive health checks and feature reporting.

---

## 6. Dataset Validation Summary

Validated on real documents from `datasets/`:
1. **Manuals (`Emerson_Control_Valve_Handbook.pdf`, 1.89M chars, ~1000 pages):** Cleaned in 6.02s; 90 protected engineering tokens detected and verified intact.
2. **Manuals (`fisher_control_valve_handbook.pdf`, 180K chars):** Cleaned in 369ms; 25 protected tokens intact.
3. **Manuals (`fisher_ic2_control_valve_handbook.pdf`, 99K chars):** Cleaned in 188ms; 12 protected tokens intact.
4. **Safety Docs (`OISD-STD-105.pdf`, `OISD-STD-116.pdf`, `OISD-STD-117.pdf`):** Cleaned in ~500ms each; standards and revision codes intact.
5. **Inspection Reports (`boiler_inspection_003.md`, `centrifugal_pump_inspection_005.md`, `compressor_inspection_006.md`):** Cleaned in <35ms; equipment tags (`B-101`, `P-005`), temperatures (`71 °C`, `84 °C`), and flow rates (`118 m³/h`) 100% preserved.
6. **Maintenance (`ai4i2020_maintenance_analysis_10000.csv`, 2.23M chars, 10,000 rows):** Cleaned in 13.4s without memory spikes.

---

## 7. Remaining Risks & Mitigations

| Risk | Impact | Mitigation |
|:---|:---|:---|
| Multi-thousand row CSV files creating large in-memory markdown strings | Memory overhead in downstream chunker | `TableCleaner` generates markdown table representations with up to 500 rows preview while preserving 100% of rows in structured attributes. |
| Mixed character sets in legacy scanned documents | Potential OCR character drift | Unicode NFKC and encoding normalizers clean control characters and normalize homoglyphs before chunking. |

---

## 8. Integration Notes for Chunking Engine (Milestone 5)

When implementing Milestone 5:
1. **Input Schema:** The Chunking Engine must accept `CleanParsedDocument` as its primary input.
2. **Token Boundary Integrity:** Use `clean_doc.protected_tokens` as unbreakable tokens so that equipment tags (e.g. `Pump P-203`, `MOV-101`) and physical units (`10 bar`, `250°C`) are **never split across chunk boundaries**.
3. **Citation & Page Coordinates:** Use `clean_doc.page_map` to assign `page_number`, `char_offset_start`, and `char_offset_end` to every generated `Chunk`.
4. **Hierarchical Section-Aware Chunking:** Leverage `clean_doc.sections` (H1-H6 hierarchy) to generate section-aware chunks that inherit parent section titles in their metadata.
5. **Table-Aware Chunking:** Leverage `clean_doc.tables` to chunk tables as semantic units, retaining header rows with each chunk if a table spans multiple chunks.

---

## 9. Recommended Implementation Strategy for Milestone 5

1. **Chunk Schemas (`rag_engine/schemas/chunk.py`):**
   - Extend `Chunk` and `ChunkMetadata` with: `chunk_id`, `document_id`, `chunk_type` (`TEXT`, `TABLE`, `EQUIPMENT`, `WARNING`), `heading_path` (e.g., `["Section 1", "Subsection 1.2"]`), `equipment_tags: list[str]`, `page_number: int`, `char_start: int`, `char_end: int`, `token_count: int`.
2. **Chunking Strategies (`rag_engine/chunking/`):**
   - `FixedSizeChunker`: Sliding window with configurable token overlap.
   - `SentenceAwareChunker`: Chunking respecting sentence boundaries.
   - `HierarchicalSectionChunker`: Respects H1-H6 section boundaries from `CleanParsedDocument`.
   - `TableChunker`: Chunks table rows while repeating column headers for context preservation.
   - `EquipmentChunker`: Chunks text surrounding equipment entities and their graph relations.
3. **Chunking Framework:**
   - `BaseChunker` (ABC).
   - `ChunkerRegistry` & `ChunkerFactory` (thread-safe with `RLock`).
   - `ChunkingMetrics` & `ChunkingEventBus`.
4. **Quality & Validation:**
   - Strict verification that zero protected tokens are bisected.
   - Automated test suite with 100% pass rate.
