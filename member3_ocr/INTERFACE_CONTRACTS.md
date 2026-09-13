# Interface Contracts: Member 3 OCR → Downstream Consumers

> **Status:** Active — grounded in live schema inspection of `rag_engine/schemas/` (2026-09-12)
> **Owner:** Member 3 (OCR & Vision Intelligence)
> **Consumers:** Member 1 (RAG Engine), Agent components (TBD)

---

## Overview

`document_parser.py` is the **sole integration boundary** between the OCR pipeline and all
downstream components.  It converts an `OCRDocumentResult` (from `ocr_pipeline.py`) into a
`ParsedDocument` (from `rag_engine.schemas.parsed_document`).  No downstream consumer
should import from `ocr_pipeline.py` directly.

```
OCRPipeline → OCRDocumentResult
                    ↓
             DocumentParser.parse_ocr_result()
                    ↓
              ParsedDocument          ← this is the contract boundary
             /            \
  RAG Chunking Engine   Agent Query Interface
        ↓                        ↓
  list[Chunk]         AgentQueryableDocument (stub)
```

---

## Contract 1: ParsedDocument (Member 3 → Member 1 RAG)

### Source
`rag_engine.schemas.parsed_document.ParsedDocument` (Pydantic v2 model, frozen=False)

### Guarantee
`DocumentParser.parse_ocr_result()` always returns a `ParsedDocument` with
`lifecycle_state = DocumentLifecycleState.PARSED`.  Callers must not assume
`lifecycle_state = CLEANED`; that is set by Member 1's Cleaning & Normalization Engine.

### Mandatory fields (always populated, never None)

| Field | Type | Description |
|-------|------|-------------|
| `document_id` | `str` | `"parsed_{ocr_result.document_id}"` — deterministic, stable |
| `raw_document_id` | `str` | Source OCR document ID or caller-supplied override |
| `title` | `str` | Heuristically detected or caller-supplied (may be `""`) |
| `category` | `str` | Caller-supplied category label (default: `"Unknown"`) |
| `sections` | `list[Section]` | ≥ 0 entries; each has `section_id`, `title`, `level`, `content`, `paragraphs`, `citation_coords` |
| `tables` | `list[Table]` | ≥ 0 entries; converted from `OCRPageResult.tables` |
| `page_map` | `dict[int, PageMapEntry]` | One entry per OCR page; keys are 1-indexed page numbers |
| `statistics` | `DocumentStatistics` | Computed from sections/tables/pages |
| `processing_history` | `list[dict]` | At least one entry with key `"stage" = "document_parser"` |
| `validation_report` | `ValidationReport` | Always present; `is_valid=False` only on OCR errors |
| `lifecycle_state` | `DocumentLifecycleState` | Always `PARSED` |
| `metadata` | `ParsedMetadata` | `title` and `category` are populated; all other fields default to `None` or `[]` |
| `entity_graph` | `EntityGraph` | Empty in Phase 1 (no entity extraction) |
| `cross_references` | `list[CrossReference]` | Extracted via regex from section text |

### Optional / phase-gated fields

| Field | Phase 1 value | Future |
|-------|--------------|--------|
| `equipment` | `[]` | Populated by entity extractor (Phase 2) |
| `warnings` | `[]` | Populated by safety warning extractor (Phase 2) |
| `drawing_metadata` | `None` | Populated when source is a technical drawing |
| `cleaning_status` | `"RAW"` | Set to `"CLEANED"` by Member 1 Cleaning Engine |
| `cleaning_statistics` | `None` | Set by Member 1 Cleaning Engine |

### Section schema

Each `Section` object carries:

```python
Section(
    section_id   = "sec_{page_number}_{section_index}",   # deterministic
    title        = str,       # heading text or "Document Content"
    level        = int,       # 1-6 (H1-H6), default 1
    content      = str,       # body text (may be empty for headings-only)
    raw_text     = str,       # same as content (Phase 1: no normalization)
    normalized_text = str,    # same as content (Phase 1: no normalization)
    paragraphs   = list[str], # grouped paragraph strings
    page_number  = int,       # 1-indexed
    confidence   = float,     # average OCR word confidence (0.0–1.0)
    citation_coords = CitationCoordinates(
        page_number       = int,
        section_title     = str,
        paragraph_index   = 0,
        char_offset_start = int,   # byte offset in canonical document text
        char_offset_end   = int,
        citation_id       = "cit_sec_{page}_{idx}",
    ),
)
```

### Table schema

```python
Table(
    table_id    = "tbl_{page_number}_{table_index}_{ocr_table_id}",
    caption     = "",          # not populated in Phase 1
    headers     = list[str],   # from OCRTable.cells where is_header=True
    rows        = list[list[str]],
    row_count   = int,
    col_count   = int,
    page_number = int,
    cells       = list[TableCell],
    confidence  = float,       # from OCRTable.confidence
    citation_coords = CitationCoordinates(...),
)
```

> **Note:** The PaddleOCR basic backend does **not** extract tables. `tables` will be
> empty for all current documents. The schema is present to be populated by a
> future PP-StructureV3 backend or a dedicated table-detection model.

> [!WARNING]
> **Cross-Team Risk & Follow-Up: `TableCell.confidence` Defaulting to `1.0`**
> - **Schema Ownership:** `rag_engine.schemas.parsed_document.TableCell` is owned by Member 1 (RAG Engine). It specifies `confidence: float = Field(default=1.0, ge=0.0, le=1.0)`.
> - **Member 3 Population:** In `member3_ocr.document_parser` (line 784) and `member3_ocr.table_extractor` (lines 109, 306, 360), unmeasured cell confidence falls back to `1.0` to satisfy this non-optional float contract.
> - **Operational Risk:** Downstream RAG filtering and Agent reasoning tools consuming `TableCell.confidence` must **not** treat a `1.0` value as proof of 100% extraction certainty or ground-truth calibration. Any logic checking `if cell.confidence > 0.9` will silently trust cells where confidence was never measured.
> - **Recommended Cross-Team Action:** Raise a schema change request with Member 1/2 to migrate `TableCell.confidence` to `Optional[float] = None` (with `null` semantics denoting "unmeasured"), consistent with the calibration fix already implemented for Member 3's own `DrawingAnalysisRecord` (Contract 4).


### Geometry provenance (bounding boxes)

`CitationCoordinates` has **no bbox field**.  All bounding-box data is preserved in
`processing_history[0]["ocr_geometry"]`:

```json
{
    "stage": "document_parser",
    "ocr_geometry": [
        {
            "page_number": 1,
            "block_id": "text-1",
            "text_snippet": "MRPL PUMP INSPECTION REPORT",
            "bbox": {"left": 50, "top": 30, "right": 750, "bottom": 60,
                     "coordinate_space": "processed_pixels"},
            "confidence": 0.95
        },
        ...
    ],
    "key_value_fields": [
        {
            "page_number": 1,
            "key": "Equipment Tag",
            "value": "P-203",
            "confidence": 0.97,
            "extraction_method": "backend"
        },
        ...
    ]
}
```

> Downstream bounding-box highlighting and citation navigation must read from
> `processing_history[0]["ocr_geometry"]` until `CitationCoordinates` is extended.

### Confidence score derivation

Section `confidence` is the **arithmetic mean of OCR word confidence scores** for all
`TextBlock` objects in that section's body.  If no words have a confidence score, the
section inherits the heading-classification confidence (0.80–0.95).  These are raw
PaddleOCR recognition probabilities, not independently calibrated.

### Canonical text and character offsets

`DocumentParser.build_canonical_text(doc)` reconstructs the full document text that
`page_map` and `citation_coords` offsets refer to.  The relationship is:

```
canonical_text = INTER_PAGE_SEPARATOR.join(
    SECTION_SEPARATOR.join(
        section_canonical_text(s) for s in page_sections
    )
    for page_num in sorted(page_map.keys())
)
```

Where `INTER_PAGE_SEPARATOR = "\n\n"` and `SECTION_SEPARATOR = "\n\n"`.

Character offsets in `CitationCoordinates` are **byte offsets into this canonical text**,
not into the raw OCR text or the original PDF.

### Known limitations (Phase 1)

1. `CitationCoordinates` has no `bbox` field — geometry is in `processing_history`.
2. `ParsedDocument` has no top-level `key_value_fields` list — KV pairs are in `processing_history`.
3. `equipment`, `warnings`, and entity graph are always empty.
4. `section.raw_text` == `section.normalized_text` == `section.content` (no cleaning applied at this stage).
5. Reading order is top-to-bottom, left-to-right with 10px rounding tolerance; multi-column
   layouts are not handled.
6. Upstream OCR token-merge limitation: In dense document areas where inter-token spacing is
   very narrow (e.g. <= 5px horizontal gap such as 'FAX NO.' abutting '(614) 466-5087' in
   `82092117.png`), PaddleOCR's DBNet detector can segment both elements into a single bounding
   box, yielding an unspaced merged token (`FAXNO.(614)466-5087`). Without explicit punctuation
   delimiters (: or ;), downstream form extraction cannot split these without risk of false
   positives on alphanumeric equipment tags. Upstream OCR segmentation tuning will address this.

---

## Contract 2: RagChunk — ParsedDocument → Chunk

### Source
`rag_engine.schemas.chunk.Chunk` and `rag_engine.schemas.chunk.ChunkMetadata`

### Status: Implemented in member3_ocr.document_parser (`export_for_rag`)

Member 3 provides a concrete implementation of `export_for_rag(parsed_doc: ParsedDocument, *, prefix_modality: bool = False) -> list[Chunk]`
co-located in `member3_ocr.document_parser`. It maps section and table content into retrievable `Chunk` objects:

```python
# For each section in parsed_doc.sections:
Chunk.create(
    document_id   = parsed_doc.document_id,
    content       = section.content,   # or section paragraph
    chunk_index   = <sequential_index>,
    document_name = parsed_doc.title,
    page_number   = section.page_number,
    section_title = section.title,
    section_id    = section.section_id,
    heading_path  = [section.title],   # extend for nested sections
    category      = parsed_doc.category,
    char_start    = section.citation_coords.char_offset_start,
    char_end      = section.citation_coords.char_offset_end,
    chunk_strategy = "section",
)

# For each table in parsed_doc.tables:
Chunk.create(
    document_id   = parsed_doc.document_id,
    content       = table.normalized_text or table.raw_text,
    chunk_index   = <sequential_index>,
    page_number   = table.page_number,
    is_table_chunk = True,
    table_id      = table.table_id,
    chunk_strategy = "table",
)
```

### Required ParsedDocument fields for RAG chunking

| Field | Why needed |
|-------|-----------|
| `document_id` | Chunk ID generation and provenance |
| `sections[*].section_id` | `ChunkMetadata.section_id` |
| `sections[*].citation_coords.char_offset_start/end` | `ChunkMetadata.char_start/char_end` |
| `sections[*].page_number` | `ChunkMetadata.page_number` |
| `sections[*].content` | `Chunk.content` |
| `tables[*].table_id` | `ChunkMetadata.table_id` |
| `page_map` | Cross-page citation validation |
| `processing_history[0]["ocr_geometry"]` | Bounding-box highlighting at retrieval time |

---

## Contract 3: AgentQueryableDocument

### Status: Implemented in member3_ocr.document_parser (`AgentQueryableDocument`, `export_for_agent`)

Member 3 provides a concrete read-only implementation of `AgentQueryableDocument` and `export_for_agent(parsed_doc: ParsedDocument) -> AgentQueryableDocument`
co-located in `member3_ocr.document_parser`:

```python
class AgentQueryableDocument:
    """Read-only view of a ParsedDocument for agent query access (Contract 3)."""

    @property
    def document_id(self) -> str:
        """Stable unique ID for this parsed document."""
        ...

    @property
    def title(self) -> str:
        """Document title (may be empty string)."""
        ...

    @property
    def category(self) -> str:
        """Operational category (e.g., 'inspection_report', 'PID')."""
        ...

    def get_full_text(self) -> str:
        """Canonical document text. Uses ParsedDocument.get_full_text()."""
        ...

    def get_sections(self) -> list[dict]:
        """
        Returns section summaries as plain dicts for agent tool calls.
        Each dict has: section_id, title, level, content, page_number, confidence.
        """
        ...

    def get_tables(self) -> list[dict]:
        """
        Returns table summaries as plain dicts.
        Each dict has: table_id, headers, rows, page_number, confidence.
        """
        ...

    def get_key_value_fields(self) -> list[dict]:
        """
        Returns KV pairs from processing_history provenance log.
        Each dict has: key, value, confidence, page_number.
        """
        ...

    def get_cross_references(self) -> list[dict]:
        """
        Returns cross-reference summaries.
        Each dict has: ref_type, target, page_number, confidence.
        """
        ...

    def citation_for_section(self, section_id: str) -> dict | None:
        """
        Returns CitationCoordinates as a dict for the given section_id,
        or None if not found.
        Keys: page_number, section_title, char_offset_start, char_offset_end, citation_id.
        """
        ...
```

### Implementation guidance for Agent component

1. **Do not import from `ocr_pipeline.py`** — consume only `ParsedDocument`.
2. Call `DocumentParser.build_canonical_text(doc)` to reconstruct text with correct offsets.
3. Bounding boxes are in `doc.processing_history[0]["ocr_geometry"]` — index by `block_id`.
4. KV pairs are in `doc.processing_history[0]["key_value_fields"]`.
5. `doc.get_full_text()` joins section content only; it does **not** include table text.

---

## Contract 4: DrawingAnalysisRecord — Engineering Drawing Schema

### Status: Active (`member3_ocr.drawing_analyzer.DrawingAnalysisRecord`)

Engineering drawings (P&IDs, PFDs, equipment diagrams, electrical schematics) possess structural,
spatial, and topological semantics distinct from narrative text documents. `DrawingAnalysisRecord`
serves as the single authoritative schema produced by Member 3 for engineering drawings.

```python
@dataclass(frozen=True)
class TitleBlockInfo:
    drawing_number: str | None = None
    revision: str | None = None
    title: str | None = None
    date: str | None = None
    scale: str | None = None
    plant_unit: str | None = None
    source: str = "unknown"

@dataclass(frozen=True)
class EquipmentEntry:
    tag: str
    equipment_type: str
    source: str  # "ocr" | "vlm" | "both"
    confidence: float | None = None
    bbox: BoundingBox | None = None
    evidence: str = ""

@dataclass(frozen=True)
class InstrumentEntry:
    tag: str
    instrument_type: str
    function_code: str
    loop_number: str
    source: str = "ocr"  # "ocr" | "vlm" | "both"
    confidence: float | None = None
    bbox: BoundingBox | None = None
    evidence: str = ""

@dataclass(frozen=True)
class ConnectionEntry:
    from_tag: str
    to_tag: str
    line_tag: str | None = None
    evidence_type: str = "explicit_label"  # "drawn_line" | "explicit_label"
    source_evidence: str = ""
    confidence: float | None = None

@dataclass(frozen=True)
class DrawingCrossReference:
    target: str
    ref_type: str  # "drawing" | "standard" | "figure" | "table" | "section"
    source_text: str = ""
    confidence: float | None = None

@dataclass(frozen=True)
class DrawingAnalysisRecord:
    drawing_id: str
    source_path: str
    drawing_type: DrawingType
    title_block: TitleBlockInfo
    equipment: tuple[EquipmentEntry, ...] = ()
    instruments: tuple[InstrumentEntry, ...] = ()
    connections: tuple[ConnectionEntry, ...] = ()
    cross_references: tuple[DrawingCrossReference, ...] = ()
    extraction_metadata: dict[str, Any] = field(default_factory=dict)
    hallucination_flags: tuple[str, ...] = ()
    schema_version: str = "1.0"
```

### Confidence Semantics & Downstream Trust Contract

> [!IMPORTANT]
> **Confidence Value Semantics: Measured vs. Unmeasured (`null`)**
> 1. **`null` / `None` indicates "Not Measured"**: A missing or `null` confidence value explicitly denotes that no empirical or statistical confidence measurement was performed for this field (e.g. geometric deductions, metadata-derived tags, or models returning uncalibrated text).
> 2. **Never Default to Trust**: Downstream consumers (RAG retrieval rankers, Agent reasoning tools, and verification guards) must treat `null` as **untrusted** by default. Any trust condition must use `entry.confidence is not None and entry.confidence >= threshold`. An unmeasured field must **never** pass a trust threshold check.
> 3. **Aggregation Handling**: Aggregate or mean confidence routines (`compute_aggregate_confidence`, `record.aggregate_confidence()`) strictly exclude `null` entries to avoid artificially pulling down the average to 0 or crashing. If an entire record consists of unmeasured items, the aggregate confidence is `null`.
> 4. **Display & Serialization**: In JSON serialization, unmeasured fields emit `"confidence": null`. In human-readable RAG context chunks, unmeasured fields explicitly render as `(conf: null (unmeasured))` rather than fabricating a misleading 1.0 score.

### Critical Topology & Connectivity Rule

> [!WARNING]
> **Connections are only as reliable as the drawing's visual/textual clarity.**
> A missing connection in the output means no explicit evidence was found on the drawing,
> **not** that no connection exists physically in the refinery. Downstream consumers and agents
> must **never treat absence of evidence as a negative physical claim**. Furthermore,
> connections are never inferred from spatial proximity alone — proximity-only connections
> are rejected as hallucinations.

---

## Contract 5: Drawing → Chunk Mapping & AgentQueryableDrawing

### 1. RAG Chunk Export (`export_for_rag`)

Member 3 provides `export_for_rag(record: DrawingAnalysisRecord) -> list[Chunk]` mapping the drawing
into retrievable text chunks using authoritative `rag_engine.schemas.chunk.Chunk`:

| Chunk Target | Strategy | Heading Breadcrumb | Content Summary |
|---|---|---|---|
| **Title Block** | `title_block` | `[DrawingType, DwgNum, "Title Block"]` | Drawing number, revision, title, plant unit, date, scale |
| **Equipment & Instruments** | `equipment_list` | `[DrawingType, DwgNum, "Equipment & Instrumentation"]` | Itemized equipment tags, types, sources, ISA-5.1 instrument loops |
| **Process Topology** | `connectivity` | `[DrawingType, DwgNum, "Process Connectivity"]` | Evidenced line connections and stream transfers (`from_tag -> to_tag`) |
| **Cross-References** | `cross_references` | `[DrawingType, DwgNum, "Cross-References"]` | Referenced drawings (`SEE DWG-042`) and standards (`API 650`, `OISD-105`) |

### 2. Agent Queryable Interface (`AgentQueryableDrawing`)

Member 3 provides `export_for_agent(record: DrawingAnalysisRecord) -> AgentQueryableDrawing` exposing:

```python
class AgentQueryableDrawing:
    @property
    def drawing_id(self) -> str: ...
    @property
    def drawing_type(self) -> str: ...
    def get_title_block(self) -> dict[str, Any]: ...
    def get_equipment_list(self) -> list[dict[str, Any]]: ...
    def get_connections_for(self, tag: str) -> list[dict[str, Any]]: ...
    def get_instrument_readings(self) -> list[dict[str, Any]]: ...
    def citation_for_equipment(self, tag: str) -> dict[str, Any] | None: ...
    def get_drawing_summary(self) -> str: ...
```

---

## Contract 6: MultimodalProcessingResult (Top-Level Entry Point)

> [!IMPORTANT]
> **Canonical Integration Point**: `.orchestrate()` is the canonical, recommended integration point for RAG, Agent, and external callers.
> The legacy method `.process(...) -> MultimodalOutput` is retained strictly for backward compatibility with pre-existing internal callers and unit tests.
> New integrations **must not use `.process()`**, since it bypasses routing classification and always incurs the runtime/GPU cost of whichever sub-pipelines are manually enabled.

This is the **primary contract** that downstream consumers (RAG and Agent) should integrate against.
Rather than managing the 5 individual Member 3 sub-pipelines (`image_preprocessing.py`, `ocr_pipeline.py`,
`vision_pipeline.py`, `document_parser.py`, `drawing_analyzer.py`), callers submit raw inputs to
`MultimodalProcessor.orchestrate()` or `process_document()`.

### 1. Unified Envelope Schema (`MultimodalProcessingResult`)

```python
@dataclass(frozen=True)
class ProcessingError:
    stage: str          # "input_validation" | "pdf_rendering" | "image_preprocessing" | "ocr" | "vision_pipeline" | "drawing_analyzer" | "document_parser"
    code: str           # E.g. "OCR_EXECUTION_FAILED", "FILE_NOT_FOUND"
    message: str
    severity: str = "error"  # "error" | "warning" | "info"
    details: dict[str, Any] | None = None

@dataclass(frozen=True)
class MultimodalProcessingResult:
    document_id: str
    source_path: str
    routing_decision: str      # "plain_document" | "engineering_drawing" | "visual_inspection" | "mixed"
    routing_signals: dict[str, Any]
    parsed_document: ParsedDocument | None = None
    drawing_analysis: DrawingAnalysisRecord | None = None
    vision_analysis: VisionAnalysisRecord | None = None
    processing_errors: tuple[ProcessingError, ...] = ()
    pipeline_versions: dict[str, str] = field(default_factory=dict)
    processing_metadata: dict[str, Any] = field(default_factory=dict)
    modality_statuses: dict[str, str] = field(default_factory=dict)
    schema_version: str = "1.0"
    success: bool = True

    @property
    def has_document(self) -> bool:
        return self.parsed_document is not None

    @property
    def has_drawing(self) -> bool:
        return self.drawing_analysis is not None

    @property
    def has_vision_analysis(self) -> bool:
        return self.vision_analysis is not None

    @property
    def has_errors(self) -> bool:
        return any(e.severity == "error" for e in self.processing_errors)

    def to_dict(self) -> dict[str, Any]: ...
    def to_json(self, *, indent: int = 2) -> str: ...
    def get_full_text(self) -> str: ...
```

### 2. Fast Non-VLM Heuristic Routing Guarantee

Routing decisions are determined using lightweight, deterministic signals (image aspect ratio, OCR text density,
drawing/document keyword matching, equipment tag frequency, and inspection path patterns) in milliseconds.

| Content Type | Routing Path | VLM Invoked? | Expected Latency |
|---|---|---|---|
| **Plain Document** (reports, invoices, forms, tables) | `image_preprocessing` → `ocr_pipeline` → `document_parser` | **NO** | ~1 - 3 seconds |
| **Engineering Drawing** (P&ID, PFD, SLD, diagrams) | `image_preprocessing` → `ocr_pipeline` + `vision_pipeline` → `drawing_analyzer` | **YES** | ~4 - 9.5 minutes (CPU) |
| **Visual Inspection** (surface defects, PCB flaws, infrared) | `image_preprocessing` → `vision_pipeline` → `VisionAnalysisRecord` | **YES** | ~3 - 8 minutes (CPU) |
| **Mixed / Ambiguous** | Full multimodal pipeline with enabled paths executed | **YES** | ~4 - 9.5 minutes (CPU) |

**CLI / Runtime Override**: Callers can bypass heuristic classification with `--force-route {plain_document, engineering_drawing, visual_inspection, both}`.

### 3. Unified RAG Chunk Export (`export_for_rag`)

```python
def export_for_rag(result: MultimodalProcessingResult) -> list[Chunk]:
    ...
```

- If `result.parsed_document` is populated: delegates directly to `member3_ocr.document_parser.export_for_rag()` (Contract 2).
- If `result.drawing_analysis` is populated: delegates directly to `member3_ocr.drawing_analyzer.export_for_rag()` (Contract 5).
- If `result.vision_analysis` is populated: delegates directly to `member3_ocr.vision_pipeline.export_for_rag()` (Contract 7).
- If **multiple** are populated (mixed): emits a combined ordered list of chunks with `heading_path` prefixed by `["Document", ...]`, `["Drawing", ...]`, or `["VisualInspection", ...]`, enabling downstream vector/hybrid search to filter or weight chunks by content modality.

### 4. Unified Agent Interface (`export_for_agent`)

Delegates to `member3_ocr.document_parser.export_for_agent()`, `member3_ocr.drawing_analyzer.export_for_agent()`, and `member3_ocr.vision_pipeline.export_for_agent()`:

```python
class AgentQueryableMultimodalDocument:
    @property
    def document_id(self) -> str: ...
    @property
    def routing_decision(self) -> str: ...
    @property
    def has_document(self) -> bool: ...
    @property
    def has_drawing(self) -> bool: ...
    @property
    def has_vision_analysis(self) -> bool: ...
    @property
    def document(self) -> AgentQueryableDocument | None: ...
    @property
    def drawing(self) -> AgentQueryableDrawing | None: ...
    @property
    def visual_inspection(self) -> AgentQueryableVisionResult | None: ...

    # Pass-through querying methods:
    def get_full_text(self) -> str: ...
    def get_sections(self) -> list[dict[str, Any]]: ...
    def get_tables(self) -> list[dict[str, Any]]: ...
    def get_key_value_fields(self) -> list[dict[str, Any]]: ...
    def get_equipment_list(self) -> list[dict[str, Any]]: ...
    def get_connections_for(self, tag: str) -> list[dict[str, Any]]: ...
    def get_instrument_readings(self) -> list[dict[str, Any]]: ...
    def get_visual_observations(self) -> list[dict[str, Any]]: ...
    def has_defects(self) -> bool: ...
    def citation_for_section(self, section_id: str) -> dict[str, Any] | None: ...
    def citation_for_equipment(self, tag: str) -> dict[str, Any] | None: ...
    def get_summary(self) -> dict[str, Any]: ...
```

### 5. Error Handling & Zero Silent Mock Rule

- Pipeline failures never raise uncaught exceptions; failures are isolated and returned inside `processing_errors: tuple[ProcessingError, ...]`.
- Partial results are preserved (e.g. OCR text remains accessible even if semantic document parsing fails).
- The orchestrator **never silently substitutes a mock backend** for real inference. Backend identities and models used are recorded explicitly in `pipeline_versions` and provenance logs.

### 6. Concurrency & Windows OpenMP Safety Invariants

- **Deadlock Assessment**: Concurrent and sequential initialization of PaddleOCR (C++ inference engine linking Intel OpenMP `libiomp5md.dll`) and PyTorch (linking OpenMP runtime) was tested empirically in the same Windows process under multi-threaded execution.
- **Empirical Results Across Configurations**:
  - **With `KMP_DUPLICATE_LIB_OK=TRUE`**: Clean execution across 10 stress cycles (2.33s total); zero deadlock, hang, or error.
  - **Without `KMP_DUPLICATE_LIB_OK`**: An explicit confirmatory test was run with the environment variable cleared from the process environment (`os.environ.pop("KMP_DUPLICATE_LIB_OK", None)`). Real PaddleOCR and PyTorch models initialized and performed inference concurrently across 3 multi-threaded stress cycles without conflict, hang, or process abort (Cycle 1 cold load: 13.92s, Cycles 2 & 3: 1.32s & 1.04s; 100% success).
- **Classification as Precautionary Setting**: Because no conflict or deadlock was reproduced under either configuration on this platform, `os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"` is classified as a **precautionary setting** (following industry best practices for mixed OpenMP runtimes on Windows) rather than a load-bearing fix for an active deadlock. It is set unconditionally at import time in `multimodal_processor.py`.
- **Sequential Invariant**: Under standard operation, the orchestrator invokes OCR and Vision sequentially within a single request. Plain documents bypass the Vision/PyTorch pipeline entirely.
- **Guidance for Multi-Threaded Callers**: While multi-threaded backend initialization has not demonstrated conflicts in verified tests, downstream server architectures distributing concurrent requests across threads should keep `KMP_DUPLICATE_LIB_OK=TRUE` in place as a safeguard, or employ process-isolated workers.

### 7. Known Limitation: OCR Confidence Miscalibration on Real-World Scans

> [!WARNING]
> **OCR Confidence is NOT a Valid Proxy for Absolute Correctness on Real Scanned Documents**
> While PP-OCRv5 confidence scores show monotonic rank-order correlation (low-confidence blocks `<0.60` suffer 73.0% CER / 89.3% WER, whereas very high-confidence blocks `>0.95` achieve 14.2% CER / 40.4% WER), the model is severely **overconfident**:
> - Real scanned data (FUNSD benchmark) demonstrates an overall mean confidence of **91.40%** alongside a mean Word Error Rate of **44.03%**.
> - Even in the highest confidence band (`>0.95`, mean reported confidence `98.59%`), **40.42% of words** remain wrong or mis-segmented when matched against ground truth.
> - Root causes: lack of space tokenization in continuous text (e.g. `TO:George`, `CONFIDENTIALFACSIMILE`), dot-matrix scan artifacts, and font visual ambiguity without language model priors.
> 
> **Downstream Invariant**:
> Downstream consumers (Document Parser, Drawing Analyzer, RAG retrieval, and Agent reasoning) **MUST NOT** make automated trust, acceptance, or exclusive arbitration decisions based on raw OCR confidence exceeding a threshold (e.g., trusting OCR fields unconditionally over VLM fields). Confidence may be safely used as a **negative filter** to prune low-confidence noise/speckles (`<0.60`), but never as an affirmative guarantee of factual accuracy.

#### Codebase Audit of OCR Confidence Usage (Tracked Follow-up Items)
The following concrete locations in `member3_ocr` currently calculate, filter, or record confidence and must be audited in downstream integration tasks:
1. `member3_ocr/table_extractor.py`:
   - `_is_ocr_noise_artifact(block: TextBlock)` (Lines 109–149): Discards table cells if `conf < 0.60` or short height with low/medium confidence. *Assessment*: Valid usage as negative noise filter, but threshold must remain conservative.
   - `extract_tables_from_blocks` (Lines 306, 360, 419): Computes cell and table confidence via arithmetic mean of block confidences. *Assessment*: Reporting only; must not be treated as a probability of table structural accuracy.
2. `member3_ocr/form_extractor.py`:
   - `extract_form_key_values` (Lines 423–432, 517–526): Computes joint field confidence `joint_conf = round(sum(confs)/len(confs), 4)` by averaging key and value block confidences. *Assessment*: Exposes confidence to consumers; must be documented as an uncalibrated proxy.
3. `member3_ocr/document_parser.py`:
   - `Section` builder (Lines 411–425): Sets section confidence to `avg_conf = sum(conf_vals) / len(conf_vals)`. *Assessment*: Preserves confidence for downstream queries; consumers must not gate RAG indexing purely on `section.confidence > 0.9`.
   - `Table` parser (Lines 783–837): Propagates table cell confidence.
4. `member3_ocr/drawing_analyzer.py`:
   - `DrawingAnalyzerConfig.confidence_threshold: float = 0.5` (Lines 163, 183): Detection cutoff threshold.
   - Schema elements (Lines 214–391): `EquipmentItem`, `InstrumentItem`, `TitleBlockData` store `confidence: float = 1.0` default.

---

## Contract 7: Visual Inspection Intelligence (`VisionAnalysisRecord` → RAG Chunks & `AgentQueryableVisionResult`)

### Source
- `member3_ocr.vision_pipeline` (`VisionAnalysisRecord`, `create_vision_analysis_record`, `export_for_rag`, `export_for_agent`)
- `member3_ocr.multimodal_processor` (`RoutingDecision.VISUAL_INSPECTION`, Route C orchestration)

### Overview
Addresses industrial inspection and quality-control observations (NEU surface defects, PCB flaws, infrared photovoltaic anomalies, and field equipment photographs) that are neither text documents nor engineering diagrams. Provides a direct, structured pathway for vision-only observation outputs to reach downstream RAG retrieval and Agent reasoning tools, closing the interface gap in Requirement 4 (Vision Model) and Requirement 7 (RAG/Agent interface).

### 1. Schema (`VisionAnalysisRecord`)

```python
@dataclass(frozen=True)
class VisionAnalysisRecord:
    """Structured non-drawing visual analysis record for RAG and Agent export (Contract 7)."""

    image_id: str
    source_path: str
    image_type: str = "visual_inspection"
    scene_type: str = "unknown"
    caption: str = ""
    observations: tuple[VisualObservation, ...] = ()
    equipment: tuple[EquipmentItem, ...] = ()
    detected_objects: tuple[DetectedObject, ...] = ()
    visible_text: tuple[VisibleText, ...] = ()
    confidence: float | None = None
    extraction_metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "1.0"
```

### 2. RAG Chunk Export (`export_for_rag`)

`member3_ocr.vision_pipeline.export_for_rag(record: VisionAnalysisRecord) -> list[Chunk]` extracts up to 4 retrievable text chunks conforming strictly to Member 1's `Chunk` schema:

1. **Inspection Summary Chunk**:
   - `section_title`: `"Inspection Summary"`
   - `chunk_strategy`: `"visual_inspection_summary"`
   - Content: Image ID, source filename, inspection category, scene type, and high-level visual caption.
2. **Visual Observations & Anomaly Findings Chunk**:
   - `section_title`: `"Visual Observations"`
   - `chunk_strategy`: `"visual_observations"`
   - Content: Bulleted list of qualitative findings (e.g. crazing, pitting, cracks, hotspots, missing holes), per-finding confidence, and uncertainty notes.
3. **Visually Recognized Equipment Chunk** (emitted if equipment detected):
   - `section_title`: `"Recognized Equipment"`
   - `chunk_strategy`: `"visual_equipment"`
   - Content: Equipment type, visible tag (if verified), confidence, and visual evidence.
4. **Visible Text & Markings Chunk** (emitted if visible text detected):
   - `section_title`: `"Visible Markings"`
   - `chunk_strategy`: `"visual_markings"`
   - Content: Text strings observed directly by the vision model.

### 3. Agent Query Interface (`AgentQueryableVisionResult`)

`member3_ocr.vision_pipeline.export_for_agent(record: VisionAnalysisRecord) -> AgentQueryableVisionResult` provides an interactive query surface for Agent tool calling:

```python
class AgentQueryableVisionResult:
    def get_summary(self) -> dict[str, Any]: ...
    def get_observations(self) -> list[dict[str, Any]]: ...
    def get_equipment_list(self) -> list[dict[str, Any]]: ...
    def get_visible_text(self) -> list[dict[str, Any]]: ...
    def has_defects(self) -> bool: ...
    def to_dict(self) -> dict[str, Any]: ...
```

- `has_defects()`: Returns `True` if any observation contains defect/anomaly keywords (`defect`, `crack`, `crazing`, `pitting`, `corrosion`, `anomaly`, `scratch`, `inclusion`, `hotspot`, `missing`, `open circuit`, `short`).
- `get_observations()`: Returns serializable dictionaries with description, confidence, evidence, uncertainty, and optional bounding boxes.

---

## Section 8: Real Industrial Table OCR Validation & Data Availability Status

### Status: DATA AVAILABILITY GAP
- **Implementation State**: Fully implemented in `member3_ocr/table_extractor.py` (heuristic rule-based table extraction supporting bordered grids, row/column alignment, and markdown/HTML table export).
- **Synthetic Quantitative Benchmark**: Completed and quantified across 12 synthetic tables (cell text accuracy = 95.73%, exact match = 90.58%, row reconstruction = 100%, column reconstruction = 100%).
- **Real Industrial Table Ground Truth Audit**: An audit of all candidate datasets (`datasets/ocr`, `datasets/engineering_drawings`, `datasets/maintenance`, `datasets/inspection_reports`, `datasets/manuals`, `datasets/safety_docs`, `datasets/handwritten_notes`, `datasets/templates`) confirmed that zero cell-level ground truth annotations exist for real industrial tables. The candidate industrial PDF templates (`equipment_maintenance_log.pdf`, `corrective_action_plan.pdf`, etc.) are either blank unpopulated forms or contain non-uniform multi-row merged key-value headers rather than annotated tabular ground truth grids.
- **Contract & Traceability Statement**: *"Real industrial table quantitative validation is currently blocked by the absence of suitable cell-level ground truth in the available project datasets."*
- **Requirement 1 Traceability Status**: Requirement 1 remains **Partially Met** with the exact justification: *"Real industrial table quantitative validation is blocked by availability of suitable cell-level ground truth in the current project datasets; synthetic table validation remains quantified."*

---

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-09-12 | Initial version — grounded in live schema inspection | Member 3 |
| 2026-09-13 | Added Contract 4 (DrawingAnalysisRecord) & Contract 5 (RAG/Agent export) | Member 3 |
| 2026-09-13 | Added Contract 6 (MultimodalProcessingResult, top-level orchestrator & router) | Member 3 |
| 2026-09-13 | Promoted Contracts 2 & 3 to Implemented in `document_parser.py`; documented OpenMP concurrency verification | Member 3 |
| 2026-09-13 | Marked `.orchestrate()` as canonical over `.process()`; added no-flag OpenMP stress test empirical findings | Member 3 |
| 2026-09-13 | Added Section 7: Known limitation on OCR confidence miscalibration on real scans, and concrete audit list of codebase confidence usage | Member 3 |
| 2026-09-13 | Added Contract 7 (Visual Inspection Intelligence: `VisionAnalysisRecord` → RAG Chunks & `AgentQueryableVisionResult`) and Route C in `MultimodalProcessor` | Member 3 |
| 2026-09-13 | Added Section 8: Documented Real Industrial Table OCR Data Availability Gap and Requirement 1 traceability justification | Member 3 |



