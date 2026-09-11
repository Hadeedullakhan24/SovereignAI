"""Schema: ParsedDocument and structured refinery knowledge representations.

Defines rich, semantic representations of parsed documents:
- Sections (hierarchical H1-H6)
- Tables (headers, rows, cells, coordinates)
- Equipment entities (pumps, valves, compressors, etc. with operating limits)
- Entity relationships and EntityGraph
- Safety warnings and regulatory standards (OISD, PNGRB, API, ASME)
- Cross references, citations, and validation reports
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from rag_engine.schemas.document import DocumentLifecycleState


class EquipmentType(StrEnum):
    """Refinery equipment classifications."""

    VALVE = "VALVE"
    PUMP = "PUMP"
    MOTOR = "MOTOR"
    COMPRESSOR = "COMPRESSOR"
    HEAT_EXCHANGER = "HEAT_EXCHANGER"
    PIPELINE = "PIPELINE"
    PRESSURE_VESSEL = "PRESSURE_VESSEL"
    INSTRUMENT = "INSTRUMENT"
    FURNACE = "FURNACE"
    TANK = "TANK"
    OTHER = "OTHER"


class RelationType(StrEnum):
    """Refinery equipment relationship types."""

    CONNECTED_TO = "connected_to"
    FEEDS = "feeds"
    REGULATES = "regulates"
    MONITORS = "monitors"
    POWERS = "powers"
    BYPASSES = "bypasses"
    ISOLATES = "isolates"
    MEASURES = "measures"
    ASSOCIATED_WITH = "associated_with"


class SafetyWarningSeverity(StrEnum):
    """Severity levels for safety warnings."""

    DANGER = "DANGER"
    WARNING = "WARNING"
    CAUTION = "CAUTION"
    NOTICE = "NOTICE"


class CitationCoordinates(BaseModel):
    """Exact source coordinates for citation and auditability."""

    model_config = ConfigDict(frozen=True)

    page_number: Optional[int] = Field(default=None, description="1-indexed source page")
    section_title: Optional[str] = Field(default=None, description="Heading or section name")
    paragraph_index: Optional[int] = Field(default=None, description="0-indexed paragraph number")
    char_offset_start: Optional[int] = Field(default=None, description="Start character offset")
    char_offset_end: Optional[int] = Field(default=None, description="End character offset")
    citation_id: str = Field(
        default_factory=lambda: f"cit_{uuid.uuid4().hex[:8]}",
        description="Deterministic or unique citation identifier",
    )


class Section(BaseModel):
    """Hierarchical document section (H1-H6)."""

    section_id: str = Field(description="Unique section identifier")
    title: str = Field(description="Section heading or title")
    level: int = Field(default=1, ge=1, le=6, description="Heading level 1-6")
    content: str = Field(default="", description="Normalized body content of section")
    raw_text: str = Field(default="", description="Raw unnormalized text")
    normalized_text: str = Field(default="", description="Sanitized and normalized text")
    paragraphs: list[str] = Field(default_factory=list, description="Extracted paragraph strings")
    bullet_points: list[str] = Field(default_factory=list, description="Extracted bullet points")
    numbered_items: list[str] = Field(default_factory=list, description="Extracted numbered list items")
    page_number: Optional[int] = Field(default=None, description="Starting page number")
    subsections: list[Section] = Field(default_factory=list, description="Nested child subsections")
    citation_coords: Optional[CitationCoordinates] = Field(default=None, description="Citation coordinates")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Extraction confidence score")


class TableCell(BaseModel):
    """Individual table cell with coordinate preservation."""

    row_idx: int = Field(ge=0, description="0-indexed row position")
    col_idx: int = Field(ge=0, description="0-indexed column position")
    value: str = Field(default="", description="Normalized cell value")
    raw_value: str = Field(default="", description="Raw string value from source")
    is_header: bool = Field(default=False, description="True if cell belongs to a header row")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Extraction confidence")


class Table(BaseModel):
    """Structured tabular data extracted from document."""

    table_id: str = Field(description="Unique table identifier")
    caption: str = Field(default="", description="Table caption, title, or label")
    headers: list[str] = Field(default_factory=list, description="Column header strings")
    rows: list[list[str]] = Field(default_factory=list, description="Row values as list of string cells")
    row_count: int = Field(default=0, ge=0, description="Total number of data rows")
    col_count: int = Field(default=0, ge=0, description="Total number of columns")
    page_number: Optional[int] = Field(default=None, description="Source page number")
    dataframe_dict: dict[str, list[Any]] = Field(
        default_factory=dict, description="Dictionary of columns for DataFrame ingestion"
    )
    raw_text: str = Field(default="", description="Raw delimited or visual representation")
    normalized_text: str = Field(default="", description="Markdown or clean representation")
    cells: list[TableCell] = Field(default_factory=list, description="Coordinate-indexed cell objects")
    citation_coords: Optional[CitationCoordinates] = Field(default=None, description="Source coordinates")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Table extraction confidence")


class EquipmentEntity(BaseModel):
    """Refinery equipment entity extracted from text or schematics."""

    entity_id: str = Field(description="Unique entity identifier")
    tag: str = Field(description="Equipment tag or ID, e.g. 'P-203', 'V-12', 'MOV-101'")
    name: str = Field(description="Human readable name or description")
    equipment_type: EquipmentType = Field(default=EquipmentType.OTHER, description="Equipment category")
    operating_pressure: Optional[str] = Field(default=None, description="Extracted pressure specification")
    operating_temperature: Optional[str] = Field(default=None, description="Extracted temperature specification")
    units: dict[str, str] = Field(default_factory=dict, description="Physical units mapped to parameter types")
    loop_number: Optional[str] = Field(default=None, description="Control or instrumentation loop tag")
    pipe_number: Optional[str] = Field(default=None, description="Associated pipe or line number")
    location_section: Optional[str] = Field(default=None, description="Document section where found")
    page_number: Optional[int] = Field(default=None, description="Page number where detected")
    raw_text: str = Field(default="", description="Raw verbatim mention in text")
    normalized_text: str = Field(default="", description="Normalized representation")
    char_offset: Optional[int] = Field(default=None, description="Character offset in document")
    citation_id: str = Field(default="", description="Citation identifier")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Extraction confidence")


class EntityRelationship(BaseModel):
    """Directed connection between two equipment entities."""

    source_tag: str = Field(description="Source equipment tag, e.g. 'P-203'")
    target_tag: str = Field(description="Target equipment tag, e.g. 'V-12'")
    relation_type: RelationType = Field(description="Type of connection or flow")
    context_sentence: str = Field(default="", description="Verbatim sentence establishing the connection")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Relationship confidence score")


class EntityGraph(BaseModel):
    """Graph of connected refinery equipment entities."""

    nodes: list[EquipmentEntity] = Field(default_factory=list, description="Unique equipment nodes")
    edges: list[EntityRelationship] = Field(default_factory=list, description="Directed relationship edges")

    def add_node(self, entity: EquipmentEntity) -> None:
        """Add node if tag does not already exist."""
        existing_tags = {n.tag for n in self.nodes}
        if entity.tag not in existing_tags:
            self.nodes.append(entity)

    def add_edge(self, relation: EntityRelationship) -> None:
        """Add directed relationship edge."""
        self.edges.append(relation)

    def get_neighbors(self, tag: str) -> list[tuple[RelationType, str]]:
        """Return list of (relation_type, target_tag) for a source tag."""
        return [(e.relation_type, e.target_tag) for e in self.edges if e.source_tag == tag]

    def to_dict(self) -> dict[str, Any]:
        """Serialize graph to dictionary."""
        return {
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "nodes": [n.model_dump() for n in self.nodes],
            "edges": [e.model_dump() for e in self.edges],
        }


class SafetyWarning(BaseModel):
    """Industrial safety warning, precaution, or regulatory clause."""

    warning_id: str = Field(description="Unique warning identifier")
    severity: SafetyWarningSeverity = Field(description="Severity level")
    text: str = Field(description="Normalized warning text")
    raw_text: str = Field(default="", description="Raw verbatim warning text")
    normalized_text: str = Field(default="", description="Sanitized warning text")
    standards: list[str] = Field(
        default_factory=list, description="Associated standards (e.g. OISD-105, PNGRB)"
    )
    page_number: Optional[int] = Field(default=None, description="Source page number")
    section: Optional[str] = Field(default=None, description="Section heading")
    citation_id: str = Field(default="", description="Citation identifier")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Extraction confidence")


class CrossReference(BaseModel):
    """Cross reference link to figure, table, section, or standard."""

    ref_type: str = Field(description="Type of reference: FIGURE | TABLE | SECTION | STANDARD")
    target: str = Field(description="Target identifier, e.g. 'Fig. 3.2', 'Table 4'")
    raw_text: str = Field(default="", description="Verbatim reference text")
    page_number: Optional[int] = Field(default=None, description="Source page number")
    citation_id: str = Field(default="", description="Citation identifier")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Extraction confidence")


class DrawingMetadata(BaseModel):
    """Metadata extracted from technical drawings and schematics."""

    drawing_name: str = Field(default="", description="Drawing title from title block")
    drawing_number: str = Field(default="", description="Official drawing number")
    title_block: dict[str, Any] = Field(default_factory=dict, description="Parsed title block fields")
    revision: str = Field(default="", description="Revision code or number")
    dimensions: tuple[int, int] = Field(default=(0, 0), description="(width_px, height_px)")
    resolution_dpi: tuple[int, int] = Field(default=(72, 72), description="(dpi_x, dpi_y)")
    file_size_bytes: int = Field(default=0, ge=0, description="Asset file size")
    image_reference: str = Field(default="", description="Path or URI to image file")


class ParsedMetadata(BaseModel):
    """Structured document-level operational metadata."""

    model_config = ConfigDict(extra="allow")

    title: str = Field(default="", description="Detected or declared document title")
    author: Optional[str] = Field(default=None, description="Document author or creator")
    category: str = Field(default="Unknown", description="Classified category")
    plant_unit: Optional[str] = Field(default=None, description="Refinery plant or operating unit")
    revision_numbers: list[str] = Field(default_factory=list, description="Extracted revision tags")
    engineer_names: list[str] = Field(default_factory=list, description="Detected engineer signatures")
    inspection_dates: list[str] = Field(default_factory=list, description="Extracted inspection dates")
    maintenance_dates: list[str] = Field(default_factory=list, description="Extracted maintenance dates")
    standards_referenced: list[str] = Field(default_factory=list, description="Industrial standards cited")
    extra_metadata: dict[str, Any] = Field(default_factory=dict, description="Format-specific attributes")


class DocumentStatistics(BaseModel):
    """Aggregate statistics for the parsed document."""

    total_pages: int = Field(default=1, ge=0)
    total_sections: int = Field(default=0, ge=0)
    total_paragraphs: int = Field(default=0, ge=0)
    total_words: int = Field(default=0, ge=0)
    total_characters: int = Field(default=0, ge=0)
    total_tables: int = Field(default=0, ge=0)
    total_entities: int = Field(default=0, ge=0)
    total_warnings: int = Field(default=0, ge=0)
    total_relations: int = Field(default=0, ge=0)


class ValidationIssue(BaseModel):
    """Detailed validation issue identified in document structure."""

    severity: str = Field(description="ERROR | WARNING | INFO")
    code: str = Field(description="Standardized error or warning code")
    message: str = Field(description="Human readable explanation")
    location: str = Field(default="", description="Path, section, or page number")


class ValidationReport(BaseModel):
    """Structural validation report produced by ParserValidator."""

    is_valid: bool = Field(default=True, description="True if no blocking ERROR issues exist")
    issues: list[ValidationIssue] = Field(default_factory=list, description="List of issues")
    error_count: int = Field(default=0, ge=0, description="Total count of ERROR issues")
    warning_count: int = Field(default=0, ge=0, description="Total count of WARNING issues")


class PageMapEntry(BaseModel):
    """Page number coordinate mapping preserving citation tracking."""

    page_number: int = Field(ge=1, description="1-indexed source document page")
    char_start: int = Field(default=0, ge=0, description="Start character offset in full text")
    char_end: int = Field(default=0, ge=0, description="End character offset in full text")
    section_ids: list[str] = Field(default_factory=list, description="Section IDs on this page")
    table_ids: list[str] = Field(default_factory=list, description="Table IDs on this page")


class CleaningStatistics(BaseModel):
    """Observability metrics and counters produced by the Cleaning & Normalization Engine."""

    execution_time_ms: float = Field(default=0.0, ge=0.0, description="Execution time in milliseconds")
    characters_removed: int = Field(default=0, ge=0, description="Total characters removed")
    characters_normalized: int = Field(default=0, ge=0, description="Total characters modified/normalized")
    headers_removed: int = Field(default=0, ge=0, description="Recurring header instances removed")
    footers_removed: int = Field(default=0, ge=0, description="Recurring footer instances removed")
    whitespace_reductions: int = Field(default=0, ge=0, description="Excess whitespace characters collapsed")
    protected_tokens_count: int = Field(default=0, ge=0, description="Count of protected engineering tokens")
    warnings_count: int = Field(default=0, ge=0, description="Count of non-fatal cleaning warnings")
    failures_count: int = Field(default=0, ge=0, description="Count of stage failures")


class ParsedDocument(BaseModel):
    """Complete parsed document containing structured refinery knowledge."""

    document_id: str = Field(description="Unique parsed document identifier")
    raw_document_id: str = Field(description="ID of source Document from Milestone 3A")
    title: str = Field(default="", description="Cleaned document title")
    category: str = Field(default="Unknown", description="Classified document category")
    sections: list[Section] = Field(default_factory=list, description="Hierarchical document sections")
    tables: list[Table] = Field(default_factory=list, description="Extracted tables with coordinates")
    equipment: list[EquipmentEntity] = Field(default_factory=list, description="Extracted equipment entities")
    entity_graph: EntityGraph = Field(default_factory=EntityGraph, description="Connected equipment graph")
    warnings: list[SafetyWarning] = Field(default_factory=list, description="Safety and regulatory notices")
    cross_references: list[CrossReference] = Field(default_factory=list, description="Document cross-references")
    drawing_metadata: Optional[DrawingMetadata] = Field(default=None, description="Drawing asset metadata")
    metadata: ParsedMetadata = Field(default_factory=ParsedMetadata, description="Operational metadata")
    statistics: DocumentStatistics = Field(default_factory=DocumentStatistics, description="Document statistics")
    processing_history: list[dict[str, Any]] = Field(
        default_factory=list, description="Audit log of pipeline stages and transformations"
    )
    validation_report: Optional[ValidationReport] = Field(
        default=None, description="Automated validation findings"
    )
    lifecycle_state: DocumentLifecycleState = Field(
        default=DocumentLifecycleState.PARSED, description="Pipeline state: PARSED"
    )

    # Cleaning and normalization extensions (Milestone 4)
    cleaning_status: str = Field(default="RAW", description="Cleaning status: RAW | CLEANED | FAILED")
    cleaning_statistics: Optional[CleaningStatistics] = Field(
        default=None, description="Cleaning metrics and counters"
    )
    normalization_version: str = Field(default="1.0.0", description="Normalization engine version")
    page_map: dict[int, PageMapEntry] = Field(
        default_factory=dict, description="Page mapping and citation coordinates"
    )
    removed_headers: list[str] = Field(
        default_factory=list, description="List of recurring headers removed"
    )
    removed_footers: list[str] = Field(
        default_factory=list, description="List of recurring footers removed"
    )
    protected_tokens: list[str] = Field(
        default_factory=list, description="Refinery engineering tokens protected during cleaning"
    )
    cleaning_warnings: list[str] = Field(
        default_factory=list, description="Warnings emitted during cleaning"
    )

    def get_full_text(self) -> str:
        """Concatenate all section text into a unified body."""
        if not self.sections:
            return ""
        return "\n\n".join(s.content for s in self.sections if s.content.strip())


class CleanParsedDocument(ParsedDocument):
    """Cleaned and normalized ParsedDocument ready for Chunking Engine (Milestone 5)."""

    cleaning_status: str = Field(default="CLEANED", description="Cleaning status")
    lifecycle_state: DocumentLifecycleState = Field(
        default=DocumentLifecycleState.CLEANED, description="Pipeline state: CLEANED"
    )

