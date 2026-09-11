"""Unit tests for concrete parsers, entity extraction, tables, and relationships."""

import pytest

from rag_engine.parsers.csv_parser import CSVParser
from rag_engine.parsers.docx_parser import DOCXParser
from rag_engine.parsers.email_parser import EmailParser
from rag_engine.parsers.engineering_parser import EngineeringParser
from rag_engine.parsers.generic_text_parser import GenericTextParser
from rag_engine.parsers.image_metadata_parser import ImageMetadataParser
from rag_engine.parsers.inspection_parser import InspectionParser
from rag_engine.parsers.markdown_parser import MarkdownParser
from rag_engine.parsers.pdf_parser import PDFParser
from rag_engine.parsers.pptx_parser import PPTXParser
from rag_engine.parsers.safety_parser import SafetyParser
from rag_engine.parsers.parsing_context import ParsingContext
from rag_engine.schemas.document import Document, DocumentMetadata
from rag_engine.schemas.parsed_document import (
    DocumentLifecycleState,
    EquipmentType,
    RelationType,
    SafetyWarningSeverity,
)


def test_generic_text_parser_with_equipment_and_graph() -> None:
    parser = GenericTextParser()
    text = (
        "1.1 Primary Feed Line\n"
        "Centrifugal pump P-203 discharges to heat exchanger HX-01 at 12.5 bar and 140 °C.\n"
        "Gate valve V-12 isolates pipe LINE-101-CS from the suction side.\n"
        "DANGER: High pressure flammable hydrocarbon line! Compliance with OISD-105 is mandatory.\n"
        "Refer to Figure 2.1 and Table 3.2 for isometric details.\n"
        "Inspected by Engineer Ramesh Kumar on 15/04/2025. Rev: B."
    )
    doc = Document(
        doc_id="doc_text_01",
        content=text,
        metadata=DocumentMetadata(
            source_path="/data/manuals/sample.txt",
            file_name="sample.txt",
            file_format=".txt",
        ),
    )
    parsed = parser.parse(doc)

    assert parsed.lifecycle_state == DocumentLifecycleState.PARSED
    assert len(parsed.sections) >= 1
    assert parsed.sections[0].title == "1.1 Primary Feed Line"

    # Equipment verification
    tags = {e.tag for e in parsed.equipment}
    assert "P-203" in tags
    assert "HX-01" in tags
    assert "V-12" in tags

    pump_entity = next(e for e in parsed.equipment if e.tag == "P-203")
    assert pump_entity.equipment_type == EquipmentType.PUMP
    assert pump_entity.operating_pressure == "12.5 bar"
    assert pump_entity.operating_temperature == "140 °C"

    # EntityGraph verification
    graph = parsed.entity_graph
    assert len(graph.nodes) >= 3
    assert len(graph.edges) >= 1
    # P-203 -> HX-01
    p_edges = [e for e in graph.edges if e.source_tag == "P-203" and e.target_tag == "HX-01"]
    assert len(p_edges) == 1
    assert p_edges[0].relation_type == RelationType.FEEDS

    # Safety Warnings
    assert len(parsed.warnings) >= 1
    assert parsed.warnings[0].severity == SafetyWarningSeverity.DANGER
    assert "OISD-105" in parsed.warnings[0].standards

    # Cross References
    ref_targets = [r.target for r in parsed.cross_references]
    assert any("Figure 2.1" in t for t in ref_targets)
    assert any("Table 3.2" in t for t in ref_targets)

    # Operational Metadata
    assert "15/04/2025" in parsed.metadata.inspection_dates
    assert "B" in parsed.metadata.revision_numbers


def test_pdf_parser_page_split() -> None:
    parser = PDFParser()
    content = (
        "--- Page 1 ---\n"
        "# Section 1 Overview\n"
        "Pump P-101 operates at 5.0 bar.\n"
        "--- Page 2 ---\n"
        "# Section 2 Operations\n"
        "Valve V-202 throttles discharge to 2.5 bar.\n"
    )
    doc = Document(
        doc_id="doc_pdf_01",
        content=content,
        metadata=DocumentMetadata(
            source_path="/data/manuals/pump.pdf",
            file_name="pump.pdf",
            file_format=".pdf",
        ),
    )
    parsed = parser.parse(doc)

    assert parsed.statistics.total_pages == 2
    assert len(parsed.sections) == 2
    assert parsed.sections[0].page_number == 1
    assert parsed.sections[1].page_number == 2

    # Equipment has respective page numbers
    p101 = next(e for e in parsed.equipment if e.tag == "P-101")
    v202 = next(e for e in parsed.equipment if e.tag == "V-202")
    assert p101.page_number == 1
    assert v202.page_number == 2


def test_csv_parser_structured_table() -> None:
    parser = CSVParser()
    csv_content = (
        "Tag,Type,Design_Pressure,Design_Temp\n"
        "P-201,Centrifugal Pump,15 bar,160 C\n"
        "V-102,Gate Valve,20 bar,180 C\n"
        "C-301,Centrifugal Compressor,35 bar,220 C\n"
    )
    doc = Document(
        doc_id="doc_csv_01",
        content=csv_content,
        metadata=DocumentMetadata(
            source_path="/data/specs/equipment.csv",
            file_name="equipment.csv",
            file_format=".csv",
        ),
    )
    parsed = parser.parse(doc)

    assert len(parsed.tables) == 1
    table = parsed.tables[0]
    assert table.headers == ["Tag", "Type", "Design_Pressure", "Design_Temp"]
    assert table.row_count == 3
    assert table.col_count == 4
    assert table.rows[0] == ["P-201", "Centrifugal Pump", "15 bar", "160 C"]
    assert table.dataframe_dict["Tag"] == ["P-201", "V-102", "C-301"]

    # Also checks coordinates of cells
    cell_0_0 = next(c for c in table.cells if c.row_idx == 0 and c.col_idx == 0)
    assert cell_0_0.is_header is True
    assert cell_0_0.value == "Tag"


def test_markdown_parser_with_frontmatter_and_table() -> None:
    parser = MarkdownParser()
    md_content = (
        "---\n"
        "title: Crude Distillation Operating Procedure\n"
        "author: Process Engineering Lead\n"
        "---\n"
        "\n"
        "# Unit Overview\n"
        "Desalting and pre-flash operational steps.\n"
        "\n"
        "| Equipment | Tag | Limit |\n"
        "| --- | --- | --- |\n"
        "| Feed Pump | P-101 | 18 bar |\n"
        "| Preheater | HX-101 | 210 C |\n"
    )
    doc = Document(
        doc_id="doc_md_01",
        content=md_content,
        metadata=DocumentMetadata(
            source_path="/data/sop/cdu.md",
            file_name="cdu.md",
            file_format=".md",
        ),
    )
    parsed = parser.parse(doc)

    assert parsed.title == "Crude Distillation Operating Procedure"
    assert parsed.metadata.author == "Process Engineering Lead"
    assert len(parsed.tables) == 1
    assert parsed.tables[0].headers == ["Equipment", "Tag", "Limit"]
    assert parsed.tables[0].row_count == 2


def test_image_metadata_parser() -> None:
    parser = ImageMetadataParser()
    dwg_content = (
        "TITLE: Crude Overhead P&ID\n"
        "DRAWING NO: DWG-MRPL-CDU-001\n"
        "REV: 3\n"
        "SCALE: 1:50\n"
        "Connects pump P-101 to column T-101.\n"
    )
    doc = Document(
        doc_id="doc_dwg_01",
        content=dwg_content,
        metadata=DocumentMetadata(
            source_path="/data/drawings/dwg_mrpl_001.png",
            file_name="dwg_mrpl_001.png",
            file_format=".png",
            category="engineering_drawings",
            extra_metadata={"width": 3840, "height": 2160, "dpi_x": 300, "dpi_y": 300},
        ),
    )
    parsed = parser.parse(doc)

    assert parsed.drawing_metadata is not None
    assert parsed.drawing_metadata.drawing_name == "Crude Overhead P&ID"
    assert parsed.drawing_metadata.drawing_number == "DWG-MRPL-CDU-001"
    assert parsed.drawing_metadata.revision == "3"
    assert parsed.drawing_metadata.dimensions == (3840, 2160)
    assert parsed.drawing_metadata.resolution_dpi == (300, 300)


def test_email_parser_action_items() -> None:
    parser = EmailParser()
    email_text = (
        "From: safety.officer@mrpl.co.in\n"
        "To: maintenance.lead@mrpl.co.in\n"
        "Date: 12-May-2025\n"
        "Subject: Urgent: Valve V-105 Seal Leakage\n"
        "\n"
        "Inspection revealed minor gland leakage in Valve V-105.\n"
        "Action Item: Replace packing rings by 14-May-2025.\n"
        "Action Item: Perform leak test before commissioning.\n"
        "WARNING: Ensure lock-out tag-out (LOTO) is completed before commencing maintenance.\n"
    )
    doc = Document(
        doc_id="doc_email_01",
        content=email_text,
        metadata=DocumentMetadata(
            source_path="/data/emails/v105_leak.eml",
            file_name="v105_leak.eml",
            file_format=".eml",
        ),
    )
    parsed = parser.parse(doc)

    assert parsed.title == "Urgent: Valve V-105 Seal Leakage"
    assert parsed.metadata.author == "safety.officer@mrpl.co.in"
    action_section = next((s for s in parsed.sections if "Action Items" in s.title), None)
    assert action_section is not None
    assert len(action_section.bullet_points) == 2


def test_streaming_generator_support() -> None:
    parser = GenericTextParser()
    text = "# Section 1\nContent 1\n# Section 2\nContent 2\n# Section 3\nContent 3\n"
    doc = Document(
        doc_id="doc_stream_01",
        content=text,
        metadata=DocumentMetadata(
            source_path="/data/manual.txt",
            file_name="manual.txt",
            file_format=".txt",
        ),
    )
    stream = parser.parse_stream(doc)
    sections = list(stream)

    assert len(sections) == 3
    assert sections[0].title == "Section 1"
    assert sections[1].title == "Section 2"
    assert sections[2].title == "Section 3"


def test_engineering_parser_refinery_specs() -> None:
    parser = EngineeringParser()
    eng_text = (
        "# Crude Distillation Unit 1 (CDU-1) Manual\n"
        "Design Pressure: 16.5 bar\n"
        "Operating Temperature: 360 °C\n"
        "Feed pump P-101A supplies crude charge to preheat train HX-101.\n"
        "Governing Standard: API 610.\n"
    )
    doc = Document(
        doc_id="doc_eng_01",
        content=eng_text,
        metadata=DocumentMetadata(
            source_path="/data/manuals/cdu_spec.txt",
            file_name="cdu_spec.txt",
            file_format=".txt",
            category="manual",
        ),
    )
    parsed = parser.parse(doc)
    assert parsed.category == "Manual"
    assert "CDU-1" in parsed.metadata.extra_metadata["plant_units"]
    assert parsed.metadata.extra_metadata["design_limits_detected"] >= 2
    assert len(parsed.equipment) >= 2


def test_inspection_parser_findings() -> None:
    parser = InspectionParser()
    insp_text = (
        "Visual Inspection and NDT Report for Pipeline LINE-101-CS\n"
        "Remaining Wall Thickness: 8.4 mm\n"
        "Inspection Finding: Satisfactory\n"
        "Inspected by Engineer Suresh Kumar on 20/05/2025.\n"
        "Standards: API 570, OISD-155.\n"
    )
    doc = Document(
        doc_id="doc_insp_01",
        content=insp_text,
        metadata=DocumentMetadata(
            source_path="/data/reports/ndt_line_101.txt",
            file_name="ndt_line_101.txt",
            file_format=".txt",
            category="inspection_reports",
        ),
    )
    parsed = parser.parse(doc)
    assert parsed.category == "Inspection Report"
    assert "Satisfactory" in parsed.metadata.extra_metadata["inspection_findings"]
    assert "8.4 mm" in parsed.metadata.extra_metadata["thickness_measurements"]
    assert "20/05/2025" in parsed.metadata.inspection_dates


def test_safety_parser_ppe_and_permits() -> None:
    parser = SafetyParser()
    safety_text = (
        "Standard Operating Procedure: Hot Work Permit in Hydrocarbon Area\n"
        "Personal Protective Equipment Required: Helmet, Safety Shoes, Safety Goggles, Fire Suit.\n"
        "Permit to Work: Hot Work Permit and Confined Space Entry required.\n"
        "DANGER: Combustible gas atmosphere potential! Continuous gas monitoring required under OISD-105.\n"
    )
    doc = Document(
        doc_id="doc_safe_01",
        content=safety_text,
        metadata=DocumentMetadata(
            source_path="/data/safety/sop_hot_work.txt",
            file_name="sop_hot_work.txt",
            file_format=".txt",
            category="safety_docs",
        ),
    )
    parsed = parser.parse(doc)
    assert parsed.category == "Safety Document"
    assert len(parsed.metadata.extra_metadata["ppe_required"]) >= 3
    assert len(parsed.metadata.extra_metadata["work_permits_applicable"]) >= 1
    assert len(parsed.warnings) >= 1
    assert parsed.warnings[0].severity == SafetyWarningSeverity.DANGER


def test_docx_parser_fallback() -> None:
    parser = DOCXParser()
    docx_text = (
        "# Compressor Station K-301\n"
        "Compressor K-301 discharges gas at 45 bar.\n"
        "| Parameter | Value |\n"
        "| --- | --- |\n"
        "| Speed | 6500 RPM |\n"
    )
    doc = Document(
        doc_id="doc_docx_01",
        content=docx_text,
        metadata=DocumentMetadata(
            source_path="/nonexistent/compressor.docx",
            file_name="compressor.docx",
            file_format=".docx",
        ),
    )
    parsed = parser.parse(doc)
    assert len(parsed.sections) >= 1
    assert len(parsed.tables) == 1
    assert len(parsed.equipment) >= 1

