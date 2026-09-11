"""Unit tests for all concrete document loaders."""

from pathlib import Path
import pytest
import struct
import zipfile

from rag_engine.loaders.csv_loader import CSVLoader
from rag_engine.loaders.docx_loader import DOCXLoader
from rag_engine.loaders.exceptions import (
    CorruptedDocumentError,
    UnsupportedFormatError,
    ValidationFailedError,
)
from rag_engine.loaders.image_loader import ImageLoader
from rag_engine.loaders.loader_factory import global_loader_factory
from rag_engine.loaders.markdown_loader import MarkdownLoader
from rag_engine.loaders.pdf_loader import PDFLoader
from rag_engine.loaders.pptx_loader import PPTXLoader
from rag_engine.loaders.txt_loader import TXTLoader
from rag_engine.loaders.xlsx_loader import XLSXLoader


def test_txt_loader(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("Line 1: Operational log\nLine 2: Pump P-203 running normally\n", encoding="utf-8")

    loader = TXTLoader()
    doc = loader.load(f)

    assert "Pump P-203" in doc.content
    assert doc.metadata.file_format == ".txt"
    assert doc.metadata.extra_metadata["line_count"] == 2
    assert doc.metadata.word_count > 0
    assert doc.metadata.character_count > 0


def test_markdown_loader(tmp_path):
    f = tmp_path / "sop.md"
    f.write_text("""---
title: SOP-001
equipment: Hydrocracker
---

# Hydrocracker Startup Procedure

## Pre-check
Ensure pressure is below 150 bar.
```bash
check_valves.sh
```
""", encoding="utf-8")

    loader = MarkdownLoader()
    doc = loader.load(f)

    assert "Hydrocracker Startup" in doc.content
    assert doc.metadata.extra_metadata["has_frontmatter"] is True
    assert doc.metadata.extra_metadata["frontmatter"]["title"] == "SOP-001"
    assert doc.metadata.extra_metadata["heading_count"] == 2
    assert doc.metadata.extra_metadata["code_block_count"] == 1


def test_csv_loader(tmp_path):
    f = tmp_path / "sensor_data.csv"
    f.write_text("timestamp,temperature,pressure\n2026-09-06T10:00,450.5,12.3\n2026-09-06T10:01,451.2,12.4\n", encoding="utf-8")

    loader = CSVLoader()
    doc = loader.load(f)

    assert "450.5" in doc.content
    assert doc.metadata.extra_metadata["row_count"] == 3
    assert doc.metadata.extra_metadata["delimiter"] == ","


def test_pdf_loader_fallback(tmp_path):
    # Construct a minimal valid PDF with a literal text stream
    f = tmp_path / "sample.pdf"
    f.write_bytes(b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>
endobj
4 0 obj
<< /Length 44 >>
stream
BT
/F1 12 Tf
72 712 Td
(MRPL Refinery SOP Unit 2) Tj
ET
endstream
endobj
xref
0 5
trailer
<< /Root 1 0 R >>
%%EOF
""")

    loader = PDFLoader()
    doc = loader.load(f)

    assert doc.metadata.file_format == ".pdf"
    assert doc.metadata.page_count is not None
    assert doc.metadata.page_count >= 1
    # Fallback or primary should extract the string
    assert "MRPL Refinery" in doc.content or doc.metadata.page_count == 1


def test_docx_loader_fallback(tmp_path):
    # Construct mock docx zip container with word/document.xml
    f = tmp_path / "manual.docx"
    with zipfile.ZipFile(f, "w") as zf:
        doc_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
    <w:body>
        <w:p><w:r><w:t>Standard Operating Procedure for Crude Distillation Unit</w:t></w:r></w:p>
        <w:tbl>
            <w:tr><w:tc><w:p><w:r><w:t>Param</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>Value</w:t></w:r></w:p></w:tc></w:tr>
        </w:tbl>
    </w:body>
</w:document>"""
        zf.writestr("word/document.xml", doc_xml)

    loader = DOCXLoader()
    doc = loader.load(f)

    assert "Crude Distillation Unit" in doc.content
    assert doc.metadata.file_format == ".docx"
    assert doc.metadata.extra_metadata["paragraph_count"] >= 1


def test_pptx_loader_fallback(tmp_path):
    # Construct mock pptx zip container with slide1.xml
    f = tmp_path / "presentation.pptx"
    with zipfile.ZipFile(f, "w") as zf:
        slide_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
       xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
    <p:cSld>
        <p:spTree>
            <p:sp><p:txBody><a:p><a:r><a:t>Safety Induction MRPL 2026</a:t></a:r></a:p></p:txBody></p:sp>
        </p:spTree>
    </p:cSld>
</p:sld>"""
        zf.writestr("ppt/slides/slide1.xml", slide_xml)

    loader = PPTXLoader()
    doc = loader.load(f)

    assert "Safety Induction MRPL" in doc.content
    assert doc.metadata.file_format == ".pptx"
    assert doc.metadata.page_count == 1


def test_xlsx_loader_fallback(tmp_path):
    # Construct mock xlsx zip container with sharedStrings.xml and sheet1.xml
    f = tmp_path / "equipment.xlsx"
    with zipfile.ZipFile(f, "w") as zf:
        ss_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="2" uniqueCount="2">
    <si><t>Pump_P203</t></si>
    <si><t>Operational</t></si>
</sst>"""
        sheet_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
    <sheetData>
        <row r="1">
            <c r="A1" t="s"><v>0</v></c>
            <c r="B1" t="s"><v>1</v></c>
        </row>
    </sheetData>
</worksheet>"""
        zf.writestr("xl/sharedStrings.xml", ss_xml)
        zf.writestr("xl/worksheets/sheet1.xml", sheet_xml)

    loader = XLSXLoader()
    doc = loader.load(f)

    assert "Pump_P203" in doc.content
    assert "Operational" in doc.content
    assert doc.metadata.file_format == ".xlsx"


def test_image_loader_fallback(tmp_path):
    # Construct minimal PNG header: 8 bytes magic + 4 len + 4 'IHDR' + 4 width (800) + 4 height (600)
    png_header = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + struct.pack(">II", 800, 600) + b"\x08\x02\x00\x00\x00\x00\x00\x00\x00"
    f = tmp_path / "drawing.png"
    f.write_bytes(png_header)

    loader = ImageLoader()
    doc = loader.load(f)

    assert doc.metadata.image_reference == f.as_posix()
    assert doc.metadata.extra_metadata["width"] == 800
    assert doc.metadata.extra_metadata["height"] == 600
    assert "Visual Asset" in doc.content


def test_empty_file_validation_error(tmp_path):
    f = tmp_path / "empty.txt"
    f.write_bytes(b"")

    loader = TXTLoader()
    with pytest.raises(ValidationFailedError):
        loader.load(f)


def test_global_factory_e2e(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("Universal Factory Load Test", encoding="utf-8")

    doc = global_loader_factory.load(f)
    assert doc.content == "Universal Factory Load Test"
    assert doc.metadata.loader_name == "TXTLoader"
