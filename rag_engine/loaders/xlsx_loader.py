"""
XLSX Loader — Microsoft Excel Spreadsheet Loading with Dual-Driver Resiliency.

Sovereign On-Premise Agentic AI Workbench (SIH26117 / MRPL)
Primary driver: openpyxl (if available)
Fallback driver: Standard library zipfile + xml.etree.ElementTree (air-gapped)
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any
import xml.etree.ElementTree as ET
import zipfile

from rag_engine.loaders.base_loader import BaseLoader
from rag_engine.loaders.loader_registry import register_loader

try:
    import openpyxl
    _OPENPYXL_AVAILABLE = True
except ImportError:
    _OPENPYXL_AVAILABLE = False


@register_loader(".xlsx", ".xls")
class XLSXLoader(BaseLoader):
    """Universal loader for Microsoft Excel spreadsheets."""

    def supported_formats(self) -> list[str]:
        return [".xlsx", ".xls"]

    def is_primary_driver_available(self) -> bool:
        return _OPENPYXL_AVAILABLE

    def dependencies(self) -> dict[str, bool]:
        return {"openpyxl": _OPENPYXL_AVAILABLE}

    def _load_primary(self, file_path: Path) -> tuple[str, dict[str, Any]]:
        """Load XLSX using openpyxl."""
        wb = openpyxl.load_workbook(str(file_path), data_only=True, read_only=True)
        sheets_data: list[str] = []
        total_rows = 0

        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            sheet_lines: list[str] = [f"=== Sheet: {sheet_name} ==="]
            row_count = 0
            for row in ws.iter_rows(values_only=True):
                # Filter non-empty cells
                cells = [str(c).strip() for c in row if c is not None and str(c).strip()]
                if cells:
                    sheet_lines.append(" | ".join(cells))
                    row_count += 1

            if len(sheet_lines) > 1:
                sheets_data.append("\n".join(sheet_lines))
                total_rows += row_count

        wb.close()
        full_content = "\n\n".join(sheets_data)
        meta: dict[str, Any] = {
            "sheet_count": len(wb.sheetnames),
            "sheet_names": wb.sheetnames,
            "total_rows": total_rows,
            "driver": "openpyxl",
        }
        return full_content, meta

    def _load_fallback(self, file_path: Path) -> tuple[str, dict[str, Any]]:
        """Air-gapped fallback: parse sharedStrings.xml and sheet*.xml from ZIP archive."""
        sheets_data: list[str] = []
        sheet_names: list[str] = []
        total_rows = 0
        meta: dict[str, Any] = {"driver": "stdlib_xlsx_fallback"}

        with zipfile.ZipFile(file_path, "r") as zf:
            namelist = zf.namelist()

            # 1. Parse shared strings table if present
            shared_strings: list[str] = []
            if "xl/sharedStrings.xml" in namelist:
                ss_data = zf.read("xl/sharedStrings.xml")
                ss_root = ET.fromstring(ss_data)
                ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
                for si in ss_root.findall(".//s:si", ns):
                    # Text can be direct in <t> or inside rich text runs <r><t>
                    texts = [t.text for t in si.findall(".//s:t", ns) if t.text]
                    shared_strings.append("".join(texts))

            # 2. Find sheets in xl/worksheets/
            sheet_files = [f for f in namelist if re.match(r"xl/worksheets/sheet\d+\.xml", f)]
            sheet_files.sort(key=lambda name: int(re.search(r"\d+", name).group()))

            ns_sheet = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

            for idx, sheet_file in enumerate(sheet_files):
                s_name = f"Sheet{idx + 1}"
                sheet_names.append(s_name)
                xml_data = zf.read(sheet_file)
                root = ET.fromstring(xml_data)

                sheet_lines: list[str] = [f"=== Sheet: {s_name} ==="]
                row_count = 0

                for row_elem in root.findall(".//s:row", ns_sheet):
                    row_cells: list[str] = []
                    for c_elem in row_elem.findall("s:c", ns_sheet):
                        cell_type = c_elem.get("t")
                        val_elem = c_elem.find("s:v", ns_sheet)
                        if val_elem is not None and val_elem.text is not None:
                            val = val_elem.text
                            if cell_type == "s":
                                # Lookup in shared strings
                                try:
                                    idx_s = int(val)
                                    val = shared_strings[idx_s] if idx_s < len(shared_strings) else val
                                except (ValueError, IndexError):
                                    pass
                            val_clean = str(val).strip()
                            if val_clean:
                                row_cells.append(val_clean)

                    if row_cells:
                        sheet_lines.append(" | ".join(row_cells))
                        row_count += 1

                if len(sheet_lines) > 1:
                    sheets_data.append("\n".join(sheet_lines))
                    total_rows += row_count

        full_content = "\n\n".join(sheets_data)
        meta["sheet_count"] = len(sheet_names)
        meta["sheet_names"] = sheet_names
        meta["total_rows"] = total_rows

        return full_content, meta
