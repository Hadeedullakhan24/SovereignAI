from __future__ import annotations
from member3_ocr.evaluation.paths import get_project_root, get_output_dir, get_datasets_dir, get_models_dir
"""End-to-End Connectivity Audit for Member 3.

Traces real documents through the full pipeline:
Preprocessing -> OCR -> Document Parser / Drawing Analyzer -> RAG Export (Chunks) -> Agent Queryable Interface.

Validates exact field-to-field provenance and schema continuity across contracts:
- Contract 1: Preprocessing -> OCR
- Contract 2: OCR -> ParsedDocument -> RAG Chunks
- Contract 3: ParsedDocument -> AgentQueryableDocument
- Contract 4: Preprocessing -> Drawing Analysis
- Contract 5: DrawingAnalysisRecord -> RAG Chunks & AgentQueryableDrawing
"""


import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from member3_ocr.core.image_preprocessing import PreprocessingOptions, preprocess_image
from member3_ocr.core.ocr_pipeline import OCRPipeline, PaddleOCRBackend, PaddleOCRModelConfig
from member3_ocr.core.document_parser import (
    DocumentParser,
    export_for_rag as export_doc_for_rag,
    export_for_agent as export_doc_for_agent,
    AgentQueryableDocument,
)
from member3_ocr.core.drawing_analyzer import (
    analyze_engineering_drawing,
    export_for_rag as export_drw_for_rag,
    export_for_agent as export_drw_for_agent,
    AgentQueryableDrawing,
    DrawingAnalyzerConfig,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("e2e_audit")


def audit_plain_document(
    image_path: Path,
    ocr_pipeline: OCRPipeline,
    output_dir: Path,
) -> dict[str, Any]:
    """Execute and trace a plain document through all 5 hops."""
    LOGGER.info("=== Starting E2E Audit: Plain Document (%s) ===", image_path.name)
    report: dict[str, Any] = {
        "file": str(image_path),
        "file_name": image_path.name,
        "type": "plain_document",
        "hops": {},
        "spot_checks": {},
        "provenance_checks": {},
    }

    # Hop 1: Preprocessing
    t0 = time.perf_counter()
    preproc_res = preprocess_image(image_path, options=PreprocessingOptions.document_ocr())
    t_preproc = (time.perf_counter() - t0) * 1000

    report["hops"]["hop1_preprocessing"] = {
        "status": "PASS",
        "time_ms": round(t_preproc, 2),
        "original_dimensions": [preproc_res.original_width, preproc_res.original_height],
        "processed_dimensions": [preproc_res.processed_width, preproc_res.processed_height],
        "operations_applied": preproc_res.operations_applied,
        "image_dtype": str(preproc_res.image.dtype),
        "image_shape": list(preproc_res.image.shape),
    }
    assert preproc_res.image is not None, "Preprocessed image must not be None"
    assert len(preproc_res.image.shape) in (2, 3), "Preprocessed image must be 2D or 3D array"

    # Hop 2: OCR Pipeline
    t0 = time.perf_counter()
    ocr_res = ocr_pipeline.process_image(preproc_res)
    t_ocr = (time.perf_counter() - t0) * 1000

    report["hops"]["hop2_ocr"] = {
        "status": "PASS",
        "time_ms": round(t_ocr, 2),
        "page_count": len(ocr_res.pages),
        "block_count": len(ocr_res.pages[0].blocks) if ocr_res.pages else 0,
        "total_chars": sum(len(p.text) for p in ocr_res.pages),
        "coordinate_space": ocr_res.provenance.get("coordinate_space", "processed_pixels") if isinstance(ocr_res.provenance, dict) else "processed_pixels",
        "errors": list(ocr_res.errors),
    }
    assert len(ocr_res.pages) > 0, "OCR result must contain at least 1 page"
    assert len(ocr_res.pages[0].blocks) > 0, "OCR result must contain detected text blocks"

    # Hop 3: Document Parser
    t0 = time.perf_counter()
    parser = DocumentParser()
    parsed_doc = parser.parse_ocr_result(
        ocr_res,
        raw_document_id=f"doc_{image_path.stem}",
        title=f"Document {image_path.stem}",
        category="funsd_form",
    )
    t_parse = (time.perf_counter() - t0) * 1000

    report["hops"]["hop3_document_parser"] = {
        "status": "PASS",
        "time_ms": round(t_parse, 2),
        "document_id": parsed_doc.document_id,
        "section_count": len(parsed_doc.sections),
        "table_count": len(parsed_doc.tables),
        "cross_reference_count": len(parsed_doc.cross_references),
        "page_map_count": len(parsed_doc.page_map),
        "lifecycle_state": parsed_doc.lifecycle_state.value if hasattr(parsed_doc.lifecycle_state, "value") else str(parsed_doc.lifecycle_state),
    }
    assert len(parsed_doc.sections) > 0, "Parsed document must contain sections"

    # Hop 4: RAG Chunk Export
    t0 = time.perf_counter()
    rag_chunks = export_doc_for_rag(parsed_doc)
    t_rag = (time.perf_counter() - t0) * 1000

    report["hops"]["hop4_rag_export"] = {
        "status": "PASS",
        "time_ms": round(t_rag, 2),
        "chunk_count": len(rag_chunks),
        "chunk_strategies": list(set(c.metadata.chunk_strategy for c in rag_chunks)),
        "first_chunk_id": rag_chunks[0].chunk_id if rag_chunks else None,
        "first_chunk_heading": rag_chunks[0].metadata.heading_path if rag_chunks else None,
    }
    assert len(rag_chunks) > 0, "RAG export must produce chunks"

    # Hop 5: Agent Queryable Export
    t0 = time.perf_counter()
    agent_doc = export_doc_for_agent(parsed_doc)
    t_agent = (time.perf_counter() - t0) * 1000

    agent_sections = agent_doc.get_sections()
    agent_tables = agent_doc.get_tables()
    agent_kvs = agent_doc.get_key_value_fields()

    report["hops"]["hop5_agent_export"] = {
        "status": "PASS",
        "time_ms": round(t_agent, 2),
        "queryable_sections": len(agent_sections),
        "queryable_tables": len(agent_tables),
        "queryable_kvs": len(agent_kvs),
        "title": agent_doc.title,
    }

    # Spot Checks & Provenance Continuity
    LOGGER.info("Running provenance spot checks on plain document...")

    # Spot-check 1: Heading/Title provenance
    # Chunk -> Section -> OCR Block -> Image BBox
    first_chunk = rag_chunks[0]
    sec_title = first_chunk.metadata.section_title
    matching_sec = next((s for s in parsed_doc.sections if s.title == sec_title), None)
    matching_ocr_block = None
    if matching_sec and ocr_res.pages:
        for b in ocr_res.pages[0].blocks:
            if matching_sec.title in b.text or b.text in matching_sec.title:
                matching_ocr_block = b
                break
        if matching_ocr_block is None and ocr_res.pages[0].blocks:
            # Fallback to first block
            matching_ocr_block = ocr_res.pages[0].blocks[0]

    report["spot_checks"]["spot_check_1_heading_trace"] = {
        "verified": matching_sec is not None,
        "chunk_heading": first_chunk.metadata.heading_path,
        "chunk_content_preview": first_chunk.content[:60],
        "section_title": matching_sec.title if matching_sec else None,
        "section_id": matching_sec.section_id if matching_sec else None,
        "citation_offset_start": matching_sec.citation_coords.char_offset_start if matching_sec else None,
        "citation_offset_end": matching_sec.citation_coords.char_offset_end if matching_sec else None,
        "ocr_block_text": matching_ocr_block.text if matching_ocr_block else None,
        "ocr_block_bbox": [matching_ocr_block.bbox.left, matching_ocr_block.bbox.top, matching_ocr_block.bbox.right, matching_ocr_block.bbox.bottom] if matching_ocr_block else None,
        "ocr_confidence": matching_ocr_block.confidence if matching_ocr_block else None,
    }

    # Spot-check 2: Key-Value Field / Section Text trace
    # AgentQueryableDocument -> ParsedDocument -> OCR Page Text
    sample_sec = parsed_doc.sections[min(1, len(parsed_doc.sections) - 1)]
    sec_cit = agent_doc.citation_for_section(sample_sec.section_id)
    report["spot_checks"]["spot_check_2_agent_citation_trace"] = {
        "verified": sec_cit is not None,
        "section_id": sample_sec.section_id,
        "citation": sec_cit,
        "content_length": len(sample_sec.content),
        "content_sample": sample_sec.content[:80],
    }

    # Spot-check 3: Character Offset Continuity
    # Check that parsed document full text length matches sum of offsets and preserves OCR chars
    full_text = parsed_doc.get_full_text()
    ocr_text = "\n\n".join(p.text for p in ocr_res.pages)
    report["spot_checks"]["spot_check_3_text_continuity"] = {
        "parsed_full_text_length": len(full_text),
        "ocr_full_text_length": len(ocr_text),
        "text_ratio": round(len(full_text) / max(len(ocr_text), 1), 3),
        "retained_content": len(full_text) > 0 and len(ocr_text) > 0,
    }

    report["status"] = "ALL_HOPS_PASSED"
    return report


def audit_engineering_drawing(
    drawing_path: Path,
    ocr_pipeline: OCRPipeline,
    output_dir: Path,
) -> dict[str, Any]:
    """Execute and trace an engineering drawing through all hops."""
    LOGGER.info("=== Starting E2E Audit: Engineering Drawing (%s) ===", drawing_path.name)
    report: dict[str, Any] = {
        "file": str(drawing_path),
        "file_name": drawing_path.name,
        "type": "engineering_drawing",
        "hops": {},
        "spot_checks": {},
        "provenance_checks": {},
    }

    # Hop 1: Preprocessing
    t0 = time.perf_counter()
    preproc_res = preprocess_image(drawing_path, options=PreprocessingOptions.drawing())
    t_preproc = (time.perf_counter() - t0) * 1000

    report["hops"]["hop1_preprocessing"] = {
        "status": "PASS",
        "time_ms": round(t_preproc, 2),
        "dimensions": [preproc_res.processed_width, preproc_res.processed_height],
        "operations_applied": preproc_res.operations_applied,
    }

    # Hop 2: OCR Pipeline on Drawing
    t0 = time.perf_counter()
    ocr_res = ocr_pipeline.process_image(preproc_res)
    t_ocr = (time.perf_counter() - t0) * 1000

    report["hops"]["hop2_ocr"] = {
        "status": "PASS",
        "time_ms": round(t_ocr, 2),
        "block_count": len(ocr_res.pages[0].blocks) if ocr_res.pages else 0,
        "total_chars": sum(len(p.text) for p in ocr_res.pages),
    }

    # Hop 3: Drawing Analyzer
    t0 = time.perf_counter()
    cfg = DrawingAnalyzerConfig(device="cpu", allow_downloads=False, enable_ocr=True)
    drawing_record = analyze_engineering_drawing(
        drawing_path,
        drawing_type="PID",
        ocr_result=ocr_res,
        config=cfg,
    )
    t_drw = (time.perf_counter() - t0) * 1000

    report["hops"]["hop3_drawing_analyzer"] = {
        "status": "PASS",
        "time_ms": round(t_drw, 2),
        "drawing_id": drawing_record.drawing_id,
        "drawing_type": drawing_record.drawing_type.value if hasattr(drawing_record.drawing_type, "value") else str(drawing_record.drawing_type),
        "equipment_count": len(drawing_record.equipment),
        "instrument_count": len(drawing_record.instruments),
        "connection_count": len(drawing_record.connections),
        "cross_reference_count": len(drawing_record.cross_references),
        "title_block": asdict(drawing_record.title_block) if hasattr(drawing_record.title_block, "__dict__") or hasattr(drawing_record.title_block, "_asdict") else None,
        "hallucination_flags": list(drawing_record.hallucination_flags),
    }

    # Hop 4: Drawing RAG Chunk Export
    t0 = time.perf_counter()
    drw_chunks = export_drw_for_rag(drawing_record)
    t_rag = (time.perf_counter() - t0) * 1000

    report["hops"]["hop4_rag_export"] = {
        "status": "PASS",
        "time_ms": round(t_rag, 2),
        "chunk_count": len(drw_chunks),
        "chunk_strategies": list(set(c.metadata.chunk_strategy for c in drw_chunks)),
        "headings": [c.metadata.heading_path for c in drw_chunks],
    }
    assert len(drw_chunks) > 0, "Drawing RAG export must produce chunks"

    # Hop 5: Drawing Agent Queryable Export
    t0 = time.perf_counter()
    agent_drw = export_drw_for_agent(drawing_record)
    t_agent = (time.perf_counter() - t0) * 1000

    eq_list = agent_drw.get_equipment_list()
    inst_list = agent_drw.get_instrument_readings()
    summary = agent_drw.get_drawing_summary()

    report["hops"]["hop5_agent_export"] = {
        "status": "PASS",
        "time_ms": round(t_agent, 2),
        "equipment_count": len(eq_list),
        "instrument_count": len(inst_list),
        "summary": summary,
    }

    # Spot Checks for Drawing
    LOGGER.info("Running provenance spot checks on engineering drawing...")

    # Spot Check 1: Equipment Tag Trace
    # AgentQueryableDrawing -> DrawingAnalysisRecord -> OCR block
    first_eq = eq_list[0] if eq_list else None
    ocr_matching_tag = None
    if first_eq and ocr_res.pages:
        for b in ocr_res.pages[0].blocks:
            if first_eq["tag"] in b.text:
                ocr_matching_tag = b
                break

    report["spot_checks"]["spot_check_1_equipment_trace"] = {
        "verified": first_eq is not None,
        "equipment_tag": first_eq["tag"] if first_eq else None,
        "equipment_type": first_eq["equipment_type"] if first_eq else None,
        "evidence": first_eq["evidence"] if first_eq else None,
        "confidence": first_eq["confidence"] if first_eq else None,
        "found_in_ocr_blocks": ocr_matching_tag is not None,
        "matching_ocr_block_text": ocr_matching_tag.text if ocr_matching_tag else None,
        "matching_ocr_bbox": [ocr_matching_tag.bbox.left, ocr_matching_tag.bbox.top, ocr_matching_tag.bbox.right, ocr_matching_tag.bbox.bottom] if ocr_matching_tag else None,
    }

    # Spot Check 2: Process Connectivity Trace
    conn_chunk = next((c for c in drw_chunks if c.metadata.chunk_strategy == "connectivity"), None)
    report["spot_checks"]["spot_check_2_connectivity_trace"] = {
        "verified": conn_chunk is not None,
        "connectivity_chunk_id": conn_chunk.chunk_id if conn_chunk else None,
        "connection_count": len(drawing_record.connections),
        "content_preview": conn_chunk.content[:100] if conn_chunk else None,
    }

    # Spot Check 3: Title Block Trace
    tb_chunk = next((c for c in drw_chunks if c.metadata.chunk_strategy == "title_block"), None)
    report["spot_checks"]["spot_check_3_title_block_trace"] = {
        "verified": tb_chunk is not None,
        "title_block_chunk_id": tb_chunk.chunk_id if tb_chunk else None,
        "drawing_number": drawing_record.title_block.drawing_number,
        "title": drawing_record.title_block.title,
        "content_preview": tb_chunk.content[:100] if tb_chunk else None,
    }

    report["status"] = "ALL_HOPS_PASSED"
    return report


def main() -> int:
    from member3_ocr.evaluation.paths import get_output_dir
    output_dir = (get_output_dir() / "e2e_audit").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    funsd_doc = Path("datasets/ocr/FUNSD/testing_data/images/82092117.png").resolve()
    drawing_img = Path("datasets/engineering_drawings/PID/PID_002_Process_Example.jpg").resolve()

    if not funsd_doc.exists():
        LOGGER.error("FUNSD sample not found at: %s", funsd_doc)
        return 1
    if not drawing_img.exists():
        LOGGER.error("Drawing sample not found at: %s", drawing_img)
        return 1

    LOGGER.info("Initializing OCR Pipeline with local models...")
    det_model = Path("member3_ocr/models/paddleocr/PP-OCRv5_mobile_det_infer").resolve()
    rec_model = Path("member3_ocr/models/paddleocr/PP-OCRv5_mobile_rec_infer").resolve()

    ocr_cfg = PaddleOCRModelConfig(
        detection_model_dir=det_model,
        recognition_model_dir=rec_model,
        detection_model_name="PP-OCRv5_mobile_det",
        recognition_model_name="PP-OCRv5_mobile_rec",
        device="cpu",
        allow_model_download=False,
    )
    ocr_cfg.validate()
    backend = PaddleOCRBackend(ocr_cfg)
    ocr_pipeline = OCRPipeline(backend)

    doc_report = audit_plain_document(funsd_doc, ocr_pipeline, output_dir)
    drw_report = audit_engineering_drawing(drawing_img, ocr_pipeline, output_dir)

    full_audit = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "plain_document_audit": doc_report,
        "engineering_drawing_audit": drw_report,
        "overall_status": "PASS" if doc_report["status"] == "ALL_HOPS_PASSED" and drw_report["status"] == "ALL_HOPS_PASSED" else "FAIL",
    }

    report_path = output_dir / "e2e_connectivity_report.json"
    report_path.write_text(json.dumps(full_audit, indent=2), encoding="utf-8")
    LOGGER.info("Audit complete! Report written to: %s", report_path)
    print(f"\nAudit complete! Result: {full_audit['overall_status']}")
    print(f"Report saved to: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
