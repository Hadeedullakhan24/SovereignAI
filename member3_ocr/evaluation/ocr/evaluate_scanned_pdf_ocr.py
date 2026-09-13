from __future__ import annotations
from member3_ocr.evaluation.paths import get_project_root, get_output_dir, get_datasets_dir, get_models_dir
"""Targeted batch evaluation harness for Scanned PDF OCR in Member 3.

Evaluates:
- Scanned PDF rasterization via pdf_rendering.py (pypdfium2)
- Multi-page and single-page local PP-OCRv5 recognition
- Accuracy metrics (CER, WER) against ground-truth reference texts
- Performance metrics (render time, OCR time, total throughput)
- Outputs results to member3_ocr/output/evaluation/pdf_ocr_accuracy/summary.json
"""


import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from member3_ocr.evaluation.ocr.ocr_evaluator import compute_cer, compute_wer
from member3_ocr.core.ocr_pipeline import (
    OCRPipeline,
    PaddleOCRBackend,
    PaddleOCRModelConfig,
)
from member3_ocr.core.pdf_rendering import get_pdf_page_count

LOGGER = logging.getLogger(__name__)


def run_scanned_pdf_evaluation(
    output_dir: Path | str = r"c:\SovereignAI\member3_ocr\output\evaluation\pdf_ocr_accuracy",
    limit_samples: int = 6,
) -> dict[str, Any]:
    """Execute batch evaluation on scanned PDFs with ground truth."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    det_dir = Path(r"C:\SovereignAI\member3_ocr\models\paddleocr\PP-OCRv5_mobile_det_infer")
    rec_dir = Path(r"C:\SovereignAI\member3_ocr\models\paddleocr\PP-OCRv5_mobile_rec_infer")

    config = PaddleOCRModelConfig(
        detection_model_dir=det_dir,
        recognition_model_dir=rec_dir,
        device="cpu",
    )
    backend = PaddleOCRBackend(config)
    pipeline = OCRPipeline(backend)

    pdf_base = Path(r"c:\SovereignAI\datasets\ocr\synthetic_ocr_dataset\pdf")
    gt_base = Path(r"c:\SovereignAI\datasets\ocr\synthetic_ocr_dataset\text")

    # Select representative samples: forms, manuals, tables
    target_stems = [
        "form_001",
        "form_002",
        "manual_001",
        "manual_002",
        "table_001",
        "table_002",
    ][:limit_samples]

    results: list[dict[str, Any]] = []
    cers: list[float] = []
    wers: list[float] = []
    total_pages = 0
    total_render_ms = 0.0
    total_ocr_ms = 0.0

    print(f"Starting Scanned PDF Batch Evaluation on {len(target_stems)} PDFs...")

    for stem in target_stems:
        pdf_file = pdf_base / f"{stem}.pdf"
        gt_file = gt_base / f"{stem}.txt"

        if not pdf_file.exists() or not gt_file.exists():
            print(f"Skipping {stem}: file not found.")
            continue

        with open(gt_file, "r", encoding="utf-8") as f:
            gt_text = f.read().strip()

        start_t = time.perf_counter()
        doc_res = pipeline.process_pdf(pdf_file, dpi=150)
        total_time_ms = (time.perf_counter() - start_t) * 1000.0

        # Combine text across pages
        hyp_text = "\n".join(p.text for p in doc_res.pages).strip()
        num_pages = len(doc_res.pages)
        total_pages += num_pages

        cer = compute_cer(gt_text, hyp_text)
        wer = compute_wer(gt_text, hyp_text)
        cers.append(cer)
        wers.append(wer)

        render_ms = doc_res.provenance.get("total_render_time_ms", 0.0)
        ocr_ms = doc_res.provenance.get("total_ocr_time_ms", total_time_ms - render_ms)
        total_render_ms += render_ms
        total_ocr_ms += ocr_ms

        rec = {
            "pdf_name": pdf_file.name,
            "pages": num_pages,
            "characters_gt": len(gt_text),
            "characters_ocr": len(hyp_text),
            "cer": round(cer, 4),
            "wer": round(wer, 4),
            "render_time_ms": round(render_ms, 2),
            "ocr_time_ms": round(ocr_ms, 2),
            "total_time_ms": round(total_time_ms, 2),
        }
        results.append(rec)
        print(f"  [{pdf_file.name}] Pages: {num_pages} | CER: {cer:.4f} | WER: {wer:.4f} | Time: {total_time_ms:.1f}ms")

    # Real industrial safety doc verification (OISD-STD-105 first page)
    oisd_path = Path(r"c:\SovereignAI\datasets\safety_docs\OISD-STD-105.pdf")
    oisd_res: dict[str, Any] | None = None
    if oisd_path.exists():
        import pypdfium2 as pdfium
        import cv2
        start_t = time.perf_counter()
        with pdfium.PdfDocument(oisd_path) as doc:
            total_doc_pages = len(doc)
            page0 = doc[0]
            bitmap = page0.render(scale=150 / 72.0)
            pil_img = bitmap.to_pil()
            page1_arr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        doc_res = pipeline.process_image(page1_arr, document_id="OISD_STD_105_p1")
        oisd_ms = (time.perf_counter() - start_t) * 1000.0
        oisd_text = doc_res.pages[0].text if doc_res.pages else ""
        oisd_res = {
            "pdf_name": oisd_path.name,
            "pages_processed": 1,
            "total_document_pages": total_doc_pages,
            "ocr_blocks": len(doc_res.pages[0].blocks) if doc_res.pages else 0,
            "characters_extracted": len(oisd_text),
            "total_time_ms": round(oisd_ms, 2),
            "contains_expected_keywords": any(kw in oisd_text.upper() for kw in ["OISD", "STANDARD", "SAFETY", "PETROLEUM"]),
        }
        print(f"  [{oisd_path.name} (page 1)] Blocks: {oisd_res['ocr_blocks']} | Chars: {oisd_res['characters_extracted']} | Time: {oisd_ms:.1f}ms")

    summary = {
        "evaluation_mode": "scanned_pdf_batch_accuracy",
        "model": "PP-OCRv5_mobile_cpu",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": {
            "total_synthetic_pdfs": len(results),
            "total_pages_evaluated": total_pages,
            "mean_cer": round(float(np.mean(cers)), 4) if cers else 0.0,
            "mean_wer": round(float(np.mean(wers)), 4) if wers else 0.0,
            "mean_render_ms_per_doc": round(total_render_ms / max(1, len(results)), 2),
            "mean_ocr_ms_per_doc": round(total_ocr_ms / max(1, len(results)), 2),
            "mean_total_ms_per_doc": round((total_render_ms + total_ocr_ms) / max(1, len(results)), 2),
        },
        "per_sample_results": results,
        "real_safety_doc_validation": oisd_res,
    }

    sum_json = out_path / "summary.json"
    with open(sum_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    readme_md = out_path / "README.md"
    with open(readme_md, "w", encoding="utf-8") as f:
        f.write("# Scanned PDF Batch OCR Accuracy Evaluation Report\n\n")
        f.write(f"- **Model**: PP-OCRv5 Mobile (local CPU)\n")
        f.write(f"- **PDF Renderer**: pypdfium2 (local, offline)\n")
        f.write(f"- **Evaluated**: {len(results)} multi-page & single-page scanned PDFs + real industrial standard\n")
        f.write(f"- **Mean Character Error Rate (CER)**: {summary['metrics']['mean_cer']:.4f}\n")
        f.write(f"- **Mean Word Error Rate (WER)**: {summary['metrics']['mean_wer']:.4f}\n")
        f.write(f"- **Mean Processing Time per Document**: {summary['metrics']['mean_total_ms_per_doc']:.2f} ms\n\n")
        f.write("## Per-Sample Results\n\n")
        f.write("| PDF Document | Pages | Chars (GT / OCR) | CER | WER | Render (ms) | OCR (ms) | Total (ms) |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for r in results:
            f.write(f"| {r['pdf_name']} | {r['pages']} | {r['characters_gt']} / {r['characters_ocr']} | {r['cer']:.4f} | {r['wer']:.4f} | {r['render_time_ms']} | {r['ocr_time_ms']} | {r['total_time_ms']} |\n")
        if oisd_res:
            f.write(f"\n## Real Safety Document Validation ({oisd_res['pdf_name']})\n\n")
            f.write(f"- Pages processed: {oisd_res['pages_processed']} of {oisd_res['total_document_pages']}\n")
            f.write(f"- Blocks extracted: {oisd_res['ocr_blocks']} ({oisd_res['characters_extracted']} chars)\n")
            f.write(f"- Expected industry keywords verified: {oisd_res['contains_expected_keywords']}\n")
            f.write(f"- Latency: {oisd_res['total_time_ms']} ms\n")

    print(f"\nCompleted Scanned PDF Batch Evaluation. Summary written to {sum_json}")
    print(f"Mean CER: {summary['metrics']['mean_cer']:.4f} | Mean WER: {summary['metrics']['mean_wer']:.4f}")
    return summary


if __name__ == "__main__":
    run_scanned_pdf_evaluation()
