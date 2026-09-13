from __future__ import annotations
from member3_ocr.evaluation.paths import get_project_root, get_output_dir, get_datasets_dir, get_models_dir
"""Evaluation harness for table extraction against synthetic OCR dataset.

Evaluates:
- Cell exact-match accuracy
- Cell text accuracy (normalized)
- Header accuracy
- Row/column reconstruction accuracy
- Table-level exact match rate
- Precision, Recall, F1 score
- Mean confidence
- Processing latency (ms)

Generates evaluation artifacts under:
C:\\SovereignAI\\member3_ocr\\output\\evaluation\\tables
- summary.json
- per_sample.csv
- failures.csv
- README.md
"""


import csv
import glob
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = get_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from member3_ocr.core.ocr_pipeline import (
    OCRPipeline,
    PaddleOCRBackend,
    PaddleOCRModelConfig,
)

LOGGER = logging.getLogger("table_evaluator")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def normalize_text(text: str) -> str:
    """Normalize text for evaluation comparison."""
    if not text:
        return ""
    # Strip whitespace, lowercase, remove thousands commas/periods in numbers if appropriate
    norm = " ".join(text.strip().lower().split())
    # Also standardize comma vs period in numbers for OCR noise tolerance
    return norm.replace(",", "").replace(".", "")


def load_ground_truth_table(annotation_path: Path) -> dict[str, Any]:
    """Load ground truth table structure and cells from annotation JSON."""
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    doc_id = data["doc_id"]
    degraded = bool(data.get("degraded", False))
    meta = data.get("meta", {})
    elements = data.get("elements", [])

    headers = [e.get("text", "").strip() for e in elements if e.get("type") == "table_header"]
    raw_cells = [e for e in elements if e.get("type") == "table_cell"]

    num_cols = len(headers)
    num_rows = meta.get("rows", len(raw_cells) // max(1, num_cols))

    gt_grid: dict[tuple[int, int], str] = {}
    # Row 0: headers
    for c_idx, h_text in enumerate(headers):
        gt_grid[(0, c_idx)] = h_text

    # Rows 1..num_rows: data cells
    for idx, c_elem in enumerate(raw_cells):
        r_idx = (idx // num_cols) + 1
        c_idx = idx % num_cols
        gt_grid[(r_idx, c_idx)] = c_elem.get("text", "").strip()

    return {
        "doc_id": doc_id,
        "degraded": degraded,
        "num_rows": num_rows,
        "num_cols": num_cols,
        "headers": headers,
        "gt_grid": gt_grid,
        "total_cells": len(gt_grid),
    }


def evaluate_all_tables(
    dataset_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run table extraction evaluation across all synthetic table samples."""
    base_dir = dataset_dir or Path(r"C:\SovereignAI\datasets\ocr\synthetic_ocr_dataset\synthetic_tables")
    out_dir = output_dir or Path(r"C:\SovereignAI\member3_ocr\output\evaluation\tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    det_dir = Path(r"C:\SovereignAI\member3_ocr\models\paddleocr\PP-OCRv5_mobile_det_infer")
    rec_dir = Path(r"C:\SovereignAI\member3_ocr\models\paddleocr\PP-OCRv5_mobile_rec_infer")

    config = PaddleOCRModelConfig(detection_model_dir=det_dir, recognition_model_dir=rec_dir)
    backend = PaddleOCRBackend(config)
    pipeline = OCRPipeline(backend, extract_tables=True, extract_key_values=False)

    annot_files = sorted(glob.glob(str(base_dir / "annotations" / "table_*.json")))
    LOGGER.info("Found %d table annotation files", len(annot_files))

    per_sample_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    total_gt_cells = 0
    total_pred_cells = 0
    total_exact_matches = 0
    total_norm_matches = 0
    total_gt_headers = 0
    total_matched_headers = 0
    row_reconstruction_matches = 0
    col_reconstruction_matches = 0
    table_exact_matches = 0
    total_confidences: list[float] = []
    total_latencies_ms: list[float] = []

    clean_exact_matches = 0
    clean_gt_cells = 0
    degraded_exact_matches = 0
    degraded_gt_cells = 0

    for annot_str in annot_files:
        annot_path = Path(annot_str)
        gt = load_ground_truth_table(annot_path)
        doc_id = gt["doc_id"]
        degraded = gt["degraded"]
        gt_grid = gt["gt_grid"]
        gt_rows = gt["num_rows"]
        gt_cols = gt["num_cols"]
        gt_headers = gt["headers"]

        img_candidates = [
            base_dir / "images" / f"{doc_id}.png",
            base_dir / "images" / f"{doc_id}.jpg",
        ]
        img_path = next((p for p in img_candidates if p.exists()), None)
        if not img_path:
            LOGGER.warning("Missing image for %s", doc_id)
            continue

        start_time = time.perf_counter()
        doc_result = pipeline.process_image(img_path)
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        total_latencies_ms.append(latency_ms)

        page = doc_result.pages[0] if doc_result.pages else None
        extracted_tables = page.tables if page else ()

        if not extracted_tables:
            LOGGER.warning("No table extracted for %s", doc_id)
            total_gt_headers += len(gt_headers)
            for (r, c), gtext in gt_grid.items():
                total_gt_cells += 1
                if not degraded:
                    clean_gt_cells += 1
                else:
                    degraded_gt_cells += 1
                failure_rows.append({
                    "doc_id": doc_id,
                    "degraded": degraded,
                    "row_idx": r,
                    "col_idx": c,
                    "failure_type": "missing_table",
                    "ground_truth_text": gtext,
                    "extracted_text": "",
                    "confidence": 0.0,
                })
            per_sample_rows.append({
                "doc_id": doc_id,
                "degraded": degraded,
                "gt_rows": gt_rows,
                "pred_rows": 0,
                "gt_cols": gt_cols,
                "pred_cols": 0,
                "gt_cells": len(gt_grid),
                "pred_cells": 0,
                "exact_matches": 0,
                "norm_matches": 0,
                "cell_exact_accuracy": 0.0,
                "cell_norm_accuracy": 0.0,
                "header_accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "table_exact_match": False,
                "latency_ms": latency_ms,
            })
            continue

        table = extracted_tables[0]
        # Build pred_grid
        pred_grid: dict[tuple[int, int], tuple[str, float]] = {}
        for c in table.cells:
            r_idx = int(c.get("row_idx", 0))
            c_idx = int(c.get("col_idx", 0))
            text = str(c.get("text", "")).strip()
            conf = float(c.get("confidence", 1.0))
            if (r_idx, c_idx) in pred_grid:
                prev_text, prev_conf = pred_grid[(r_idx, c_idx)]
                pred_grid[(r_idx, c_idx)] = (f"{prev_text} {text}".strip(), min(prev_conf, conf))
            else:
                pred_grid[(r_idx, c_idx)] = (text, conf)
            total_confidences.append(conf)

        pred_rows = max((r for r, _ in pred_grid.keys()), default=0)
        pred_cols = max((c for _, c in pred_grid.keys()), default=0) + 1

        # Check row/column structure reconstruction
        row_match = (pred_rows == gt_rows)
        col_match = (pred_cols == gt_cols)
        if row_match:
            row_reconstruction_matches += 1
        if col_match:
            col_reconstruction_matches += 1

        sample_exact_matches = 0
        sample_norm_matches = 0
        sample_header_matches = 0

        # Evaluate headers (row 0)
        total_gt_headers += len(gt_headers)
        for c_idx, h_text in enumerate(gt_headers):
            if (0, c_idx) in pred_grid:
                p_text, _ = pred_grid[(0, c_idx)]
                if normalize_text(p_text) == normalize_text(h_text):
                    sample_header_matches += 1
                    total_matched_headers += 1

        # Evaluate all cells
        for (r, c), gtext in gt_grid.items():
            total_gt_cells += 1
            if not degraded:
                clean_gt_cells += 1
            else:
                degraded_gt_cells += 1

            if (r, c) in pred_grid:
                p_text, p_conf = pred_grid[(r, c)]
                if p_text == gtext:
                    sample_exact_matches += 1
                    total_exact_matches += 1
                    sample_norm_matches += 1
                    total_norm_matches += 1
                    if not degraded:
                        clean_exact_matches += 1
                    else:
                        degraded_exact_matches += 1
                elif normalize_text(p_text) == normalize_text(gtext):
                    sample_norm_matches += 1
                    total_norm_matches += 1
                    if not degraded:
                        clean_exact_matches += 1
                    else:
                        degraded_exact_matches += 1
                else:
                    failure_rows.append({
                        "doc_id": doc_id,
                        "degraded": degraded,
                        "row_idx": r,
                        "col_idx": c,
                        "failure_type": "text_mismatch",
                        "ground_truth_text": gtext,
                        "extracted_text": p_text,
                        "confidence": p_conf,
                    })
            else:
                failure_rows.append({
                    "doc_id": doc_id,
                    "degraded": degraded,
                    "row_idx": r,
                    "col_idx": c,
                    "failure_type": "missing_cell",
                    "ground_truth_text": gtext,
                    "extracted_text": "",
                    "confidence": 0.0,
                })

        total_pred_cells += len(pred_grid)

        # Table level exact match
        is_table_exact = (sample_exact_matches == len(gt_grid))
        if is_table_exact:
            table_exact_matches += 1

        sample_tp = sample_norm_matches
        sample_fp = len(pred_grid) - sample_tp
        sample_fn = len(gt_grid) - sample_tp
        sample_prec = sample_tp / max(1, sample_tp + sample_fp)
        sample_rec = sample_tp / max(1, sample_tp + sample_fn)
        sample_f1 = (2 * sample_prec * sample_rec) / max(1e-6, sample_prec + sample_rec)

        per_sample_rows.append({
            "doc_id": doc_id,
            "degraded": degraded,
            "gt_rows": gt_rows,
            "pred_rows": pred_rows,
            "gt_cols": gt_cols,
            "pred_cols": pred_cols,
            "gt_cells": len(gt_grid),
            "pred_cells": len(pred_grid),
            "exact_matches": sample_exact_matches,
            "norm_matches": sample_norm_matches,
            "cell_exact_accuracy": round(sample_exact_matches / max(1, len(gt_grid)), 4),
            "cell_norm_accuracy": round(sample_norm_matches / max(1, len(gt_grid)), 4),
            "header_accuracy": round(sample_header_matches / max(1, len(gt_headers)), 4),
            "precision": round(sample_prec, 4),
            "recall": round(sample_rec, 4),
            "f1_score": round(sample_f1, 4),
            "table_exact_match": is_table_exact,
            "latency_ms": latency_ms,
        })

    # Summary calculations
    cell_em_acc = total_exact_matches / max(1, total_gt_cells)
    cell_norm_acc = total_norm_matches / max(1, total_gt_cells)
    hdr_acc = total_matched_headers / max(1, total_gt_headers)
    row_recon_acc = row_reconstruction_matches / max(1, len(per_sample_rows))
    col_recon_acc = col_reconstruction_matches / max(1, len(per_sample_rows))
    prec = total_norm_matches / max(1, total_pred_cells)
    rec = total_norm_matches / max(1, total_gt_cells)
    f1 = (2 * prec * rec) / max(1e-6, prec + rec)
    mean_conf = sum(total_confidences) / max(1, len(total_confidences))
    mean_lat = sum(total_latencies_ms) / max(1, len(total_latencies_ms))

    clean_acc = clean_exact_matches / max(1, clean_gt_cells)
    deg_acc = degraded_exact_matches / max(1, degraded_gt_cells)

    summary_data = {
        "dataset": "synthetic_tables",
        "total_samples": len(per_sample_rows),
        "total_gt_cells": total_gt_cells,
        "total_extracted_cells": total_pred_cells,
        "metrics": {
            "cell_exact_match_accuracy": round(cell_em_acc, 4),
            "cell_text_accuracy": round(cell_norm_acc, 4),
            "header_accuracy": round(hdr_acc, 4),
            "row_reconstruction_accuracy": round(row_recon_acc, 4),
            "col_reconstruction_accuracy": round(col_recon_acc, 4),
            "table_exact_match_rate": round(table_exact_matches / max(1, len(per_sample_rows)), 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "mean_confidence": round(mean_conf, 4),
            "mean_latency_ms": round(mean_lat, 2),
        },
        "breakdown": {
            "clean_samples": sum(1 for r in per_sample_rows if not r["degraded"]),
            "clean_cell_accuracy": round(clean_acc, 4),
            "degraded_samples": sum(1 for r in per_sample_rows if r["degraded"]),
            "degraded_cell_accuracy": round(deg_acc, 4),
        },
    }

    # Write summary.json
    (out_dir / "summary.json").write_text(json.dumps(summary_data, indent=2), encoding="utf-8")

    # Write per_sample.csv
    if per_sample_rows:
        keys = list(per_sample_rows[0].keys())
        with open(out_dir / "per_sample.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(per_sample_rows)

    # Write failures.csv
    fail_keys = ["doc_id", "degraded", "row_idx", "col_idx", "failure_type", "ground_truth_text", "extracted_text", "confidence"]
    with open(out_dir / "failures.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fail_keys)
        writer.writeheader()
        writer.writerows(failure_rows)

    # Write README.md
    readme_content = f"""# Table Extraction Evaluation Report

**Evaluation Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}
**Backend:** PaddleOCR (PP-OCRv5 local mobile detection + recognition)
**Target Dataset:** `datasets/ocr/synthetic_ocr_dataset/synthetic_tables`
**Total Samples:** {len(per_sample_rows)} (Clean: {summary_data['breakdown']['clean_samples']}, Degraded: {summary_data['breakdown']['degraded_samples']})

## Summary Metrics

| Metric | Score | Description |
| :--- | :--- | :--- |
| **Cell Text Accuracy** | **{summary_data['metrics']['cell_text_accuracy'] * 100:.1f}%** | Normalized cell string matching accuracy across all grid cells |
| **Cell Exact-Match Accuracy** | **{summary_data['metrics']['cell_exact_match_accuracy'] * 100:.1f}%** | Exact character-for-character match rate |
| **Header Accuracy** | **{summary_data['metrics']['header_accuracy'] * 100:.1f}%** | Accuracy of reconstructed column headers |
| **Row Reconstruction Accuracy** | **{summary_data['metrics']['row_reconstruction_accuracy'] * 100:.1f}%** | Fraction of tables where total row count perfectly matches ground truth |
| **Col Reconstruction Accuracy** | **{summary_data['metrics']['col_reconstruction_accuracy'] * 100:.1f}%** | Fraction of tables where column count perfectly matches ground truth |
| **Precision** | **{summary_data['metrics']['precision'] * 100:.1f}%** | TP / (TP + FP) across all grid cell extractions |
| **Recall** | **{summary_data['metrics']['recall'] * 100:.1f}%** | TP / (TP + FN) across all ground-truth cells |
| **F1 Score** | **{summary_data['metrics']['f1_score'] * 100:.1f}%** | Harmonic mean of precision and recall |
| **Mean Cell Confidence** | **{summary_data['metrics']['mean_confidence']:.4f}** | Average OCR confidence of table cells |
| **Mean Latency** | **{summary_data['metrics']['mean_latency_ms']:.1f} ms** | Average processing latency per page |

## Clean vs. Degraded Performance

- **Clean Scans ({summary_data['breakdown']['clean_samples']} docs):** **{clean_acc * 100:.1f}%** Cell Accuracy
- **Degraded Scans ({summary_data['breakdown']['degraded_samples']} docs):** **{deg_acc * 100:.1f}%** Cell Accuracy

## Generated Evaluation Artifacts

1. `summary.json`: Aggregate metrics, counts, and category breakdown.
2. `per_sample.csv`: Sample-by-sample scores, row/col counts, accuracy, precision, recall, and latency.
3. `failures.csv`: Cell-by-cell discrepancy analysis categorizing mismatches and OCR noise.
"""
    (out_dir / "README.md").write_text(readme_content, encoding="utf-8")

    LOGGER.info("Table evaluation complete! Summary saved to %s", out_dir / "summary.json")
    return summary_data


if __name__ == "__main__":
    summary = evaluate_all_tables()
    print(json.dumps(summary, indent=2))
