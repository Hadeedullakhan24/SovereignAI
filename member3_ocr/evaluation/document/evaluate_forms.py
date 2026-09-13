from __future__ import annotations
from member3_ocr.evaluation.paths import get_project_root, get_output_dir, get_datasets_dir, get_models_dir
"""Evaluation harness for form key-value extraction against synthetic OCR dataset.

Evaluates:
- Key-Value exact-match accuracy
- Key accuracy
- Value accuracy
- Field precision
- Field recall
- Field F1
- Average confidence
- Latency (ms)

Generates evaluation artifacts under:
C:\\SovereignAI\\member3_ocr\\output\\evaluation\\forms
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

# Ensure project root is in sys.path
PROJECT_ROOT = get_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from member3_ocr.core.ocr_pipeline import (
    OCRPipeline,
    PaddleOCRBackend,
    PaddleOCRModelConfig,
)

LOGGER = logging.getLogger("form_evaluator")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def normalize_text(text: str) -> str:
    """Normalize text for evaluation comparison."""
    if not text:
        return ""
    # Strip whitespace, lowercase, normalize internal spacing
    return " ".join(text.strip().lower().split())


def load_ground_truth(annotation_path: Path) -> dict[str, Any]:
    """Extract ground-truth key-value pairs from synthetic form annotation file."""
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    doc_id = data["doc_id"]
    degraded = bool(data.get("degraded", False))
    elements = data.get("elements", [])

    gt_fields: dict[str, str] = {}

    # Extract inline headers/status
    for el in elements:
        t = el.get("text", "").strip()
        etype = el.get("type", "")
        if etype == "header" and "Form No:" in t:
            parts = t.split(":", 1)
            gt_fields[normalize_text(parts[0])] = parts[1].strip()
        elif etype == "field_value" and t.startswith("Status:"):
            parts = t.split(":", 1)
            gt_fields[normalize_text(parts[0])] = parts[1].strip()

    # Extract horizontally adjacent field_label + field_value
    for el in elements:
        if el.get("type") == "field_label":
            key_raw = el.get("text", "").rstrip(":").strip()
            norm_k = normalize_text(key_raw)
            label_box = el.get("bbox", [0, 0, 0, 0])

            # Find matching field_value to the right on the same row
            candidates = []
            for other in elements:
                if other.get("type") == "field_value" and other != el:
                    obox = other.get("bbox", [0, 0, 0, 0])
                    # must be to the right and vertical center within 15px
                    if obox[0] > label_box[0] and abs(obox[1] - label_box[1]) < 20:
                        candidates.append((obox[0], other.get("text", "").strip()))

            if candidates:
                candidates.sort(key=lambda x: x[0])
                gt_fields[norm_k] = candidates[0][1]
            elif norm_k == "findings":
                # findings in form ground truth is a field label with empty inline value
                gt_fields[norm_k] = ""

    return {
        "doc_id": doc_id,
        "degraded": degraded,
        "fields": gt_fields,
        "annotation_path": str(annotation_path),
    }


def evaluate_all_forms(
    dataset_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run full evaluation on synthetic form samples."""
    base_dir = dataset_dir or Path(r"C:\SovereignAI\datasets\ocr\synthetic_ocr_dataset\synthetic_forms")
    out_dir = output_dir or Path(r"C:\SovereignAI\member3_ocr\output\evaluation\forms")
    out_dir.mkdir(parents=True, exist_ok=True)

    det_dir = Path(r"C:\SovereignAI\member3_ocr\models\paddleocr\PP-OCRv5_mobile_det_infer")
    rec_dir = Path(r"C:\SovereignAI\member3_ocr\models\paddleocr\PP-OCRv5_mobile_rec_infer")

    config = PaddleOCRModelConfig(detection_model_dir=det_dir, recognition_model_dir=rec_dir)
    backend = PaddleOCRBackend(config)
    pipeline = OCRPipeline(backend, extract_key_values=True)

    annot_files = sorted(glob.glob(str(base_dir / "annotations" / "form_*.json")))
    LOGGER.info("Found %d form annotation files", len(annot_files))

    per_sample_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    total_gt_fields = 0
    total_pred_fields = 0
    total_key_matches = 0
    total_exact_matches = 0
    total_confidences: list[float] = []
    total_latencies_ms: list[float] = []

    clean_exact_matches = 0
    clean_gt_fields = 0
    degraded_exact_matches = 0
    degraded_gt_fields = 0

    for annot_str in annot_files:
        annot_path = Path(annot_str)
        gt_data = load_ground_truth(annot_path)
        doc_id = gt_data["doc_id"]
        degraded = gt_data["degraded"]
        gt_fields = gt_data["fields"]

        # Find matching image (PNG clean or JPG degraded)
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
        extracted_kvs = page.key_value_fields if page else ()

        pred_fields: dict[str, tuple[str, float | None]] = {}
        for f in extracted_kvs:
            norm_k = normalize_text(f.key)
            pred_fields[norm_k] = (f.value.strip(), f.confidence)
            if f.confidence is not None:
                total_confidences.append(f.confidence)

        # Compute sample level metrics
        sample_key_matches = 0
        sample_exact_matches = 0

        for k, gt_val in gt_fields.items():
            total_gt_fields += 1
            if not degraded:
                clean_gt_fields += 1
            else:
                degraded_gt_fields += 1

            if k in pred_fields:
                sample_key_matches += 1
                total_key_matches += 1
                pred_val, conf = pred_fields[k]

                # Compare normalized values
                if normalize_text(pred_val) == normalize_text(gt_val):
                    sample_exact_matches += 1
                    total_exact_matches += 1
                    if not degraded:
                        clean_exact_matches += 1
                    else:
                        degraded_exact_matches += 1
                else:
                    failure_rows.append({
                        "doc_id": doc_id,
                        "degraded": degraded,
                        "failure_type": "value_mismatch",
                        "field_key": k,
                        "ground_truth_value": gt_val,
                        "extracted_value": pred_val,
                        "confidence": conf,
                    })
            else:
                failure_rows.append({
                    "doc_id": doc_id,
                    "degraded": degraded,
                    "failure_type": "missing_field",
                    "field_key": k,
                    "ground_truth_value": gt_val,
                    "extracted_value": "",
                    "confidence": None,
                })

        total_pred_fields += len(pred_fields)

        # Check for false positive fields
        for pk, (pval, pconf) in pred_fields.items():
            if pk not in gt_fields:
                failure_rows.append({
                    "doc_id": doc_id,
                    "degraded": degraded,
                    "failure_type": "spurious_field",
                    "field_key": pk,
                    "ground_truth_value": "",
                    "extracted_value": pval,
                    "confidence": pconf,
                })

        # Precision, Recall, F1 for this sample
        sample_tp = sample_exact_matches
        sample_fp = len(pred_fields) - sample_tp
        sample_fn = len(gt_fields) - sample_tp
        sample_prec = sample_tp / max(1, (sample_tp + sample_fp))
        sample_rec = sample_tp / max(1, (sample_tp + sample_fn))
        sample_f1 = (2 * sample_prec * sample_rec) / max(1e-6, (sample_prec + sample_rec))

        per_sample_rows.append({
            "doc_id": doc_id,
            "degraded": degraded,
            "image_filename": img_path.name,
            "gt_field_count": len(gt_fields),
            "extracted_field_count": len(pred_fields),
            "key_matches": sample_key_matches,
            "exact_matches": sample_exact_matches,
            "key_accuracy": round(sample_key_matches / max(1, len(gt_fields)), 4),
            "value_accuracy": round(sample_exact_matches / max(1, sample_key_matches), 4),
            "exact_match_accuracy": round(sample_exact_matches / max(1, len(gt_fields)), 4),
            "precision": round(sample_prec, 4),
            "recall": round(sample_rec, 4),
            "f1_score": round(sample_f1, 4),
            "latency_ms": latency_ms,
        })

    # Overall dataset metrics
    precision = total_exact_matches / max(1, total_pred_fields)
    recall = total_exact_matches / max(1, total_gt_fields)
    f1_score = (2 * precision * recall) / max(1e-6, (precision + recall))
    key_acc = total_key_matches / max(1, total_gt_fields)
    val_acc = total_exact_matches / max(1, total_key_matches)
    em_acc = total_exact_matches / max(1, total_gt_fields)
    avg_conf = sum(total_confidences) / max(1, len(total_confidences))
    avg_latency = sum(total_latencies_ms) / max(1, len(total_latencies_ms))

    clean_em_acc = clean_exact_matches / max(1, clean_gt_fields)
    deg_em_acc = degraded_exact_matches / max(1, degraded_gt_fields)

    summary_data = {
        "dataset": "synthetic_forms",
        "total_samples": len(per_sample_rows),
        "total_gt_fields": total_gt_fields,
        "total_extracted_fields": total_pred_fields,
        "total_exact_matches": total_exact_matches,
        "metrics": {
            "key_accuracy": round(key_acc, 4),
            "value_accuracy": round(val_acc, 4),
            "exact_match_accuracy": round(em_acc, 4),
            "field_precision": round(precision, 4),
            "field_recall": round(recall, 4),
            "field_f1": round(f1_score, 4),
            "average_confidence": round(avg_conf, 4),
            "average_latency_ms": round(avg_latency, 2),
        },
        "breakdown": {
            "clean_samples": sum(1 for r in per_sample_rows if not r["degraded"]),
            "clean_exact_match_accuracy": round(clean_em_acc, 4),
            "degraded_samples": sum(1 for r in per_sample_rows if r["degraded"]),
            "degraded_exact_match_accuracy": round(deg_em_acc, 4),
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
    fail_fields = ["doc_id", "degraded", "failure_type", "field_key", "ground_truth_value", "extracted_value", "confidence"]
    with open(out_dir / "failures.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fail_fields)
        writer.writeheader()
        writer.writerows(failure_rows)

    # Write README.md
    readme_content = f"""# Form Key-Value Extraction Evaluation Report

**Evaluation Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}
**Backend:** PaddleOCR (PP-OCRv5 local mobile detection + recognition)
**Target Dataset:** `datasets/ocr/synthetic_ocr_dataset/synthetic_forms`
**Total Samples:** {len(per_sample_rows)} (Clean: {summary_data['breakdown']['clean_samples']}, Degraded: {summary_data['breakdown']['degraded_samples']})

## Summary Metrics

| Metric | Score | Description |
| :--- | :--- | :--- |
| **Key-Value Exact-Match Accuracy** | **{summary_data['metrics']['exact_match_accuracy'] * 100:.1f}%** | Percentage of ground-truth fields where both key and value match exactly |
| **Key Accuracy** | **{summary_data['metrics']['key_accuracy'] * 100:.1f}%** | Percentage of ground-truth field labels identified |
| **Value Accuracy** | **{summary_data['metrics']['value_accuracy'] * 100:.1f}%** | Accuracy of extracted values for identified keys |
| **Field Precision** | **{summary_data['metrics']['field_precision'] * 100:.1f}%** | TP / (TP + FP) across all extracted key-value fields |
| **Field Recall** | **{summary_data['metrics']['field_recall'] * 100:.1f}%** | TP / (TP + FN) across all ground-truth fields |
| **Field F1 Score** | **{summary_data['metrics']['field_f1'] * 100:.1f}%** | Harmonic mean of field precision and recall |
| **Average Confidence** | **{summary_data['metrics']['average_confidence']:.4f}** | Conservative joint confidence derived from OCR blocks |
| **Average Latency** | **{summary_data['metrics']['average_latency_ms']:.1f} ms** | End-to-end processing latency per page |

## Clean vs. Degraded Performance

- **Clean Scans ({summary_data['breakdown']['clean_samples']} docs):** **{clean_em_acc * 100:.1f}%** Exact-Match Accuracy
- **Degraded Scans ({summary_data['breakdown']['degraded_samples']} docs):** **{deg_em_acc * 100:.1f}%** Exact-Match Accuracy

## Generated Evaluation Artifacts

1. `summary.json`: Aggregate metrics, counts, and category breakdown.
2. `per_sample.csv`: Sample-by-sample scores, ground-truth count, precision, recall, and latency.
3. `failures.csv`: Discrepancy analysis categorizing value mismatches, missing fields, or spurious fields.
"""
    (out_dir / "README.md").write_text(readme_content, encoding="utf-8")

    LOGGER.info("Evaluation complete! Summary saved to %s", out_dir / "summary.json")
    return summary_data


if __name__ == "__main__":
    summary = evaluate_all_forms()
    print(json.dumps(summary, indent=2))
