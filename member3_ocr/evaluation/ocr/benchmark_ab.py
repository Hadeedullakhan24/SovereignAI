"""Controlled Preprocessing A/B Benchmark for Synthetic OCR Dataset (Member 3).

Compares:
Config A: Current production preprocessing (grayscale, denoise=3, CLAHE clip=2.0, adaptive binarization)
Config B: Raw/minimal preprocessing (preserve 3-channel RGB/BGR, no grayscale, no denoise, no CLAHE, no binarization)

Dataset: C:\\SovereignAI\\datasets\\ocr\\synthetic_ocr_dataset (READ-ONLY)
Output:  C:\\SovereignAI\\member3_ocr\\output\\evaluation\\preprocessing_ab\\
"""
from __future__ import annotations


import argparse
import csv
import json
import logging
import platform
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import paddlex

from member3_ocr.core.image_preprocessing import PreprocessingOptions
from member3_ocr.evaluation.ocr.ocr_evaluator import (
    EvaluationSample,
    GroupMetrics,
    OCREvaluator,
    SampleEvaluationResult,
    aggregate_group_metrics,
    load_synthetic_dataset_samples,
    validate_safe_output_path,
)
from member3_ocr.core.ocr_pipeline import (
    OCRPipeline,
    PaddleOCRBackend,
    PaddleOCRModelConfig,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)


def build_ab_comparison_row(
    group_name: str,
    metrics_a: GroupMetrics,
    metrics_b: GroupMetrics,
) -> dict[str, Any]:
    """Build a comparative row showing Config A, Config B, and Delta (B - A)."""
    return {
        "group": group_name,
        "sample_count": metrics_a.sample_count,
        # Config A (Production)
        "config_a_cer": metrics_a.mean_cer,
        "config_a_wer": metrics_a.mean_wer,
        "config_a_exact_match_rate": metrics_a.exact_match_rate,
        "config_a_exact_match_count": metrics_a.exact_match_count,
        "config_a_mean_confidence": metrics_a.mean_confidence,
        "config_a_mean_latency_ms": metrics_a.mean_processing_time_ms,
        "config_a_total_blocks": metrics_a.total_blocks_detected,
        "config_a_images_per_sec": metrics_a.images_per_second,
        # Config B (Raw RGB)
        "config_b_cer": metrics_b.mean_cer,
        "config_b_wer": metrics_b.mean_wer,
        "config_b_exact_match_rate": metrics_b.exact_match_rate,
        "config_b_exact_match_count": metrics_b.exact_match_count,
        "config_b_mean_confidence": metrics_b.mean_confidence,
        "config_b_mean_latency_ms": metrics_b.mean_processing_time_ms,
        "config_b_total_blocks": metrics_b.total_blocks_detected,
        "config_b_images_per_sec": metrics_b.images_per_second,
        # Deltas: Raw (B) - Production (A)
        # Note: For CER & WER, negative delta means Raw has LOWER error (better).
        # For Confidence, positive delta means Raw has HIGHER confidence (better).
        "delta_cer": round(metrics_b.mean_cer - metrics_a.mean_cer, 4),
        "delta_wer": round(metrics_b.mean_wer - metrics_a.mean_wer, 4),
        "delta_confidence": round(metrics_b.mean_confidence - metrics_a.mean_confidence, 4),
        "delta_latency_ms": round(metrics_b.mean_processing_time_ms - metrics_a.mean_processing_time_ms, 2),
        "delta_blocks": metrics_b.total_blocks_detected - metrics_a.total_blocks_detected,
    }


def run_preprocessing_ab_benchmark(
    dataset_dir: Path,
    output_dir: Path,
    det_model_dir: Path,
    rec_model_dir: Path,
) -> dict[str, Any]:
    """Execute the controlled A/B preprocessing evaluation."""
    validate_safe_output_path(output_dir, dataset_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load samples
    samples = load_synthetic_dataset_samples(dataset_dir)
    if not samples:
        raise RuntimeError(f"No samples loaded from {dataset_dir}")
    LOGGER.info("Loaded %d synthetic samples for A/B benchmarking", len(samples))

    # 2. Shared backend
    model_config = PaddleOCRModelConfig(
        detection_model_dir=det_model_dir,
        recognition_model_dir=rec_model_dir,
    )
    backend = PaddleOCRBackend(model_config)
    backend.initialize()

    # Define Configurations
    config_a_options = PreprocessingOptions.document_ocr(
        grayscale=True,
        denoise=True,
        contrast_enhancement=True,
        binarize=True,
    )
    config_b_options = PreprocessingOptions(
        max_width=None,
        max_height=None,
        allow_upscale=False,
        grayscale=False,
        denoise=False,
        contrast_enhancement=False,
        deskew=False,
        binarize=False,
        crop_box=None,
    )

    # 3. Run Configuration A (Production)
    LOGGER.info("--- Running Benchmark for Configuration A (Production Preprocessing) ---")
    dir_a = output_dir / "config_a"
    pipeline_a = OCRPipeline(backend, default_preprocessing=config_a_options)
    evaluator_a = OCREvaluator(pipeline_a, output_dir=dir_a, dataset_root=dataset_dir)

    def _progress_a(current: int, total: int, res: SampleEvaluationResult) -> None:
        LOGGER.info("[Config A %d/%d] %s (%s) CER=%.4f WER=%.4f (%.0fms)", current, total, res.sample_id, res.category, res.cer, res.wer, res.processing_time_ms)

    summary_a, results_a = evaluator_a.evaluate_all(samples, progress_callback=_progress_a)
    evaluator_a.export_results(summary_a, results_a)

    # 4. Run Configuration B (Raw RGB)
    LOGGER.info("--- Running Benchmark for Configuration B (Raw / Minimal RGB) ---")
    dir_b = output_dir / "config_b"
    pipeline_b = OCRPipeline(backend, default_preprocessing=config_b_options)
    evaluator_b = OCREvaluator(pipeline_b, output_dir=dir_b, dataset_root=dataset_dir)

    def _progress_b(current: int, total: int, res: SampleEvaluationResult) -> None:
        LOGGER.info("[Config B %d/%d] %s (%s) CER=%.4f WER=%.4f (%.0fms)", current, total, res.sample_id, res.category, res.cer, res.wer, res.processing_time_ms)

    summary_b, results_b = evaluator_b.evaluate_all(samples, progress_callback=_progress_b)
    evaluator_b.export_results(summary_b, results_b)

    # 5. Build Group Metrics & Comparison
    groups = ["overall", "synthetic_forms", "synthetic_manual_pages", "synthetic_tables", "clean", "degraded"]
    comparison_rows: list[dict[str, Any]] = []

    def _filter_results(results_list: list[SampleEvaluationResult], grp: str) -> list[SampleEvaluationResult]:
        if grp == "overall":
            return results_list
        if grp in ("synthetic_forms", "synthetic_manual_pages", "synthetic_tables"):
            return [r for r in results_list if r.category == grp]
        if grp == "clean":
            return [r for r in results_list if not r.degraded]
        if grp == "degraded":
            return [r for r in results_list if r.degraded]
        return []

    for grp in groups:
        sub_a = _filter_results(results_a, grp)
        sub_b = _filter_results(results_b, grp)
        metrics_a = aggregate_group_metrics(sub_a)
        metrics_b = aggregate_group_metrics(sub_b)
        row = build_ab_comparison_row(grp, metrics_a, metrics_b)
        comparison_rows.append(row)

    # 6. Export Comparison Artifacts
    # JSON
    benchmark_payload = {
        "benchmark_timestamp": datetime.now(timezone.utc).isoformat(),
        "paddlex_version": paddlex.__version__,
        "dataset_path": str(dataset_dir),
        "sample_count": len(samples),
        "configurations": {
            "config_a": {
                "name": "Current Production Preprocessing",
                "options": asdict(config_a_options),
            },
            "config_b": {
                "name": "Raw / Minimal Preprocessing (3-channel RGB)",
                "options": asdict(config_b_options),
            },
        },
        "comparison": comparison_rows,
        "important_notes": [
            "Table CER and WER reflect 1D flattened transcript reading order vs 2D detected text blocks, not purely OCR character recognition quality.",
            "Delta = Config B (Raw) - Config A (Production). For CER/WER, negative delta means Raw is lower/better. For Confidence, positive delta means Raw is higher/better.",
        ],
    }

    comp_json_path = output_dir / "ab_comparison.json"
    comp_json_path.write_text(json.dumps(benchmark_payload, indent=2), encoding="utf-8")

    # CSV
    comp_csv_path = output_dir / "ab_comparison.csv"
    if comparison_rows:
        with comp_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(comparison_rows[0].keys()))
            writer.writeheader()
            writer.writerows(comparison_rows)

    # Per-Sample Delta CSV
    per_sample_comp_rows: list[dict[str, Any]] = []
    map_b = {r.sample_id: r for r in results_b}
    for ra in results_a:
        rb = map_b.get(ra.sample_id)
        if not rb:
            continue
        per_sample_comp_rows.append({
            "sample_id": ra.sample_id,
            "category": ra.category,
            "degraded": ra.degraded,
            "cer_a": ra.cer,
            "cer_b": rb.cer,
            "delta_cer": round(rb.cer - ra.cer, 4),
            "wer_a": ra.wer,
            "wer_b": rb.wer,
            "delta_wer": round(rb.wer - ra.wer, 4),
            "conf_a": ra.ocr_confidence,
            "conf_b": rb.ocr_confidence,
            "delta_conf": round(rb.ocr_confidence - ra.ocr_confidence, 4),
            "blocks_a": ra.num_blocks,
            "blocks_b": rb.num_blocks,
            "time_ms_a": ra.processing_time_ms,
            "time_ms_b": rb.processing_time_ms,
            "exact_match_a": ra.exact_match,
            "exact_match_b": rb.exact_match,
        })

    per_sample_csv_path = output_dir / "per_sample_delta.csv"
    if per_sample_comp_rows:
        with per_sample_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_sample_comp_rows[0].keys()))
            writer.writeheader()
            writer.writerows(per_sample_comp_rows)

    # Markdown README
    readme_path = output_dir / "README.md"
    readme_text = f"""# Preprocessing A/B Benchmark Results

**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
**PaddleX Version:** `{paddlex.__version__}`
**Dataset:** `{dataset_dir}` (36 samples: 12 forms, 12 manuals, 12 tables; 22 clean, 14 degraded)

## Configurations
- **Config A (Production):** Grayscale, Denoise (strength=3), CLAHE (clip=2.0, grid=8), Adaptive Binarization.
- **Config B (Raw RGB):** Original 3-channel image, No Grayscale, No Denoise, No CLAHE, No Binarization, No Deskew.

## Summary Comparison Table

| Category | Samples | CER (A) | CER (B) | Δ CER | WER (A) | WER (B) | Δ WER | Conf (A) | Conf (B) | Blocks (A) | Blocks (B) | Latency A (ms) | Latency B (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for r in comparison_rows:
        readme_text += (
            f"| **{r['group']}** | {r['sample_count']} | "
            f"{r['config_a_cer']:.4f} | {r['config_b_cer']:.4f} | **{r['delta_cer']:+.4f}** | "
            f"{r['config_a_wer']:.4f} | {r['config_b_wer']:.4f} | **{r['delta_wer']:+.4f}** | "
            f"{r['config_a_mean_confidence']:.4f} | {r['config_b_mean_confidence']:.4f} | "
            f"{r['config_a_total_blocks']} | {r['config_b_total_blocks']} | "
            f"{r['config_a_mean_latency_ms']:.1f} | {r['config_b_mean_latency_ms']:.1f} |\n"
        )

    readme_text += """
> [!IMPORTANT]
> **Table Metric Limitation:** Table CER and WER reflect 1D flattened transcript serialization vs 2D spatial block layout. Discrepancies between OCR block reading order and transcript table column sequence do not necessarily denote OCR character recognition misidentifications.
"""
    readme_path.write_text(readme_text, encoding="utf-8")

    LOGGER.info("Exported A/B benchmark artifacts to %s", output_dir)
    return benchmark_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Preprocessing A/B Benchmark")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(r"C:\SovereignAI\datasets\ocr\synthetic_ocr_dataset"),
        help="Path to synthetic_ocr_dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"C:\SovereignAI\member3_ocr\output\evaluation\preprocessing_ab"),
        help="Output directory for benchmark results",
    )
    parser.add_argument(
        "--det-model-dir",
        type=Path,
        default=Path(r"C:\SovereignAI\member3_ocr\models\paddleocr\PP-OCRv5_mobile_det_infer"),
        help="Detection model directory",
    )
    parser.add_argument(
        "--rec-model-dir",
        type=Path,
        default=Path(r"C:\SovereignAI\member3_ocr\models\paddleocr\PP-OCRv5_mobile_rec_infer"),
        help="Recognition model directory",
    )
    args = parser.parse_args()

    run_preprocessing_ab_benchmark(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        det_model_dir=args.det_model_dir,
        rec_model_dir=args.rec_model_dir,
    )


if __name__ == "__main__":
    main()
