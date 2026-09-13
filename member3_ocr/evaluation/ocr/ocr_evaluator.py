"""OCR Dataset Evaluator for Sovereign AI Workbench (Member 3).

Evaluates OCR pipeline accuracy, robustness, and performance across:
1. Synthetic OCR documents (forms, manuals/SOPs, tables, clean vs degraded).
2. FUNSD benchmark documents (training and testing splits).

Computes Character Error Rate (CER), Word Error Rate (WER), Exact Match Rate,
Confidence, Block Counts, and Latency/Throughput.

Strict Safety Guarantees:
- Dataset files are READ-ONLY; never modified, moved, or deleted.
- Output files are strictly prohibited from being written into datasets/.
- Fully offline and reproducible.
"""
from __future__ import annotations


import argparse
import csv
import json
import logging
import platform
import re
import sys
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from member3_ocr.core.image_preprocessing import PreprocessingOptions, preprocess_image
from member3_ocr.core.ocr_pipeline import (
    BackendCapabilities,
    BackendInfo,
    BoundingBox,
    OCRDocumentResult,
    OCRPipeline,
    PaddleOCRBackend,
    PaddleOCRModelConfig,
    TextBlock,
)

LOGGER = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Metric Utilities (Levenshtein, CER, WER, Normalization)
# ──────────────────────────────────────────────────────────────────────────────

def normalize_text_for_eval(text: str) -> str:
    """Normalize text for metric computation.

    Policy:
    - Unicode NFKC normalization.
    - Standardize whitespace: convert tabs/newlines/multiple spaces into single spaces.
    - Strip leading and trailing whitespace.
    - Preserves case and all punctuation (colons, hyphens, periods, degree symbols)
      essential for refinery equipment tags, pressures, and status values.
    """
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKC", text)
    # Collapse whitespace sequences into single space
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def levenshtein_distance(seq1: Sequence[Any], seq2: Sequence[Any]) -> int:
    """Compute Levenshtein edit distance between two sequences with O(min(N, M)) memory."""
    if len(seq1) < len(seq2):
        seq1, seq2 = seq2, seq1

    if not seq2:
        return len(seq1)

    previous_row = list(range(len(seq2) + 1))
    for i, c1 in enumerate(seq1):
        current_row = [i + 1] + [0] * len(seq2)
        for j, c2 in enumerate(seq2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (0 if c1 == c2 else 1)
            current_row[j + 1] = min(insertions, deletions, substitutions)
        previous_row = current_row

    return previous_row[-1]


def compute_cer(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate (CER).

    CER = Levenshtein(ref_chars, hyp_chars) / max(1, len(ref_chars))
    """
    ref_norm = normalize_text_for_eval(reference)
    hyp_norm = normalize_text_for_eval(hypothesis)

    if not ref_norm and not hyp_norm:
        return 0.0
    if not ref_norm:
        return 1.0

    dist = levenshtein_distance(list(ref_norm), list(hyp_norm))
    return dist / len(ref_norm)


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate (WER).

    WER = Levenshtein(ref_words, hyp_words) / max(1, len(ref_words))
    """
    ref_norm = normalize_text_for_eval(reference)
    hyp_norm = normalize_text_for_eval(hypothesis)

    ref_words = ref_norm.split()
    hyp_words = hyp_norm.split()

    if not ref_words and not hyp_words:
        return 0.0
    if not ref_words:
        return 1.0

    dist = levenshtein_distance(ref_words, hyp_words)
    return dist / len(ref_words)


# ──────────────────────────────────────────────────────────────────────────────
# Data Models
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EvaluationSample:
    """One document image and ground-truth pair to be evaluated."""

    sample_id: str
    dataset: str  # "synthetic" | "funsd_train" | "funsd_test"
    category: str  # "synthetic_forms" | "synthetic_manual_pages" | "synthetic_tables" | "funsd"
    degraded: bool
    image_path: Path
    ground_truth_path: Path
    ground_truth_text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BlockConfidenceRecord:
    """Per-block confidence, ground truth text, and error metrics for calibration."""

    block_id: str
    confidence: float
    block_text: str
    ground_truth_text: str = ""
    cer: float = 0.0
    wer: float = 0.0


@dataclass(frozen=True)
class ConfidenceCalibrationBucket:
    """Aggregated accuracy stats for blocks within a confidence range."""

    label: str           # e.g. "high (>0.90)"
    min_conf: float
    max_conf: float
    block_count: int
    percent_of_blocks: float = 0.0
    mean_confidence: float = 0.0
    mean_cer: float = 0.0
    mean_wer: float = 0.0
    non_empty_rate: float = 0.0
    total_chars: int = 0


@dataclass(frozen=True)
class SampleEvaluationResult:
    """Evaluation output and metric scores for a single document sample."""

    sample_id: str
    dataset: str
    category: str
    degraded: bool
    image_path: str
    ground_truth_length: int
    ocr_text_length: int
    exact_match: bool
    cer: float
    wer: float
    ocr_confidence: float
    num_blocks: int
    processing_time_ms: float
    ocr_text: str
    ground_truth_text: str
    error_message: Optional[str] = None
    block_records: tuple[BlockConfidenceRecord, ...] = ()


@dataclass
class GroupMetrics:
    """Aggregated statistics for a subset of evaluated samples."""

    sample_count: int = 0
    exact_match_count: int = 0
    exact_match_rate: float = 0.0
    mean_cer: float = 0.0
    mean_wer: float = 0.0
    mean_confidence: float = 0.0
    mean_processing_time_ms: float = 0.0
    images_per_second: float = 0.0
    total_blocks_detected: int = 0
    error_count: int = 0


@dataclass
class EvaluationSummary:
    """Complete summary of an OCR evaluation run across all splits."""

    evaluation_timestamp: str
    python_version: str
    backend_info: dict[str, Any]
    preprocessing_options: dict[str, Any]
    total_samples: int
    overall: GroupMetrics
    by_dataset: dict[str, GroupMetrics]
    by_category: dict[str, GroupMetrics]
    by_degradation: dict[str, GroupMetrics]
    confidence_calibration: list[dict[str, Any]] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset Loaders & Safety Validation
# ──────────────────────────────────────────────────────────────────────────────

def validate_safe_output_path(output_dir: Path, dataset_root: Optional[Path] = None) -> None:
    """Ensure evaluation outputs are strictly written outside any datasets directory."""
    resolved_out = output_dir.resolve()
    from member3_ocr.evaluation.paths import get_project_root, get_output_dir, get_models_dir, get_datasets_dir
    repo_datasets = get_datasets_dir()

    forbidden_roots = [repo_datasets.resolve()]
    if dataset_root is not None:
        forbidden_roots.append(dataset_root.resolve())

    for forbidden in forbidden_roots:
        if resolved_out == forbidden or forbidden in resolved_out.parents:
            raise ValueError(
                f"Safety violation: evaluation output directory {output_dir} cannot be "
                f"inside datasets directory {forbidden}. Evaluation is strictly read-only on datasets."
            )


def load_synthetic_dataset_samples(dataset_dir: Path) -> list[EvaluationSample]:
    """Load samples from synthetic_ocr_dataset using manifest.json, images, and text."""
    manifest_file = dataset_dir / "manifest.json"
    if not manifest_file.exists():
        LOGGER.warning("manifest.json not found in %s", dataset_dir)
        return []

    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    samples: list[EvaluationSample] = []

    images_dir = dataset_dir / "images"
    text_dir = dataset_dir / "text"

    for entry in sorted(manifest, key=lambda x: x["doc_id"]):
        doc_id = entry["doc_id"]
        category = entry.get("category", "unknown")
        degraded = bool(entry.get("degraded", False))

        # PNG for clean, JPG for scan-degraded (fallback to whichever exists)
        img_candidate = images_dir / f"{doc_id}.jpg" if degraded else images_dir / f"{doc_id}.png"
        if not img_candidate.exists():
            img_candidate = images_dir / f"{doc_id}.png"
        if not img_candidate.exists():
            img_candidate = images_dir / f"{doc_id}.jpg"

        txt_candidate = text_dir / f"{doc_id}.txt"

        if not img_candidate.exists() or not txt_candidate.exists():
            LOGGER.warning("Skipping synthetic sample %s: missing image or text file", doc_id)
            continue

        gt_text = txt_candidate.read_text(encoding="utf-8")
        gt_boxes: list[dict[str, Any]] = []
        ann_candidate = dataset_dir / "annotations" / f"{doc_id}.json"
        if ann_candidate.exists():
            try:
                ann_data = json.loads(ann_candidate.read_text(encoding="utf-8"))
                for el in ann_data.get("elements", []):
                    t = el.get("text", "").strip()
                    if t and "bbox" in el:
                        gt_boxes.append({"box": el["bbox"], "text": t})
            except Exception:
                pass

        meta_dict = dict(entry.get("meta", {}))
        meta_dict["gt_boxes"] = gt_boxes

        samples.append(
            EvaluationSample(
                sample_id=doc_id,
                dataset="synthetic",
                category=category,
                degraded=degraded,
                image_path=img_candidate,
                ground_truth_path=txt_candidate,
                ground_truth_text=gt_text,
                metadata=meta_dict,
            )
        )

    return samples


def load_funsd_dataset_samples(funsd_root: Path, split: str = "training_data") -> list[EvaluationSample]:
    """Load samples from FUNSD split (training_data or testing_data).

    Reconstructs ground-truth text by sorting form annotation elements in
    top-to-bottom reading order. Also extracts word bounding boxes for block-level calibration.
    """
    split_dir = funsd_root / split
    if not split_dir.is_dir():
        LOGGER.warning("FUNSD split directory not found: %s", split_dir)
        return []

    images_dir = split_dir / "images"
    ann_dir = split_dir / "annotations"

    if not images_dir.is_dir() or not ann_dir.is_dir():
        return []

    samples: list[EvaluationSample] = []
    split_tag = "funsd_train" if "train" in split else "funsd_test"

    for ann_file in sorted(ann_dir.glob("*.json"), key=lambda p: p.stem):
        doc_id = ann_file.stem
        img_file = images_dir / f"{doc_id}.png"
        if not img_file.exists():
            img_file = images_dir / f"{doc_id}.jpg"
        if not img_file.exists():
            continue

        try:
            data = json.loads(ann_file.read_text(encoding="utf-8"))
            form_elements = data.get("form", [])

            # Sort in standard top-to-bottom, left-to-right reading order
            def _sort_key(el: dict[str, Any]) -> tuple[int, int]:
                box = el.get("box", [0, 0, 0, 0])
                # Row grouping tolerance of 12 pixels
                return (round(box[1] / 12) * 12, box[0])

            sorted_elements = sorted(form_elements, key=_sort_key)
            gt_text = "\n".join(el["text"] for el in sorted_elements if el.get("text"))

            # Extract word-level bounding boxes for block-level matching
            all_words: list[dict[str, Any]] = []
            for el in form_elements:
                for w in el.get("words", []):
                    t = w.get("text", "").strip()
                    if t and "box" in w:
                        all_words.append({"box": w["box"], "text": t})
        except Exception as exc:
            LOGGER.error("Failed to parse FUNSD annotation %s: %s", ann_file, exc)
            continue

        samples.append(
            EvaluationSample(
                sample_id=doc_id,
                dataset=split_tag,
                category="funsd",
                degraded=False,
                image_path=img_file,
                ground_truth_path=ann_file,
                ground_truth_text=gt_text,
                metadata={"total_form_elements": len(form_elements), "gt_boxes": all_words},
            )
        )

    return samples



# ──────────────────────────────────────────────────────────────────────────────
# Confidence Calibration
# ──────────────────────────────────────────────────────────────────────────────

def match_block_to_gt_boxes(
    block_box: tuple[float, float, float, float],
    gt_boxes: Sequence[dict[str, Any]],
) -> str:
    """Find ground-truth text elements that spatially overlap an OCR block box.

    Uses intersection-over-GT-box area (>=0.35) or center-point containment
    to associate ground truth words/elements with detected OCR blocks.
    """
    if not gt_boxes:
        return ""
    bx1, by1, bx2, by2 = block_box
    matched: list[dict[str, Any]] = []
    for item in gt_boxes:
        ibox = item.get("box") or item.get("bbox")
        if not ibox or len(ibox) < 4:
            continue
        wx1, wy1, wx2, wy2 = ibox[0], ibox[1], ibox[2], ibox[3]
        cx = (wx1 + wx2) / 2.0
        cy = (wy1 + wy2) / 2.0
        center_in = (bx1 <= cx <= bx2) and (by1 <= cy <= by2)
        ix1 = max(wx1, bx1)
        iy1 = max(wy1, by1)
        ix2 = min(wx2, bx2)
        iy2 = min(wy2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        w_area = max(1.0, (wx2 - wx1) * (wy2 - wy1))
        frac = inter / w_area
        if center_in or frac >= 0.35:
            matched.append(item)
    if not matched:
        return ""

    def _box_key(it: dict[str, Any]) -> tuple[int, float]:
        ib = it.get("box") or it.get("bbox") or [0, 0, 0, 0]
        return (round(ib[1] / 10.0) * 10, ib[0])

    matched.sort(key=_box_key)
    return " ".join(m.get("text", "").strip() for m in matched if m.get("text"))


# Confidence buckets: (label, min_conf, max_conf)
_CONF_BUCKETS: list[tuple[str, float, float]] = [
    ("very_high (>0.95)", 0.95, 1.01),
    ("high (0.80-0.95)", 0.80, 0.95),
    ("medium (0.60-0.80)", 0.60, 0.80),
    ("low (<0.60)", 0.0, 0.60),
]


def compute_confidence_calibration(
    results: Sequence[SampleEvaluationResult],
) -> list[ConfidenceCalibrationBucket]:
    """Compute confidence calibration by bucketing block-level confidence scores.

    Each block's confidence is bucketed, and we report: block count, percent of blocks,
    mean confidence, mean CER, mean WER, non-empty proportion, and total characters.
    """
    # Collect all block records across all samples
    all_blocks: list[BlockConfidenceRecord] = []
    for r in results:
        all_blocks.extend(r.block_records)

    total_blocks_count = len(all_blocks)
    buckets: list[ConfidenceCalibrationBucket] = []
    for label, lo, hi in _CONF_BUCKETS:
        bucket_blocks = [b for b in all_blocks if lo <= b.confidence < hi]
        if not bucket_blocks:
            buckets.append(ConfidenceCalibrationBucket(
                label=label, min_conf=lo, max_conf=hi if hi < 1.01 else 1.0,
                block_count=0, percent_of_blocks=0.0, mean_confidence=0.0,
                mean_cer=0.0, mean_wer=0.0, non_empty_rate=0.0, total_chars=0,
            ))
            continue
        mean_conf = float(np.mean([b.confidence for b in bucket_blocks]))
        mean_cer = float(np.mean([b.cer for b in bucket_blocks]))
        mean_wer = float(np.mean([b.wer for b in bucket_blocks]))
        pct_blocks = (len(bucket_blocks) / total_blocks_count * 100.0) if total_blocks_count > 0 else 0.0
        non_empty = sum(1 for b in bucket_blocks if b.block_text.strip())
        total_chars = sum(len(b.block_text) for b in bucket_blocks)
        buckets.append(ConfidenceCalibrationBucket(
            label=label,
            min_conf=lo,
            max_conf=hi if hi < 1.01 else 1.0,
            block_count=len(bucket_blocks),
            percent_of_blocks=round(pct_blocks, 2),
            mean_confidence=round(mean_conf, 4),
            mean_cer=round(mean_cer, 4),
            mean_wer=round(mean_wer, 4),
            non_empty_rate=round(non_empty / len(bucket_blocks), 4),
            total_chars=total_chars,
        ))
    return buckets


# ──────────────────────────────────────────────────────────────────────────────
# Aggregation & Evaluation Engine
# ──────────────────────────────────────────────────────────────────────────────

def aggregate_group_metrics(results: Sequence[SampleEvaluationResult]) -> GroupMetrics:
    """Aggregate a group of sample results into summary metrics."""
    if not results:
        return GroupMetrics()

    n = len(results)
    exact_matches = sum(1 for r in results if r.exact_match)
    errors = sum(1 for r in results if r.error_message is not None)
    total_blocks = sum(r.num_blocks for r in results)

    mean_cer = sum(r.cer for r in results) / n
    mean_wer = sum(r.wer for r in results) / n
    mean_conf = sum(r.ocr_confidence for r in results) / n
    mean_time = sum(r.processing_time_ms for r in results) / n

    total_time_sec = sum(r.processing_time_ms for r in results) / 1000.0
    ips = n / total_time_sec if total_time_sec > 0 else 0.0

    return GroupMetrics(
        sample_count=n,
        exact_match_count=exact_matches,
        exact_match_rate=round(exact_matches / n, 4),
        mean_cer=round(mean_cer, 4),
        mean_wer=round(mean_wer, 4),
        mean_confidence=round(mean_conf, 4),
        mean_processing_time_ms=round(mean_time, 2),
        images_per_second=round(ips, 2),
        total_blocks_detected=total_blocks,
        error_count=errors,
    )


class OCREvaluator:
    """Evaluates an OCRPipeline against real or synthetic dataset samples."""

    def __init__(
        self,
        pipeline: OCRPipeline,
        *,
        output_dir: Path,
        dataset_root: Optional[Path] = None,
        poor_cer_threshold: float = 0.15,
    ) -> None:
        self.pipeline = pipeline
        self.output_dir = output_dir
        self.dataset_root = dataset_root
        self.poor_cer_threshold = poor_cer_threshold

        validate_safe_output_path(self.output_dir, self.dataset_root)

    def evaluate_sample(self, sample: EvaluationSample) -> SampleEvaluationResult:
        """Run OCR on one sample image, compute metrics against ground truth.

        In addition to document-level CER/WER, this collects per-block
        confidence records for downstream calibration analysis.
        """
        start_time = time.perf_counter()
        ocr_text = ""
        confidence = 0.0
        num_blocks = 0
        error_msg: Optional[str] = None
        block_records: list[BlockConfidenceRecord] = []

        try:
            doc_result = self.pipeline.process_image(
                source=sample.image_path,
                document_id=sample.sample_id,
            )

            if doc_result.errors:
                error_msg = "; ".join(e.message for e in doc_result.errors)

            if doc_result.pages:
                page = doc_result.pages[0]
                ocr_text = page.text
                num_blocks = len(page.blocks)
                valid_confs = [b.confidence for b in page.blocks if b.confidence is not None]
                confidence = float(np.mean(valid_confs)) if valid_confs else 0.0
                # Collect per-block records for confidence calibration
                gt_boxes = sample.metadata.get("gt_boxes", [])
                for block in page.blocks:
                    if block.confidence is not None:
                        block_box = (block.bbox.left, block.bbox.top, block.bbox.right, block.bbox.bottom)
                        gt_matched = match_block_to_gt_boxes(block_box, gt_boxes) if gt_boxes else ""
                        if gt_matched:
                            b_cer = compute_cer(gt_matched, block.text)
                            b_wer = compute_wer(gt_matched, block.text)
                        elif gt_boxes:
                            b_cer = 0.0 if not block.text.strip() else 1.0
                            b_wer = 0.0 if not block.text.strip() else 1.0
                        else:
                            b_cer = 0.0
                            b_wer = 0.0
                        block_records.append(BlockConfidenceRecord(
                            block_id=block.id,
                            confidence=float(block.confidence),
                            block_text=block.text,
                            ground_truth_text=gt_matched,
                            cer=round(b_cer, 4),
                            wer=round(b_wer, 4),
                        ))
        except Exception as exc:
            error_msg = str(exc)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Calculate metrics
        cer = compute_cer(sample.ground_truth_text, ocr_text)
        wer = compute_wer(sample.ground_truth_text, ocr_text)
        exact = (
            normalize_text_for_eval(sample.ground_truth_text)
            == normalize_text_for_eval(ocr_text)
        )

        return SampleEvaluationResult(
            sample_id=sample.sample_id,
            dataset=sample.dataset,
            category=sample.category,
            degraded=sample.degraded,
            image_path=str(sample.image_path),
            ground_truth_length=len(sample.ground_truth_text),
            ocr_text_length=len(ocr_text),
            exact_match=exact,
            cer=round(cer, 4),
            wer=round(wer, 4),
            ocr_confidence=round(confidence, 4),
            num_blocks=num_blocks,
            processing_time_ms=round(elapsed_ms, 2),
            ocr_text=ocr_text,
            ground_truth_text=sample.ground_truth_text,
            error_message=error_msg,
            block_records=tuple(block_records),
        )

    def evaluate_all(
        self,
        samples: Sequence[EvaluationSample],
        *,
        progress_callback: Optional[Callable[[int, int, SampleEvaluationResult], None]] = None,
    ) -> Tuple[EvaluationSummary, list[SampleEvaluationResult]]:
        """Evaluate a collection of samples and aggregate breakdowns."""
        results: list[SampleEvaluationResult] = []
        total = len(samples)

        for i, sample in enumerate(samples):
            res = self.evaluate_sample(sample)
            results.append(res)
            if progress_callback is not None:
                progress_callback(i + 1, total, res)

        # Breakdowns
        by_dataset: dict[str, list[SampleEvaluationResult]] = {}
        by_category: dict[str, list[SampleEvaluationResult]] = {}
        by_degradation: dict[str, list[SampleEvaluationResult]] = {"clean": [], "degraded": []}

        for r in results:
            by_dataset.setdefault(r.dataset, []).append(r)
            by_category.setdefault(r.category, []).append(r)
            if r.degraded:
                by_degradation["degraded"].append(r)
            else:
                by_degradation["clean"].append(r)

        calibration_buckets = compute_confidence_calibration(results)
        calibration_dicts = [
            {
                "label": b.label,
                "min_conf": b.min_conf,
                "max_conf": b.max_conf,
                "block_count": b.block_count,
                "percent_of_blocks": b.percent_of_blocks,
                "mean_confidence": b.mean_confidence,
                "mean_cer": b.mean_cer,
                "mean_wer": b.mean_wer,
                "non_empty_rate": b.non_empty_rate,
                "total_chars": b.total_chars,
            }
            for b in calibration_buckets
        ]


        summary = EvaluationSummary(
            evaluation_timestamp=datetime.now(timezone.utc).isoformat(),
            python_version=platform.python_version(),
            backend_info=asdict(self.pipeline.backend.backend_info),
            preprocessing_options=asdict(self.pipeline.default_preprocessing),
            total_samples=total,
            overall=aggregate_group_metrics(results),
            by_dataset={k: aggregate_group_metrics(v) for k, v in by_dataset.items()},
            by_category={k: aggregate_group_metrics(v) for k, v in by_category.items()},
            by_degradation={k: aggregate_group_metrics(v) for k, v in by_degradation.items()},
            confidence_calibration=calibration_dicts,
        )

        return summary, results

    def export_results(
        self,
        summary: EvaluationSummary,
        results: Sequence[SampleEvaluationResult],
    ) -> dict[str, Path]:
        """Export summary.json, per_sample.csv, failures.csv, and README.md outside datasets/."""
        validate_safe_output_path(self.output_dir, self.dataset_root)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        exported_paths: dict[str, Path] = {}

        # 1. summary.json
        summary_path = self.output_dir / "summary.json"
        summary_path.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
        exported_paths["summary"] = summary_path

        # 2. per_sample.csv
        per_sample_path = self.output_dir / "per_sample.csv"
        fieldnames = [
            "sample_id",
            "dataset",
            "category",
            "degraded",
            "exact_match",
            "cer",
            "wer",
            "ocr_confidence",
            "num_blocks",
            "processing_time_ms",
            "ground_truth_length",
            "ocr_text_length",
            "error_message",
            "image_path",
        ]
        with open(per_sample_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                row = {k: getattr(r, k) for k in fieldnames}
                writer.writerow(row)
        exported_paths["per_sample"] = per_sample_path

        # 3. failures.csv (poor samples with CER > threshold or errors)
        failures_path = self.output_dir / "failures.csv"
        fail_fieldnames = fieldnames + ["ground_truth_snippet", "ocr_snippet"]
        with open(failures_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fail_fieldnames)
            writer.writeheader()
            for r in results:
                if r.cer > self.poor_cer_threshold or r.error_message is not None:
                    row = {k: getattr(r, k) for k in fieldnames}
                    # Compact snippets to avoid huge lines
                    row["ground_truth_snippet"] = r.ground_truth_text[:120].replace("\n", " ")
                    row["ocr_snippet"] = r.ocr_text[:120].replace("\n", " ")
                    writer.writerow(row)
        exported_paths["failures"] = failures_path

        # 4. README.md
        readme_path = self.output_dir / "README.md"
        readme_content = self._generate_readme(summary)
        readme_path.write_text(readme_content, encoding="utf-8")
        exported_paths["readme"] = readme_path

        return exported_paths

    def _generate_readme(self, summary: EvaluationSummary) -> str:
        """Generate human-readable Markdown evaluation report."""
        lines = [
            "# Member 3 — OCR Pipeline Evaluation Report",
            "",
            f"- **Timestamp**: `{summary.evaluation_timestamp}`",
            f"- **Python Version**: `{summary.python_version}`",
            f"- **Backend**: `{summary.backend_info.get('name')}` (Device: `{summary.backend_info.get('device')}`)",
            f"- **Total Samples Evaluated**: `{summary.total_samples}`",
            "",
            "## 1. Overall Performance",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Exact Match Rate | {summary.overall.exact_match_rate * 100:.2f}% ({summary.overall.exact_match_count}/{summary.overall.sample_count}) |",
            f"| Mean Character Error Rate (CER) | {summary.overall.mean_cer:.4f} |",
            f"| Mean Word Error Rate (WER) | {summary.overall.mean_wer:.4f} |",
            f"| Mean OCR Confidence | {summary.overall.mean_confidence:.4f} |",
            f"| Mean Processing Time | {summary.overall.mean_processing_time_ms:.2f} ms |",
            f"| Throughput | {summary.overall.images_per_second:.2f} img/s |",
            f"| Errors Encountered | {summary.overall.error_count} |",
            "",
            "## 2. Breakdown by Dataset",
            "",
            "| Dataset | Samples | Exact Match % | Mean CER | Mean WER | Mean Conf | Throughput (img/s) |",
            "|---|---|---|---|---|---|---|",
        ]
        for name, m in sorted(summary.by_dataset.items()):
            lines.append(
                f"| {name} | {m.sample_count} | {m.exact_match_rate*100:.1f}% | {m.mean_cer:.4f} | "
                f"{m.mean_wer:.4f} | {m.mean_confidence:.4f} | {m.images_per_second:.2f} |"
            )

        lines.extend([
            "",
            "## 3. Breakdown by Document Category",
            "",
            "| Category | Samples | Exact Match % | Mean CER | Mean WER | Mean Conf | Mean Time (ms) |",
            "|---|---|---|---|---|---|---|",
        ])
        for name, m in sorted(summary.by_category.items()):
            lines.append(
                f"| {name} | {m.sample_count} | {m.exact_match_rate*100:.1f}% | {m.mean_cer:.4f} | "
                f"{m.mean_wer:.4f} | {m.mean_confidence:.4f} | {m.mean_processing_time_ms:.2f} |"
            )

        lines.extend([
            "",
            "## 4. Breakdown by Degradation State",
            "",
            "| State | Samples | Exact Match % | Mean CER | Mean WER | Mean Conf |",
            "|---|---|---|---|---|---|",
        ])
        for name, m in sorted(summary.by_degradation.items()):
            lines.append(
                f"| {name} | {m.sample_count} | {m.exact_match_rate*100:.1f}% | {m.mean_cer:.4f} | "
                f"{m.mean_wer:.4f} | {m.mean_confidence:.4f} |"
            )

        lines.extend([
            "",
            "## 5. Confidence Calibration",
            "",
            "| Confidence Bin | Block Count | % of Blocks | Mean Confidence | Mean CER (this bin) | Mean WER (this bin) |",
            "|---|---|---|---|---|---|",
        ])
        for bucket in summary.confidence_calibration:
            lines.append(
                f"| {bucket['label']} | {bucket['block_count']} | {bucket.get('percent_of_blocks', 0.0):.2f}% | "
                f"{bucket['mean_confidence']:.4f} | {bucket.get('mean_cer', 0.0):.4f} | {bucket.get('mean_wer', 0.0):.4f} |"
            )

        lines.extend([
            "",
            "## 6. Artifact Files",
            "- `summary.json`: Complete JSON summary of metrics and configuration.",
            "- `per_sample.csv`: Granular row-by-row metric results for all samples.",
            "- `failures.csv`: Subset of samples exceeding CER threshold or encountering errors.",
            "",
        ])
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# CLI Entrypoint
# ──────────────────────────────────────────────────────────────────────────────

def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Member 3 OCR Pipeline on datasets/ocr.")
    parser.add_argument(
        "--dataset-root",
        default=str(get_datasets_dir() / "ocr"),
        help="Root directory containing FUNSD and synthetic_ocr_dataset",
    )
    parser.add_argument(
        "--output-dir",
        default=str(get_output_dir() / "evaluation"),
        help="Target output directory for evaluation artifacts (must be outside datasets/)",
    )
    parser.add_argument(
        "--det-model-dir",
        default=str(get_models_dir() / "paddleocr" / "PP-OCRv5_mobile_det_infer"),
        help="Local path to detection model",
    )
    parser.add_argument(
        "--rec-model-dir",
        default=str(get_models_dir() / "paddleocr" / "PP-OCRv5_mobile_rec_infer"),
        help="Local path to recognition model",
    )
    parser.add_argument("--limit", type=int, help="Optional limit on number of samples per split for quick evaluation")
    parser.add_argument(
        "--split",
        choices=("all", "synthetic", "funsd_train", "funsd_test"),
        default="all",
        help="Which dataset split(s) to evaluate",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_cli_parser().parse_args(argv)
    dataset_root = Path(args.dataset_root).resolve()
    output_dir = Path(args.output_dir).resolve()

    try:
        validate_safe_output_path(output_dir, dataset_root)
    except ValueError as exc:
        LOGGER.error("%s", exc)
        return 2

    # Collect samples based on split
    samples: list[EvaluationSample] = []
    synthetic_dir = dataset_root / "synthetic_ocr_dataset"
    funsd_dir = dataset_root / "FUNSD"

    if args.split in ("all", "synthetic") and synthetic_dir.is_dir():
        synth_samples = load_synthetic_dataset_samples(synthetic_dir)
        samples.extend(synth_samples[: args.limit] if args.limit else synth_samples)

    if args.split in ("all", "funsd_train") and funsd_dir.is_dir():
        funsd_train = load_funsd_dataset_samples(funsd_dir, split="training_data")
        samples.extend(funsd_train[: args.limit] if args.limit else funsd_train)

    if args.split in ("all", "funsd_test") and funsd_dir.is_dir():
        funsd_test = load_funsd_dataset_samples(funsd_dir, split="testing_data")
        samples.extend(funsd_test[: args.limit] if args.limit else funsd_test)

    if not samples:
        LOGGER.error("No evaluation samples found under %s", dataset_root)
        return 1

    LOGGER.info("Loaded %d samples for OCR evaluation", len(samples))

    # Configure local PaddleOCR backend
    det_dir = Path(args.det_model_dir)
    rec_dir = Path(args.rec_model_dir)
    config = PaddleOCRModelConfig(
        detection_model_dir=det_dir,
        recognition_model_dir=rec_dir,
        detection_model_name="PP-OCRv5_mobile_det",
        recognition_model_name="PP-OCRv5_mobile_rec",
        device="cpu",
        allow_model_download=False,
    )
    config.validate()

    backend = PaddleOCRBackend(config)
    pipeline = OCRPipeline(backend, default_preprocessing=PreprocessingOptions.document_ocr())
    evaluator = OCREvaluator(pipeline, output_dir=output_dir, dataset_root=dataset_root)

    def _on_progress(idx: int, total: int, r: SampleEvaluationResult) -> None:
        status = "OK" if r.error_message is None else "ERR"
        print(f"[{idx}/{total}] [{status}] {r.sample_id} ({r.category}) CER={r.cer:.3f} WER={r.wer:.3f} ({r.processing_time_ms:.0f}ms)")

    summary, results = evaluator.evaluate_all(samples, progress_callback=_on_progress)
    exported = evaluator.export_results(summary, results)

    print("\nEvaluation Completed!")
    print(f"Total Samples: {summary.total_samples}")
    print(f"Exact Match Rate: {summary.overall.exact_match_rate * 100:.2f}%")
    print(f"Mean CER: {summary.overall.mean_cer:.4f}")
    print(f"Mean WER: {summary.overall.mean_wer:.4f}")
    print(f"Mean Confidence: {summary.overall.mean_confidence:.4f}")
    print(f"Throughput: {summary.overall.images_per_second:.2f} images/sec")

    print("\nConfidence Calibration:")
    print(f"{'Confidence Bin':<22} | {'Blocks':<7} | {'% Blocks':<8} | {'Mean Conf':<9} | {'Mean CER':<9} | {'Mean WER':<9}")
    print("-" * 75)
    for b in summary.confidence_calibration:
        print(f"{b['label']:<22} | {b['block_count']:<7} | {b.get('percent_of_blocks', 0.0):<7.2f}% | {b['mean_confidence']:<9.4f} | {b.get('mean_cer', 0.0):<9.4f} | {b.get('mean_wer', 0.0):<9.4f}")

    print("\nExported Artifacts:")
    for k, p in exported.items():
        print(f"  {k}: {p}")

    return 0



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())
