from __future__ import annotations
from member3_ocr.evaluation.paths import get_project_root, get_output_dir, get_datasets_dir, get_models_dir
"""Image preprocessing correctness evaluation for member3_ocr.

Validates that preprocessing algorithms produce measurable, correct improvements:
- Deskew: does it reduce the measured skew angle?
- Denoise: does it reduce local pixel variance in flat regions?
- CLAHE: does it increase histogram spread (contrast)?
- Downstream: does preprocessing improve OCR CER/WER on degraded samples?

Dataset: datasets/ocr/synthetic_ocr_dataset (36 images with documented
rotation_deg, noise_amount, gaussian_blur per-sample in manifest.json)

Safety guarantee: datasets/ is read-only. No files written inside it.

Usage:
    python -m member3_ocr.evaluate_preprocessing_correctness
    python -m member3_ocr.evaluate_preprocessing_correctness --output-dir member3_ocr/output/preprocessing
"""


import argparse
import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from member3_ocr.core.image_preprocessing import (
    ImagePreprocessor,
    PreprocessingOptions,
    estimate_skew_angle,
    preprocess_image,
    validate_image,
)

LOGGER = logging.getLogger(__name__)

PREPROCESSOR = ImagePreprocessor()


# ──────────────────────────────────────────────────────────────────────────────
# Metric Utilities
# ──────────────────────────────────────────────────────────────────────────────

def _as_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def measure_skew_angle(image: np.ndarray) -> float | None:
    """Return estimated skew angle in degrees, or None if not detectable."""
    return estimate_skew_angle(image, max_angle=15.0)


def measure_local_variance(image: np.ndarray, region_fraction: float = 0.2) -> float:
    """Measure mean local variance in a central flat region of the image.

    A high variance indicates noise; a lower value after denoising indicates
    that noise was actually removed.
    """
    gray = _as_gray(image)
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    dh = max(1, int(h * region_fraction / 2))
    dw = max(1, int(w * region_fraction / 2))
    patch = gray[cy - dh : cy + dh, cx - dw : cx + dw]
    if patch.size == 0:
        return 0.0
    # Local variance: variance within a 5x5 sliding kernel
    blurred = cv2.GaussianBlur(patch.astype(np.float32), (5, 5), 0)
    diff = patch.astype(np.float32) - blurred
    return float(np.mean(diff ** 2))


def measure_histogram_spread(image: np.ndarray) -> float:
    """Measure contrast as P95 - P5 percentile spread of pixel intensities.

    A higher spread indicates better contrast. CLAHE should increase this.
    """
    gray = _as_gray(image)
    flat = gray.flatten().astype(np.float32)
    p5 = float(np.percentile(flat, 5))
    p95 = float(np.percentile(flat, 95))
    return p95 - p5


# ──────────────────────────────────────────────────────────────────────────────
# Data Models
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SampleMeasurements:
    """Before/after preprocessing measurements for one synthetic sample."""

    sample_id: str
    category: str
    degraded: bool
    rotation_deg: float | None
    noise_amount: float | None
    gaussian_blur: float | None

    # Deskew measurement
    angle_before_deg: float | None
    angle_after_deskew_deg: float | None
    deskew_applied: bool

    # Denoise measurement (local variance)
    local_variance_before: float
    local_variance_after_denoise: float
    variance_delta_pct: float   # positive = reduction (good)

    # CLAHE measurement (histogram spread)
    hist_spread_before: float
    hist_spread_after_clahe: float
    spread_delta_pct: float     # positive = improvement (good)


@dataclass
class OCRDeltaMeasurement:
    """CER/WER with and without preprocessing for one sample."""

    sample_id: str
    cer_raw: float
    cer_preprocessed: float
    cer_delta: float     # negative = improvement
    wer_raw: float
    wer_preprocessed: float
    wer_delta: float     # negative = improvement
    error_raw: str | None
    error_preprocessed: str | None


@dataclass
class PreprocessingCorrectnessReport:
    """Full report of preprocessing correctness evaluation."""

    evaluation_timestamp: str
    dataset_path: str
    total_samples: int
    degraded_samples: int

    # Deskew stats
    deskew_samples_tested: int
    deskew_mean_angle_before: float
    deskew_mean_angle_after: float
    deskew_success_rate: float   # fraction where |angle_after| < |angle_before|

    # Denoise stats
    denoise_mean_variance_before: float
    denoise_mean_variance_after: float
    denoise_mean_reduction_pct: float

    # CLAHE stats
    clahe_mean_spread_before: float
    clahe_mean_spread_after: float
    clahe_mean_improvement_pct: float

    # OCR delta stats (if pipeline available)
    ocr_delta_available: bool
    ocr_mean_cer_raw: float = 0.0
    ocr_mean_cer_preprocessed: float = 0.0
    ocr_mean_cer_delta: float = 0.0
    ocr_mean_wer_raw: float = 0.0
    ocr_mean_wer_preprocessed: float = 0.0
    ocr_mean_wer_delta: float = 0.0

    per_sample: list[dict[str, Any]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.per_sample is None:
            self.per_sample = []


# ──────────────────────────────────────────────────────────────────────────────
# Dataset Loading
# ──────────────────────────────────────────────────────────────────────────────

def load_synthetic_manifest(dataset_dir: Path) -> list[dict[str, Any]]:
    """Load the synthetic_ocr_dataset manifest.json entries."""
    manifest_file = dataset_dir / "manifest.json"
    if not manifest_file.exists():
        raise FileNotFoundError(f"manifest.json not found in {dataset_dir}")
    entries = json.loads(manifest_file.read_text(encoding="utf-8"))
    LOGGER.info("Loaded %d entries from manifest", len(entries))
    return entries


def _find_image(dataset_dir: Path, doc_id: str, degraded: bool) -> Path | None:
    images_dir = dataset_dir / "images"
    for ext in (".jpg", ".jpeg", ".png"):
        candidate = images_dir / f"{doc_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def _load_image_array(path: Path) -> np.ndarray | None:
    encoded = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if image is None or image.size == 0:
        return None
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    return image


# ──────────────────────────────────────────────────────────────────────────────
# Per-Sample Measurement
# ──────────────────────────────────────────────────────────────────────────────

def measure_sample(entry: dict[str, Any], dataset_dir: Path) -> SampleMeasurements | None:
    """Compute before/after preprocessing measurements for one sample."""
    doc_id = entry["doc_id"]
    category = entry.get("category", "unknown")
    degraded = bool(entry.get("degraded", False))
    deg_meta = entry.get("degradation") or {}

    rotation_deg = deg_meta.get("rotation_deg")
    noise_amount = deg_meta.get("noise_amount")
    gaussian_blur = deg_meta.get("gaussian_blur")

    img_path = _find_image(dataset_dir, doc_id, degraded)
    if img_path is None:
        LOGGER.warning("Image not found for %s, skipping", doc_id)
        return None

    image = _load_image_array(img_path)
    if image is None:
        LOGGER.warning("Could not load image for %s, skipping", doc_id)
        return None

    # 1. Deskew measurement
    angle_before = measure_skew_angle(image)
    deskew_applied = False
    angle_after = None
    if angle_before is not None and abs(angle_before) >= 0.3:
        deskewed_result = PREPROCESSOR.process_array(
            image.copy(),
            PreprocessingOptions(deskew=True, deskew_min_angle=0.3, deskew_max_angle=15.0),
        )
        angle_after = measure_skew_angle(deskewed_result.image)
        deskew_applied = True

    # 2. Denoise measurement (local variance)
    variance_before = measure_local_variance(image)
    denoised_result = PREPROCESSOR.process_array(
        image.copy(),
        PreprocessingOptions(denoise=True, denoise_strength=5),
    )
    variance_after = measure_local_variance(denoised_result.image)
    variance_delta_pct = 0.0
    if variance_before > 0:
        variance_delta_pct = round((variance_before - variance_after) / variance_before * 100, 2)

    # 3. CLAHE measurement (histogram spread)
    spread_before = measure_histogram_spread(image)
    clahe_result = PREPROCESSOR.process_array(
        image.copy(),
        PreprocessingOptions(contrast_enhancement=True),
    )
    spread_after = measure_histogram_spread(clahe_result.image)
    spread_delta_pct = 0.0
    if spread_before > 0:
        spread_delta_pct = round((spread_after - spread_before) / spread_before * 100, 2)

    return SampleMeasurements(
        sample_id=doc_id,
        category=category,
        degraded=degraded,
        rotation_deg=rotation_deg,
        noise_amount=noise_amount,
        gaussian_blur=gaussian_blur,
        angle_before_deg=round(angle_before, 3) if angle_before is not None else None,
        angle_after_deskew_deg=round(angle_after, 3) if angle_after is not None else None,
        deskew_applied=deskew_applied,
        local_variance_before=round(variance_before, 4),
        local_variance_after_denoise=round(variance_after, 4),
        variance_delta_pct=variance_delta_pct,
        hist_spread_before=round(spread_before, 2),
        hist_spread_after_clahe=round(spread_after, 2),
        spread_delta_pct=spread_delta_pct,
    )


# ──────────────────────────────────────────────────────────────────────────────
# OCR Delta (optional — requires models to be available)
# ──────────────────────────────────────────────────────────────────────────────

def measure_ocr_delta(
    entries: list[dict[str, Any]],
    dataset_dir: Path,
    pipeline: Any,
    limit: int | None = None,
) -> list[OCRDeltaMeasurement]:
    """Measure CER/WER with and without preprocessing for degraded samples."""
    from member3_ocr.evaluation.ocr.ocr_evaluator import compute_cer, compute_wer

    results: list[OCRDeltaMeasurement] = []
    text_dir = dataset_dir / "text"
    count = 0

    for entry in entries:
        if limit is not None and count >= limit:
            break

        doc_id = entry["doc_id"]
        img_path = _find_image(dataset_dir, doc_id, entry.get("degraded", False))
        if img_path is None:
            continue

        gt_path = text_dir / f"{doc_id}.txt"
        if not gt_path.exists():
            continue

        gt_text = gt_path.read_text(encoding="utf-8").strip()
        if not gt_text:
            continue

        # Raw OCR (no preprocessing)
        cer_raw, wer_raw, err_raw = 1.0, 1.0, None
        try:
            raw_opts = PreprocessingOptions()  # No ops
            result_raw = pipeline.process_image(
                source=img_path, document_id=f"{doc_id}_raw",
                preprocessing_options=raw_opts,
            )
            if result_raw.errors:
                err_raw = "; ".join(e.message for e in result_raw.errors)
            elif result_raw.pages:
                ocr_text_raw = result_raw.pages[0].text
                cer_raw = compute_cer(gt_text, ocr_text_raw)
                wer_raw = compute_wer(gt_text, ocr_text_raw)
        except Exception as exc:
            err_raw = str(exc)

        # Preprocessed OCR (denoise + deskew + CLAHE)
        cer_pre, wer_pre, err_pre = 1.0, 1.0, None
        try:
            pre_opts = PreprocessingOptions(
                denoise=True, denoise_strength=3,
                deskew=True, deskew_min_angle=0.3, deskew_max_angle=10.0,
                contrast_enhancement=True,
            )
            result_pre = pipeline.process_image(
                source=img_path, document_id=f"{doc_id}_preprocessed",
                preprocessing_options=pre_opts,
            )
            if result_pre.errors:
                err_pre = "; ".join(e.message for e in result_pre.errors)
            elif result_pre.pages:
                ocr_text_pre = result_pre.pages[0].text
                cer_pre = compute_cer(gt_text, ocr_text_pre)
                wer_pre = compute_wer(gt_text, ocr_text_pre)
        except Exception as exc:
            err_pre = str(exc)

        results.append(OCRDeltaMeasurement(
            sample_id=doc_id,
            cer_raw=round(cer_raw, 4),
            cer_preprocessed=round(cer_pre, 4),
            cer_delta=round(cer_pre - cer_raw, 4),
            wer_raw=round(wer_raw, 4),
            wer_preprocessed=round(wer_pre, 4),
            wer_delta=round(wer_pre - wer_raw, 4),
            error_raw=err_raw,
            error_preprocessed=err_pre,
        ))
        count += 1
        print(f"  [{count}] {doc_id}: CER {cer_raw:.3f}→{cer_pre:.3f} (Δ{cer_pre-cer_raw:+.3f})  "
              f"WER {wer_raw:.3f}→{wer_pre:.3f} (Δ{wer_pre-wer_raw:+.3f})")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Report Builder
# ──────────────────────────────────────────────────────────────────────────────

def build_report(
    measurements: list[SampleMeasurements],
    dataset_dir: Path,
    ocr_deltas: list[OCRDeltaMeasurement] | None = None,
) -> PreprocessingCorrectnessReport:
    """Aggregate measurements into a summary report."""
    total = len(measurements)
    degraded = sum(1 for m in measurements if m.degraded)

    # Deskew
    deskew_tested = [m for m in measurements if m.deskew_applied and m.angle_before_deg is not None]
    deskew_success = sum(
        1 for m in deskew_tested
        if m.angle_after_deskew_deg is not None
        and abs(m.angle_after_deskew_deg) < abs(m.angle_before_deg)
    )
    mean_angle_before = float(np.mean([abs(m.angle_before_deg) for m in deskew_tested])) if deskew_tested else 0.0
    mean_angle_after = float(np.mean([
        abs(m.angle_after_deskew_deg)
        for m in deskew_tested
        if m.angle_after_deskew_deg is not None
    ])) if deskew_tested else 0.0

    # Denoise
    var_before = float(np.mean([m.local_variance_before for m in measurements]))
    var_after = float(np.mean([m.local_variance_after_denoise for m in measurements]))
    var_reduction_pct = float(np.mean([m.variance_delta_pct for m in measurements]))

    # CLAHE
    spread_before = float(np.mean([m.hist_spread_before for m in measurements]))
    spread_after = float(np.mean([m.hist_spread_after_clahe for m in measurements]))
    spread_improvement_pct = float(np.mean([m.spread_delta_pct for m in measurements]))

    # OCR delta
    ocr_available = bool(ocr_deltas)
    cer_raw_mean, cer_pre_mean, cer_delta_mean = 0.0, 0.0, 0.0
    wer_raw_mean, wer_pre_mean, wer_delta_mean = 0.0, 0.0, 0.0
    if ocr_deltas:
        cer_raw_mean = float(np.mean([d.cer_raw for d in ocr_deltas]))
        cer_pre_mean = float(np.mean([d.cer_preprocessed for d in ocr_deltas]))
        cer_delta_mean = float(np.mean([d.cer_delta for d in ocr_deltas]))
        wer_raw_mean = float(np.mean([d.wer_raw for d in ocr_deltas]))
        wer_pre_mean = float(np.mean([d.wer_preprocessed for d in ocr_deltas]))
        wer_delta_mean = float(np.mean([d.wer_delta for d in ocr_deltas]))

    per_sample = [asdict(m) for m in measurements]
    if ocr_deltas:
        delta_map = {d.sample_id: asdict(d) for d in ocr_deltas}
        for entry in per_sample:
            entry["ocr_delta"] = delta_map.get(entry["sample_id"])

    return PreprocessingCorrectnessReport(
        evaluation_timestamp=datetime.now(timezone.utc).isoformat(),
        dataset_path=str(dataset_dir),
        total_samples=total,
        degraded_samples=degraded,
        deskew_samples_tested=len(deskew_tested),
        deskew_mean_angle_before=round(mean_angle_before, 3),
        deskew_mean_angle_after=round(mean_angle_after, 3),
        deskew_success_rate=round(deskew_success / len(deskew_tested), 4) if deskew_tested else 0.0,
        denoise_mean_variance_before=round(var_before, 4),
        denoise_mean_variance_after=round(var_after, 4),
        denoise_mean_reduction_pct=round(var_reduction_pct, 2),
        clahe_mean_spread_before=round(spread_before, 2),
        clahe_mean_spread_after=round(spread_after, 2),
        clahe_mean_improvement_pct=round(spread_improvement_pct, 2),
        ocr_delta_available=ocr_available,
        ocr_mean_cer_raw=round(cer_raw_mean, 4),
        ocr_mean_cer_preprocessed=round(cer_pre_mean, 4),
        ocr_mean_cer_delta=round(cer_delta_mean, 4),
        ocr_mean_wer_raw=round(wer_raw_mean, 4),
        ocr_mean_wer_preprocessed=round(wer_pre_mean, 4),
        ocr_mean_wer_delta=round(wer_delta_mean, 4),
        per_sample=per_sample,
    )


def _print_report(report: PreprocessingCorrectnessReport) -> None:
    print("\n" + "=" * 60)
    print("PREPROCESSING CORRECTNESS EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nDataset   : {report.dataset_path}")
    print(f"Samples   : {report.total_samples} total ({report.degraded_samples} degraded)")

    print("\n--- Deskew ---")
    print(f"  Samples tested          : {report.deskew_samples_tested}")
    print(f"  Mean angle before       : {report.deskew_mean_angle_before:.3f}°")
    print(f"  Mean angle after deskew : {report.deskew_mean_angle_after:.3f}°")
    print(f"  Success rate            : {report.deskew_success_rate*100:.1f}%")

    print("\n--- Denoise ---")
    print(f"  Mean local variance before : {report.denoise_mean_variance_before:.4f}")
    print(f"  Mean local variance after  : {report.denoise_mean_variance_after:.4f}")
    verdict = "REDUCED" if report.denoise_mean_reduction_pct > 0 else "INCREASED"
    print(f"  Mean variance change       : {report.denoise_mean_reduction_pct:+.1f}% ({verdict})")

    print("\n--- CLAHE Contrast Enhancement ---")
    print(f"  Mean spread before : {report.clahe_mean_spread_before:.2f}")
    print(f"  Mean spread after  : {report.clahe_mean_spread_after:.2f}")
    verdict_c = "IMPROVED" if report.clahe_mean_improvement_pct > 0 else "REDUCED"
    print(f"  Mean change        : {report.clahe_mean_improvement_pct:+.1f}% ({verdict_c})")

    if report.ocr_delta_available:
        print("\n--- OCR CER/WER Delta (raw vs. fully preprocessed) ---")
        print(f"  Mean CER: {report.ocr_mean_cer_raw:.4f} → {report.ocr_mean_cer_preprocessed:.4f}  "
              f"(Δ {report.ocr_mean_cer_delta:+.4f})")
        print(f"  Mean WER: {report.ocr_mean_wer_raw:.4f} → {report.ocr_mean_wer_preprocessed:.4f}  "
              f"(Δ {report.ocr_mean_wer_delta:+.4f})")
    print()


def export_report(report: PreprocessingCorrectnessReport, output_dir: Path) -> Path:
    """Export the report JSON to output_dir (must be outside datasets/)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dict = asdict(report)
    report_path = output_dir / "preprocessing_correctness_report.json"
    report_path.write_text(json.dumps(report_dict, indent=2, ensure_ascii=False), encoding="utf-8")
    return report_path


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate image preprocessing correctness on synthetic_ocr_dataset."
    )
    parser.add_argument(
        "--dataset-dir",
        default=str(get_datasets_dir() / "ocr" / "synthetic_ocr_dataset"),
        help="Path to synthetic_ocr_dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        default=str(get_output_dir() / "preprocessing"),
        help="Output directory (must be outside datasets/)",
    )
    parser.add_argument(
        "--with-ocr-delta",
        action="store_true",
        help="Run before/after OCR CER/WER comparison (requires local PaddleOCR models)",
    )
    parser.add_argument(
        "--det-model-dir",
        default=str(get_models_dir() / "paddleocr" / "PP-OCRv5_mobile_det_infer"),
    )
    parser.add_argument(
        "--rec-model-dir",
        default=str(get_models_dir() / "paddleocr" / "PP-OCRv5_mobile_rec_infer"),
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit the number of samples for quick validation",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_cli_parser().parse_args(argv)
    dataset_dir = Path(args.dataset_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    # Safety: output must not be inside datasets/
    repo_datasets = get_datasets_dir()
    try:
        output_dir.relative_to(repo_datasets.resolve())
        print(f"ERROR: output_dir {output_dir} is inside datasets/ — not allowed.")
        return 2
    except ValueError:
        pass  # Safe — output is outside datasets/

    if not dataset_dir.is_dir():
        print(f"ERROR: dataset_dir not found: {dataset_dir}")
        return 1

    print(f"\nLoading synthetic manifest from {dataset_dir} ...")
    entries = load_synthetic_manifest(dataset_dir)
    if args.limit:
        entries = entries[: args.limit]

    print(f"Measuring {len(entries)} samples (deskew, denoise, CLAHE) ...")
    measurements: list[SampleMeasurements] = []
    for i, entry in enumerate(entries, 1):
        m = measure_sample(entry, dataset_dir)
        if m is not None:
            measurements.append(m)
            print(
                f"  [{i}/{len(entries)}] {m.sample_id}: "
                f"angle={m.angle_before_deg}° var_delta={m.variance_delta_pct:+.1f}% "
                f"spread_delta={m.spread_delta_pct:+.1f}%"
            )

    if not measurements:
        print("No measurements collected. Check dataset_dir path.")
        return 1

    ocr_deltas: list[OCRDeltaMeasurement] | None = None
    if args.with_ocr_delta:
        print("\nRunning OCR CER/WER before/after comparison ...")
        from member3_ocr.core.ocr_pipeline import OCRPipeline, PaddleOCRBackend, PaddleOCRModelConfig
        config = PaddleOCRModelConfig(
            detection_model_dir=Path(args.det_model_dir),
            recognition_model_dir=Path(args.rec_model_dir),
            detection_model_name="PP-OCRv5_mobile_det",
            recognition_model_name="PP-OCRv5_mobile_rec",
            device="cpu",
            allow_model_download=False,
        )
        config.validate()
        backend = PaddleOCRBackend(config)
        pipeline = OCRPipeline(backend, default_preprocessing=PreprocessingOptions())
        ocr_deltas = measure_ocr_delta(entries, dataset_dir, pipeline, limit=args.limit)

    report = build_report(measurements, dataset_dir, ocr_deltas)
    _print_report(report)
    report_path = export_report(report, output_dir)
    print(f"Report saved: {report_path}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())
