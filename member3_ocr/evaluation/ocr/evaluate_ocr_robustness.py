from __future__ import annotations
from member3_ocr.evaluation.paths import get_project_root, get_output_dir, get_datasets_dir, get_models_dir
"""Adversarial OCR robustness harness for member3_ocr.

Tests the OCR pipeline against degenerate and adversarial inputs to verify
that all error paths produce structured Issue objects rather than crashes.

Safety guarantee: no dataset files are read or written by this script.
All adversarial images are created in a temporary directory at runtime.

Usage:
    python -m member3_ocr.evaluate_ocr_robustness
    python -m member3_ocr.evaluate_ocr_robustness --output-dir member3_ocr/output/robustness

Exit code: 0 if all cases pass, 1 if any case fails unexpectedly.
"""


import argparse
import json
import logging
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

LOGGER = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Data Models
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RobustnessCase:
    """Definition of one adversarial test case."""

    case_id: str
    description: str
    expected_behaviour: str  # "structured_error", "empty_result", "graceful"


@dataclass
class RobustnessResult:
    """Outcome of running one adversarial test case."""

    case_id: str
    description: str
    expected_behaviour: str
    outcome: str        # "PASS" | "FAIL"
    observed_behaviour: str
    elapsed_ms: float
    error_raised: str | None = None
    issue_codes: list[str] = field(default_factory=list)
    num_blocks: int = 0


# ──────────────────────────────────────────────────────────────────────────────
# Image Generators (all write to a temp path, never to datasets/)
# ──────────────────────────────────────────────────────────────────────────────

def _write_zero_byte_png(path: Path) -> Path:
    """Create a 0-byte file with .png extension."""
    path.write_bytes(b"")
    return path


def _write_corrupt_png_header(path: Path) -> Path:
    """Create a file with a valid PNG signature but corrupted IHDR chunk."""
    png_sig = b"\x89PNG\r\n\x1a\n"
    bad_ihdr = b"\x00\x00\x00\rIHDRCorruptPayload"
    path.write_bytes(png_sig + bad_ihdr)
    return path


def _write_all_white_png(path: Path, size: tuple[int, int] = (640, 480)) -> Path:
    """Create a blank white image with no text."""
    arr = np.full((size[1], size[0], 3), 255, dtype=np.uint8)
    Image.fromarray(arr).save(path)
    return path


def _write_solid_colour_png(path: Path, colour: tuple[int, int, int] = (80, 80, 80)) -> Path:
    """Create a solid non-white, non-black image with no text content."""
    arr = np.full((480, 640, 3), colour, dtype=np.uint8)
    Image.fromarray(arr).save(path)
    return path


def _write_sub_pixel_png(path: Path) -> Path:
    """Create a valid but extremely tiny 3x3 image."""
    arr = np.zeros((3, 3, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)
    return path


def _write_salt_pepper_noise(path: Path, size: tuple[int, int] = (640, 480)) -> Path:
    """Create a high-noise image with no recognisable text."""
    rng = np.random.default_rng(seed=42)
    arr = rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    arr[arr < 128] = 0
    arr[arr >= 128] = 255
    Image.fromarray(arr).save(path)
    return path


def _write_non_image_bytes_with_png_ext(path: Path) -> Path:
    """Write arbitrary non-image content with a .png extension."""
    path.write_bytes(b"This is definitely not a PNG image file\x00\x01\x02")
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Test Runner
# ──────────────────────────────────────────────────────────────────────────────

def run_robustness_case(
    case: RobustnessCase,
    image_path: Path,
    pipeline: Any,
) -> RobustnessResult:
    """Run a single adversarial case and classify the outcome."""
    start = time.perf_counter()
    error_raised: str | None = None
    issue_codes: list[str] = []
    num_blocks: int = 0
    observed: str = ""

    try:
        result = pipeline.process_image(source=image_path, document_id=case.case_id)
        if result.errors:
            issue_codes = [e.code for e in result.errors]
            observed = f"structured_error [{', '.join(issue_codes)}]"
        elif result.pages and result.pages[0].blocks:
            num_blocks = len(result.pages[0].blocks)
            observed = f"non_empty_result ({num_blocks} blocks)"
        else:
            observed = "empty_result (0 blocks)"
    except Exception as exc:
        error_raised = f"{type(exc).__name__}: {exc}"
        observed = f"unhandled_exception: {type(exc).__name__}"

    elapsed_ms = (time.perf_counter() - start) * 1000.0

    expected = case.expected_behaviour
    if expected == "structured_error":
        passed = bool(issue_codes) and error_raised is None
    elif expected == "empty_result":
        passed = num_blocks == 0 and error_raised is None
    elif expected == "graceful":
        # Any outcome except an unhandled crash is acceptable
        passed = error_raised is None
    else:
        passed = False

    return RobustnessResult(
        case_id=case.case_id,
        description=case.description,
        expected_behaviour=expected,
        outcome="PASS" if passed else "FAIL",
        observed_behaviour=observed,
        elapsed_ms=round(elapsed_ms, 2),
        error_raised=error_raised,
        issue_codes=issue_codes,
        num_blocks=num_blocks,
    )


def _print_result(r: RobustnessResult) -> None:
    symbol = "OK" if r.outcome == "PASS" else "FAIL"
    print(f"  [{symbol}] {r.case_id}: {r.outcome}")
    print(f"       Expected: {r.expected_behaviour}")
    print(f"       Observed: {r.observed_behaviour}")
    if r.error_raised:
        print(f"       Error   : {r.error_raised}")
    print(f"       Time    : {r.elapsed_ms:.0f}ms")


def run_all_robustness_cases(pipeline: Any, output_dir: Path) -> list[RobustnessResult]:
    """Run all 7 adversarial cases and return results."""
    results: list[RobustnessResult] = []

    cases: list[tuple[RobustnessCase, Callable[[Path], Path]]] = [
        (
            RobustnessCase("ADV-001", "Zero-byte file with .png extension", "structured_error"),
            _write_zero_byte_png,
        ),
        (
            RobustnessCase("ADV-002", "Corrupted PNG header (valid signature, bad IHDR)", "structured_error"),
            _write_corrupt_png_header,
        ),
        (
            RobustnessCase("ADV-003", "Non-image bytes with .png extension", "structured_error"),
            _write_non_image_bytes_with_png_ext,
        ),
        (
            RobustnessCase("ADV-004", "All-white image (no text content, 640x480)", "graceful"),
            _write_all_white_png,
        ),
        (
            RobustnessCase("ADV-005", "Solid grey image (no text, non-white, 640x480)", "graceful"),
            _write_solid_colour_png,
        ),
        (
            RobustnessCase("ADV-006", "Sub-pixel image (3x3 pixels)", "graceful"),
            _write_sub_pixel_png,
        ),
        (
            RobustnessCase("ADV-007", "Pure salt-and-pepper noise image (640x480)", "graceful"),
            _write_salt_pepper_noise,
        ),
    ]

    with tempfile.TemporaryDirectory(prefix="ocr_robustness_") as tmpdir:
        tmp_path = Path(tmpdir)
        print("\n=== OCR Adversarial Robustness Evaluation ===\n")

        for case, img_generator in cases:
            img_path = tmp_path / f"{case.case_id}.png"
            try:
                img_generator(img_path)
            except Exception as exc:
                LOGGER.error("Failed to generate test image for %s: %s", case.case_id, exc)
                continue

            result = run_robustness_case(case, img_path, pipeline)
            results.append(result)
            _print_result(result)
            print()

    passed = sum(1 for r in results if r.outcome == "PASS")
    failed = sum(1 for r in results if r.outcome == "FAIL")
    print(f"\n{'=' * 50}")
    print(f"Robustness: {passed}/{len(results)} cases PASSED, {failed} FAILED")

    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        "total_cases": len(results),
        "passed": passed,
        "failed": failed,
        "results": [
            {
                "case_id": r.case_id,
                "description": r.description,
                "expected_behaviour": r.expected_behaviour,
                "outcome": r.outcome,
                "observed_behaviour": r.observed_behaviour,
                "elapsed_ms": r.elapsed_ms,
                "error_raised": r.error_raised,
                "issue_codes": r.issue_codes,
                "num_blocks": r.num_blocks,
            }
            for r in results
        ],
    }
    report_path = output_dir / "robustness_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport: {report_path}")

    return results


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Adversarial OCR robustness evaluation for member3_ocr."
    )
    parser.add_argument(
        "--output-dir",
        default=str(get_output_dir() / "robustness"),
        help="Output directory for robustness report JSON",
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_cli_parser().parse_args(argv)
    output_dir = Path(args.output_dir).resolve()

    from member3_ocr.core.image_preprocessing import PreprocessingOptions
    from member3_ocr.core.ocr_pipeline import (
        OCRPipeline,
        PaddleOCRBackend,
        PaddleOCRModelConfig,
    )

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
    pipeline = OCRPipeline(backend, default_preprocessing=PreprocessingOptions.document_ocr())

    results = run_all_robustness_cases(pipeline, output_dir)
    return 1 if any(r.outcome == "FAIL" for r in results) else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())
