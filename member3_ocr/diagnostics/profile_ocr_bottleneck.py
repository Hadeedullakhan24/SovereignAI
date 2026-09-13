from __future__ import annotations
from member3_ocr.evaluation.paths import get_project_root, get_output_dir, get_datasets_dir, get_models_dir
"""Step 0: Profile real OCR bottleneck before building chunking logic.

Measures wall-clock time and basic output statistics for:
  1. image_preprocessing.py alone (load + resize)
  2. Full OCRPipeline.process_image() on a small handwritten-note image
  3. OCRPipeline for a PDF (first page only) via page-capped rendering

Outputs JSON to:
  member3_ocr/output/evaluation/document_parser_real/step0_profiling.json

Usage:
  python member3_ocr/profile_ocr_bottleneck.py

Read-only guarantee:
  Reads from datasets/ but never writes to it.
  All output goes to member3_ocr/output/evaluation/document_parser_real/.
"""


import gc
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = get_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Windows stdout UTF-8 guard
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

OUTPUT_DIR = PROJECT_ROOT / "member3_ocr" / "output" / "evaluation" / "document_parser_real"
OUTPUT_JSON = OUTPUT_DIR / "step0_profiling.json"

# ── Test assets (read-only) ───────────────────────────────────────────────────
TEST_IMAGE = PROJECT_ROOT / "datasets" / "handwritten_notes" / "HN_0002_quick_Reactor_R516.jpg"
TEST_IMAGE_MEDIUM = PROJECT_ROOT / "datasets" / "handwritten_notes" / "HN_0005_structured_Pump_P203.jpg"
TEST_PDF = PROJECT_ROOT / "datasets" / "safety_docs" / "OISD-STD-105.pdf"

# ── PaddleOCR model paths (same as test_form_extractor.py real integration test)
DET_DIR = PROJECT_ROOT / "member3_ocr" / "models" / "paddleocr" / "PP-OCRv5_mobile_det_infer"
REC_DIR = PROJECT_ROOT / "member3_ocr" / "models" / "paddleocr" / "PP-OCRv5_mobile_rec_infer"


def _check_prerequisites() -> dict:
    """Check which test assets and models are available."""
    return {
        "test_image_small_exists": TEST_IMAGE.exists(),
        "test_image_medium_exists": TEST_IMAGE_MEDIUM.exists(),
        "test_pdf_exists": TEST_PDF.exists(),
        "paddle_det_model_exists": DET_DIR.exists(),
        "paddle_rec_model_exists": REC_DIR.exists(),
        "models_available": DET_DIR.exists() and REC_DIR.exists(),
    }


def profile_preprocess_only(image_path: Path) -> dict:
    """Profile image_preprocessing.py standalone (no OCR)."""
    from member3_ocr.core.image_preprocessing import PreprocessingOptions, preprocess_image

    opts = PreprocessingOptions.document_ocr(max_width=1600, max_height=1200)
    t0 = time.perf_counter()
    result = preprocess_image(image_path, opts, save_output=False)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "preprocess_only_ms": round(elapsed_ms, 2),
        "original_size": f"{result.original_width}x{result.original_height}",
        "processed_size": f"{result.processed_width}x{result.processed_height}",
        "operations_applied": result.operations_applied,
        "test_image": str(image_path.relative_to(PROJECT_ROOT)),
    }


def profile_ocr_image(image_path: Path, det_dir: Path, rec_dir: Path) -> dict:
    """Profile full OCRPipeline.process_image() including preprocessing."""
    from member3_ocr.core.ocr_pipeline import (
        OCRPipeline, PaddleOCRBackend, PaddleOCRModelConfig, PreprocessingOptions
    )

    config = PaddleOCRModelConfig(
        detection_model_dir=det_dir,
        recognition_model_dir=rec_dir,
    )
    backend = PaddleOCRBackend(config)
    pipeline = OCRPipeline(
        backend,
        default_preprocessing=PreprocessingOptions.document_ocr(max_width=1600, max_height=1200),
        extract_key_values=True,
        extract_tables=False,
    )

    # Warm-up: initialize backend (loads model into memory)
    print("  [Step 0] Warming up PaddleOCR backend (loading model)...")
    t_warm = time.perf_counter()
    backend.initialize()
    warm_ms = (time.perf_counter() - t_warm) * 1000.0
    print(f"  [Step 0] Backend warm-up: {warm_ms:.0f} ms")

    # Timed run
    t0 = time.perf_counter()
    result = pipeline.process_image(image_path)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    page = result.pages[0] if result.pages else None
    blocks = len(page.blocks) if page else 0
    chars = sum(len(b.text) for b in page.blocks) if page else 0
    avg_conf = (
        sum(b.confidence for b in page.blocks if b.confidence is not None) / max(blocks, 1)
        if page else 0.0
    )
    errors = [e.message for e in result.errors]

    return {
        "ocr_image_ms": round(elapsed_ms, 2),
        "backend_warmup_ms": round(warm_ms, 2),
        "ocr_blocks_extracted": blocks,
        "ocr_chars_extracted": chars,
        "avg_ocr_confidence": round(avg_conf, 4),
        "errors": errors,
        "test_image": str(image_path.relative_to(PROJECT_ROOT)),
    }


def profile_ocr_pdf_first_page(pdf_path: Path, det_dir: Path, rec_dir: Path) -> dict:
    """Profile OCR on just the first page of a PDF (no warm-up needed after image test)."""
    from member3_ocr.core.ocr_pipeline import (
        OCRPipeline, PaddleOCRBackend, PaddleOCRModelConfig, PreprocessingOptions
    )
    from member3_ocr.core.pdf_rendering import render_pdf_pages

    # Render only first page (DPI=150 for validation speed)
    t_render = time.perf_counter()
    rendered = render_pdf_pages(pdf_path, dpi=150)
    render_ms = (time.perf_counter() - t_render) * 1000.0
    total_pages = len(rendered)
    first_page = rendered[0] if rendered else None

    if first_page is None or first_page.image is None:
        return {
            "ocr_pdf_first_page_ms": None,
            "pdf_render_ms": round(render_ms, 2),
            "pdf_total_pages": 0,
            "error": "PDF render produced no pages or first page has no image",
            "test_pdf": str(pdf_path.relative_to(PROJECT_ROOT)),
        }

    config = PaddleOCRModelConfig(
        detection_model_dir=det_dir,
        recognition_model_dir=rec_dir,
    )
    backend = PaddleOCRBackend(config)
    pipeline = OCRPipeline(
        backend,
        default_preprocessing=PreprocessingOptions.document_ocr(max_width=1600, max_height=1200),
        extract_key_values=True,
        extract_tables=False,
    )
    backend.initialize()

    import numpy as np
    from member3_ocr.core.image_preprocessing import PreprocessingOptions as PrepOpts, preprocess_image

    # preprocess the rendered page image
    t0 = time.perf_counter()
    prep_opts = PrepOpts.document_ocr(max_width=1600, max_height=1200)

    # The rendered page is a numpy array — wrap it in a PreprocessingResult-compatible call
    # by saving to a temp buffer (in-memory only, not to disk)
    from member3_ocr.core.image_preprocessing import ImagePreprocessor
    preprocessor = ImagePreprocessor()
    prep_result = preprocessor.process_array(
        first_page.image,
        source_path=pdf_path,
        options=prep_opts,
    )
    page_result = pipeline.process_image(prep_result)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    page = page_result.pages[0] if page_result.pages else None
    blocks = len(page.blocks) if page else 0
    chars = sum(len(b.text) for b in page.blocks) if page else 0
    errors = [e.message for e in page_result.errors]

    return {
        "ocr_pdf_first_page_ms": round(elapsed_ms, 2),
        "pdf_render_ms": round(render_ms, 2),
        "pdf_total_pages": total_pages,
        "pdf_page1_image_size": f"{first_page.image.shape[1]}x{first_page.image.shape[0]}",
        "ocr_blocks_extracted": blocks,
        "ocr_chars_extracted": chars,
        "errors": errors,
        "test_pdf": str(pdf_path.relative_to(PROJECT_ROOT)),
        "rendered_dpi": 150,
    }


def derive_session_eta(ocr_image_ms: float, ocr_pdf_first_page_ms: float | None) -> dict:
    """Derive per-document time estimate and session sizing."""
    # Conservative estimate: use image OCR as baseline
    # PDF pages are similar cost; cap at 3 pages → 3x image cost
    pdf_ms = ocr_pdf_first_page_ms if ocr_pdf_first_page_ms else ocr_image_ms * 1.5
    per_pdf_doc_ms = pdf_ms * 3  # 3 pages cap
    per_image_doc_ms = ocr_image_ms

    # Manifest: 6 image docs + 1 PDF doc (3-page cap) + 1 unavailable
    total_ms_estimate = 6 * per_image_doc_ms + 1 * per_pdf_doc_ms
    total_min_estimate = total_ms_estimate / 60_000.0

    docs_per_15min_image = int(15 * 60_000 / per_image_doc_ms) if per_image_doc_ms > 0 else 1
    docs_per_15min_mixed = max(1, int(15 * 60_000 / max(per_image_doc_ms, per_pdf_doc_ms / 3)))

    return {
        "per_image_doc_estimate_ms": round(per_image_doc_ms, 0),
        "per_pdf_page_estimate_ms": round(pdf_ms, 0),
        "per_pdf_doc_3page_estimate_ms": round(per_pdf_doc_ms, 0),
        "full_7doc_estimate_minutes": round(total_min_estimate, 1),
        "docs_per_15min_image_only": docs_per_15min_image,
        "session_recommendation": (
            f"Run 2-3 image docs per session (~{docs_per_15min_mixed} fit in 15 min). "
            f"PDF doc (3-page cap) counts as ~{round(per_pdf_doc_ms/per_image_doc_ms, 1):.1f}x an image doc. "
            f"Full 7-doc run estimated at {round(total_min_estimate, 1)} min."
        ),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prereqs = _check_prerequisites()

    print("\n" + "=" * 65)
    print("  Step 0 — OCR Bottleneck Profiling")
    print("=" * 65)
    for k, v in prereqs.items():
        status = "OK" if v else "MISSING"
        print(f"  {k:<40} {status}")
    print()

    result: dict = {
        "profile_timestamp": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
        "prerequisites": prereqs,
    }

    if not prereqs["models_available"]:
        print("  [SKIP] PaddleOCR models not found.")
        print(f"  Expected: {DET_DIR}")
        print(f"  Expected: {REC_DIR}")
        result["ocr_available"] = False
        result["note"] = "Models not found; profiling skipped. Cannot run real-document validation."
        _write(result)
        return

    result["ocr_available"] = True

    # ── Stage 1: Preprocessing only ─────────────────────────────────────────
    image_for_preprocess = TEST_IMAGE if TEST_IMAGE.exists() else TEST_IMAGE_MEDIUM
    if image_for_preprocess.exists():
        print(f"[Stage 1] Profiling image_preprocessing.py on {image_for_preprocess.name} ...")
        preprocess_result = profile_preprocess_only(image_for_preprocess)
        result["stage1_preprocess_only"] = preprocess_result
        print(f"  preprocess_only_ms: {preprocess_result['preprocess_only_ms']:.1f}")
        print(f"  original_size:      {preprocess_result['original_size']}")
        print(f"  processed_size:     {preprocess_result['processed_size']}")
    else:
        print("[Stage 1] SKIP — test image not found")
        result["stage1_preprocess_only"] = {"error": "test image not found"}
        preprocess_result = {}

    print()

    # ── Stage 2: Full OCR on image ───────────────────────────────────────────
    image_for_ocr = TEST_IMAGE_MEDIUM if TEST_IMAGE_MEDIUM.exists() else TEST_IMAGE
    if image_for_ocr.exists():
        print(f"[Stage 2] Profiling OCRPipeline.process_image() on {image_for_ocr.name} ...")
        try:
            ocr_result = profile_ocr_image(image_for_ocr, DET_DIR, REC_DIR)
            result["stage2_ocr_image"] = ocr_result
            print(f"  ocr_image_ms:        {ocr_result['ocr_image_ms']:.1f}")
            print(f"  backend_warmup_ms:   {ocr_result['backend_warmup_ms']:.1f}")
            print(f"  blocks_extracted:    {ocr_result['ocr_blocks_extracted']}")
            print(f"  chars_extracted:     {ocr_result['ocr_chars_extracted']}")
            print(f"  avg_confidence:      {ocr_result['avg_ocr_confidence']:.3f}")
            if ocr_result.get("errors"):
                print(f"  errors:              {ocr_result['errors']}")
        except Exception as exc:
            print(f"  ERROR: {exc}")
            result["stage2_ocr_image"] = {"error": str(exc)}
            ocr_result = {}
    else:
        print("[Stage 2] SKIP — test image not found")
        result["stage2_ocr_image"] = {"error": "test image not found"}
        ocr_result = {}

    print()

    # ── Stage 3: PDF first page ──────────────────────────────────────────────
    if TEST_PDF.exists():
        print(f"[Stage 3] Profiling OCR on first page of {TEST_PDF.name} (dpi=150) ...")
        try:
            pdf_result = profile_ocr_pdf_first_page(TEST_PDF, DET_DIR, REC_DIR)
            result["stage3_ocr_pdf_page1"] = pdf_result
            print(f"  pdf_render_ms:           {pdf_result['pdf_render_ms']:.1f}")
            print(f"  pdf_total_pages:         {pdf_result.get('pdf_total_pages', '?')}")
            print(f"  pdf_page1_image_size:    {pdf_result.get('pdf_page1_image_size', '?')}")
            if pdf_result.get("ocr_pdf_first_page_ms") is not None:
                print(f"  ocr_pdf_first_page_ms:   {pdf_result['ocr_pdf_first_page_ms']:.1f}")
                print(f"  blocks_extracted:        {pdf_result.get('ocr_blocks_extracted', 0)}")
            if pdf_result.get("errors"):
                print(f"  errors:                  {pdf_result['errors']}")
        except Exception as exc:
            import traceback
            print(f"  ERROR: {exc}")
            traceback.print_exc()
            result["stage3_ocr_pdf_page1"] = {"error": str(exc)}
            pdf_result = {}
    else:
        print("[Stage 3] SKIP — PDF not found")
        result["stage3_ocr_pdf_page1"] = {"error": "PDF not found"}
        pdf_result = {}

    print()

    # ── Session ETA ──────────────────────────────────────────────────────────
    ocr_ms = ocr_result.get("ocr_image_ms")
    pdf_ms = pdf_result.get("ocr_pdf_first_page_ms")

    if ocr_ms:
        eta = derive_session_eta(ocr_ms, pdf_ms)
        result["session_eta"] = eta
        print("[Session ETA]")
        print(f"  per-image doc:     ~{eta['per_image_doc_estimate_ms']:.0f} ms")
        print(f"  per-PDF page:      ~{eta['per_pdf_page_estimate_ms']:.0f} ms")
        print(f"  full 7-doc run:    ~{eta['full_7doc_estimate_minutes']:.1f} min")
        print(f"  recommendation:    {eta['session_recommendation']}")
    else:
        result["session_eta"] = {"note": "Could not derive ETA — OCR profiling failed or was skipped."}

    print()

    # ── Bottleneck summary ───────────────────────────────────────────────────
    preprocess_ms = preprocess_result.get("preprocess_only_ms", 0.0) or 0.0
    ocr_ms_val = ocr_result.get("ocr_image_ms", 0.0) or 0.0
    warmup_ms = ocr_result.get("backend_warmup_ms", 0.0) or 0.0
    pure_ocr_ms = ocr_ms_val - preprocess_ms if ocr_ms_val > preprocess_ms else ocr_ms_val

    if ocr_ms_val > 0:
        preprocess_pct = round(preprocess_ms / ocr_ms_val * 100, 1)
        ocr_pct = round(pure_ocr_ms / ocr_ms_val * 100, 1)
        result["bottleneck_analysis"] = {
            "preprocess_pct": preprocess_pct,
            "ocr_recognition_pct": ocr_pct,
            "backend_warmup_ms": round(warmup_ms, 0),
            "conclusion": (
                "OCR recognition is the bottleneck."
                if pure_ocr_ms > preprocess_ms
                else "Preprocessing is comparable to OCR — unexpected, check image size."
            ),
        }
        print("[Bottleneck Analysis]")
        print(f"  preprocessing:     ~{preprocess_pct}% of total")
        print(f"  ocr_recognition:   ~{ocr_pct}% of total")
        print(f"  backend_warmup:    {warmup_ms:.0f} ms (one-time, not per-doc)")
        print(f"  conclusion:        {result['bottleneck_analysis']['conclusion']}")

    _write(result)
    print(f"\n  JSON written to: {OUTPUT_JSON}")
    print("=" * 65 + "\n")


def _write(result: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
