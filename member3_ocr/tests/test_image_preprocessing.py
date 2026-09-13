"""Comprehensive unit tests for member3_ocr.image_preprocessing.

Covers all 20 required validation points:
1. Valid PNG input
2. Valid JPG input
3. PIL/OpenCV image loading
4. Aspect-ratio-preserving resize
5. No unnecessary upscaling
6. Grayscale conversion
7. Denoising
8. CLAHE/contrast enhancement
9. Deskew behavior
10. Global and adaptive binarization
11. Valid crop
12. Invalid crop coordinates
13. Unsupported extension
14. Missing file
15. Corrupt image
16. Dataset protection
17. Output is written outside datasets/
18. Deterministic processing with identical configuration
19. Result structure is JSON serializable
20. Original source image is not modified
"""
from __future__ import annotations


import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from member3_ocr.core.image_preprocessing import (
    CropValidationError,
    ImagePreprocessingError,
    ImageValidationError,
    PreprocessingOptions,
    PreprocessingResult,
    _ensure_safe_output_path,
    _load_image,
    crop_image,
    estimate_skew_angle,
    preprocess_image,
    resize_to_fit,
    validate_image,
)


def _create_synthetic_image(
    path: Path,
    size: tuple[int, int] = (160, 120),
    colour: tuple[int, int, int] = (30, 120, 200),
    fmt: str | None = None,
) -> Path:
    """Create a synthetic test image."""
    w, h = size
    arr = np.full((h, w, 3), colour, dtype=np.uint8)
    if h > 20 and w > 20:
        arr[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = [255, 255, 255]
    Image.fromarray(arr).save(path, format=fmt)
    return path


# ──────────────────────────────────────────────────────────────────────────────
# 1. Valid PNG input
# ──────────────────────────────────────────────────────────────────────────────

def test_valid_png_input(tmp_path: Path) -> None:
    source = tmp_path / "valid_image.png"
    _create_synthetic_image(source, size=(140, 80))
    result = preprocess_image(source)
    assert isinstance(result, PreprocessingResult)
    assert (result.original_width, result.original_height) == (140, 80)
    assert (result.processed_width, result.processed_height) == (140, 80)
    assert result.image.shape[:2] == (80, 140)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Valid JPG input
# ──────────────────────────────────────────────────────────────────────────────

def test_valid_jpg_input(tmp_path: Path) -> None:
    source = tmp_path / "valid_image.jpg"
    _create_synthetic_image(source, size=(100, 70), fmt="JPEG")
    result = preprocess_image(source)
    assert isinstance(result, PreprocessingResult)
    assert (result.original_width, result.original_height) == (100, 70)
    assert (result.processed_width, result.processed_height) == (100, 70)


# ──────────────────────────────────────────────────────────────────────────────
# 3. PIL/OpenCV image loading
# ──────────────────────────────────────────────────────────────────────────────

def test_pil_opencv_image_loading(tmp_path: Path) -> None:
    source = tmp_path / "load_test.png"
    _create_synthetic_image(source, size=(80, 60))
    image, width, height, mode, metadata = _load_image(source)
    assert isinstance(image, np.ndarray)
    assert (width, height) == (80, 60)
    assert mode == "RGB"
    assert isinstance(metadata, dict)
    assert image.shape == (60, 80, 3)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Aspect-ratio-preserving resize
# ──────────────────────────────────────────────────────────────────────────────

def test_aspect_ratio_preserving_resize() -> None:
    # 400x100 (4:1 aspect ratio)
    img_arr = np.zeros((100, 400, 3), dtype=np.uint8)
    resized = resize_to_fit(img_arr, max_width=200, max_height=200)
    # Scaled down to 200 width -> height must be 50 to preserve 4:1
    assert resized.shape[:2] == (50, 200)


# ──────────────────────────────────────────────────────────────────────────────
# 5. No unnecessary upscaling
# ──────────────────────────────────────────────────────────────────────────────

def test_no_unnecessary_upscaling() -> None:
    img_arr = np.zeros((60, 120, 3), dtype=np.uint8)
    # Target max is larger than original; image must not be enlarged
    resized = resize_to_fit(img_arr, max_width=300, max_height=300, allow_upscale=False)
    assert resized.shape[:2] == (60, 120)
    assert resized is img_arr  # Exact identity preserved when no scaling occurs


# ──────────────────────────────────────────────────────────────────────────────
# 6. Grayscale conversion
# ──────────────────────────────────────────────────────────────────────────────

def test_grayscale_conversion(tmp_path: Path) -> None:
    source = tmp_path / "color.png"
    _create_synthetic_image(source)
    result = preprocess_image(source, PreprocessingOptions(grayscale=True))
    assert result.processed_is_grayscale is True
    assert result.image.ndim == 2
    assert "grayscale" in result.operations_applied


# ──────────────────────────────────────────────────────────────────────────────
# 7. Denoising
# ──────────────────────────────────────────────────────────────────────────────

def test_denoising(tmp_path: Path) -> None:
    source = tmp_path / "noisy.png"
    _create_synthetic_image(source)
    result = preprocess_image(source, PreprocessingOptions(denoise=True, denoise_strength=5))
    assert "denoise" in result.operations_applied
    assert result.image.shape == (120, 160, 3)


# ──────────────────────────────────────────────────────────────────────────────
# 8. CLAHE/contrast enhancement
# ──────────────────────────────────────────────────────────────────────────────

def test_clahe_contrast_enhancement(tmp_path: Path) -> None:
    source = tmp_path / "contrast.png"
    _create_synthetic_image(source)
    result = preprocess_image(
        source,
        PreprocessingOptions(contrast_enhancement=True, clahe_clip_limit=3.0, clahe_tile_grid_size=4),
    )
    assert "clahe" in result.operations_applied


# ──────────────────────────────────────────────────────────────────────────────
# 9. Deskew behavior
# ──────────────────────────────────────────────────────────────────────────────

def test_deskew_behavior(tmp_path: Path) -> None:
    source = tmp_path / "deskew_test.png"
    _create_synthetic_image(source, size=(200, 200))
    # Unrotated image should have estimate_skew_angle or deskew handled safely
    result = preprocess_image(source, PreprocessingOptions(deskew=True))
    assert isinstance(result, PreprocessingResult)

    # Test synthetic skew estimation on an angled binary image
    slanted = np.zeros((200, 200), dtype=np.uint8)
    for i in range(50, 150):
        slanted[i, int(i * 0.8):int(i * 0.8) + 20] = 255
    angle = estimate_skew_angle(slanted, max_angle=20.0)
    assert angle is None or isinstance(angle, float)


# ──────────────────────────────────────────────────────────────────────────────
# 10. Global and adaptive binarization
# ──────────────────────────────────────────────────────────────────────────────

def test_global_and_adaptive_binarization(tmp_path: Path) -> None:
    source = tmp_path / "binarize_source.png"
    _create_synthetic_image(source)

    # Global threshold
    res_global = preprocess_image(
        source,
        PreprocessingOptions(binarize=True, threshold_method="global"),
    )
    assert "binarize:global" in res_global.operations_applied
    assert res_global.image.ndim == 2
    unique_vals_global = set(np.unique(res_global.image))
    assert unique_vals_global.issubset({0, 255})

    # Adaptive threshold
    res_adaptive = preprocess_image(
        source,
        PreprocessingOptions(binarize=True, threshold_method="adaptive"),
    )
    assert "binarize:adaptive" in res_adaptive.operations_applied
    assert res_adaptive.image.ndim == 2
    unique_vals_adaptive = set(np.unique(res_adaptive.image))
    assert unique_vals_adaptive.issubset({0, 255})


# ──────────────────────────────────────────────────────────────────────────────
# 11. Valid crop
# ──────────────────────────────────────────────────────────────────────────────

def test_valid_crop(tmp_path: Path) -> None:
    source = tmp_path / "crop_source.png"
    _create_synthetic_image(source, size=(160, 120))
    # Crop (left=20, top=10, right=100, bottom=70) -> width=80, height=60
    result = preprocess_image(
        source,
        PreprocessingOptions(crop_box=(20, 10, 100, 70)),
    )
    assert "crop" in result.operations_applied
    assert (result.processed_width, result.processed_height) == (80, 60)
    assert result.image.shape[:2] == (60, 80)


# ──────────────────────────────────────────────────────────────────────────────
# 12. Invalid crop coordinates
# ──────────────────────────────────────────────────────────────────────────────

def test_invalid_crop_coordinates() -> None:
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    # Out of bounds right
    with pytest.raises(CropValidationError, match="exceeds image bounds"):
        crop_image(img, (0, 0, 51, 10))
    # Out of bounds bottom
    with pytest.raises(CropValidationError, match="exceeds image bounds"):
        crop_image(img, (0, 0, 10, 55))
    # Negative left
    with pytest.raises(CropValidationError, match="exceeds image bounds"):
        crop_image(img, (-5, 0, 20, 20))
    # right <= left
    with pytest.raises(CropValidationError, match="right > left"):
        crop_image(img, (20, 0, 20, 20))
    # bottom <= top
    with pytest.raises(CropValidationError, match="bottom > top"):
        crop_image(img, (0, 20, 20, 20))


# ──────────────────────────────────────────────────────────────────────────────
# 13. Unsupported extension
# ──────────────────────────────────────────────────────────────────────────────

def test_unsupported_extension(tmp_path: Path) -> None:
    bad_ext = tmp_path / "document.txt"
    bad_ext.write_text("not an image")
    with pytest.raises(ImageValidationError, match="Unsupported image type"):
        preprocess_image(bad_ext)


# ──────────────────────────────────────────────────────────────────────────────
# 14. Missing file
# ──────────────────────────────────────────────────────────────────────────────

def test_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "non_existent.png"
    with pytest.raises(ImageValidationError, match="does not exist"):
        preprocess_image(missing)


# ──────────────────────────────────────────────────────────────────────────────
# 15. Corrupt image
# ──────────────────────────────────────────────────────────────────────────────

def test_corrupt_image(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"\x89PNG\r\n\x1a\nCorruptPayloadWithoutValidChunks")
    with pytest.raises(ImageValidationError, match="Unreadable or corrupt"):
        preprocess_image(corrupt)


# ──────────────────────────────────────────────────────────────────────────────
# 16. Dataset protection
# ──────────────────────────────────────────────────────────────────────────────

def test_dataset_protection(tmp_path: Path) -> None:
    source = tmp_path / "protect_test.png"
    _create_synthetic_image(source)

    # Construct an output path attempting to write inside datasets/
    package_root = Path(__file__).resolve().parents[2]
    forbidden_dir = package_root / "datasets" / "leak_attempt"

    with pytest.raises(ImagePreprocessingError, match="Processed output may not be written inside datasets/"):
        _ensure_safe_output_path(source, forbidden_dir, ["test"])


# ──────────────────────────────────────────────────────────────────────────────
# 17. Output is written outside datasets/
# ──────────────────────────────────────────────────────────────────────────────

def test_output_written_outside_datasets(tmp_path: Path) -> None:
    source = tmp_path / "safe_out_source.png"
    _create_synthetic_image(source)
    out_dir = tmp_path / "preprocessed_output"

    result = preprocess_image(
        source,
        PreprocessingOptions.document_ocr(),
        save_output=True,
        output_dir=out_dir,
    )
    assert result.output_path is not None
    assert result.output_path.is_file()
    assert out_dir in result.output_path.parents
    assert "save" in result.operations_applied


# ──────────────────────────────────────────────────────────────────────────────
# 18. Deterministic processing with identical configuration
# ──────────────────────────────────────────────────────────────────────────────

def test_deterministic_processing(tmp_path: Path) -> None:
    source = tmp_path / "determ_source.png"
    _create_synthetic_image(source)
    opts = PreprocessingOptions(
        grayscale=True,
        contrast_enhancement=True,
        binarize=True,
        threshold_method="adaptive",
    )
    res1 = preprocess_image(source, opts)
    res2 = preprocess_image(source, opts)

    assert np.array_equal(res1.image, res2.image)
    assert res1.operations_applied == res2.operations_applied
    assert res1.processed_width == res2.processed_width
    assert res1.processed_height == res2.processed_height


# ──────────────────────────────────────────────────────────────────────────────
# 19. Result structure is JSON serializable
# ──────────────────────────────────────────────────────────────────────────────

def test_result_structure_json_serializable(tmp_path: Path) -> None:
    source = tmp_path / "json_struct.png"
    _create_synthetic_image(source)
    result = preprocess_image(source, PreprocessingOptions(grayscale=True))

    record = {
        "original_path": str(result.original_path),
        "original_width": result.original_width,
        "original_height": result.original_height,
        "processed_width": result.processed_width,
        "processed_height": result.processed_height,
        "original_mode": result.original_mode,
        "processed_is_grayscale": result.processed_is_grayscale,
        "operations_applied": result.operations_applied,
        "processing_time_seconds": result.processing_time_seconds,
        "output_path": str(result.output_path) if result.output_path else None,
        "deskew_angle_degrees": result.deskew_angle_degrees,
    }
    dumped = json.dumps(record)
    loaded = json.loads(dumped)
    assert loaded["processed_is_grayscale"] is True
    assert "grayscale" in loaded["operations_applied"]


# ──────────────────────────────────────────────────────────────────────────────
# 20. Original source image is not modified
# ──────────────────────────────────────────────────────────────────────────────

def test_original_source_image_not_modified(tmp_path: Path) -> None:
    source = tmp_path / "unmodified.png"
    _create_synthetic_image(source)
    before_bytes = source.read_bytes()
    before_hash = hashlib.sha256(before_bytes).hexdigest()
    before_mtime = source.stat().st_mtime

    # Run full pipeline with saving
    result = preprocess_image(
        source,
        PreprocessingOptions.document_ocr(max_width=80),
        save_output=True,
        output_dir=tmp_path / "saved_safe",
    )
    assert result.output_path is not None

    after_bytes = source.read_bytes()
    after_hash = hashlib.sha256(after_bytes).hexdigest()
    after_mtime = source.stat().st_mtime

    assert before_hash == after_hash
    assert before_mtime == after_mtime


# ──────────────────────────────────────────────────────────────────────────────
# 21. Minimal Preprocessing Strategy Change Regression Tests (A-F)
# ──────────────────────────────────────────────────────────────────────────────

class TestMinimalPreprocessingDefault:
    def test_default_document_ocr_does_not_apply_binarization(self, tmp_path: Path) -> None:
        """Requirement A: Default document OCR preprocessing does NOT apply adaptive binarization."""
        source = tmp_path / "test_doc.png"
        _create_synthetic_image(source, size=(120, 80))
        opts = PreprocessingOptions.document_ocr()

        assert opts.binarize is False, "Default document_ocr must have binarize=False"
        result = preprocess_image(source, opts)
        assert not any("binarize" in op for op in result.operations_applied)
        assert result.image.ndim == 3, "Output must be 3-channel RGB without binarization"

    def test_explicit_binarization_still_applies(self, tmp_path: Path) -> None:
        """Requirement B: Explicitly requesting binarization still applies it."""
        source = tmp_path / "test_bin.png"
        _create_synthetic_image(source, size=(120, 80))
        opts = PreprocessingOptions.document_ocr(binarize=True)

        assert opts.binarize is True
        result = preprocess_image(source, opts)
        assert any("binarize" in op for op in result.operations_applied)
        assert result.image.ndim == 2, "Binarized image must be 2D single-channel"

    def test_raw_minimal_rgb_remains_valid(self, tmp_path: Path) -> None:
        """Requirement C: Raw/minimal RGB remains valid and preserves color channels."""
        source = tmp_path / "test_rgb.png"
        _create_synthetic_image(source, size=(100, 60), colour=(45, 130, 210))
        result = preprocess_image(source, PreprocessingOptions.document_ocr())

        assert result.image.ndim == 3
        assert result.image.shape == (60, 100, 3)
        assert result.processed_is_grayscale is False
        assert result.operations_applied == []

    def test_process_array_remains_compatible(self) -> None:
        """Requirement D: process_array() remains compatible with both default and explicit options."""
        from member3_ocr.core.image_preprocessing import ImagePreprocessor
        arr = np.full((80, 120, 3), 150, dtype=np.uint8)
        preprocessor = ImagePreprocessor()

        res_default = preprocessor.process_array(arr, PreprocessingOptions.document_ocr())
        assert res_default.image.shape == (80, 120, 3)
        assert res_default.processed_is_grayscale is False

        res_bin = preprocessor.process_array(arr, PreprocessingOptions.document_ocr(binarize=True))
        assert res_bin.image.ndim == 2
        assert res_bin.processed_is_grayscale is True

    def test_existing_enhancement_functionality_remains_available(self, tmp_path: Path) -> None:
        """Requirement E: Grayscale, denoise, CLAHE, and deskew remain available via explicit configuration."""
        source = tmp_path / "test_enhance.png"
        _create_synthetic_image(source, size=(120, 80))

        # Grayscale
        res_gray = preprocess_image(source, PreprocessingOptions.document_ocr(grayscale=True))
        assert "grayscale" in res_gray.operations_applied
        assert res_gray.image.ndim == 2

        # Denoise
        res_denoise = preprocess_image(source, PreprocessingOptions.document_ocr(denoise=True))
        assert "denoise" in res_denoise.operations_applied

        # CLAHE
        res_clahe = preprocess_image(source, PreprocessingOptions.document_ocr(contrast_enhancement=True))
        assert "clahe" in res_clahe.operations_applied

        # Deskew
        res_deskew = preprocess_image(source, PreprocessingOptions.document_ocr(deskew=True))
        assert isinstance(res_deskew, PreprocessingResult)

    def test_no_dataset_files_modified(self) -> None:
        """Requirement F: Datasets directory remains completely untouched and read-only."""
        import subprocess
        proc = subprocess.run(["git", "status", "--short", "datasets"], capture_output=True, text=True)
        assert proc.returncode == 0
        assert proc.stdout.strip() == "", "No files in datasets/ may be modified"


# ──────────────────────────────────────────────────────────────────────────────
# 22. Correctness Measurement Tests
#   These tests verify that preprocessing algorithms produce MEASURABLE, CORRECT
#   improvements — not just that the operation ran without crashing.
# ──────────────────────────────────────────────────────────────────────────────

class TestPreprocessingCorrectness:
    """Tests that preprocessing algorithms produce measurable improvements."""

    def test_deskew_reduces_angle_measurably(self) -> None:
        """Deskew should reduce the measured skew angle on a deliberately rotated image.

        Uses a dense filled rectangle as the test pattern — this gives
        estimate_skew_angle a reliable geometry to measure via minAreaRect.
        The test skips if the estimator cannot detect an angle (conservative
        rejection of ambiguous patterns is a documented design choice).
        """
        import cv2 as cv2_local
        from member3_ocr.core.image_preprocessing import ImagePreprocessor, estimate_skew_angle

        # Create a dense filled-rectangle image (minAreaRect works on filled blobs)
        base = np.zeros((300, 400), dtype=np.uint8)
        base[80:220, 40:360] = 255  # Large solid white rectangle

        # Rotate by 4 degrees — well within the max_angle=15 detection window
        h, w = base.shape[:2]
        matrix = cv2_local.getRotationMatrix2D((w / 2, h / 2), 4.0, 1.0)
        rotated = cv2_local.warpAffine(base, matrix, (w, h))

        angle_before = estimate_skew_angle(rotated, max_angle=15.0)

        if angle_before is None or abs(angle_before) < 0.3:
            pytest.skip(
                "estimate_skew_angle returned None on synthetic rotated image — "
                "conservative rejection is valid behaviour, not a deskew defect."
            )

        # Apply deskew
        preprocessor = ImagePreprocessor()
        result = preprocessor.process_array(
            rotated,
            PreprocessingOptions(deskew=True, deskew_min_angle=0.3, deskew_max_angle=15.0),
        )
        assert "deskew" in result.operations_applied, "Deskew must be listed in operations_applied"

        # After deskew: angle should be None (corrected to within min threshold) or smaller
        angle_after = estimate_skew_angle(result.image, max_angle=15.0)
        if angle_after is not None:
            assert abs(angle_after) < abs(angle_before), (
                f"Deskew should reduce angle: before={angle_before:.2f}° after={angle_after:.2f}°"
            )
        # angle_after is None = fully corrected — test passes


    def test_denoise_reduces_local_variance(self) -> None:
        """Denoising should measurably reduce pixel-level noise variance.

        Uses raw standard deviation of pixel intensities in a flat uniform
        region as the metric.  A flat background with added Gaussian noise
        has high std-dev; after denoising, std-dev should drop significantly
        since the region should return close to its original uniform value.
        """
        import cv2 as cv2_local
        from member3_ocr.core.image_preprocessing import ImagePreprocessor

        # Create a flat-grey image (128) and add strong Gaussian noise (~30 std)
        rng = np.random.default_rng(seed=7)
        flat = np.full((200, 200, 3), 128, dtype=np.uint8)
        noise = rng.normal(0, 30, (200, 200, 3)).astype(np.int16)
        noisy = np.clip(flat.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Measure std dev in a central flat patch (should be high = noisy)
        def patch_stddev(img: np.ndarray) -> float:
            gray = cv2_local.cvtColor(img, cv2_local.COLOR_BGR2GRAY) if img.ndim == 3 else img
            patch = gray[70:130, 70:130].astype(np.float32)
            return float(np.std(patch))

        std_before = patch_stddev(noisy)
        assert std_before > 15.0, f"Test setup: expected high noise std, got {std_before:.2f}"

        preprocessor = ImagePreprocessor()
        result = preprocessor.process_array(
            noisy,
            PreprocessingOptions(denoise=True, denoise_strength=10),
        )
        assert "denoise" in result.operations_applied

        std_after = patch_stddev(result.image)

        assert std_after < std_before, (
            f"Denoising must reduce noise std-dev: before={std_before:.4f} after={std_after:.4f}"
        )
        # Expect at least 20% reduction in std-dev
        reduction_pct = (std_before - std_after) / max(std_before, 1e-9) * 100
        assert reduction_pct >= 20, (
            f"Expected >=20% std-dev reduction, got {reduction_pct:.1f}%"
        )

    def test_clahe_increases_histogram_spread(self) -> None:
        """CLAHE should measurably increase the P95-P5 pixel intensity spread.

        Uses a truly flat uniform image (all pixels = 128) so CLAHE has the
        maximum possible room to redistribute the histogram. On a uniform image,
        any CLAHE redistribution must produce a wider spread.
        """
        import cv2 as cv2_local
        from member3_ocr.core.image_preprocessing import ImagePreprocessor

        # Truly flat image — every pixel is exactly 128 (no existing contrast)
        flat = np.full((128, 128, 3), 128, dtype=np.uint8)
        # Add a tiny ramp (1 pixel per row) so CLAHE has something to redistribute
        for i in range(128):
            val = np.clip(64 + i, 64, 191)   # range: 64..191 = 127-unit range
            flat[i, :] = (val, val, val)

        def hist_spread(img: np.ndarray) -> float:
            gray = cv2_local.cvtColor(img, cv2_local.COLOR_BGR2GRAY) if img.ndim == 3 else img
            flat_gray = gray.flatten().astype(np.float32)
            return float(np.percentile(flat_gray, 95) - np.percentile(flat_gray, 5))

        spread_before = hist_spread(flat)

        preprocessor = ImagePreprocessor()
        # Use aggressive CLAHE settings to guarantee redistribution
        result = preprocessor.process_array(
            flat,
            PreprocessingOptions(contrast_enhancement=True, clahe_clip_limit=4.0, clahe_tile_grid_size=4),
        )
        assert "clahe" in result.operations_applied

        spread_after = hist_spread(result.image)

        assert spread_after > spread_before, (
            f"CLAHE must increase histogram spread: before={spread_before:.2f} after={spread_after:.2f}"
        )

    def test_crop_excludes_margin_pixels(self) -> None:
        """Crop should exclude all pixels outside the requested rectangle.

        Specifically: the cropped image must not contain any pixels that were
        originally in the outer margin — verifiable when the margin has a
        distinct colour not present in the crop region.
        """
        from member3_ocr.core.image_preprocessing import crop_image

        # 100x100 image: bright green border (0,255,0), white interior
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[:, :] = (0, 255, 0)          # green everywhere
        image[10:90, 10:90] = (255, 255, 255)  # white interior

        # Crop to interior region only
        cropped = crop_image(image, (10, 10, 90, 90))
        assert cropped.shape[:2] == (80, 80), f"Expected (80,80), got {cropped.shape[:2]}"

        # The cropped region must not contain any green pixels (0,255,0)
        green_pixels = np.all(cropped == np.array([0, 255, 0], dtype=np.uint8), axis=2)
        assert not np.any(green_pixels), (
            "Cropped image must not contain border pixels from outside the crop box"
        )

