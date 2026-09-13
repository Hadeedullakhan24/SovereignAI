"""Safe, configurable image preprocessing for OCR, drawings, and vision inputs.

This module reads source images without changing them.  Processed output is only
written when explicitly requested and is never allowed inside ``datasets/``.
"""
from __future__ import annotations


import argparse
import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

LOGGER = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})
ThresholdMethod = Literal["global", "adaptive"]


class ImagePreprocessingError(RuntimeError):
    """Base exception for an image preprocessing failure."""


class ImageValidationError(ImagePreprocessingError):
    """Raised when an input does not exist, is unsupported, or is unreadable."""


class CropValidationError(ImagePreprocessingError):
    """Raised when a requested crop rectangle is invalid."""


@dataclass(frozen=True)
class PreprocessingOptions:
    """Options for a preprocessing run.

    All enhancement operations are opt-in.  ``document_ocr`` is a conservative
    preset, while drawing and general-vision presets preserve colour and detail.
    """

    max_width: int | None = None
    max_height: int | None = None
    allow_upscale: bool = False
    grayscale: bool = False
    denoise: bool = False
    denoise_strength: int = 3
    contrast_enhancement: bool = False
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 8
    deskew: bool = False
    deskew_min_angle: float = 0.3
    deskew_max_angle: float = 10.0
    binarize: bool = False
    threshold_method: ThresholdMethod = "adaptive"
    crop_box: tuple[int, int, int, int] | None = None

    @classmethod
    def document_ocr(cls, **overrides: Any) -> "PreprocessingOptions":
        """Return conservative settings appropriate for text documents.

        By default, preserves original RGB color and detail without destructive
        adaptive binarization. Enhancement operations (grayscale, denoise, CLAHE,
        deskew, and binarization) remain available as opt-in capabilities.
        """
        values: dict[str, Any] = {
            "grayscale": False,
            "denoise": False,
            "contrast_enhancement": False,
            "binarize": False,
        }
        values.update(overrides)
        return cls(**values)

    @classmethod
    def drawing(cls, **overrides: Any) -> "PreprocessingOptions":
        """Return detail-preserving settings appropriate for engineering drawings."""
        values: dict[str, Any] = {"contrast_enhancement": True}
        values.update(overrides)
        return cls(**values)

    @classmethod
    def general_vision(cls, **overrides: Any) -> "PreprocessingOptions":
        """Return a no-transform preset for colour-sensitive vision workflows."""
        return cls(**overrides)


@dataclass
class PreprocessingResult:
    """The processed image together with provenance and applied operations."""

    image: np.ndarray = field(repr=False)
    original_path: Path
    original_width: int
    original_height: int
    processed_width: int
    processed_height: int
    original_mode: str
    processed_is_grayscale: bool
    original_metadata: Mapping[str, Any]
    operations_applied: list[str]
    processing_time_seconds: float
    output_path: Path | None = None
    deskew_angle_degrees: float | None = None


def validate_image(path: str | Path) -> Path:
    """Validate and return a readable supported image path.

    Pillow performs an inexpensive verification before OpenCV decodes pixels, so
    callers receive a useful error for corrupt images or unsupported formats.
    """
    image_path = Path(path).expanduser()
    if not image_path.is_file():
        raise ImageValidationError(f"Image does not exist or is not a file: {image_path}")
    if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ImageValidationError(
            f"Unsupported image type '{image_path.suffix}'. Supported types: "
            f"{', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    try:
        with Image.open(image_path) as image:
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageValidationError(f"Unreadable or corrupt image: {image_path}") from exc
    return image_path


def _load_image(path: Path) -> tuple[np.ndarray, int, int, str, Mapping[str, Any]]:
    """Load pixels and practical Pillow metadata without altering the source."""
    try:
        with Image.open(path) as pil_image:
            width, height = pil_image.size
            mode = pil_image.mode
            metadata = {
                key: value
                for key, value in pil_image.info.items()
                if isinstance(value, (str, int, float, bytes, tuple))
            }
        # imdecode supports Windows paths containing non-ASCII characters.
        encoded = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    except (OSError, ValueError, cv2.error) as exc:
        raise ImageValidationError(f"Could not load image pixels: {path}") from exc
    if image is None or image.size == 0:
        raise ImageValidationError(f"Could not decode image pixels: {path}")
    return image, width, height, mode, metadata


def _as_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def resize_to_fit(
    image: np.ndarray,
    max_width: int | None = None,
    max_height: int | None = None,
    *,
    allow_upscale: bool = False,
) -> np.ndarray:
    """Resize within maximum dimensions while retaining the exact aspect ratio."""
    if max_width is not None and max_width <= 0:
        raise ValueError("max_width must be positive when provided")
    if max_height is not None and max_height <= 0:
        raise ValueError("max_height must be positive when provided")
    height, width = image.shape[:2]
    limits = [1.0]
    if max_width is not None:
        limits.append(max_width / width)
    if max_height is not None:
        limits.append(max_height / height)
    scale = min(limits)
    if not allow_upscale:
        scale = min(scale, 1.0)
    if abs(scale - 1.0) < 1e-9:
        return image
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(image, new_size, interpolation=interpolation)


def crop_image(image: np.ndarray, box: Sequence[int]) -> np.ndarray:
    """Return a strict in-bounds crop for ``(left, top, right, bottom)``."""
    if len(box) != 4 or any(not isinstance(value, (int, np.integer)) for value in box):
        raise CropValidationError("crop_box must contain four integer coordinates: (left, top, right, bottom)")
    left, top, right, bottom = (int(value) for value in box)
    height, width = image.shape[:2]
    if left < 0 or top < 0 or right > width or bottom > height:
        raise CropValidationError(f"crop_box {tuple(box)} exceeds image bounds 0..{width}, 0..{height}")
    if right <= left or bottom <= top:
        raise CropValidationError("crop_box must have right > left and bottom > top")
    return image[top:bottom, left:right].copy()


def _denoise(image: np.ndarray, strength: int) -> np.ndarray:
    if not 1 <= strength <= 30:
        raise ValueError("denoise_strength must be between 1 and 30")
    if image.ndim == 2:
        return cv2.fastNlMeansDenoising(image, None, h=strength, templateWindowSize=7, searchWindowSize=21)
    if image.ndim == 3 and image.shape[2] == 4:
        bgr = cv2.fastNlMeansDenoisingColored(image[:, :, :3], None, strength, strength, 7, 21)
        return np.dstack((bgr, image[:, :, 3]))
    return cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)


def _apply_clahe(image: np.ndarray, clip_limit: float, tile_size: int) -> np.ndarray:
    if clip_limit <= 0 or tile_size <= 0:
        raise ValueError("CLAHE clip limit and tile grid size must be positive")
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    if image.ndim == 2:
        return clahe.apply(image)
    bgr = image[:, :, :3]
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return np.dstack((enhanced, image[:, :, 3])) if image.shape[2] == 4 else enhanced


def estimate_skew_angle(image: np.ndarray, max_angle: float = 10.0) -> float | None:
    """Estimate a conservative document skew angle, or return ``None``.

    The estimator rejects sparse content and strong non-document angles, which
    prevents drawings with many deliberate lines from being automatically rotated.
    """
    gray = _as_gray(image)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    points = np.column_stack(np.where(binary > 0))
    if len(points) < 100:
        return None
    angle = cv2.minAreaRect(points.astype(np.float32))[-1]
    angle = angle + 90 if angle < -45 else angle
    if abs(angle) > max_angle:
        return None
    return float(angle)


def _rotate(image: np.ndarray, angle: float) -> np.ndarray:
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    border = 255 if image.ndim == 2 else (255, 255, 255, 0) if image.shape[2] == 4 else (255, 255, 255)
    return cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE, borderValue=border)


def _binarize(image: np.ndarray, method: ThresholdMethod) -> np.ndarray:
    gray = _as_gray(image)
    if method == "global":
        return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    if method == "adaptive":
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11)
    raise ValueError("threshold_method must be 'global' or 'adaptive'")


def _ensure_safe_output_path(source: Path, output_dir: Path, operations: Sequence[str]) -> Path:
    package_root = Path(__file__).resolve().parents[2]
    datasets_root = package_root / "datasets"
    resolved_output = output_dir.expanduser().resolve()
    try:
        resolved_output.relative_to(datasets_root.resolve())
    except ValueError:
        pass
    else:
        raise ImagePreprocessingError("Processed output may not be written inside datasets/")
    output_hash = hashlib.sha256((str(source.resolve()) + "|" + "|".join(operations)).encode()).hexdigest()[:12]
    return resolved_output / f"{source.stem}_processed_{output_hash}.png"


class ImagePreprocessor:
    """Reusable pipeline that applies selected preprocessing operations in order."""

    def process_array(
        self,
        image: np.ndarray,
        options: PreprocessingOptions | None = None,
        *,
        source_path: Path | None = None,
        original_mode: str = "RGB",
        original_metadata: Mapping[str, Any] | None = None,
        save_output: bool = False,
        output_dir: str | Path | None = None,
        _start_time: float | None = None,
        _orig_size: tuple[int, int] | None = None,
    ) -> PreprocessingResult:
        """Process an in-memory numpy image array through the preprocessing pipeline."""
        start = _start_time if _start_time is not None else time.perf_counter()
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise ImageValidationError("Image must be a non-empty numpy ndarray")
        if image.dtype != np.uint8:
            raise ImageValidationError(f"Image must have dtype uint8, got {image.dtype}")

        if _orig_size is not None:
            original_width, original_height = _orig_size
        else:
            original_height, original_width = image.shape[:2]

        image = image.copy()
        options = options or PreprocessingOptions()
        operations: list[str] = []
        deskew_angle: float | None = None

        if options.crop_box is not None:
            image = crop_image(image, options.crop_box)
            operations.append("crop")
        resized = resize_to_fit(image, options.max_width, options.max_height, allow_upscale=options.allow_upscale)
        if resized is not image:
            image = resized
            operations.append("resize")
        if options.grayscale:
            image = _as_gray(image)
            operations.append("grayscale")
        if options.denoise:
            image = _denoise(image, options.denoise_strength)
            operations.append("denoise")
        if options.contrast_enhancement:
            image = _apply_clahe(image, options.clahe_clip_limit, options.clahe_tile_grid_size)
            operations.append("clahe")
        if options.deskew:
            angle = estimate_skew_angle(image, options.deskew_max_angle)
            if angle is not None and abs(angle) >= options.deskew_min_angle:
                image = _rotate(image, angle)
                deskew_angle = angle
                operations.append("deskew")
            else:
                LOGGER.debug("Deskew skipped: no reliable angle")
        if options.binarize:
            image = _binarize(image, options.threshold_method)
            operations.append(f"binarize:{options.threshold_method}")

        saved_path: Path | None = None
        source = source_path or Path("in_memory_image.png")
        if save_output:
            destination = Path(output_dir) if output_dir is not None else Path(__file__).resolve().parents[1] / "output"
            saved_path = _ensure_safe_output_path(source, destination, operations)
            saved_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(saved_path), image):
                raise ImagePreprocessingError(f"Could not save processed image: {saved_path}")
            operations.append("save")

        height, width = image.shape[:2]
        return PreprocessingResult(
            image=image,
            original_path=source,
            original_width=original_width,
            original_height=original_height,
            processed_width=width,
            processed_height=height,
            original_mode=original_mode,
            processed_is_grayscale=image.ndim == 2,
            original_metadata=original_metadata or {},
            operations_applied=operations,
            processing_time_seconds=time.perf_counter() - start,
            output_path=saved_path,
            deskew_angle_degrees=deskew_angle,
        )

    def process(
        self,
        input_path: str | Path | np.ndarray,
        options: PreprocessingOptions | None = None,
        *,
        save_output: bool = False,
        output_dir: str | Path | None = None,
        source_path: Path | None = None,
    ) -> PreprocessingResult:
        """Process one image file or array and optionally save a PNG outside the dataset directory."""
        if isinstance(input_path, np.ndarray):
            return self.process_array(
                input_path,
                options,
                source_path=source_path,
                save_output=save_output,
                output_dir=output_dir,
            )
        start = time.perf_counter()
        source = validate_image(input_path)
        image, original_width, original_height, mode, metadata = _load_image(source)
        return self.process_array(
            image,
            options,
            source_path=source,
            original_mode=mode,
            original_metadata=metadata,
            save_output=save_output,
            output_dir=output_dir,
            _start_time=start,
            _orig_size=(original_width, original_height),
        )


def preprocess_image(
    input_path: str | Path | np.ndarray,
    options: PreprocessingOptions | None = None,
    *,
    save_output: bool = False,
    output_dir: str | Path | None = None,
    source_path: Path | None = None,
) -> PreprocessingResult:
    """Convenience function for a one-off image preprocessing run."""
    return ImagePreprocessor().process(
        input_path,
        options,
        save_output=save_output,
        output_dir=output_dir,
        source_path=source_path,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess a single image safely for OCR or vision.")
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--output", help="Directory for the processed PNG (defaults to member3_ocr/output)")
    parser.add_argument("--profile", choices=("general", "document", "drawing"), default="general")
    parser.add_argument("--max-width", type=int)
    parser.add_argument("--max-height", type=int)
    parser.add_argument("--grayscale", action="store_true")
    parser.add_argument("--denoise", action="store_true")
    parser.add_argument("--contrast", action="store_true")
    parser.add_argument("--deskew", action="store_true")
    parser.add_argument("--binarize", choices=("global", "adaptive"))
    parser.add_argument("--crop", nargs=4, type=int, metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the small manual-testing CLI."""
    args = _build_parser().parse_args(argv)
    presets = {
        "general": PreprocessingOptions.general_vision,
        "document": PreprocessingOptions.document_ocr,
        "drawing": PreprocessingOptions.drawing,
    }
    options = presets[args.profile](
        max_width=args.max_width,
        max_height=args.max_height,
        grayscale=args.grayscale or (args.profile == "document"),
        denoise=args.denoise or (args.profile == "document"),
        contrast_enhancement=args.contrast or (args.profile in {"document", "drawing"}),
        deskew=args.deskew,
        binarize=bool(args.binarize) or (args.profile == "document"),
        threshold_method=args.binarize or "adaptive",
        crop_box=tuple(args.crop) if args.crop else None,
    )
    try:
        result = preprocess_image(args.input, options, save_output=True, output_dir=args.output)
    except ImagePreprocessingError as exc:
        LOGGER.error("Preprocessing failed: %s", exc)
        return 2
    print(f"Saved: {result.output_path}")
    print(f"Size: {result.original_width}x{result.original_height} -> {result.processed_width}x{result.processed_height}")
    print(f"Operations: {', '.join(result.operations_applied) or 'none'}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())
