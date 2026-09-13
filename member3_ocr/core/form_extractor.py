"""Form Key-Value Extraction for Member 3 OCR.

Extracts structured key-value pairs (KeyValueField) from normalized OCR TextBlocks.
Supports:
- Inline colon patterns: "Key: Value", "Key : Value", "Key:Value"
- Horizontally adjacent label and value blocks
- Vertically adjacent label and value blocks
- Clearly labelled fields with empty values (e.g. "Remarks:")
- Ambiguity detection and conservative prose filtering without LLMs or external calls.

Strictly offline, deterministic, and preserves OCR bounding-box geometry and confidence.
"""
from __future__ import annotations


import logging
import re
from dataclasses import dataclass
from typing import Any, Sequence

from .ocr_pipeline import BoundingBox, Issue, KeyValueField, TextBlock

LOGGER = logging.getLogger(__name__)

# Common industrial form label names that may appear without a trailing colon
KNOWN_FORM_LABELS: frozenset[str] = frozenset(
    {
        "equipment",
        "equipment id",
        "equipment name",
        "location",
        "unit",
        "plant",
        "date",
        "inspection date",
        "report date",
        "time",
        "inspector",
        "inspector name",
        "technician",
        "operator",
        "shift",
        "status",
        "form no",
        "form number",
        "formno",
        "report no",
        "report number",
        "tag",
        "tag no",
        "tag number",
        "serial no",
        "model",
        "pressure",
        "operating pressure",
        "temperature",
        "operating temperature",
        "findings",
        "action required",
        "remarks",
        "approved by",
        "reviewed by",
    }
)

# Prefix words that indicate prose or narrative rather than a form field label
PROSE_PREFIXES: frozenset[str] = frozenset(
    {
        "note",
        "note that",
        "please",
        "ensure",
        "ensure that",
        "verify",
        "check that",
        "warning",
        "caution",
        "danger",
        "step",
        "section",
        "chapter",
        "figure",
        "table",
        "in case of",
        "as per",
        "according to",
    }
)

# Field labels that typically introduce long, multi-line values (notes, remarks, disclaimers)
LONG_VALUE_FIELD_LABELS: frozenset[str] = frozenset(
    {
        "note",
        "notes",
        "comment",
        "comments",
        "remark",
        "remarks",
        "special instructions",
        "instructions",
        "disclaimer",
        "description",
        "scope",
        "action required",
        "corrective action",
    }
)


def clean_field_key(text: str) -> str:
    """Normalize a key string by stripping whitespace and trailing separators."""
    t = text.strip()
    # Strip trailing punctuation colons, hyphens, or semicolons
    while t and (t.endswith(":") or t.endswith("-") or t.endswith(";")):
        t = t[:-1].strip()
    if t.lower() == "formno":
        return "Form No"
    return t


def is_prose_text(text: str) -> bool:
    """Check if text is likely narrative prose rather than a form field."""
    s = text.strip().lower()
    if not s:
        return False
    # Check bullet points or dashes indicating a bulleted narrative line
    if text.strip().startswith(("-", "•", "*", "–")):
        return True
    # If the text is an isolated label token like "NOTE:", "NOTE -", "REMARKS:", it is a field label, not prose
    if (s.endswith(":") or s.endswith("-")) and len(s.split()) <= 3:
        key_candidate = s.rstrip(":-").strip()
        if key_candidate in LONG_VALUE_FIELD_LABELS or key_candidate in KNOWN_FORM_LABELS:
            return False
    # Check prose prefixes
    for prefix in PROSE_PREFIXES:
        if s == prefix or s.startswith(prefix + " ") or s.startswith(prefix + ":") or s.startswith(prefix + "-"):
            return True
    # Check multiple sentence endings
    if len(re.findall(r"[.!?]\s+[A-Z]", text)) > 0:
        return True
    return False


@dataclass(frozen=True)
class FormExtractorConfig:
    """Configuration options for FormKeyValueExtractor."""

    max_label_words: int = 6
    max_label_chars: int = 50
    max_horizontal_gap_px: float = 450.0
    max_vertical_gap_px: float = 80.0
    min_vertical_overlap_ratio: float = 0.35
    min_horizontal_overlap_ratio: float = 0.35
    ambiguity_gap_threshold_px: float = 18.0
    allow_empty_values: bool = True


class FormKeyValueExtractor:
    """Deterministic extractor of structured KeyValueField objects from OCR TextBlocks."""

    def __init__(self, config: FormExtractorConfig | None = None) -> None:
        self.config = config or FormExtractorConfig()

    def is_candidate_label(self, block: TextBlock) -> bool:
        """Determine if an OCR block looks like a form field label."""
        text = block.text.strip()
        if not text:
            return False
        if is_prose_text(text):
            return False

        # Ends with a colon
        if text.endswith(":"):
            key_part = text[:-1].strip()
            if 1 <= len(key_part) <= self.config.max_label_chars:
                words = key_part.split()
                if 1 <= len(words) <= self.config.max_label_words:
                    return any(c.isalpha() for c in key_part)
            return False

        # Known form label without colon
        norm_text = text.lower()
        if norm_text in KNOWN_FORM_LABELS:
            return True

        return False

    def _interpolate_inline_bboxes(
        self, block: TextBlock, key_str: str, val_str: str
    ) -> tuple[BoundingBox, BoundingBox]:
        """Estimate bounding boxes for inline key and value by proportional split."""
        bbox = block.bbox
        total_len = len(block.text)
        if total_len <= 0:
            return bbox, bbox

        key_len = len(key_str)
        width = max(1.0, bbox.right - bbox.left)
        ratio = min(1.0, max(0.0, (key_len + 1.0) / total_len))
        split_x = round(bbox.left + width * ratio, 2)

        key_bbox = BoundingBox(
            left=bbox.left,
            top=bbox.top,
            right=split_x,
            bottom=bbox.bottom,
            coordinate_space=bbox.coordinate_space,
        )
        val_bbox = BoundingBox(
            left=split_x,
            top=bbox.top,
            right=bbox.right,
            bottom=bbox.bottom,
            coordinate_space=bbox.coordinate_space,
        )
        return key_bbox, val_bbox

    def _collect_multiline_paragraph(
        self,
        first_val: TextBlock,
        all_blocks: Sequence[TextBlock],
        label_block: TextBlock,
        consumed_ids: set[str],
        max_lines: int = 15,
    ) -> list[TextBlock]:
        """Collect contiguous vertical text lines below first_val forming a multi-line value/paragraph."""
        val_blocks = [first_val]
        col_left = first_val.bbox.left
        col_right = first_val.bbox.right
        current_bottom = first_val.bbox.bottom

        for _ in range(max_lines):
            next_line = None
            min_gap = 999.0
            for b in all_blocks:
                if b.id in consumed_ids or b.id == label_block.id or any(b.id == v.id for v in val_blocks):
                    continue
                gap_y = b.bbox.top - current_bottom
                # Tight line spacing typical of continuous multi-line paragraphs
                if -8.0 <= gap_y <= 25.0:
                    overlap_x = min(col_right, b.bbox.right) - max(col_left, b.bbox.left)
                    width = min(col_right - col_left, b.bbox.right - b.bbox.left)
                    if width > 0 and (overlap_x / width >= 0.35 or abs(b.bbox.left - col_left) <= 35.0):
                        clean_t = b.text.strip()
                        if clean_t.startswith(("-", "•", "*", "–")):
                            continue
                        if clean_t.endswith(":") and len(clean_t.split()) <= 4:
                            continue
                        if gap_y < min_gap:
                            min_gap = gap_y
                            next_line = b
            if next_line is not None:
                val_blocks.append(next_line)
                current_bottom = next_line.bbox.bottom
                col_left = min(col_left, next_line.bbox.left)
                col_right = max(col_right, next_line.bbox.right)
            else:
                break
        return val_blocks

    def extract(
        self,
        blocks: Sequence[TextBlock],
        *,
        page_number: int = 1,
    ) -> tuple[tuple[KeyValueField, ...], tuple[Issue, ...]]:
        """Extract key-value fields from a sequence of OCR TextBlocks.

        Parameters
        ----------
        blocks:
            Detected text blocks for a page.
        page_number:
            1-based page number for provenance tracking.

        Returns
        -------
        tuple[tuple[KeyValueField, ...], tuple[Issue, ...]]
            Extracted KeyValueField objects and any ambiguity/extraction warnings.
        """
        if not blocks:
            return (), ()

        extracted: list[KeyValueField] = []
        issues: list[Issue] = []
        consumed_block_ids: set[str] = set()

        # Sort blocks in reading order: top-to-bottom, then left-to-right
        sorted_blocks = sorted(blocks, key=lambda b: (round(b.bbox.top / 10) * 10, b.bbox.left))

        # ── Step 1: Detect Inline Colon Fields ─────────────────────────────────
        for block in sorted_blocks:
            text = block.text.strip()
            if not text:
                continue
            if is_prose_text(text):
                continue

            sep = None
            if ":" in text:
                sep = ":"
            elif ";" in text:
                prefix = text.split(";", 1)[0].strip().lower()
                if prefix in KNOWN_FORM_LABELS or prefix in ("form no", "formno", "form number"):
                    sep = ";"
            if not sep:
                continue

            parts = text.split(sep, 1)
            key_candidate = parts[0].strip()
            val_candidate = parts[1].strip()

            # Validate key candidate
            if not key_candidate or len(key_candidate) > self.config.max_label_chars:
                continue
            clean_k = clean_field_key(key_candidate)
            if clean_k.lower() in PROSE_PREFIXES:
                continue
            words = key_candidate.split()
            if not (1 <= len(words) <= self.config.max_label_words):
                continue
            if not any(c.isalpha() for c in key_candidate):
                continue
            if any(p in key_candidate for p in (".", "!", "?")):
                continue

            # If value is present in the same block, extract as inline_colon
            if val_candidate:
                key_bbox, val_bbox = self._interpolate_inline_bboxes(block, key_candidate, val_candidate)
                extracted.append(
                    KeyValueField(
                        key=clean_k,
                        value=val_candidate,
                        confidence=block.confidence,
                        key_bbox=key_bbox,
                        value_bbox=val_bbox,
                        extraction_method="inline_colon",
                        page_number=page_number,
                    )
                )
                consumed_block_ids.add(block.id)

        # ── Step 2: Separate-Block Horizontal Matching ─────────────────────────
        remaining_blocks = [b for b in sorted_blocks if b.id not in consumed_block_ids]
        label_blocks = [b for b in remaining_blocks if self.is_candidate_label(b)]

        for label in label_blocks:
            if label.id in consumed_block_ids:
                continue

            clean_key = clean_field_key(label.text)
            label_h = max(1.0, label.bbox.bottom - label.bbox.top)

            # Search for candidate value blocks to the right
            candidates: list[tuple[float, TextBlock]] = []
            for val in remaining_blocks:
                if val.id in consumed_block_ids or val.id == label.id:
                    continue
                # Must be to the right of label left edge
                if val.bbox.left < label.bbox.left:
                    continue
                # Horizontal gap between label right and val left
                gap_x = val.bbox.left - label.bbox.right
                if gap_x < -15.0 or gap_x > self.config.max_horizontal_gap_px:
                    continue

                # Vertical overlap check
                overlap_y = min(label.bbox.bottom, val.bbox.bottom) - max(label.bbox.top, val.bbox.top)
                min_h = min(label_h, max(1.0, val.bbox.bottom - val.bbox.top))
                overlap_ratio = overlap_y / min_h

                # Center y distance check
                cy_label = (label.bbox.top + label.bbox.bottom) / 2.0
                cy_val = (val.bbox.top + val.bbox.bottom) / 2.0
                cy_dist = abs(cy_label - cy_val)

                if overlap_ratio >= self.config.min_vertical_overlap_ratio or cy_dist <= (0.6 * label_h):
                    # Do not match another label block as a value
                    if not self.is_candidate_label(val):
                        candidates.append((max(0.0, gap_x), val))

            if not candidates:
                continue

            # Sort by horizontal gap
            candidates.sort(key=lambda x: x[0])
            best_gap, best_val = candidates[0]

            # Ambiguity check: multiple values with almost identical gap
            if len(candidates) > 1:
                second_gap, second_val = candidates[1]
                if abs(second_gap - best_gap) <= self.config.ambiguity_gap_threshold_px:
                    issues.append(
                        Issue(
                            code="ambiguous_form_field",
                            message=(
                                f"Ambiguous horizontal field match on page {page_number}: "
                                f"label '{clean_key}' has multiple competing values "
                                f"('{best_val.text}' vs '{second_val.text}')"
                            ),
                            severity="warning",
                            recoverable=True,
                        )
                    )
                    consumed_block_ids.add(label.id)
                    continue

            # Valid horizontal link established
            if clean_key.lower() in LONG_VALUE_FIELD_LABELS:
                val_blocks = self._collect_multiline_paragraph(
                    best_val, sorted_blocks, label, consumed_block_ids
                )
            else:
                val_blocks = [best_val]
            val_text = " ".join(b.text.strip() for b in val_blocks)
            val_bbox = BoundingBox(
                left=min(b.bbox.left for b in val_blocks),
                top=val_blocks[0].bbox.top,
                right=max(b.bbox.right for b in val_blocks),
                bottom=val_blocks[-1].bbox.bottom,
                coordinate_space=best_val.bbox.coordinate_space,
            )
            confs = [c for b in val_blocks for c in [b.confidence] if c is not None]
            if label.confidence is not None:
                confs.append(label.confidence)
            joint_conf = round(min(confs), 4) if confs else None

            extracted.append(
                KeyValueField(
                    key=clean_key,
                    value=val_text,
                    confidence=joint_conf,
                    key_bbox=label.bbox,
                    value_bbox=val_bbox,
                    extraction_method="horizontal_adjacent",
                    page_number=page_number,
                )
            )
            consumed_block_ids.add(label.id)
            for b in val_blocks:
                consumed_block_ids.add(b.id)

        # ── Step 3: Separate-Block Vertical Matching ───────────────────────────
        remaining_blocks = [b for b in sorted_blocks if b.id not in consumed_block_ids]
        label_blocks = [b for b in remaining_blocks if self.is_candidate_label(b)]

        for label in label_blocks:
            if label.id in consumed_block_ids:
                continue

            clean_key = clean_field_key(label.text)
            label_w = max(1.0, label.bbox.right - label.bbox.left)
            label_h = max(1.0, label.bbox.bottom - label.bbox.top)

            # Search for candidate value blocks directly below
            candidates_v: list[tuple[float, TextBlock]] = []
            for val in remaining_blocks:
                if val.id in consumed_block_ids or val.id == label.id:
                    continue
                # Must be below the label
                gap_y = val.bbox.top - label.bbox.bottom
                if gap_y < -5.0 or gap_y > self.config.max_vertical_gap_px:
                    continue

                # Horizontal alignment check (overlap or left-aligned)
                overlap_x = min(label.bbox.right, val.bbox.right) - max(label.bbox.left, val.bbox.left)
                min_w = min(label_w, max(1.0, val.bbox.right - val.bbox.left))
                overlap_x_ratio = overlap_x / min_w
                left_dist = abs(val.bbox.left - label.bbox.left)

                if overlap_x_ratio >= self.config.min_horizontal_overlap_ratio or left_dist <= 25.0:
                    val_clean = val.text.strip()
                    if val_clean.startswith(("-", "•", "*", "–")):
                        continue
                    if not self.is_candidate_label(val) and (
                        clean_key.lower() in LONG_VALUE_FIELD_LABELS or not is_prose_text(val.text)
                    ):
                        candidates_v.append((max(0.0, gap_y), val))

            if not candidates_v:
                continue

            candidates_v.sort(key=lambda x: x[0])
            best_gap_y, best_val_v = candidates_v[0]

            # Ambiguity check
            if len(candidates_v) > 1:
                second_gap_y, second_val_v = candidates_v[1]
                if abs(second_gap_y - best_gap_y) <= 10.0:
                    issues.append(
                        Issue(
                            code="ambiguous_form_field",
                            message=(
                                f"Ambiguous vertical field match on page {page_number}: "
                                f"label '{clean_key}' has multiple competing values below"
                            ),
                            severity="warning",
                            recoverable=True,
                        )
                    )
                    continue

            if clean_key.lower() in LONG_VALUE_FIELD_LABELS:
                val_blocks_v = self._collect_multiline_paragraph(
                    best_val_v, sorted_blocks, label, consumed_block_ids
                )
            else:
                val_blocks_v = [best_val_v]
            val_text_v = " ".join(b.text.strip() for b in val_blocks_v)
            val_bbox_v = BoundingBox(
                left=min(b.bbox.left for b in val_blocks_v),
                top=val_blocks_v[0].bbox.top,
                right=max(b.bbox.right for b in val_blocks_v),
                bottom=val_blocks_v[-1].bbox.bottom,
                coordinate_space=best_val_v.bbox.coordinate_space,
            )
            confs = [c for b in val_blocks_v for c in [b.confidence] if c is not None]
            if label.confidence is not None:
                confs.append(label.confidence)
            joint_conf = round(min(confs), 4) if confs else None

            extracted.append(
                KeyValueField(
                    key=clean_key,
                    value=val_text_v,
                    confidence=joint_conf,
                    key_bbox=label.bbox,
                    value_bbox=val_bbox_v,
                    extraction_method="vertical_adjacent",
                    page_number=page_number,
                )
            )
            consumed_block_ids.add(label.id)
            for b in val_blocks_v:
                consumed_block_ids.add(b.id)

        # ── Step 4: Empty Value Fields ─────────────────────────────────────────
        if self.config.allow_empty_values:
            remaining_labels = [
                b for b in sorted_blocks
                if b.id not in consumed_block_ids and b.text.strip().endswith(":") and self.is_candidate_label(b)
            ]
            for label in remaining_labels:
                clean_key = clean_field_key(label.text)
                extracted.append(
                    KeyValueField(
                        key=clean_key,
                        value="",
                        confidence=label.confidence,
                        key_bbox=label.bbox,
                        value_bbox=None,
                        extraction_method="empty_value",
                        page_number=page_number,
                    )
                )
                consumed_block_ids.add(label.id)

        # Sort extracted fields deterministically by visual reading position (top, left)
        def _kv_sort_key(f: KeyValueField) -> tuple[float, float]:
            if f.key_bbox is not None:
                return (round(f.key_bbox.top / 5) * 5, f.key_bbox.left)
            return (0.0, 0.0)

        extracted.sort(key=_kv_sort_key)
        return tuple(extracted), tuple(issues)


def extract_form_key_values(
    blocks: Sequence[TextBlock],
    *,
    page_number: int = 1,
    config: FormExtractorConfig | None = None,
) -> tuple[tuple[KeyValueField, ...], tuple[Issue, ...]]:
    """Convenience function to extract KeyValueFields from a sequence of TextBlocks."""
    return FormKeyValueExtractor(config).extract(blocks, page_number=page_number)
