"""Table Extraction for Member 3 OCR.

Extracts structured tables (Table) from normalized OCR TextBlocks using deterministic
geometry-based reconstruction with automated tilt compensation.

Features:
- Identifies table regions based on multi-column row alignment and pitch consistency.
- Automatically detects and compensates for scan tilt (up to +/- 3 degrees) via projection profile.
- Reconstructs grid rows and columns using coordinate clustering.
- Preserves cell bounding box, row/column indices, confidence, header status, and page provenance.
- Generates standard Markdown and HTML table representations.
- Detects ambiguous cell assignments and logs structured Issues without fabricating values.
- Completely offline, deterministic, and preserves OCR geometry.
"""
from __future__ import annotations


import logging
import math
import string
from dataclasses import dataclass
from typing import Any, Sequence

from .ocr_pipeline import BoundingBox, Issue, Table, TextBlock

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TableExtractorConfig:
    """Configuration options for TableExtractor."""

    min_table_rows: int = 2
    min_table_cols: int = 2
    row_vertical_tolerance_px: float = 12.0
    col_horizontal_tolerance_px: float = 35.0
    ambiguity_margin_px: float = 8.0
    interpolate_merged_headers: bool = True
    max_tilt_search_deg: float = 3.0
    tilt_step_deg: float = 0.1

    # Targeted OCR noise filtering
    filter_noise_fragments: bool = True
    max_noise_glyph_dimension_px: float = 12.0
    noise_min_confidence_threshold: float = 0.40
    max_noise_punct_dimension_px: float = 24.0
    noise_min_height_px: float = 11.5
    noise_short_text_len: int = 4
    noise_short_text_conf_threshold: float = 0.70
    noise_legit_min_height_px: float = 12.0
    noise_single_char_min_conf: float = 0.35


class TableExtractor:
    """Deterministic extractor of structured Table objects from OCR TextBlocks."""

    def __init__(self, config: TableExtractorConfig | None = None) -> None:
        self.config = config or TableExtractorConfig()

    def _estimate_table_tilt(self, blocks: Sequence[TextBlock]) -> float:
        """Estimate document / table tilt angle in radians using projection profile."""
        if len(blocks) < 6:
            return 0.0

        best_angle_deg = 0.0
        best_score = -1.0
        zero_score = -1.0

        n_steps = int(round(self.config.max_tilt_search_deg / self.config.tilt_step_deg))
        for step in range(-n_steps, n_steps + 1):
            deg = step * self.config.tilt_step_deg
            rad = math.radians(deg)
            tan_r = math.tan(rad)

            # Compute rotated y coordinate for block center
            bins: dict[int, int] = {}
            for b in blocks:
                cx = (b.bbox.left + b.bbox.right) / 2.0
                cy = (b.bbox.top + b.bbox.bottom) / 2.0
                y_corr = cy - tan_r * cx
                bin_idx = int(round(y_corr / 8.0))
                bins[bin_idx] = bins.get(bin_idx, 0) + 1

            score = sum(count * count for count in bins.values())
            if deg == 0.0:
                zero_score = score
            if score > best_score:
                best_score = score
                best_angle_deg = deg

        # Only apply non-zero tilt if it yields meaningful alignment improvement
        if zero_score > 0 and (best_score / zero_score) >= 1.05:
            return math.radians(best_angle_deg)
        return 0.0

    def _is_noise_fragment(self, block: TextBlock) -> tuple[bool, str]:
        """Determine deterministically whether an OCR block is a noise artifact.

        Preserves legitimate single-character, numeric, and mathematical values,
        while filtering isolated punctuation specks, hallucinated non-Latin noise glyphs,
        and tiny low-confidence fragments.
        """
        text = block.text.strip()
        if not text:
            return True, "empty_text"

        w = block.bbox.right - block.bbox.left
        h = block.bbox.bottom - block.bbox.top
        conf = block.confidence if block.confidence is not None else 1.0

        # Preserve legitimate single-character alphanumeric values with good geometry or confidence
        if len(text) == 1 and text.isalnum():
            if h >= self.config.noise_legit_min_height_px and conf >= self.config.noise_single_char_min_conf:
                return False, "legitimate_single_alphanumeric"

        # Preserve legitimate mathematical signs (+, -) with normal character geometry and confidence
        if text in ("+", "-"):
            if h >= self.config.noise_legit_min_height_px and w >= 8.0 and conf >= 0.50:
                return False, "legitimate_sign"

        # Preserve legitimate numeric values with normal height
        cleaned_num = text.replace(".", "").replace(",", "").replace("-", "").replace("+", "")
        if cleaned_num.isdigit() and h >= self.config.noise_legit_min_height_px and conf >= 0.30:
            return False, "legitimate_numeric"

        # 1. Non-alphanumeric / punctuation-only noise fragments
        punct_set = set(string.punctuation + "…·√：！↓」=+~^`￥—*°•")
        if all(ch in punct_set or ch.isspace() for ch in text):
            if (
                max(w, h) <= self.config.max_noise_punct_dimension_px
                or h <= self.config.noise_min_height_px
                or conf < 0.60
            ):
                return True, f"punctuation_noise(text={text!r}, w={w:.1f}, h={h:.1f}, conf={conf:.2f})"

        # 2. CJK / non-Latin ideograph hallucinations from noise specks
        if any("\u4e00" <= ch <= "\u9fff" for ch in text):
            if max(w, h) <= 24.0 or conf < 0.60:
                return True, f"hallucinated_cjk_glyph(text={text!r}, w={w:.1f}, h={h:.1f}, conf={conf:.2f})"

        # 3. Tiny height fragment with short text and low/medium confidence
        if (
            h <= self.config.noise_min_height_px
            and len(text) <= self.config.noise_short_text_len
            and conf < self.config.noise_short_text_conf_threshold
        ):
            return True, f"tiny_height_fragment(text={text!r}, h={h:.1f}, conf={conf:.2f})"

        # 4. Tiny bounding box overall with low confidence
        if max(w, h) <= self.config.max_noise_glyph_dimension_px and conf < self.config.noise_min_confidence_threshold:
            return True, f"tiny_low_conf_glyph(text={text!r}, w={w:.1f}, h={h:.1f}, conf={conf:.2f})"

        return False, "valid_block"

    def extract(
        self,
        blocks: Sequence[TextBlock],
        *,
        page_number: int = 1,
    ) -> tuple[tuple[Table, ...], tuple[Issue, ...]]:
        """Extract structured Table objects from page TextBlocks.

        Parameters
        ----------
        blocks:
            Detected text blocks for a page.
        page_number:
            1-based page number for provenance tracking.

        Returns
        -------
        tuple[tuple[Table, ...], tuple[Issue, ...]]
            Extracted Table objects and any ambiguity/extraction warnings.
        """
        if not blocks:
            return (), ()

        issues: list[Issue] = []

        # Filter out empty blocks and noise fragments
        valid_blocks: list[TextBlock] = []
        for b in blocks:
            text = b.text.strip()
            if not text:
                continue
            if self.config.filter_noise_fragments:
                is_noise_flag, noise_reason = self._is_noise_fragment(b)
                if is_noise_flag:
                    issues.append(
                        Issue(
                            code="noise_fragment_filtered",
                            message=(
                                f"Filtered OCR noise fragment '{text}' on page {page_number}: {noise_reason}"
                            ),
                            severity="info",
                            recoverable=True,
                            backend_detail=noise_reason,
                        )
                    )
                    continue
            valid_blocks.append(b)

        if not valid_blocks or len(valid_blocks) < (self.config.min_table_rows * self.config.min_table_cols):
            return (), tuple(issues)

        # ── Step 0: Estimate Table Tilt ────────────────────────────────────────
        tilt_rad = self._estimate_table_tilt(valid_blocks)
        tan_tilt = math.tan(tilt_rad)

        def _get_projected_y(b: TextBlock) -> float:
            cx = (b.bbox.left + b.bbox.right) / 2.0
            cy = (b.bbox.top + b.bbox.bottom) / 2.0
            return cy - tan_tilt * cx

        # ── Step 1: Cluster blocks into candidate horizontal rows ──────────────
        sorted_by_y = sorted(valid_blocks, key=lambda b: (_get_projected_y(b), b.bbox.left))
        candidate_rows: list[list[TextBlock]] = []

        for block in sorted_by_y:
            py = _get_projected_y(block)
            matched_row = None
            for row in candidate_rows:
                r_py = sum(_get_projected_y(b) for b in row) / len(row)
                if abs(py - r_py) <= self.config.row_vertical_tolerance_px:
                    matched_row = row
                    break
            if matched_row is not None:
                matched_row.append(block)
            else:
                candidate_rows.append([block])

        if not candidate_rows:
            return (), ()

        # Determine dominant multi-cell row width
        max_cells = max(len(r) for r in candidate_rows)
        if max_cells < self.config.min_table_cols:
            return (), ()

        # Filter to table rows: rows with sufficient cells (avoid titles / isolated metadata)
        min_cells_threshold = max(2, min(3, max_cells - 2))
        table_rows: list[list[TextBlock]] = []

        for r in candidate_rows:
            if len(r) >= min_cells_threshold:
                table_rows.append(sorted(r, key=lambda b: b.bbox.left))

        if len(table_rows) < self.config.min_table_rows:
            return (), ()

        # ── Step 2: Establish Column Anchors ───────────────────────────────────
        col_cluster_rows = [r for r in table_rows if len(r) >= max(3, max_cells - 1)]
        if not col_cluster_rows:
            col_cluster_rows = table_rows

        all_lefts: list[float] = []
        for r in col_cluster_rows:
            for b in r:
                all_lefts.append(b.bbox.left)

        all_lefts.sort()
        col_clusters: list[list[float]] = []
        for l_pos in all_lefts:
            matched_cluster = None
            for cl in col_clusters:
                avg_cl = sum(cl) / len(cl)
                if abs(l_pos - avg_cl) <= self.config.col_horizontal_tolerance_px:
                    matched_cluster = cl
                    break
            if matched_cluster is not None:
                matched_cluster.append(l_pos)
            else:
                col_clusters.append([l_pos])

        min_cluster_freq = min(2, len(col_cluster_rows))
        valid_clusters = [cl for cl in col_clusters if len(cl) >= min_cluster_freq]
        if len(valid_clusters) < self.config.min_table_cols:
            valid_clusters = col_clusters

        valid_clusters.sort(key=lambda cl: sum(cl) / len(cl))
        num_cols = len(valid_clusters)
        col_anchor_x = [sum(cl) / len(cl) for cl in valid_clusters]

        # ── Step 3: Assign Blocks to (row_idx, col_idx) ────────────────────────
        cells: list[dict[str, Any]] = []
        grid: list[list[str]] = [["" for _ in range(num_cols)] for _ in range(len(table_rows))]

        for r_idx, r_blocks in enumerate(table_rows):
            is_header_row = (r_idx == 0)

            for b in r_blocks:
                b_left = b.bbox.left
                b_right = b.bbox.right

                # Check if block spans across 2 columns (e.g. merged OCR header)
                spanned_cols = []
                for c_idx, c_x in enumerate(col_anchor_x):
                    if (b_left - 20.0) <= c_x <= (b_right + 20.0):
                        spanned_cols.append(c_idx)

                if len(spanned_cols) == 2 and self.config.interpolate_merged_headers and is_header_row:
                    words = b.text.split()
                    if len(words) >= 2:
                        split_cells = self._split_merged_header_block(b, spanned_cols)
                        for c_idx, text_part, part_bbox in split_cells:
                            conf = b.confidence or 1.0
                            grid[r_idx][c_idx] = text_part
                            cells.append({
                                "row_idx": r_idx,
                                "col_idx": c_idx,
                                "text": text_part,
                                "value": text_part,
                                "bbox": {
                                    "left": part_bbox.left,
                                    "top": part_bbox.top,
                                    "right": part_bbox.right,
                                    "bottom": part_bbox.bottom,
                                    "coordinate_space": part_bbox.coordinate_space,
                                },
                                "confidence": conf,
                                "is_header": is_header_row,
                                "page_number": page_number,
                            })
                        continue

                # Single column assignment
                best_col = None
                min_dist = float("inf")
                for c_idx, c_x in enumerate(col_anchor_x):
                    dist = abs(b_left - c_x)
                    if dist < min_dist:
                        min_dist = dist
                        best_col = c_idx

                # Ambiguity check
                competing = []
                for c_idx, c_x in enumerate(col_anchor_x):
                    if c_idx != best_col and abs(abs(b_left - c_x) - min_dist) <= self.config.ambiguity_margin_px:
                        competing.append(c_idx)

                if competing:
                    issues.append(
                        Issue(
                            code="ambiguous_cell_assignment",
                            message=(
                                f"Ambiguous column assignment for cell '{b.text}' at x={b_left:.1f} "
                                f"on page {page_number}: competing columns {best_col} and {competing}"
                            ),
                            severity="warning",
                            recoverable=True,
                        )
                    )
                    continue

                if best_col is not None and 0 <= best_col < num_cols:
                    existing_text = grid[r_idx][best_col]
                    cell_text = f"{existing_text} {b.text}".strip() if existing_text else b.text.strip()
                    grid[r_idx][best_col] = cell_text

                    conf = b.confidence or 1.0
                    cells.append({
                        "row_idx": r_idx,
                        "col_idx": best_col,
                        "text": b.text.strip(),
                        "value": b.text.strip(),
                        "bbox": {
                            "left": b.bbox.left,
                            "top": b.bbox.top,
                            "right": b.bbox.right,
                            "bottom": b.bbox.bottom,
                            "coordinate_space": b.bbox.coordinate_space,
                        },
                        "confidence": conf,
                        "is_header": is_header_row,
                        "page_number": page_number,
                    })

        # ── Step 4: Construct Table representation ─────────────────────────────
        headers = [grid[0][c].strip() for c in range(num_cols)]
        data_rows_grid = [[grid[r][c].strip() for c in range(num_cols)] for r in range(1, len(table_rows))]

        # Markdown representation
        md_lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * num_cols) + " |",
        ]
        for row in data_rows_grid:
            md_lines.append("| " + " | ".join(row) + " |")
        markdown_str = "\n".join(md_lines)

        # HTML representation
        html_lines = ["<table>", "  <thead>", "    <tr>"]
        for h in headers:
            html_lines.append(f"      <th>{h}</th>")
        html_lines.extend(["    </tr>", "  </thead>", "  <tbody>"])
        for row in data_rows_grid:
            html_lines.append("    <tr>")
            for cell_val in row:
                html_lines.append(f"      <td>{cell_val}</td>")
            html_lines.append("    </tr>")
        html_lines.extend(["  </tbody>", "</table>"])
        html_str = "\n".join(html_lines)

        # Enclosing table bounding box
        all_blocks_in_table = [b for r in table_rows for b in r]
        t_left = min(b.bbox.left for b in all_blocks_in_table)
        t_top = min(b.bbox.top for b in all_blocks_in_table)
        t_right = max(b.bbox.right for b in all_blocks_in_table)
        t_bottom = max(b.bbox.bottom for b in all_blocks_in_table)

        table_bbox = BoundingBox(
            left=t_left,
            top=t_top,
            right=t_right,
            bottom=t_bottom,
            coordinate_space=all_blocks_in_table[0].bbox.coordinate_space,
        )

        all_confs = [c["confidence"] for c in cells if c.get("confidence") is not None]
        avg_conf = round(sum(all_confs) / max(1, len(all_confs)), 4) if all_confs else None

        table_id = f"tbl_p{page_number}_0"
        extracted_table = Table(
            id=table_id,
            bbox=table_bbox,
            confidence=avg_conf,
            html=html_str,
            markdown=markdown_str,
            cells=tuple(cells),
            page_number=page_number,
        )

        return (extracted_table,), tuple(issues)

    def _split_merged_header_block(
        self,
        block: TextBlock,
        spanned_cols: list[int],
    ) -> list[tuple[int, str, BoundingBox]]:
        """Split a header block spanning multiple columns into constituent cells."""
        words = block.text.split()
        if len(spanned_cols) == 2 and len(words) >= 2:
            total_w = max(1.0, block.bbox.right - block.bbox.left)
            split_idx = len(words) - 1
            text1 = " ".join(words[:split_idx])
            text2 = " ".join(words[split_idx:])

            ratio = len(text1) / max(1.0, len(block.text))
            split_x = round(block.bbox.left + total_w * ratio, 2)

            box1 = BoundingBox(
                left=block.bbox.left,
                top=block.bbox.top,
                right=split_x,
                bottom=block.bbox.bottom,
                coordinate_space=block.bbox.coordinate_space,
            )
            box2 = BoundingBox(
                left=split_x,
                top=block.bbox.top,
                right=block.bbox.right,
                bottom=block.bbox.bottom,
                coordinate_space=block.bbox.coordinate_space,
            )
            return [(spanned_cols[0], text1, box1), (spanned_cols[1], text2, box2)]

        return [(spanned_cols[0], block.text, block.bbox)]


def extract_tables_from_blocks(
    blocks: Sequence[TextBlock],
    *,
    page_number: int = 1,
    config: TableExtractorConfig | None = None,
) -> tuple[tuple[Table, ...], tuple[Issue, ...]]:
    """Convenience function to extract Table objects from a sequence of TextBlocks."""
    return TableExtractor(config).extract(blocks, page_number=page_number)
