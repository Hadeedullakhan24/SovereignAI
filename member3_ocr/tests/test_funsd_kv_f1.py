"""Unit tests for FUNSD Key-Value Extraction F1 evaluation and matching logic."""
from __future__ import annotations


import pytest

from member3_ocr.evaluation.document.evaluate_funsd_kv_f1 import (
    GroundTruthField,
    compute_f1,
    compute_string_similarity,
    match_form_fields,
    normalize_key,
    normalize_text,
)


def test_compute_f1_hand_computed_example():
    """Verify compute_f1 on edge cases and standard hand-computed values."""
    # Zero counts
    assert compute_f1(0, 0, 0) == (0.0, 0.0, 0.0)

    # Perfect match
    assert compute_f1(5, 0, 0) == (1.0, 1.0, 1.0)

    # Hand-computed: TP=3, FP=1, FN=2
    # Prec = 3 / 4 = 0.75, Rec = 3 / 5 = 0.60, F1 = 2 * (0.75 * 0.6) / 1.35 = 0.6667
    prec, rec, f1 = compute_f1(3, 1, 2)
    assert prec == 0.75
    assert rec == 0.60
    assert f1 == pytest.approx(0.6667, abs=1e-4)


def test_normalize_key_and_text():
    """Verify normalization handles colons, hyphens, case, and spacing."""
    assert normalize_key("DATE:") == "date"
    assert normalize_key("  REPORT NO - ") == "report no"
    assert normalize_key("TO :") == "to"
    assert normalize_key("FAX NUMBER: ") == "fax number"

    assert normalize_text("  George   Baroody \n") == "george baroody"
    assert normalize_text("") == ""


def test_match_form_fields_hand_computed_fixture():
    """Verify field matching against a controlled, hand-computed ground truth fixture."""
    gt_fields = [
        GroundTruthField(
            key_id=1,
            raw_key="Date:",
            normalized_key="date",
            value_ids=[10],
            raw_value="2026-09-13",
            normalized_value="2026-09-13",
            is_long_value=False,
        ),
        GroundTruthField(
            key_id=2,
            raw_key="Tag:",
            normalized_key="tag",
            value_ids=[20],
            raw_value="P-101",
            normalized_value="p-101",
            is_long_value=False,
        ),
        GroundTruthField(
            key_id=3,
            raw_key="Location:",
            normalized_key="location",
            value_ids=[30],
            raw_value="Unit 2",
            normalized_value="unit 2",
            is_long_value=False,
        ),
        GroundTruthField(
            key_id=4,
            raw_key="Remarks:",
            normalized_key="remarks",
            value_ids=[40],
            raw_value="Regular inspection completed without issue.",
            normalized_value="regular inspection completed without issue.",
            is_long_value=True,
        ),
    ]

    # Predictions:
    # 1. Date -> Exact match
    # 2. Tag -> Key matches, value is "p-101a" (sim > 0.70, not exact)
    # 3. Status -> Spurious (FP)
    # 4. Remarks -> Exact match
    # (Location is missing -> FN)
    pred_fields = [
        ("Date:", "2026-09-13", 0.95),
        ("Tag", "P-101A", 0.90),
        ("Status:", "Operational", 0.85),
        ("Remarks:", "Regular inspection completed without issue.", 0.92),
    ]

    res = match_form_fields(gt_fields, pred_fields, similarity_threshold=0.70)

    assert res["total_gt"] == 4
    assert res["total_pred"] == 4

    # Exact match: TP=2 (Date, Remarks), FP=2, FN=2 -> Prec=0.5, Rec=0.5, F1=0.5
    assert res["tp_exact"] == 2
    assert res["fp_exact"] == 2
    assert res["fn_exact"] == 2
    assert res["prec_exact"] == 0.50
    assert res["rec_exact"] == 0.50
    assert res["f1_exact"] == 0.50

    # Robust match: TP=3 (Date, Tag, Remarks), FP=1 (Status), FN=1 (Location)
    assert res["tp_robust"] == 3
    assert res["fp_robust"] == 1
    assert res["fn_robust"] == 1
    assert res["prec_robust"] == 0.75
    assert res["rec_robust"] == 0.75
    assert res["f1_robust"] == 0.75

    # Key match only: TP=3, FP=1, FN=1
    assert res["tp_key"] == 3
    assert res["fp_key"] == 1
    assert res["fn_key"] == 1
    assert res["f1_key"] == 0.75


def test_string_similarity():
    """Verify SequenceMatcher similarity calculation."""
    assert compute_string_similarity("abc", "abc") == 1.0
    assert compute_string_similarity("", "") == 1.0
    assert compute_string_similarity("abc", "") == 0.0
    # minor typo
    sim = compute_string_similarity("p-101", "p-101a")
    assert sim >= 0.80
