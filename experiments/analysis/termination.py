"""Termination classes of a test-generation run and the cumulative ASR indicators built on them."""

from __future__ import annotations

import pandas as pd

TERMINATION_ORDER = [
    "Terminated with BBOX",
    "Baseline Failure",
    "No objects found",
    "JSON malformed",
    "Other value errors",
]
TERMINATION_COLORS = {
    "Terminated with BBOX": "#4daf4a",
    "Baseline Failure": "#ffb000",
    "No objects found": "#7f7f7f",
    "JSON malformed": "#e41a1c",
    "Other value errors": "#377eb8",
}

# Substrings of the parser errors in defaults/mmm (``extract_json_array`` / ``_extract_bbox``)
# that mean the VLM answer could not be read as a list of boxes.
_MALFORMED_MARKERS = (
    "Failed to decode VLM JSON array",
    "Expected VLM JSON payload",
    "Expected each VLM prediction",
    "Invalid bbox payload",
    "Prediction is missing bbox field",
)


def classify_termination(row: pd.Series) -> str:
    """Classify one testcase row by its terminal outcome."""
    status = str(row.get("status", "") or "")
    fail_code = str(row.get("fail_code", "") or "")
    pred_count_raw = row.get("pred_count", float("nan"))
    try:
        pred_count = int(pred_count_raw)
    except (TypeError, ValueError):
        pred_count = -1

    if status == "baseline_fail":
        return "Baseline Failure"
    if any(marker in fail_code for marker in _MALFORMED_MARKERS):
        return "JSON malformed"
    if pred_count == 0:
        return "No objects found"
    if fail_code:
        return "Other value errors"
    return "Terminated with BBOX"


def classify_termination_series(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with a ``termination_type`` column added."""
    out = df.copy()
    out["termination_type"] = out.apply(classify_termination, axis=1)
    return out


def add_cumulative_asr_outcomes(df: pd.DataFrame, iou_threshold: float = 0.25) -> pd.DataFrame:
    """Add cumulative ASR indicators (ASR_B, ASR_E, ASR_M); baseline failures stay out of denominators.

    A "Terminated with BBOX" row only counts as an `asr_bbox` success if the attack
    actually degraded `final_iou` below `iou_threshold` - a well-formed bbox on its own
    doesn't mean the attack worked. `asr_bbox_empty`/`asr_bbox_empty_malformed` don't
    need that check: "No objects found"/"JSON malformed" are already unambiguous
    full failures of the victim model regardless of IoU.
    """
    out = classify_termination_series(df)
    eligible = out["status"].eq("success")
    degraded = pd.to_numeric(out["final_iou"], errors="coerce") <= iou_threshold
    bbox = eligible & out["termination_type"].eq("Terminated with BBOX") & degraded
    empty = eligible & out["termination_type"].eq("No objects found")
    malformed = eligible & out["termination_type"].eq("JSON malformed")
    out["asr_bbox"] = bbox.where(eligible)
    out["asr_bbox_empty"] = (bbox | empty).where(eligible)
    out["asr_bbox_empty_malformed"] = (bbox | empty | malformed).where(eligible)
    return out
