"""Bounding-box IoU for the analysis code.

This is the same metric as ``VLMBBoxIoU`` in ``src/objectives/image_criteria/_bbox_iou.py``,
which scores every test case during generation. It is repeated here because importing ``src``
pulls in the whole runtime stack (vLLM, torch), which the analysis environment does not need.
``tests/test_box_iou.py`` checks that both implementations agree.

Labels are not used: predicted and ground-truth boxes are paired one-to-one by the
Hungarian assignment that maximises total IoU, and the matched IoUs are summed and divided
by ``max(|pred|, |gt|)``, so missing and extra boxes both count as IoU 0.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def box_iou(box_a, box_b) -> float:
    """IoU of two ``[x1, y1, x2, y2]`` boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter_area
    return float(inter_area / denom) if denom > 0.0 else 0.0


def matched_box_iou(pred_boxes, gt_boxes) -> float:
    """Mean IoU over the Hungarian-optimal pred/GT assignment, normalised by ``max(|pred|, |gt|)``."""
    pred = np.asarray(pred_boxes, dtype=np.float64).reshape(-1, 4) if len(pred_boxes) else np.zeros((0, 4))
    gt = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4) if len(gt_boxes) else np.zeros((0, 4))
    if len(pred) == 0 or len(gt) == 0:
        return 0.0

    ious = np.array([[box_iou(p, g) for g in gt] for p in pred], dtype=np.float64)
    pred_idx, gt_idx = linear_sum_assignment(-ious)
    return float(ious[pred_idx, gt_idx].sum() / max(len(pred), len(gt)))
