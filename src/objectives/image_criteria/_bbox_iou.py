from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from torch import Tensor

from .._criterion import Criterion


def _as_box_matrix(boxes: NDArray | Tensor) -> NDArray[np.float64]:
    arr = boxes.detach().cpu().numpy() if isinstance(boxes, Tensor) else boxes

    if arr.size == 0:
        return arr.reshape(0, 4)

    if arr.ndim == 1:
        if arr.size != 4:
            raise ValueError(f"Expected a 4-value box, got shape {arr.shape}.")
        return arr.reshape(1, 4)

    if arr.ndim == 2 and arr.shape[1] == 4:
        return arr

    raise ValueError(f"Expected boxes shaped (N, 4), (4,), or empty, got shape {arr.shape}.")


def _box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter_area
    return float(inter_area / denom) if denom > 0.0 else 0.0


class VLMBBoxIoU(Criterion):
    """Mean IoU over geometrically matched bounding boxes."""

    _name: str = "VLMBBoxIoU"

    def evaluate(self, *, boxes: list[Any], **_: Any) -> float:
        """Calculate mean IoU for the IoU-optimal predicted/GT assignment.

        :param boxes: Ground-truth and predicted box collections.
        :param _: Unused extra criterion inputs.
        :returns: Mean IoU across matched box pairs.
        :raises ValueError: If the box count or shapes are invalid.
        """
        if len(boxes) != 2:
            raise ValueError(f"VLMBBoxIoU expects exactly 2 box collections, got {len(boxes)}.")

        gt_boxes = _as_box_matrix(boxes[0])
        pred_boxes = _as_box_matrix(boxes[1])

        if len(pred_boxes) == 0 or len(gt_boxes) == 0:
            return 0.0

        ious = np.zeros((len(pred_boxes), len(gt_boxes)), dtype=np.float64)
        for i, pred_box in enumerate(pred_boxes):
            for j, gt_box in enumerate(gt_boxes):
                ious[i, j] = _box_iou(pred_box, gt_box)

        pred_indices, gt_indices = linear_sum_assignment(-ious)
        matched_ious = ious[pred_indices, gt_indices]
        return float(np.mean(matched_ious))
