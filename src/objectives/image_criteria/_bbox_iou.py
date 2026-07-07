from __future__ import annotations

from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from torch import Tensor

from .._criterion import Criterion


def _as_array(value: Any) -> NDArray[np.float64]:
    if isinstance(value, Tensor):
        value = value.detach().cpu().numpy()
    return cast(NDArray[np.float64], np.asarray(value, dtype=np.float64))


def _as_box_matrix(boxes: Any) -> NDArray[np.float64]:
    arr = _as_array(boxes)
    if arr.ndim == 1:
        if arr.size != 4:
            raise ValueError(f"Expected a 4-value box, got shape {arr.shape}.")
        return cast(NDArray[np.float64], arr.reshape(1, 4))
    if arr.ndim == 2 and arr.shape[1] == 4:
        return arr
    raise ValueError(f"Expected boxes shaped (N, 4) or (4,), got shape {arr.shape}.")


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
    """Mean IoU over already-prepared bounding box pairs."""

    _name: str = "VLMBBoxIoU"

    def evaluate(self, *, boxes: list[Any], **_: Any) -> float:
        """Calculate mean IoU for a matched predicted/ground-truth box set.

        :param boxes: Predicted and ground-truth box collections.
        :param _: Unused extra criterion inputs.
        :returns: Mean IoU across matched box pairs.
        :raises ValueError: If the box count or shapes are invalid.
        """
        if len(boxes) != 2:
            raise ValueError(f"VLMBBoxIoU expects exactly 2 box collections, got {len(boxes)}.")

        pred_boxes = _as_box_matrix(boxes[0])
        gt_boxes = _as_box_matrix(boxes[1])
        if pred_boxes.shape != gt_boxes.shape:
            raise ValueError(
                f"Predicted and ground-truth boxes must have the same shape, got {pred_boxes.shape} and {gt_boxes.shape}."
            )

        ious = [_box_iou(pred_boxes[i], gt_boxes[i]) for i in range(pred_boxes.shape[0])]
        return float(sum(ious) / len(ious)) if ious else 0.0
