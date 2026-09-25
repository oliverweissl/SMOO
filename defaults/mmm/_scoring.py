"""Turn parsed VLM predictions into pixel boxes and score them with ``VLMBBoxIoU``."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.manipulator.pertubation_manipulator import MMMSample
from src.objectives.image_criteria import VLMBBoxIoU
from src.sut import VLMSUT

from ._parsing import extract_json_array


def _extract_bbox(pred: dict[str, Any]) -> list[float]:
    for key in ("bbox", "bbox_2d", "bounding_box", "box"):
        if key in pred:
            bbox = pred[key]
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ValueError(f"Invalid bbox payload under key {key!r}: {bbox!r}")
            return [float(value) for value in bbox]
    raise KeyError(f"Prediction is missing bbox field: {pred!r}")


def _to_pixel_box(
    bbox: list[float],
    ref_w: int,
    ref_h: int,
    coord_scale: int | None,
    bbox_order: str,
) -> list[float]:
    if bbox_order == "yxyx":
        y1, x1, y2, x2 = bbox
    elif bbox_order == "xyxy":
        x1, y1, x2, y2 = bbox
    else:
        raise ValueError(f"Unsupported bbox order: {bbox_order}")

    return [
        x1 * ref_w / coord_scale,
        y1 * ref_h / coord_scale,
        x2 * ref_w / coord_scale,
        y2 * ref_h / coord_scale,
    ]


def prepare_bbox_pairs(
    ground_truth_boxes: list[list[int]],
    original_size: tuple[int, int],
    pred_list: list[Any],
    coord_scale: int | None,
    bbox_order: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize prediction and GT boxes for geometry-only IoU matching.

    :param ground_truth_boxes: Ground-truth boxes in pixel coordinates.
    :param original_size: Original image size as ``(width, height)``.
    :param pred_list: Parsed VLM predictions.
    :param coord_scale: Optional coordinate scale used by the VLM output format.
    :param bbox_order: Coordinate order used by predicted boxes.
    :returns: Predicted boxes and ground-truth boxes as arrays.
    :raises TypeError: If predictions are not a list of dicts.
    :raises ValueError: If box shapes are invalid.
    """
    if not isinstance(pred_list, list):
        raise TypeError(
            f"Expected parsed predictions to be a list, got {type(pred_list).__name__}."
        )

    gt_boxes = np.asarray(ground_truth_boxes, dtype=np.float64)
    if gt_boxes.size == 0:
        gt_boxes = np.zeros((0, 4), dtype=np.float64)
    elif gt_boxes.ndim != 2 or gt_boxes.shape[1] != 4:
        raise ValueError(f"Expected ground_truth_boxes shaped (N, 4), got {gt_boxes.shape}.")

    pred_boxes: list[np.ndarray] = []
    for pred in pred_list:
        if not isinstance(pred, dict):
            raise TypeError(f"Expected prediction dict, got {type(pred).__name__}.")
        pred_bbox = _extract_bbox(pred)
        pred_boxes.append(
            np.array(
                _to_pixel_box(
                    pred_bbox, original_size[0], original_size[1], coord_scale, bbox_order
                ),
                dtype=np.float64,
            )
        )

    if pred_boxes:
        pred_box_matrix = np.vstack(pred_boxes)
    else:
        pred_box_matrix = np.zeros((0, 4), dtype=np.float64)
    return pred_box_matrix, gt_boxes


def evaluate_baseline(sut: VLMSUT, sample: MMMSample) -> float:
    """Run baseline VLM inference on the clean sample and compute mean IoU.

    :param sut: VLM system under test.
    :param sample: MMM sample to evaluate.
    :returns: Baseline IoU on the clean sample.
    :raises ValueError: If the VLM returns an invalid response count.
    """
    responses = sut.process_input(([sample.clean_image_pil], [sample.original_prompt]))
    if len(responses) != 1:
        raise ValueError(f"Expected exactly one baseline response, got {len(responses)}.")

    sample.baseline_fail_code = None
    try:
        parsed_preds = extract_json_array(responses[0])
    except (ValueError, TypeError) as exc:
        sample.baseline_fail_code = str(exc)
        sample.baseline_predictions = []
        sample.baseline_iou = 0.0
        return 0.0

    try:
        pred_boxes, gt_boxes = prepare_bbox_pairs(
            sample.ground_truth_boxes,
            sample.original_size,
            parsed_preds,
            sut.coord_scale,
            sut.bbox_order,
        )
    except (ValueError, TypeError) as exc:
        sample.baseline_fail_code = str(exc)
        sample.baseline_predictions = []
        sample.baseline_iou = 0.0
        return 0.0
    baseline_iou = float(VLMBBoxIoU().evaluate(boxes=[gt_boxes, pred_boxes]))
    sample.baseline_predictions = parsed_preds
    sample.baseline_iou = baseline_iou
    return baseline_iou
