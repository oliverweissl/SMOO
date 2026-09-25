"""Load one selected MMM sample (image, prompt, ground-truth boxes) from disk."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.manipulator.pertubation_manipulator import MMMSample

from ._prompts import DETECTION_PROMPT


@dataclass
class SkippedSample(Exception):
    """Signal that a sample should be skipped before baseline evaluation."""

    folder_path: str
    original_bbox_count: int
    filtered_bbox_count: int
    threshold: float


def extract_target_objects_from_ground_truth(ground_truth: dict[str, Any]) -> list[str]:
    """Derive prompt object order from ``ground_truth`` keys when no prompt is stored."""
    if not isinstance(ground_truth, dict):
        raise TypeError(
            f"Expected ground_truth to be a dict, got {type(ground_truth).__name__}."
        )

    target_objects: list[str] = []
    for key in ground_truth:
        base_label = re.sub(r"_\d+$", "", str(key)).strip()
        if base_label and base_label not in target_objects:
            target_objects.append(base_label)

    if not target_objects:
        raise ValueError("Extracted no target objects from ground_truth.")
    return target_objects


def _normalise_label(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", label.lower()).strip()


def _labels_match(pred_label: str, valid_prompt_labels: list[str] | None) -> bool:
    """Whether a label equals, or shares a token subset with, one of ``valid_prompt_labels``.

    Only used while loading a sample, to find the ground-truth key for each prompt object.
    Predicted boxes are never filtered by label; IoU matching is geometry-only (see ``_scoring``).
    """
    pred_norm = _normalise_label(pred_label)
    pred_tokens = set(pred_norm.split())

    for label in valid_prompt_labels or []:
        label_norm = _normalise_label(label)
        label_tokens = set(label_norm.split())
        if pred_norm == label_norm:
            return True
        if (
            pred_tokens
            and label_tokens
            and (pred_tokens <= label_tokens or label_tokens <= pred_tokens)
        ):
            return True
    return False


def _bbox_dict_to_xyxy(bbox: dict[str, Any]) -> list[int]:
    required = ("xmin", "ymin", "xmax", "ymax")
    if missing := [key for key in required if key not in bbox]:
        raise KeyError(f"Ground-truth bbox is missing keys {missing}: {bbox!r}")
    return [int(bbox["xmin"]), int(bbox["ymin"]), int(bbox["xmax"]), int(bbox["ymax"])]


def _normalize_ground_truth_boxes(
    ground_truth: dict[str, Any], target_objects: list[str]
) -> list[list[int]]:
    if not isinstance(ground_truth, dict):
        raise TypeError(f"Expected ground truth to be a dict, got {type(ground_truth).__name__}.")

    remaining = list(ground_truth.items())
    normalized_boxes: list[list[int]] = []
    for target_object in target_objects:
        match_index = None
        for index, (label, bbox) in enumerate(remaining):
            gt_label = label.split("_")[0]
            if _labels_match(gt_label, [target_object]):
                match_index = index
                break
        if match_index is None:
            raise KeyError(f"Could not find ground-truth box for target object {target_object!r}.")
        _, bbox = remaining.pop(match_index)
        if not isinstance(bbox, dict):
            raise TypeError(f"Ground-truth bbox payload must be a dict, got {type(bbox).__name__}.")
        normalized_boxes.append(_bbox_dict_to_xyxy(bbox))
    return normalized_boxes


def _resolve_original_dims(base_data: dict[str, Any], fallback_size: tuple[int, int]) -> tuple[int, int]:
    dims = base_data.get("original_dims")
    if (
        isinstance(dims, (list, tuple))
        and len(dims) == 2
        and all(isinstance(value, (int, float)) and value > 0 for value in dims)
    ):
        return int(dims[0]), int(dims[1])
    return fallback_size


def _filter_ground_truth_by_area_fraction(
    ground_truth: dict[str, Any],
    original_size: tuple[int, int],
    min_bbox_area_fraction: float,
) -> dict[str, Any]:
    if min_bbox_area_fraction < 0:
        raise ValueError(
            f"min_bbox_area_fraction must be non-negative, got {min_bbox_area_fraction}."
        )
    if min_bbox_area_fraction <= 0:
        return dict(ground_truth)

    image_width, image_height = original_size
    image_area = float(image_width * image_height)
    filtered: dict[str, Any] = {}
    for label, bbox in ground_truth.items():
        if not isinstance(bbox, dict):
            raise TypeError(f"Ground-truth bbox payload must be a dict, got {type(bbox).__name__}.")
        xyxy = _bbox_dict_to_xyxy(bbox)
        box_width = max(0, xyxy[2] - xyxy[0])
        box_height = max(0, xyxy[3] - xyxy[1])
        bbox_area_fraction = (box_width * box_height) / image_area if image_area > 0 else 0.0
        if bbox_area_fraction >= min_bbox_area_fraction:
            filtered[label] = bbox
    return filtered


def load_sample(
    folder_path: str | Path,
    max_resolution: int,
    *,
    category: str,
    folder_id: str,
    min_bbox_area_fraction: float = 0.0,
) -> MMMSample:
    """Load one selected MMM sample into a typed runtime object.

    :param folder_path: Path to the MMM sample directory.
    :param max_resolution: Maximum allowed size for the longest image side.
    :param category: Category relative path for the sample.
    :param folder_id: Numeric sample identifier.
    :returns: Normalized MMM sample payload.
    :raises FileNotFoundError: If the sample directory is missing required input files.
    :raises KeyError: If required JSON fields are absent or malformed.
    """
    folder = Path(folder_path)
    input_json = folder / "original.json"
    input_img = folder / "data_point.JPEG"
    if not input_json.exists() or not input_img.exists():
        raise FileNotFoundError(f"Missing original.json or data_point.JPEG in {folder}")

    with input_json.open("r", encoding="utf-8") as handle:
        base_data = json.load(handle)

    if "ground_truth" not in base_data or not isinstance(base_data["ground_truth"], dict):
        raise KeyError(f"Missing or invalid 'ground_truth' in {input_json}")

    raw_img = Image.open(input_img).convert("RGB")
    width, height = raw_img.size
    scale = min(max_resolution / max(width, height), 1)
    clean_image = raw_img.resize(
        (round(width * scale), round(height * scale)),
        Image.Resampling.LANCZOS,
    )

    original_size = _resolve_original_dims(base_data, raw_img.size)
    original_ground_truth = base_data["ground_truth"]
    filtered_ground_truth = _filter_ground_truth_by_area_fraction(
        original_ground_truth,
        original_size,
        min_bbox_area_fraction,
    )
    if not filtered_ground_truth:
        raise SkippedSample(
            folder_path=str(folder),
            original_bbox_count=len(original_ground_truth),
            filtered_bbox_count=0,
            threshold=min_bbox_area_fraction,
        )

    target_objects = extract_target_objects_from_ground_truth(filtered_ground_truth)
    original_prompt = DETECTION_PROMPT.format(objects=", ".join(target_objects))

    ground_truth_boxes = _normalize_ground_truth_boxes(filtered_ground_truth, target_objects)

    clean_image_array = np.asarray(clean_image, dtype=np.uint8)

    return MMMSample(
        folder_path=str(folder),
        category=category,
        folder_id=folder_id,
        filename=base_data.get("image", input_img.name),
        clean_image_pil=clean_image,
        original_prompt=original_prompt,
        target_objects=target_objects,
        ground_truth_boxes=ground_truth_boxes,
        original_size=original_size,
        clean_image_array=clean_image_array,
        baseline_iou=float(base_data.get("IoU", 0.0)),
    )
