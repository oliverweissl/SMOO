from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.manipulator.pertubation_manipulator import (
    MMMSample,
    PerturbCandidate,
    PerturbCandidateList,
)
from src.objectives.image_criteria import VLMBBoxIoU
from src.sut import VLMSUT

BEST_RESULT_FILENAME = "best_result.json"
BEST_RESULT_IMAGE_FILENAME = "best_result.png"
BASELINE_FAIL_FILENAME = "baseline_fail.json"


def resize_image_smart(image: Image.Image, max_resolution: int) -> Image.Image:
    """Resize an image proportionally when its longest side exceeds the configured limit.

    :param image: Input PIL image.
    :param max_resolution: Maximum allowed size for the longest image side.
    :returns: The original image or a resized copy that respects ``max_resolution``.
    """
    width, height = image.size
    if max(width, height) <= max_resolution:
        return image

    scale = max_resolution / float(max(width, height))
    resized = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return image.resize(resized, Image.Resampling.LANCZOS)


def extract_json_array(text: str) -> list[dict[str, Any]]:
    """Decode the VLM response as prediction records.

    :param text: Raw VLM response text.
    :returns: The decoded prediction list.
    :raises ValueError: If JSON decoding fails.
    :raises TypeError: If the decoded payload cannot be normalized into a list of dict objects.
    """
    raw_text = text.strip()
    fence_match = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```",
        raw_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fence_match:
        raw_text = fence_match.group(1).strip()

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        try:
            payload = json.loads(f"[{raw_text}]")
        except json.JSONDecodeError:
            raise ValueError(f"Failed to decode VLM JSON array: {text[:400]!r}") from exc

    if isinstance(payload, dict):
        if any(key in payload for key in ("bbox", "bbox_2d", "bounding_box", "box")):
            payload = [payload]
        else:
            for key in ("predictions", "objects", "detections", "results", "boxes", "output"):
                candidate = payload.get(key)
                if isinstance(candidate, list):
                    payload = candidate
                    break
            else:
                raise TypeError(
                    "Expected VLM JSON payload to be a list or prediction container, "
                    f"got dict with keys {sorted(payload.keys())!r}."
                )

    if not isinstance(payload, list):
        raise TypeError(f"Expected VLM JSON payload to be a list, got {type(payload).__name__}.")
    for item in payload:
        if not isinstance(item, dict):
            raise TypeError(
                f"Expected each VLM prediction to be a dict, got {type(item).__name__}."
            )
    return payload


def extract_target_objects(prompt: str) -> list[str]:
    """Extract the requested object labels from the MMM detection prompt.

    :param prompt: Detection prompt text.
    :returns: Parsed object labels in prompt order.
    :raises ValueError: If the prompt does not contain any extractable target objects.
    """
    match = re.search(r'Detect the object\(s\)\s+"([^"]+)"', prompt)
    if match is None:
        match = re.search(r'"([^"]+)"', prompt)
    if match is None:
        raise ValueError(f"Could not extract target objects from prompt: {prompt!r}")

    objects = [item.strip() for item in match.group(1).split(",") if item.strip()]
    if not objects:
        raise ValueError(f"Extracted no target objects from prompt: {prompt!r}")
    return objects


def _normalise_label(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", label.lower()).strip()


def _labels_match(pred_label: str, valid_prompt_labels: list[str] | None) -> bool:
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


def load_sample(
    folder_path: str | Path,
    max_resolution: int,
    *,
    category: str,
    folder_id: str,
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

    if "prompt" not in base_data:
        raise KeyError(f"Missing 'prompt' in {input_json}")
    if "ground_truth" not in base_data or not isinstance(base_data["ground_truth"], dict):
        raise KeyError(f"Missing or invalid 'ground_truth' in {input_json}")

    raw_img = Image.open(input_img).convert("RGB")
    clean_image = resize_image_smart(raw_img, max_resolution)
    target_objects = extract_target_objects(base_data["prompt"])
    ground_truth_boxes = _normalize_ground_truth_boxes(base_data["ground_truth"], target_objects)

    clean_image_array = np.asarray(clean_image, dtype=np.uint8)

    return MMMSample(
        folder_path=str(folder),
        category=category,
        folder_id=folder_id,
        filename=base_data.get("image", input_img.name),
        clean_image_pil=clean_image,
        original_prompt=base_data["prompt"],
        target_objects=target_objects,
        ground_truth_boxes=ground_truth_boxes,
        original_size=raw_img.size,
        clean_image_array=clean_image_array,
        baseline_iou=float(base_data.get("IoU", 0.0)),
    )


def save_baseline_fail(output_dir: str | Path, sample: MMMSample) -> None:
    """Persist the baseline-failure record for a clean-image VLM miss.

    :param output_dir: Target directory for the baseline-failure artifact.
    :param sample: MMM sample with baseline evaluation attached.
    :raises ValueError: If baseline evaluation artifacts are missing.
    """
    if sample.baseline_iou is None:
        raise ValueError("Cannot save baseline fail without baseline_iou.")
    if sample.baseline_predictions is None:
        raise ValueError("Cannot save baseline fail without baseline predictions.")

    os.makedirs(output_dir, exist_ok=True)
    record = {
        "status": "baseline_fail",
        "baseline_iou": float(f"{sample.baseline_iou:.5f}"),
        "data_source": {
            "folder_path": sample.folder_path,
            "folder_id": sample.folder_id,
            "category": sample.category,
            "filename": sample.filename,
        },
        "original_prompt": sample.original_prompt,
        "ground_truth_bboxes": sample.ground_truth_boxes,
        "predicted_bboxes": sample.baseline_predictions,
    }
    if sample.baseline_fail_code is not None:
        record["fail_code"] = sample.baseline_fail_code
    with open(Path(output_dir) / BASELINE_FAIL_FILENAME, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)


def active_solution_shape(mode: str, image_dim: int, text_dim: int) -> tuple[int, ...]:
    """Return the optimizer genome shape required for the selected MMM mode.

    :param mode: MMM execution mode.
    :param image_dim: Number of image perturbation parameters.
    :param text_dim: Number of text perturbation parameters.
    :returns: Genome shape expected by the optimizer.
    :raises ValueError: If the mode is unsupported or one of the dimensions is invalid.
    """
    if image_dim <= 0:
        raise ValueError(f"Invalid image_dim: {image_dim}")
    if text_dim <= 0:
        raise ValueError(f"Invalid text_dim: {text_dim}")
    if mode == "image":
        return (image_dim,)
    if mode == "text":
        return (text_dim,)
    if mode == "multi":
        return (image_dim + text_dim,)
    raise ValueError(f"Unsupported MMM mode: {mode}")


def build_population_candidates(
    genomes: np.ndarray,
    sample: MMMSample,
    prompt: str,
    mode: str,
    image_dim: int,
    text_dim: int,
) -> PerturbCandidateList:
    """Create the immutable perturbation candidate list for one optimizer population.

    :param genomes: Population genome matrix.
    :param sample: MMM sample shared by the population.
    :param prompt: Prompt template for the detector.
    :param mode: MMM execution mode.
    :param image_dim: Number of image perturbation parameters.
    :param text_dim: Number of text perturbation parameters.
    :returns: Candidate list matching the optimizer population.
    :raises ValueError: If the genome shape is invalid for the selected mode.
    """
    if genomes.ndim == 1:
        genomes = genomes.reshape(1, -1)
    if genomes.ndim != 2:
        raise ValueError(f"Expected genomes to be 2D, got shape {genomes.shape}.")

    candidates = []
    for genome in genomes:
        vector = np.asarray(genome, dtype=float).reshape(-1)
        if mode == "image":
            if vector.size != image_dim:
                raise ValueError(
                    f"Image genome size {vector.size} does not match image_dim {image_dim}."
                )
            image_genome = vector.tolist()
            text_genome = [0.0] * text_dim
        elif mode == "text":
            if vector.size != text_dim:
                raise ValueError(
                    f"Text genome size {vector.size} does not match text_dim {text_dim}."
                )
            image_genome = [0.0] * image_dim
            text_genome = vector.tolist()
        elif mode == "multi":
            if vector.size != image_dim + text_dim:
                raise ValueError(
                    f"Multimodal genome size {vector.size} does not match image_dim + text_dim {image_dim + text_dim}."
                )
            image_genome = vector[:image_dim].tolist()
            text_genome = vector[image_dim : image_dim + text_dim].tolist()
        else:
            raise ValueError(f"Unsupported MMM mode: {mode}")

        candidates.append(
            PerturbCandidate(
                sample=sample,
                prompt_template=prompt,
                text_perturbation=text_genome,
                image_pertubation=image_genome,
            )
        )
    return PerturbCandidateList(*candidates)


# def ensure_rgb(image: np.ndarray) -> NDArray[np.uint8]:
#    """Ensure a numpy image is RGB-shaped.
#
#    :param image: Image array in grayscale, single-channel, or RGB form.
#    :returns: RGB image array.
#    :raises ValueError: If the array shape cannot be interpreted as an image.
#    """
#    if image.ndim == 2:
#        return cast(NDArray[np.uint8], np.repeat(image[..., None], 3, axis=2).astype(np.uint8))
#    if image.ndim == 3 and image.shape[2] == 1:
#        return cast(NDArray[np.uint8], np.repeat(image, 3, axis=2).astype(np.uint8))
#    if image.ndim != 3 or image.shape[2] != 3:
#       raise ValueError(f"Expected RGB-like image array, got shape {image.shape}.")
#    return cast(NDArray[np.uint8], image.astype(np.uint8, copy=False))


def _extract_bbox(pred: dict[str, Any]) -> list[float]:
    for key in ("bbox", "bbox_2d", "bounding_box", "box"):
        if key in pred:
            bbox = pred[key]
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ValueError(f"Invalid bbox payload under key {key!r}: {bbox!r}")
            return [float(value) for value in bbox]
    raise KeyError(f"Prediction is missing bbox field: {pred!r}")


# def _extract_label(pred: dict[str, Any]) -> str:
#    for key in ("label", "object", "class", "name", "category"):
#        if key in pred:
#            value = str(pred[key]).strip()
#            if not value:
#                raise ValueError(f"Prediction label under key {key!r} is empty: {pred!r}")
#            return value
#    raise KeyError(f"Prediction is missing label field: {pred!r}")


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
    baseline_iou = float(VLMBBoxIoU().evaluate(boxes=[pred_boxes, gt_boxes]))
    sample.baseline_predictions = parsed_preds
    sample.baseline_iou = baseline_iou
    return baseline_iou


def save_best_result(
    output_dir: str | Path,
    sample: MMMSample,
    best_candidate: Any,
    runtime: float,
    generations_completed: int,
    early_stop_generation: int | None,
) -> None:
    """Persist the selected best MMM testcase and its metadata.

    :param output_dir: Target directory for the saved testcase.
    :param sample: Source sample corresponding to the saved candidate.
    :param best_candidate: Optimizer candidate carrying the MMM candidate payload.
    :param runtime: Total runtime in seconds for the sample.
    :param generations_completed: Number of generations evaluated.
    :param early_stop_generation: Generation index where early stopping occurred.
    :raises ValueError: If required evaluation artifacts are missing.
    """
    os.makedirs(output_dir, exist_ok=True)

    candidate: PerturbCandidate = (
        best_candidate.data[0] if isinstance(best_candidate.data, tuple) else best_candidate.data
    )
    if (
        candidate.vlm_response is None
        or candidate.parsed_predictions is None
        or candidate.prompt_objects is None
    ):
        raise ValueError("Best candidate is missing evaluation artifacts required for saving.")
    if sample.baseline_iou is None:
        raise ValueError("Sample is missing baseline_iou required for saving.")

    fitness = [float(value) for value in best_candidate.fitness]
    record = {
        "data_source": {
            "folder_path": sample.folder_path,
            "folder_id": sample.folder_id,
            "category": sample.category,
            "filename": sample.filename,
        },
        "runtime": runtime,
        "generations_completed": generations_completed,
        "early_stop_generation": early_stop_generation,
        "baseline_iou": float(f"{sample.baseline_iou:.5f}"),
        "genome": np.asarray(best_candidate.solution).reshape(-1).tolist(),
        "objectives": {
            "iou": float(f"{fitness[0]:.5f}"),
            "img_dist": float(f"{fitness[1]:.5f}"),
            "txt_dist": float(f"{fitness[2]:.5f}"),
        },
        "original_prompt": sample.original_prompt,
        "ground_truth_bboxes": sample.ground_truth_boxes,
        "vlm_output": {
            "perturbed_prompt": candidate.format_prompt(),
            "raw_response": candidate.vlm_response,
            "parsed_predictions": candidate.parsed_predictions,
            "matched_pred_boxes": candidate.matched_pred_boxes,
            "prompt_objects": candidate.prompt_objects,
        },
    }

    if candidate.fail_code is not None:
        record["fail_code"] = candidate.fail_code

    with open(Path(output_dir) / BEST_RESULT_FILENAME, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    Image.fromarray(candidate.image_array).save(Path(output_dir) / BEST_RESULT_IMAGE_FILENAME)
