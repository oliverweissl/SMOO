"""Write the per-sample result files (``best_result.json``/``.png``, ``baseline_fail.json``)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.manipulator.pertubation_manipulator import MMMSample, PerturbCandidate

BEST_RESULT_FILENAME = "best_result.json"
BEST_RESULT_IMAGE_FILENAME = "best_result.png"
BASELINE_FAIL_FILENAME = "baseline_fail.json"


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


def save_best_result(
    output_dir: str | Path,
    sample: MMMSample,
    best_candidate: Any,
    runtime: float,
    generations_completed: int,
    early_stop_generation: int | None,
    population_size: int,
    total_evaluations: int,
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
        "population_size": population_size,
        "total_evaluations": total_evaluations,
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
