"""Shared data loading utilities for the RQ analysis notebooks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

RESULTS_ROOT = Path(__file__).parent.parent / "defaults" / "mmm" / "results"
LEGACY_RESULTS_ROOT = Path(__file__).parent.parent / "results"

IMG_CORRUPTIONS = [
    "elastic",
    "gaussian_noise",
    "defocus_blur",
    "motion_blur",
    "fog_filter",
    "snow_filter",
    "contrast",
    "false_color",
    "grayscale",
    "cutout",
    "pixelate",
    "jpeg_filter",
]
TXT_OBJECT_CORRUPTIONS = [
    "homophone",
    "synonym",
    "ata_saliency",
    "fragmentation",
    "character_noise",
]
TXT_PROMPT_CORRUPTIONS = [
    "universal_suffix_injection",
    "context_rot_injection",
    "task_reinforcement",
]
TXT_CORRUPTIONS = TXT_OBJECT_CORRUPTIONS + TXT_PROMPT_CORRUPTIONS

_IMAGE_GENOME_ORDER = [
    "jpeg_filter",
    "pixelate",
    "defocus_blur",
    "motion_blur",
    "gaussian_noise",
    "fog_filter",
    "snow_filter",
    "contrast",
    "elastic",
    "cutout",
    "false_color",
    "grayscale",
]
_TEXT_GENOME_ORDER = [
    "fragmentation",
    "character_noise",
    "homophone",
    "synonym",
    "ata_saliency",
    "universal_suffix_injection",
    "context_rot_injection",
    "task_reinforcement",
]

_EMPTY_COLUMNS = {
    "model": pd.Series(dtype="object"),
    "modality": pd.Series(dtype="object"),
    "genome_mode": pd.Series(dtype="object"),
    "obj_category": pd.Series(dtype="object"),
    "folder_id": pd.Series(dtype="object"),
    "filename": pd.Series(dtype="object"),
    "status": pd.Series(dtype="object"),
    "fail_code": pd.Series(dtype="object"),
    "pred_count": pd.Series(dtype="float64"),
    "has_predictions": pd.Series(dtype="bool"),
    "baseline_iou": pd.Series(dtype="float64"),
    "final_iou": pd.Series(dtype="float64"),
    "iou_reduction": pd.Series(dtype="float64"),
    "img_dist": pd.Series(dtype="float64"),
    "txt_dist": pd.Series(dtype="float64"),
    "txt_sim": pd.Series(dtype="float64"),
    "img_budget_used": pd.Series(dtype="float64"),
    "txt_budget_used": pd.Series(dtype="float64"),
    "budget_max": pd.Series(dtype="float64"),
    "runtime": pd.Series(dtype="float64"),
    "generations_completed": pd.Series(dtype="float64"),
    "total_evaluations": pd.Series(dtype="float64"),
    "skipped_evaluations": pd.Series(dtype="float64"),
    "early_stopped": pd.Series(dtype="bool"),
    "early_stop_generation": pd.Series(dtype="float64"),
    "pareto_index": pd.Series(dtype="float64"),
    "_best_img_path": pd.Series(dtype="object"),
    "_orig_img_folder": pd.Series(dtype="object"),
    "_result_json_path": pd.Series(dtype="object"),
    "original_prompt": pd.Series(dtype="object"),
    "perturbed_prompt": pd.Series(dtype="object"),
    "modality_label": pd.Series(dtype="object"),
    "mode_label": pd.Series(dtype="object"),
    "category_label": pd.Series(dtype="object"),
    "split": pd.Series(dtype="object"),
}
for corruption in IMG_CORRUPTIONS:
    _EMPTY_COLUMNS[f"img_{corruption}"] = pd.Series(dtype="float64")
for corruption in TXT_CORRUPTIONS:
    _EMPTY_COLUMNS[f"txt_{corruption}"] = pd.Series(dtype="float64")


def _active_results_root() -> Path:
    return RESULTS_ROOT if RESULTS_ROOT.exists() else LEGACY_RESULTS_ROOT


def _discover_models(results_root: Path) -> list[str]:
    return sorted(
        path.name for path in results_root.iterdir() if path.is_dir() and path.name != "selection"
    )


def _infer_genome_mode(parts: tuple[str, ...], model: str) -> tuple[str, str] | None:
    model_idx = next((i for i, part in enumerate(parts) if part == model), None)
    if model_idx is None or model_idx + 1 >= len(parts):
        return None

    modality = parts[model_idx + 1]
    if modality == "multimodal":
        return modality, "multi"
    if modality == "unimodal":
        if model_idx + 2 >= len(parts):
            return modality, "unknown"
        return modality, parts[model_idx + 2]
    return modality, "unknown"


def _infer_obj_category(data_source: dict[str, Any], path: Path) -> str:
    category = str(data_source.get("category", "")).strip()
    if category:
        return category

    folder_path = str(data_source.get("folder_path", ""))
    combined = f"{path} {folder_path}"
    if "single/solo" in combined:
        return "single/solo"
    if "single/multi" in combined:
        return "single/multi"
    if "udacity" in combined:
        return "udacity"
    return "multi"


def _split_genome(genome: Any, genome_mode: str) -> tuple[list[float], list[float]]:
    values = []
    if isinstance(genome, list):
        for value in genome:
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                values.append(0.0)

    image_dim = len(_IMAGE_GENOME_ORDER)
    text_dim = len(_TEXT_GENOME_ORDER)

    if genome_mode == "multi":
        image_values = values[:image_dim]
        text_values = values[image_dim : image_dim + text_dim]
    elif genome_mode == "image":
        image_values = values[:image_dim]
        text_values = []
    elif genome_mode == "text":
        image_values = []
        text_values = values[:text_dim]
    else:
        image_values = []
        text_values = []

    image_values += [0.0] * (image_dim - len(image_values))
    text_values += [0.0] * (text_dim - len(text_values))
    return image_values[:image_dim], text_values[:text_dim]


def _build_corruption_columns(genome: Any, genome_mode: str) -> dict[str, float]:
    image_values, text_values = _split_genome(genome, genome_mode)
    image_map = dict(zip(_IMAGE_GENOME_ORDER, image_values))
    text_map = dict(zip(_TEXT_GENOME_ORDER, text_values))

    row: dict[str, float] = {}
    for corruption in IMG_CORRUPTIONS:
        row[f"img_{corruption}"] = float(image_map.get(corruption, 0.0))
    for corruption in TXT_CORRUPTIONS:
        row[f"txt_{corruption}"] = float(text_map.get(corruption, 0.0))
    row["img_budget_used"] = float(sum(image_values))
    row["txt_budget_used"] = float(sum(text_values))
    return row


def _parse_best_result(path: Path, model: str) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None

    inferred = _infer_genome_mode(path.parts, model)
    if inferred is None:
        return None
    modality, genome_mode = inferred

    data_source = data.get("data_source", {})
    folder_path = str(data_source.get("folder_path", ""))
    obj_category = _infer_obj_category(data_source, path)
    objectives = data.get("objectives", {})
    baseline_iou = float(data.get("baseline_iou", float("nan")))
    final_iou = float(objectives.get("iou", float("nan")))
    vlm = data.get("vlm_output", {})

    parsed_predictions = vlm.get("parsed_predictions", [])
    pred_count = len(parsed_predictions) if isinstance(parsed_predictions, list) else float("nan")
    row: dict[str, Any] = {
        "model": model,
        "modality": modality,
        "genome_mode": genome_mode,
        "obj_category": obj_category,
        "folder_id": str(data_source.get("folder_id", path.parent.name)),
        "filename": data_source.get("filename", ""),
        "status": "success",
        "fail_code": data.get("fail_code", ""),
        "pred_count": pred_count,
        "has_predictions": bool(pred_count) if pred_count == pred_count else False,
        "baseline_iou": baseline_iou,
        "final_iou": final_iou,
        "iou_reduction": (baseline_iou - final_iou) / baseline_iou,
        "img_dist": float(objectives.get("img_dist", float("nan"))),
        "txt_dist": float(objectives.get("txt_dist", float("nan"))),
        "txt_sim": float("nan"),
        "budget_max": 1.0,
        "runtime": float(data.get("runtime", float("nan"))),
        "generations_completed": float(data.get("generations_completed", float("nan"))),
        "total_evaluations": float(data.get("total_evaluations", float("nan"))),
        "skipped_evaluations": float(data.get("skipped_evaluations", float("nan"))),
        "early_stop_generation": data.get("early_stop_generation"),
        "pareto_index": float(data.get("pareto_index", float("nan"))),
        "_best_img_path": str(path.parent / "best_result.png"),
        "_orig_img_folder": folder_path,
        "_result_json_path": str(path),
        "original_prompt": data.get("original_prompt", ""),
        "perturbed_prompt": vlm.get("perturbed_prompt", ""),
    }
    row["early_stopped"] = row["early_stop_generation"] is not None
    row.update(_build_corruption_columns(data.get("genome", []), genome_mode))
    return row


def _parse_baseline_fail(path: Path, model: str) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None

    inferred = _infer_genome_mode(path.parts, model)
    if inferred is None:
        return None
    modality, genome_mode = inferred

    data_source = data.get("data_source", {})
    folder_path = str(data_source.get("folder_path", ""))
    obj_category = _infer_obj_category(data_source, path)
    baseline_iou = float(data.get("baseline_iou", float("nan")))

    predicted_bboxes = data.get("predicted_bboxes", [])
    pred_count = len(predicted_bboxes) if isinstance(predicted_bboxes, list) else float("nan")
    row: dict[str, Any] = {
        "model": model,
        "modality": modality,
        "genome_mode": genome_mode,
        "obj_category": obj_category,
        "folder_id": str(data_source.get("folder_id", path.parent.name)),
        "filename": data_source.get("filename", ""),
        "status": "baseline_fail",
        "fail_code": data.get("fail_code", ""),
        "pred_count": pred_count,
        "has_predictions": bool(pred_count) if pred_count == pred_count else False,
        "baseline_iou": baseline_iou,
        "final_iou": baseline_iou,
        "iou_reduction": 0.0,
        "img_dist": 0.0,
        "txt_dist": 0.0,
        "txt_sim": float("nan"),
        "img_budget_used": float("nan"),
        "txt_budget_used": float("nan"),
        "budget_max": 1.0,
        "runtime": float("nan"),
        "generations_completed": float("nan"),
        "total_evaluations": float("nan"),
        "skipped_evaluations": float("nan"),
        "early_stopped": False,
        "early_stop_generation": None,
        "pareto_index": float("nan"),
        "_best_img_path": "",
        "_orig_img_folder": folder_path,
        "_result_json_path": str(path),
        "original_prompt": data.get("original_prompt", ""),
        "perturbed_prompt": "",
    }
    row.update({f"img_{corruption}": 0.0 for corruption in IMG_CORRUPTIONS})
    row.update({f"txt_{corruption}": 0.0 for corruption in TXT_CORRUPTIONS})
    return row


def _empty_results_frame() -> pd.DataFrame:
    return pd.DataFrame(_EMPTY_COLUMNS)


def load_all_results(
    models: list[str] | None = None,
    include_baseline_fail: bool = True,
) -> pd.DataFrame:
    """Load all best_result.json and optionally baseline_fail.json files."""
    results_root = _active_results_root()
    if models is None:
        models = _discover_models(results_root)

    rows: list[dict[str, Any]] = []
    for model in models:
        model_dir = results_root / model
        if not model_dir.exists() or model == "selection":
            continue
        for path in model_dir.rglob("best_result.json"):
            row = _parse_best_result(path, model)
            if row is not None:
                rows.append(row)
        if include_baseline_fail:
            for path in model_dir.rglob("baseline_fail.json"):
                row = _parse_baseline_fail(path, model)
                if row is not None:
                    rows.append(row)

    if not rows:
        return _empty_results_frame()

    df = pd.DataFrame(rows)
    for column in _EMPTY_COLUMNS:
        if column not in df.columns:
            df[column] = _EMPTY_COLUMNS[column]

    df["modality_label"] = (
        df["modality"]
        .map({"multimodal": "Multimodal", "unimodal": "Unimodal"})
        .fillna(df["modality"])
    )
    df["mode_label"] = (
        df["genome_mode"]
        .map({"multi": "Multi (img+txt)", "image": "Image only", "text": "Text only"})
        .fillna(df["genome_mode"])
    )
    df["category_label"] = (
        df["obj_category"]
        .map(
            {
                "multi": "Multi-object",
                "single/solo": "Single-object (solo)",
                "single/multi": "Single-object (multi)",
                "udacity": "Driving",
            }
        )
        .fillna(df["obj_category"])
    )
    df["split"] = df["modality"] + "/" + df["genome_mode"] + "/" + df["obj_category"]
    return df


def success_only(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows where optimization actually ran (no baseline_fail)."""
    if "status" not in df.columns:
        return _empty_results_frame().iloc[0:0].copy()
    return df[df["status"] == "success"].copy()
