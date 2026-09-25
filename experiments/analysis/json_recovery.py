"""RQ4: try to repair malformed VLM JSON with a local Ollama model and re-score it."""

from __future__ import annotations

import json
import re

import numpy as np

from .box_iou import matched_box_iou

RECOVERY_ORDER = [
    "Unrecoverable",
    "Recovered but still fail",
    "Recovered and would pass",
]
RECOVERY_COLORS = {
    "Unrecoverable": "#e41a1c",
    "Recovered but still fail": "#ffb000",
    "Recovered and would pass": "#4daf4a",
}
_RECOVERY_MODEL_SPECS = {
    "qwen": {"coord_scale": 1000, "bbox_order": "xyxy"},
    "kimi": {"coord_scale": 1, "bbox_order": "xyxy"},
    "intern": {"coord_scale": 1000, "bbox_order": "xyxy"},
    "gemma": {"coord_scale": 896, "bbox_order": "yxyx"},
    "deepseek": {"coord_scale": 999, "bbox_order": "xyxy"},
    "nemotron": {"coord_scale": 1000, "bbox_order": "xyxy"},
}
_BBOX_KEYS = ("bbox", "bbox_2d", "bounding_box", "box")


def _extract_json_array_loose(text: str) -> list[dict]:
    raw_text = str(text or "").strip()
    if not raw_text:
        raise ValueError("Empty JSON recovery payload.")

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
            raise ValueError(f"Failed to decode recovery JSON array: {raw_text[:400]!r}") from exc

    if isinstance(payload, dict):
        if any(key in payload for key in _BBOX_KEYS):
            payload = [payload]
        else:
            for key in ("predictions", "objects", "detections", "results", "boxes", "output"):
                candidate = payload.get(key)
                if isinstance(candidate, list):
                    payload = candidate
                    break
            else:
                raise TypeError(
                    "Expected recovery payload to be a list or prediction container, "
                    f"got dict with keys {sorted(payload.keys())!r}."
                )

    if not isinstance(payload, list):
        raise TypeError(f"Expected recovery payload to be a list, got {type(payload).__name__}.")

    out: list[dict] = []
    for item in payload:
        if not isinstance(item, dict):
            raise TypeError(
                f"Expected each recovered prediction to be a dict, got {type(item).__name__}."
            )
        out.append(item)
    return out


def _extract_recovery_bbox(pred: dict) -> list[float]:
    for key in _BBOX_KEYS:
        if key in pred:
            bbox = pred[key]
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ValueError(f"Invalid bbox payload under key {key!r}: {bbox!r}")
            return [float(value) for value in bbox]
    raise KeyError(f"Prediction is missing bbox field: {pred!r}")


def _to_recovery_pixel_box(
    bbox: list[float],
    ref_w: int,
    ref_h: int,
    coord_scale: int | None,
    bbox_order: str,
) -> list[float]:
    if len(bbox) != 4:
        raise ValueError(f"Expected bbox of length 4, got {bbox!r}")

    a, b, c, d = bbox
    if bbox_order == "yxyx":
        x1, y1, x2, y2 = b, a, d, c
    elif bbox_order == "xyxy":
        x1, y1, x2, y2 = a, b, c, d
    else:
        raise ValueError(f"Unsupported bbox order: {bbox_order}")

    scale = float(coord_scale) if coord_scale else 1.0
    x1 = x1 * ref_w / scale
    y1 = y1 * ref_h / scale
    x2 = x2 * ref_w / scale
    y2 = y2 * ref_h / scale

    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = min(max(x1, 0.0), float(ref_w))
    y1 = min(max(y1, 0.0), float(ref_h))
    x2 = min(max(x2, 0.0), float(ref_w))
    y2 = min(max(y2, 0.0), float(ref_h))
    return [x1, y1, x2, y2]


def recover_json_bboxes_with_ollama(
    raw_response: str,
    prompt_objects: list[str] | None,
    *,
    model: str,
    host: str,
    timeout: int = 120,
) -> list[dict] | None:
    """Ask local Ollama to repair malformed MMM bbox JSON into a strict JSON array."""
    if not str(raw_response or "").strip():
        return None

    import requests

    prompt = (
        "Repair this malformed object-detection output into a strict JSON array. "
        "Return only JSON. Each element must be an object with exactly two keys: "
        '"label" and "bbox". The bbox must be [x1, y1, x2, y2]. '
        "Use only boxes that are explicitly present in the raw text. "
        "Do not invent missing coordinates. If nothing valid can be recovered, return [].\n\n"
        f"Requested labels: {json.dumps(prompt_objects or [])}\n"
        f"Raw response:\n{raw_response}"
    )

    try:
        response = requests.post(
            f"{host}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "keep_alive": "30m",
                "options": {
                    "temperature": 0,
                    "num_predict": 128,
                    "num_ctx": 4096,
                },
            },
            timeout=timeout,
        )
        response.raise_for_status()
        content = response.json()["message"]["content"]
        recovered = _extract_json_array_loose(content)
    except Exception:
        return None

    return recovered or None


def evaluate_recovered_predictions(
    predictions: list[dict],
    ground_truth_boxes: list[list[int]],
    original_size: tuple[int, int],
    *,
    model_name: str,
) -> float:
    """Score recovered MMM predictions with the same geometry-only IoU used in testing."""
    spec = _RECOVERY_MODEL_SPECS.get(str(model_name), {"coord_scale": 1, "bbox_order": "xyxy"})
    gt_boxes = np.asarray(ground_truth_boxes, dtype=np.float64)
    if gt_boxes.size == 0:
        return 0.0

    pred_boxes = [
        _to_recovery_pixel_box(
            _extract_recovery_bbox(pred),
            original_size[0],
            original_size[1],
            spec.get("coord_scale"),
            str(spec.get("bbox_order", "xyxy")),
        )
        for pred in predictions
    ]
    return matched_box_iou(pred_boxes, gt_boxes)
