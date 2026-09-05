"""Shared computation and plot-setup utilities for RQ notebooks."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from xml.etree import ElementTree

import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
import subprocess
from scipy.optimize import linear_sum_assignment
from jiwer import cer
from PIL import Image
from sewar import msssim

from ._loader import IMG_CORRUPTIONS, RESULTS_ROOT, TXT_CORRUPTIONS

IMG_COLS = [f"img_{c}" for c in IMG_CORRUPTIONS]
TXT_COLS = [f"txt_{c}" for c in TXT_CORRUPTIONS]
ALL_CORRUPT_COLS = IMG_COLS + TXT_COLS

PALETTE = sns.color_palette("tab10")

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


def tex(s: str) -> str:
    """Escape a plain string for LaTeX text mode (underscores -> spaces)."""
    return s.replace("_", " ")


def setup_matplotlib() -> None:
    """Configure matplotlib: LaTeX rendering, serif font, 18 pt."""
    mpl.rcParams.update(
        {
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{amsmath}",
            "font.family": "serif",
            "font.size": 18,
            "axes.titlesize": 20,
            "axes.labelsize": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
            "legend.title_fontsize": 16,
            "figure.titlesize": 22,
        }
    )


def compute_text_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'ned' column to df for text-corrupted rows."""
    df = df.copy()
    ned = []
    for _, row in df.iterrows():
        orig = str(row.get("original_prompt", "") or "")
        pert = str(row.get("perturbed_prompt", "") or "")
        ned.append(cer(orig, pert) if (orig or pert) else 0.0)
    df["ned"] = ned
    return df


def compute_swad(df: pd.DataFrame) -> pd.DataFrame:
    """Add stealth and stealth-weighted adversarial degradation (SWAD).

    Distance measures are MS-SSIM (image) and NED (text). MS-SSIM is a
    similarity score (1 = identical), so it's inverted to a distance;
    NED is already a distance (0 = identical), so it's used as-is.
    """
    out = df.copy()
    baseline = pd.to_numeric(out["baseline_iou"], errors="coerce")
    final = pd.to_numeric(out["final_iou"], errors="coerce")
    image_distance = (1.0 - pd.to_numeric(out["ms_ssim"], errors="coerce")).clip(0.0, 1.0)
    text_distance = pd.to_numeric(out["ned"], errors="coerce").clip(0.0, 1.0)
    degradation = ((baseline - final) / baseline.where(baseline > 0)).clip(0.0, 1.0)
    out["stealth"] = 1.0 - (image_distance + text_distance) / 2.0
    out["swad"] = degradation * out["stealth"]
    return out


def _resolve_original_image_path(folder_value: str) -> Path:
    folder = Path(folder_value)
    if not folder.is_absolute():
        folder = RESULTS_ROOT / folder
    image_path = folder / "data_point.JPEG"
    if not image_path.exists():
        image_path = image_path.with_suffix(".jpg")
    return image_path


def compute_image_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute MS-SSIM for all rows and add an ``ms_ssim`` column."""
    df = df.copy()
    ms = []
    for _, row in df.iterrows():
        orig = _resolve_original_image_path(str(row["_orig_img_folder"]))
        pth = Path(row["_best_img_path"])
        try:
            with Image.open(orig) as ref_source, Image.open(pth) as dis_source:
                ref_img = ref_source.convert("RGB")
                dis_img = dis_source.convert("RGB").resize(size=ref_img.size)
                ref = np.asarray(ref_img, dtype=np.uint8)
                dis = np.asarray(dis_img, dtype=np.uint8)
            ms.append(msssim(ref, dis).real)
        except (OSError, ValueError):
            ms.append(np.nan)

    df["ms_ssim"] = ms
    return df

def _download_xml(xml_path: Path) -> None:
    rel = f"ILSVRC/Annotations/CLS-LOC/val/{xml_path.name}"

    subprocess.run(
        [
            "kaggle",
            "competitions",
            "download",
            "-c",
            "imagenet-object-localization-challenge",
            "-f",
            rel,
            "-p",
            str(xml_path.parent),
        ],
        check=True,
    )


def _xml_gt_boxes(xml_dir: Path, img_fn: str) -> list:
    """Return list of [xmin, ymin, xmax, ymax] from ILSVRC XML for *img_fn*."""
    stem = os.path.splitext(img_fn)[0]
    xml_path = xml_dir / (stem + ".xml")

    if not xml_path.exists():
        print(f"Downloading missing annotation: {xml_path.name}")
        _download_xml(xml_path)

    root_el = ElementTree.parse(str(xml_path)).getroot()

    boxes = []
    for obj in root_el.findall("object"):
        bb = obj.find("bndbox")
        boxes.append(
            [
                int(bb.find("xmin").text),
                int(bb.find("ymin").text),
                int(bb.find("xmax").text),
                int(bb.find("ymax").text),
            ]
        )
    return boxes


def _fmt(s: pd.Series) -> str:
    return f"{s.mean():.3f} $\\pm$ {s.std():.3f}"


def _bbox_iou(a: list, b: list) -> float:
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def classify_termination(row: pd.Series, iou_threshold: float = 0.25) -> str:
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
    malformed_markers = (
        "Failed to decode VLM JSON array",
        "Expected VLM JSON payload",
        "Expected each VLM prediction",
        "Invalid bbox payload",
        "Prediction is missing bbox field",
    )
    if any(marker in fail_code for marker in malformed_markers):
        return "JSON malformed"
    if pred_count == 0:
        return "No objects found"
    if fail_code:
        return "Other value errors"
    return "Terminated with BBOX"


def classify_termination_series(
    df: pd.DataFrame, iou_threshold: float = 0.25
) -> pd.DataFrame:
    """Return a copy of ``df`` with a ``termination_type`` column added."""
    out = df.copy()
    out["termination_type"] = out.apply(
        classify_termination, axis=1, iou_threshold=iou_threshold
    )
    return out


def add_cumulative_asr_outcomes(
    df: pd.DataFrame, iou_threshold: float = 0.25
) -> pd.DataFrame:
    """Add cumulative ASR indicators; baseline failures stay out of denominators.

    A "Terminated with BBOX" row only counts as an `asr_bbox` success if the attack
    actually degraded `final_iou` below `iou_threshold` - a well-formed bbox on its own
    doesn't mean the attack worked. `asr_bbox_empty`/`asr_bbox_empty_malformed` don't
    need that check: "No objects found"/"JSON malformed" are already unambiguous
    full failures of the victim model regardless of IoU.
    """
    out = classify_termination_series(df, iou_threshold=iou_threshold)
    eligible = out["status"].eq("success")
    degraded = pd.to_numeric(out["final_iou"], errors="coerce") <= iou_threshold
    bbox = eligible & out["termination_type"].eq("Terminated with BBOX") & degraded
    empty = eligible & out["termination_type"].eq("No objects found")
    malformed = eligible & out["termination_type"].eq("JSON malformed")
    out["asr_bbox"] = bbox.where(eligible)
    out["asr_bbox_empty"] = (bbox | empty).where(eligible)
    out["asr_bbox_empty_malformed"] = (bbox | empty | malformed).where(eligible)
    return out

_SCENE_MAP = {
    "single/solo": "Isolated",
    "single/multi": "Clustered",
    "multi": "Mixed",
    "udacity": "Driving",
}

def load_rq3_data(root: str | Path) -> pd.DataFrame:
    """Load survey + GT and return one row per (image, session)."""
    root = Path(root)
    ann_dir = root / "dataset" / "2017" / "ILSVRC" / "Annotations" / "DET" / "val"

    with open(root / "experiments" / "survey.json") as f:
        survey = json.load(f)

    records = []
    for img_fn, img_data in survey.items():
        cat_path = img_data["internal_mapping"].rsplit("/", 1)[0]
        scene_type = _SCENE_MAP.get(cat_path, "Unknown")

        gt_boxes = _xml_gt_boxes(ann_dir, img_fn)
        if not gt_boxes:
            continue

        for ann in img_data["annotations"]:
            label_ious: list[float] = []
            bbox_by_idx: dict = {}
            any_reject = False
            all_reject = True
            has_img_rej = False

            for lbl in ann["labels"]:
                resp = lbl["response_type"]
                rej = lbl.get("reject_reason")
                idx = lbl["label_index"]

                if resp == "reject" and rej not in ("unclear_image", "unclear_label"):
                    continue
                if resp == "reject":
                    label_ious.append(0.0)
                    any_reject = True
                    if rej == "unclear_image":
                        has_img_rej = True
                    continue

                all_reject = False
                hbs = [[b["xmin"], b["ymin"], b["xmax"], b["ymax"]] for b in lbl.get("bboxes", [])]
                if not hbs:
                    continue
                bbox_by_idx[idx] = hbs[0]
                iou_val = max(_bbox_iou(gt, hb) for gt in gt_boxes for hb in hbs)
                label_ious.append(iou_val)

            if not label_ious:
                continue

            if all_reject and any_reject:
                rej_code = 0 if has_img_rej else 1
            else:
                rej_code = np.nan

            records.append(
                dict(
                    filename=img_fn,
                    scene_type=scene_type,
                    session=ann["session"],
                    method=ann.get("method", ""),
                    human_iou=float(np.mean([v for v in label_ious if not np.isnan(v)])),
                    n_labels=len([v for v in label_ious if not np.isnan(v)]),
                    rejected=bool(all_reject and any_reject),
                    reject_reason=rej_code,
                    bbox_by_idx=bbox_by_idx,
                )
            )

    return pd.DataFrame(records)


_OBJ_CAT_TO_SCENE = {
    "single/solo": "Isolated",
    "single/multi": "Clustered",
    "multi": "Mixed",
    "udacity": "Driving",
}


def rq3_validity_table(
    human_df: pd.DataFrame,
    vlm_df: pd.DataFrame,
    scene_order: list[str] | None = None,
) -> pd.DataFrame:
    """Validity summary table: accept rate + human IoU + VLM IoU by scene."""
    if scene_order is None:
        scene_order = ["Isolated", "Clustered", "Mixed"]

    vlm_df = vlm_df.copy()
    vlm_df["scene_type"] = vlm_df["obj_category"].map(_OBJ_CAT_TO_SCENE)
    vlm_scene = (
        vlm_df.groupby("scene_type")["final_iou"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "vlm_mean", "std": "vlm_std"})
    )
    vlm_overall_mean = vlm_df["final_iou"].mean()
    vlm_overall_std = vlm_df["final_iou"].std()

    rows = []
    for sc in scene_order:
        sub = human_df if sc == "Overall" else human_df[human_df.scene_type == sc]
        img_bbox = sub[~sub.rejected].groupby("filename")["human_iou"].mean()
        accept = 1.0 - sub["rejected"].mean()

        if sc == "Overall":
            vlm_str = f"{vlm_overall_mean:.3f} $\\pm$ {vlm_overall_std:.3f}"
        elif sc in vlm_scene.index:
            row = vlm_scene.loc[sc]
            vlm_str = f"{row.vlm_mean:.3f} $\\pm$ {row.vlm_std:.3f}"
        else:
            vlm_str = "—"

        rows.append(
            {
                "Scene": sc,
                "N sessions": len(sub),
                "Accept rate": f"{accept:.1%}",
                "Human IoU (bbox, mean $\\pm$ std)": _fmt(img_bbox),
                "VLM IoU (post-attack, mean $\\pm$ std)": vlm_str,
            }
        )

    return pd.DataFrame(rows).set_index("Scene")


_GENOME_MODE_LABEL = {
    ("multimodal", "multi"): "Multimodal (img+txt)",
    ("unimodal", "image"): "Image-only",
    ("unimodal", "text"): "Text-only",
}


def rq3_validity_by_modality(human_df: pd.DataFrame, vlm_df: pd.DataFrame) -> pd.DataFrame:
    """Validity table grouped by attack modality instead of scene type."""
    human_bbox = human_df[~human_df.rejected].groupby("filename")["human_iou"].mean()
    human_accept = 1.0 - human_df["rejected"].mean()
    human_str = _fmt(human_bbox)
    accept_str = f"{human_accept:.1%}"

    rows = []
    for (modality, gmode), label in _GENOME_MODE_LABEL.items():
        sub = vlm_df[(vlm_df["modality"] == modality) & (vlm_df["genome_mode"] == gmode)]
        if len(sub) == 0:
            continue
        is_multimodal = modality == "multimodal"
        rows.append(
            {
                "Manipulation": label,
                "N (VLM)": len(sub),
                "Accept rate": accept_str if is_multimodal else "---",
                "Human IoU (bbox, mean $\\pm$ std)": human_str if is_multimodal else "---",
                "VLM IoU (post-attack, mean $\\pm$ std)": (
                    f"{sub['final_iou'].mean():.3f} $\\pm$ {sub['final_iou'].std():.3f}"
                ),
            }
        )

    return pd.DataFrame(rows).set_index("Manipulation")


def rq3_validity_by_model(
    human_df: pd.DataFrame,
    vlm_df: pd.DataFrame,
    model_label: dict | None = None,
) -> pd.DataFrame:
    """Validity table grouped by VLM model."""
    human_bbox = human_df[~human_df.rejected].groupby("filename")["human_iou"].mean()
    human_accept = 1.0 - human_df["rejected"].mean()
    human_str = _fmt(human_bbox)
    accept_str = f"{human_accept:.1%}"

    if model_label is None:
        model_label = {}

    rows = []
    for model in sorted(vlm_df["model"].unique()):
        sub = vlm_df[vlm_df["model"] == model]
        rows.append(
            {
                "Model": model_label.get(model, tex(model)),
                "N (VLM)": len(sub),
                "Accept rate": accept_str,
                "Human IoU (bbox, mean $\\pm$ std)": human_str,
                "VLM IoU (post-attack, mean $\\pm$ std)": (
                    f"{sub['final_iou'].mean():.3f} $\\pm$ {sub['final_iou'].std():.3f}"
                ),
            }
        )

    return pd.DataFrame(rows).set_index("Model")


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


def resolve_original_image_path(folder_value: str) -> Path:
    """Resolve the original MMM image path for one results row."""
    return _resolve_original_image_path(folder_value)


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
    for key in ("bbox", "bbox_2d", "bounding_box", "box"):
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


def _recovery_box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
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
    if gt_boxes.ndim == 1:
        gt_boxes = gt_boxes.reshape(1, 4)

    pred_boxes = [
        np.asarray(
            _to_recovery_pixel_box(
                _extract_recovery_bbox(pred),
                original_size[0],
                original_size[1],
                spec.get("coord_scale"),
                str(spec.get("bbox_order", "xyxy")),
            ),
            dtype=np.float64,
        )
        for pred in predictions
    ]
    if not pred_boxes:
        return 0.0

    pred_matrix = np.vstack(pred_boxes)
    ious = np.zeros((len(pred_matrix), len(gt_boxes)), dtype=np.float64)
    for i, pred_box in enumerate(pred_matrix):
        for j, gt_box in enumerate(gt_boxes):
            ious[i, j] = _recovery_box_iou(pred_box, gt_box)

    pred_indices, gt_indices = linear_sum_assignment(-ious)
    matched_ious = ious[pred_indices, gt_indices]
    return float(matched_ious.sum() / max(len(pred_matrix), len(gt_boxes)))


class MetricTable:
    """Render one or more metric groups in a shared LaTeX table."""

    MODES = ["multi", "image", "text"]

    MODE_LABELS = {
        "multi": r"\faImage\;+ \faFont",
        "image": r"\faImage",
        "text": r"\faFont",
    }

    def __init__(
        self,
        df: pd.DataFrame,
        title: str,
        model_mapping: dict[str, str],
        metric_mapping: dict[str, str],
        scene_mapping: dict[str, str],
        shared_metrics: set[str] | None = None,
    ) -> None:
        self.model_mapping = model_mapping
        self.scene_mapping = scene_mapping

        self.tables = [
            {
                "title": title,
                "df": df.copy(),
                "metrics": metric_mapping,
                "shared_metrics": shared_metrics or set(),
            }
        ]

    def __add__(self, other: MetricTable) -> MetricTable:
        if self.model_mapping != other.model_mapping:
            raise ValueError("Model mappings differ.")

        if self.scene_mapping != other.scene_mapping:
            raise ValueError("Scene mappings differ.")

        merged = MetricTable.__new__(MetricTable)
        merged.model_mapping = self.model_mapping
        merged.scene_mapping = self.scene_mapping
        merged.tables = self.tables + other.tables
        return merged

    @staticmethod
    def _value(value: object) -> str:
        if value is None:
            return "---"

        if isinstance(value, str):
            return value if value else "---"

        if pd.isna(value):
            return "---"

        return str(value)

    def __repr__(self) -> str:
        metric_count = sum(len(table["metrics"]) for table in self.tables)

        separator_count = len(self.tables) - 1
        column_count = 2 + metric_count + separator_count
        alignment = "ll" + "c" * (metric_count + separator_count)

        lines = [
            rf"\begin{{tabular}}{{{alignment}}}",
            r"\toprule",
        ]

        if len(self.tables) == 1:
            table = self.tables[0]
            metric_titles = " & ".join(table["metrics"].values())

            lines.append(f"Model & Scene & {metric_titles} \\\\")
        else:
            group_headers = []
            detail_headers = ["", ""]
            cmidrules = []

            column = 3

            for index, table in enumerate(self.tables):
                width = len(table["metrics"])
                title = table["title"]

                group_headers.append(rf"\multicolumn{{{width}}}{{c}}{{\textsc{{{title}}}}}")

                detail_headers.extend(table["metrics"].values())

                cmidrules.append(rf"\cmidrule(lr){{{column}-{column + width - 1}}}")

                column += width

                if index < len(self.tables) - 1:
                    group_headers.append("")
                    detail_headers.append("")
                    column += 1

            lines.append(
                r"\multirow{2}{*}{Model} & "
                r"\multirow{2}{*}{Scene} & " + " & ".join(group_headers) + r" \\"
            )
            lines.extend(cmidrules)
            lines.append(" & ".join(detail_headers) + r" \\")

        lines.append(r"\midrule")

        model_items = list(self.model_mapping.items())
        scene_items = list(self.scene_mapping.items())

        for model_index, (model_key, model_label) in enumerate(model_items):
            model_span = 3 * len(scene_items)
            first_model_row = True

            for scene_index, (scene_key, scene_label) in enumerate(scene_items):
                for mode_index, mode in enumerate(self.MODES):
                    cells = []

                    if first_model_row:
                        cells.append(rf"\multirow{{{model_span}}}{{*}}{{{model_label}}}")
                        first_model_row = False
                    else:
                        cells.append("")

                    if mode_index == 0:
                        cells.append(rf"\multirow{{3}}{{*}}{{{scene_label}}}")
                    else:
                        cells.append("")

                    for table_index, table in enumerate(self.tables):
                        df = table["df"]
                        metrics = table["metrics"]
                        shared_metrics = table["shared_metrics"]

                        selected = df[
                            (df["model"] == model_key)
                            & (df["scene"] == scene_key)
                            & (df["genome_mode"] == mode)
                        ]

                        row = selected.iloc[0] if not selected.empty else None

                        for metric in metrics:
                            if metric in shared_metrics and mode_index > 0:
                                cells.append("")
                                continue

                            value = self._value(row[metric]) if row is not None else "---"

                            if metric in shared_metrics:
                                value = rf"\multirow{{3}}{{*}}{{{value}}}"

                            cells.append(value)

                        if table_index < len(self.tables) - 1:
                            cells.append("")

                    lines.append(" & ".join(cells) + r" \\")

                if scene_index < len(scene_items) - 1:
                    lines.append(rf"\cmidrule(lr){{2-{column_count}}}")

            if model_index < len(model_items) - 1:
                lines.append(r"\midrule")

        lines.extend(
            [
                r"\bottomrule",
                r"\end{tabular}",
            ]
        )

        return "\n".join(lines)
