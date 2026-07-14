"""Shared computation and plot-setup utilities for RQ notebooks."""

from __future__ import annotations

import json
import os
from pathlib import Path
from xml.etree import ElementTree

import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
from jiwer import cer
from PIL import Image
from sewar import msssim

from ._loader import IMG_CORRUPTIONS, RESULTS_ROOT, TXT_CORRUPTIONS

IMG_COLS = [f"img_{c}" for c in IMG_CORRUPTIONS]
TXT_COLS = [f"txt_{c}" for c in TXT_CORRUPTIONS]
ALL_CORRUPT_COLS = IMG_COLS + TXT_COLS

PALETTE = sns.color_palette("tab10")

TERMINATION_ORDER = [
    "Success",
    "BBOX IoU too small",
    "No objects found",
    "JSON malformed",
    "Other value errors",
]
TERMINATION_COLORS = {
    "Success": "#4daf4a",
    "BBOX IoU too small": "#ffb000",
    "No objects found": "#7f7f7f",
    "JSON malformed": "#e41a1c",
    "Other value errors": "#377eb8",
}

SPLIT_ORDER = [
    ("multimodal", "multi", "multi"),
    ("multimodal", "multi", "single/multi"),
    ("multimodal", "multi", "single/solo"),
    ("unimodal", "image", "multi"),
    ("unimodal", "image", "single/multi"),
    ("unimodal", "image", "single/solo"),
    ("unimodal", "text", "multi"),
    ("unimodal", "text", "single/multi"),
    ("unimodal", "text", "single/solo"),
]

SPLIT_LABEL = {
    ("multimodal", "multi", "multi"): "multimodal-multi",
    ("multimodal", "multi", "single/multi"): "multimodal-single-multi",
    ("multimodal", "multi", "single/solo"): "multimodal-single-solo",
    ("unimodal", "image", "multi"): "image-multi",
    ("unimodal", "image", "single/multi"): "image-single-multi",
    ("unimodal", "image", "single/solo"): "image-single-solo",
    ("unimodal", "text", "multi"): "text-multi",
    ("unimodal", "text", "single/multi"): "text-single-multi",
    ("unimodal", "text", "single/solo"): "text-single-solo",
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


def _resize_image_smart(image: Image.Image, max_resolution: int) -> Image.Image:
    """Resize an image proportionally when its longest side exceeds the limit."""
    width, height = image.size
    if max(width, height) <= max_resolution:
        return image

    scale = max_resolution / float(max(width, height))
    resized = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return image.resize(resized, Image.Resampling.LANCZOS)


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
        ref_img = _resize_image_smart(Image.open(orig).convert("RGB"), 1024)
        dis_img = Image.open(pth).convert("RGB")
        ref = np.asarray(ref_img, dtype=np.uint8)
        dis = np.asarray(dis_img, dtype=np.uint8)
        ms.append(msssim(ref, dis))

    df["ms_ssim"] = ms
    return df


_SCENE_MAP = {
    "single/solo": "Isolated",
    "single/multi": "Clustered",
    "multi": "Mixed",
}

_REJECT_REASON_CODE = {
    "unclear_image": 0,
    "unclear_label": 1,
}


def _xml_gt_boxes(xml_dir: Path, img_fn: str) -> list:
    """Return list of [xmin, ymin, xmax, ymax] from ILSVRC XML for *img_fn*."""
    stem = os.path.splitext(img_fn)[0]
    xml_path = xml_dir / (stem + ".xml")
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


def classify_termination(row: pd.Series) -> str:
    """Classify one testcase row by its terminal outcome."""
    status = str(row.get("status", "") or "")
    fail_code = str(row.get("fail_code", "") or "")
    pred_count_raw = row.get("pred_count", float("nan"))
    try:
        pred_count = int(pred_count_raw)
    except (TypeError, ValueError):
        pred_count = -1

    if "Failed to decode VLM JSON array" in fail_code:
        return "JSON malformed"
    if pred_count == 0:
        return "No objects found"
    if status == "success" and not fail_code:
        return "Success"
    if status == "baseline_fail" and not fail_code:
        return "BBOX IoU too small"
    if fail_code:
        return "Other value errors"
    return "BBOX IoU too small"


def classify_termination_series(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with a ``termination_type`` column added."""
    out = df.copy()
    out["termination_type"] = out.apply(classify_termination, axis=1)
    return out


def load_rq3_data(root: str | Path) -> pd.DataFrame:
    """Load survey + GT and return one row per (image, session)."""
    root = Path(root)
    ann_dir = root / "dataset" / "2017" / "ILSVRC" / "Annotations" / "DET" / "val"

    with open(root / "analysis" / "survey.json") as f:
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
