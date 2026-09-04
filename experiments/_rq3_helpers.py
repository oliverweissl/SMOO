import json
from pathlib import Path

import numpy as np
import pandas as pd
from irrCAC.raw import CAC
from scipy.optimize import linear_sum_assignment

CATEGORIES = ["single/solo", "single/multi", "multi", "udacity"]
CATEGORY_LABELS = {"single/solo": "SC-SI", "single/multi": "SC-MI", "multi": "MC", "udacity": "Driving"}
MODELS = ["qwen", "nemotron", "intern", "kimi"]
MODEL_LABELS = {"qwen": "Qwen3-VL", "kimi": "Kimi-VL", "intern": "InternVL-3.5", "nemotron": "Nemotron3-ON"}
MODALITIES = ["unimodal/text", "unimodal/image", "multimodal"]
MODALITY_LABELS = {
    "unimodal/text": "Text-Manipulation",
    "unimodal/image": "Image-Manipulation",
    "multimodal": "Combined-Manipulation",
}


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


def bbox_set_iou(pred_boxes, gt_boxes) -> float:
    """Mean IoU over the Hungarian-optimal pred/GT box assignment.

    Same method as src/objectives/image_criteria/_bbox_iou.py (VLMBBoxIoU, branch `mmm`),
    so human and model IoU are computed identically.
    """
    pred_boxes = np.asarray(pred_boxes, dtype=np.float64).reshape(-1, 4) if len(pred_boxes) else np.zeros((0, 4))
    gt_boxes = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4) if len(gt_boxes) else np.zeros((0, 4))

    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return 0.0

    ious = np.zeros((len(pred_boxes), len(gt_boxes)), dtype=np.float64)
    for i, pred_box in enumerate(pred_boxes):
        for j, gt_box in enumerate(gt_boxes):
            ious[i, j] = _box_iou(pred_box, gt_box)

    pred_indices, gt_indices = linear_sum_assignment(-ious)
    matched_ious = ious[pred_indices, gt_indices]
    denominator = max(len(pred_boxes), len(gt_boxes))
    return float(matched_ious.sum() / denominator)


_SURVEY_MODALITY_MAP = {"text": "unimodal/text", "image": "unimodal/image", "multimodal": "multimodal"}


def load_case_metadata(emmt_root: str) -> dict:
    """The EMMT_Survey repo's `samples/manifest.json` + per-case `metadata.json` is the
    authoritative record of which (model, modality) generated each of the 144 survey
    cases, and carries that exact case's ground-truth boxes and model IoU straight from
    the `best_result.json`/`baseline_fail.json` used to build it - so human and model
    IoU can be compared on the literal same testcase instead of a heuristic guess.

    case_id -> {model, modality, category, folder_id, ground_truth_boxes, model_iou}.
    """
    manifest = json.load(open(f"{emmt_root}/samples/manifest.json"))
    case_meta = {}
    for c in manifest["cases"]:
        result = json.load(open(f"{emmt_root}/samples/{c['directory']}/metadata.json"))["result"]
        case_meta[c["survey_case_id"]] = {
            "model": c["model"],
            "modality": _SURVEY_MODALITY_MAP[c["modality"]],
            "category": result["data_source"]["category"],
            "folder_id": result["data_source"]["folder_id"],
            "ground_truth_boxes": result["ground_truth_bboxes"],
            "model_iou": result.get("objectives", {}).get("iou", result.get("baseline_iou", 0.0)),
        }
    return case_meta


def load_case_lookup(emmt_root: str) -> dict:
    """(variant, category, filename, label) -> case_id, from `data/variants/variant-{1,2,3}.json`.

    This is an exact join key: the survey frontend (app.js) sets a task's `category`/
    `filename`/`label` fields verbatim from the matching variant-file entry
    (`label = item.labels.join(', ')`), so every real (non-attention) survey row round-
    trips back to exactly one case with no fuzzy matching needed.
    """
    lookup = {}
    for variant in (1, 2, 3):
        entries = json.load(open(f"{emmt_root}/data/variants/variant-{variant}.json"))
        for e in entries:
            lookup[(variant, e["category"], e["filename"], ", ".join(e["labels"]))] = e["case_id"]
    return lookup


def load_survey(csv_path: str) -> pd.DataFrame:
    """`csv_path` is already filtered (no attention-check rows, no annotators who never
    gave a single bbox) and anonymized (annotator_id/session_id are stable per-person
    pseudonyms, not real MTurk worker IDs) - this just parses it for analysis.
    """
    df = pd.read_csv(csv_path)

    def _clamp(box, width, height):
        x1, y1, x2, y2 = box
        return [
            min(max(x1, 0), width),
            min(max(y1, 0), height),
            min(max(x2, 0), width),
            min(max(y2, 0), height),
        ]

    def _parse_bboxes(row):
        if row["response_type"] not in ("bbox", "reject", "skip"):
            return []
        raw = row["bboxes"]
        if pd.isna(raw):
            return []
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            return []
        # the UI is responsible for keeping boxes on-image; clamp here for the rows where it didn't
        return [_clamp(b, row["image_width"], row["image_height"]) for b in parsed]

    df["bboxes_parsed"] = df.apply(_parse_bboxes, axis=1)
    df["reject_reason"] = df["reject_reason"].fillna("")
    return df


def compute_human_task_iou(df_valid: pd.DataFrame, case_lookup: dict, case_meta: dict) -> pd.DataFrame:
    """One survey row already *is* one full task: `label`/`bboxes` cover every target
    object of that task at once (e.g. label="bench, dog" with 2 boxes), and
    (annotator_id, session_id, task_index) is one-to-one with the row. So the
    "task-level" unit for IoU is just the row itself, no grouping needed.
    """
    rows = []
    for _, row in df_valid.iterrows():
        case_id = case_lookup.get((row["variant"], row["category"], row["filename"], row["label"]))
        case = case_meta.get(case_id)
        if case is None:
            continue

        interpretable = row["response_type"] == "bbox"
        pred_boxes = row["bboxes_parsed"] if interpretable else []
        iou = bbox_set_iou(pred_boxes, case["ground_truth_boxes"])

        rows.append({
            "annotator_id": row["annotator_id"],
            "session_id": row["session_id"],
            "case_id": case_id,
            "model": case["model"],
            "modality": case["modality"],
            "category": row["category"],
            "filename": row["filename"],
            "iou": iou,
            "iou_adjusted": iou if interpretable else 0.0,
            "interpretable": interpretable,
        })
    return pd.DataFrame(rows)


def case_human_iou(human_iou: pd.DataFrame) -> pd.DataFrame:
    """Collapse `compute_human_task_iou`'s per-rater rows to one row per case.

    Each case is rated by 10-38 raters, so pooling raw rows in a category/model mean
    pseudo-replicates the true sample size (n cases, not n ratings) and hides that
    raters don't all accept or reject a case together - a case where 60% give a bbox
    and 40% reject is not the same as one every rater rejects, even though a row-level
    average can't tell them apart.

    accept_rate: fraction of this case's raters who gave a usable bbox.
    iou_given_accept: mean IoU among only the accepting raters (undefined, NaN, if none did).
    iou_robust: accept_rate * iou_given_accept - non-accepting raters count as 0 IoU
        rather than being dropped, same quantity `iou_adjusted` already encodes per
        row, made explicit and case-level here.
    consensus: "unanimous_accept" / "unanimous_reject" / "split" (mixed).
    """
    def _agg(g):
        n = len(g)
        n_accept = int(g["interpretable"].sum())
        accept_rate = n_accept / n
        return pd.Series({
            "model": g["model"].iloc[0],
            "modality": g["modality"].iloc[0],
            "category": g["category"].iloc[0],
            "n_raters": n,
            "n_accept": n_accept,
            "accept_rate": accept_rate,
            "iou_given_accept": g.loc[g["interpretable"], "iou"].mean() if n_accept else float("nan"),
            "iou_robust": g["iou_adjusted"].mean(),
            "consensus": "unanimous_accept" if accept_rate == 1.0
            else "unanimous_reject" if accept_rate == 0.0
            else "split",
        })

    return human_iou.groupby("case_id").apply(_agg, include_groups=False)


def acceptance_consensus(case_df: pd.DataFrame, by: str | list[str]) -> pd.DataFrame:
    """Per `by` (e.g. "category" or "model"): how often cases are cleanly accepted or
    rejected by every rater vs. split between them, plus the mean per-case accept rate.
    """
    def _summ(g):
        n = len(g)
        return pd.Series({
            "n_cases": n,
            "mean_accept_rate": g["accept_rate"].mean(),
            "frac_unanimous_accept": (g["consensus"] == "unanimous_accept").sum() / n,
            "frac_split": (g["consensus"] == "split").sum() / n,
            "frac_unanimous_reject": (g["consensus"] == "unanimous_reject").sum() / n,
        })

    return case_df.groupby(by).apply(_summ, include_groups=False)


def human_vs_model_iou_by_model(case_df: pd.DataFrame, model_iou_cases: pd.DataFrame) -> pd.DataFrame:
    """Human vs. model mean IoU aggregated over modality and category - a per-stratum
    (model x modality x category) comparison leaves each cell only ~3-12 cases; this
    pools every case for a model into one estimate."""
    human = case_df.groupby("model").agg(
        human_iou=("iou_given_accept", "mean"),
        human_iou_robust=("iou_robust", "mean"),
        n_human_cases=("iou_robust", "count"),
    )
    model = model_iou_cases.groupby("model")["iou"].agg(model_iou="mean", n_model_cases="count")
    return human.join(model)


def reject_skip_fractions(df_valid: pd.DataFrame) -> pd.DataFrame:
    def _fracs(g):
        n = len(g)
        return pd.Series({
            "n_rows": n,
            "frac_reject_unclear_image": (g["reject_reason"] == "unclear_image").sum() / n,
            "frac_reject_unclear_label": (g["reject_reason"] == "unclear_label").sum() / n,
            "frac_reject_other": (g["reject_reason"] == "other").sum() / n,
            "frac_skip": (g["response_type"] == "skip").sum() / n,
        })
    return df_valid.groupby("category").apply(_fracs, include_groups=False)


def _response_class(row) -> str:
    """Abstracts the 3 reject-reason subtypes into one "reject" class: the specific
    reason is a subjective free-choice label, not a real agreement dimension, and
    folding it into Fleiss'/Cohen's kappa's 5-class scheme was adding noise to (and
    truncating items out of, via the modal-rater-count restriction) a statistic meant
    to measure whether raters agree on accept/reject/skip, not on *why* they rejected.
    """
    if row["response_type"] == "bbox":
        return "accept"
    if row["response_type"] == "reject":
        return "reject"
    if row["response_type"] == "skip":
        return "skip"
    raise ValueError(f"unexpected response_type {row['response_type']!r}")


_CLASSES = ["accept", "reject", "skip"]


def _raw_ratings_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per rated item, one column per rater slot, NaN-padded to the widest
    item - the "raw ratings" format `irrCAC.raw.CAC` expects (rater identity in the
    columns is arbitrary; CAC only reads per-row category counts, so padding with NaN
    correctly represents items rated fewer times without assuming any correspondence
    between column N on one row and column N on another).

    `label` (a task's full, possibly multi-object target description, e.g. "bench, dog")
    together with (category, filename) identifies one rated item; `label_index` is not
    used as it is always null in this survey export. Different annotators/variants rate
    the same item a variable number of times (observed range ~10-38).
    """
    df = df.copy()
    df["response_class"] = df.apply(_response_class, axis=1)
    item_cols = ["category", "filename", "label"]
    per_item = df.groupby(item_cols)["response_class"].apply(list)
    width = per_item.map(len).max()
    return pd.DataFrame(
        [row + [np.nan] * (width - len(row)) for row in per_item],
        index=per_item.index,
    )


def krippendorff_alpha_by_category(df_valid: pd.DataFrame) -> dict:
    """Krippendorff's alpha (nominal) via `irrCAC.raw.CAC.krippendorff`, on the
    abstracted 3-class accept/reject/skip response.

    Chosen over Fleiss'/Cohen's kappa for this scenario because it (a) handles a
    variable, unequal number of raters per item natively - no modal-count truncation,
    no mismatched "overall" vs. per-category item subsample - and (b) generalizes
    cleanly to ordinal/interval/ratio measurement and custom distance functions, so
    the same framework could later fold in the continuous IoU spatial agreement
    between accepted boxes (distance = 1 - IoU) instead of only the accept/reject/skip
    decision, without switching statistics.
    """
    def _alpha(df):
        ratings = _raw_ratings_table(df)
        result = CAC(ratings, categories=_CLASSES).krippendorff()["est"]
        return {"alpha": result["coefficient_value"], "n_items": len(ratings)}

    result = {"overall": _alpha(df_valid)}
    for category, group in df_valid.groupby("category"):
        result[category] = _alpha(group)
    return result


def load_model_results(mmm_root: str) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        for modality in MODALITIES:
            base = Path(mmm_root) / model / modality
            for best_path in base.glob("**/best_result.json"):
                doc = json.load(open(best_path))
                rows.append({
                    "model": model,
                    "modality": modality,
                    "category": doc["data_source"]["category"],
                    "folder_id": doc["data_source"]["folder_id"],
                    "iou": doc["objectives"]["iou"],
                })
            for fail_path in base.glob("**/baseline_fail.json"):
                doc = json.load(open(fail_path))
                rows.append({
                    "model": model,
                    "modality": modality,
                    "category": doc["data_source"]["category"],
                    "folder_id": doc["data_source"]["folder_id"],
                    "iou": 0.0,
                })
    return pd.DataFrame(rows)


def case_model_iou(case_meta: dict) -> pd.DataFrame:
    """Model IoU restricted to just the 144 cases actually shown to humans (3 per
    model x modality x category stratum) - the fair comparison set, as opposed to
    `load_model_results`'s full ~100-per-category population."""
    return pd.DataFrame(
        {
            "model": c["model"],
            "modality": c["modality"],
            "category": c["category"],
            "folder_id": c["folder_id"],
            "iou": c["model_iou"],
        }
        for c in case_meta.values()
    )
