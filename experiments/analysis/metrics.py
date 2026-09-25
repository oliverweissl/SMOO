"""Per-test-case metrics reported in RQ1/RQ2: MS-SSIM, NED and SWTD."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from jiwer import cer
from PIL import Image
from sewar import msssim

from .loader import RESULTS_ROOT

SWTD_EPS = 1e-8

_RESULTS_MARKER = "defaults/mmm/results/"
METRIC_CACHE_DIR = Path(__file__).parents[1] / "cache"

# Worker count for MS-SSIM. Each worker holds two full-resolution float64 images; more than a few
# workers saturate RAM on a 16 GB machine and freeze the desktop. Set to 1 for strictly sequential.
MS_SSIM_JOBS = 2


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


def compute_swtd(df: pd.DataFrame) -> pd.DataFrame:
    """Add stealth and Stealth-Weighted Task Degradation (SWTD, Eq. swtd in the paper).

    SWTD = (T0 - TM) / (T0 + eps) * (1 - mean_{m in M} a_m), where M is the set of
    *manipulated* modalities (from ``genome_mode``) and a_m is the normalized
    aggressiveness: 1 - MS-SSIM for the image, NED for the text. Unmanipulated
    modalities are excluded from the mean rather than counted as zero, so
    uni-modal and joint test cases are scored on the same scale.
    """
    out = df.copy()
    baseline = pd.to_numeric(out["baseline_iou"], errors="coerce")
    final = pd.to_numeric(out["final_iou"], errors="coerce")
    mode = out["genome_mode"]
    a_img = (1.0 - pd.to_numeric(out["ms_ssim"], errors="coerce")).clip(0.0, 1.0)
    a_txt = pd.to_numeric(out["ned"], errors="coerce").clip(0.0, 1.0)
    a_img = a_img.where(mode.isin(["multi", "image"]))
    a_txt = a_txt.where(mode.isin(["multi", "text"]))
    out["stealth"] = 1.0 - pd.concat([a_img, a_txt], axis=1).mean(axis=1, skipna=True)
    degradation = (baseline - final) / (baseline + SWTD_EPS)
    out["swtd"] = degradation * out["stealth"]
    return out


def _localize(path_value: str) -> Path:
    """Result JSONs store absolute paths from the run machine; re-root them under the local RESULTS_ROOT."""
    s = str(path_value)
    if _RESULTS_MARKER in s:
        return RESULTS_ROOT / s.split(_RESULTS_MARKER, 1)[1]
    p = Path(s)
    return p if p.is_absolute() else RESULTS_ROOT / p


def _run_key(path_value: str) -> str:
    """Machine-independent id of one run: its result JSON path relative to RESULTS_ROOT."""
    s = str(path_value)
    return s.split(_RESULTS_MARKER, 1)[1] if _RESULTS_MARKER in s else s


def cached_metric(
    df: pd.DataFrame,
    name: str,
    fn,
    recompute: bool = False,
    chunk_size: int = 100,
    cache_dir: Path = METRIC_CACHE_DIR,
) -> pd.DataFrame:
    """Add column ``name`` to ``df``, computing ``fn`` only for runs not yet in ``cache_dir/<name>.csv``.

    ``fn(sub_df) -> sequence`` returns one value per row of ``sub_df``. Rows are keyed by result JSON,
    new values are flushed to disk every ``chunk_size`` rows (an interrupted run keeps its progress),
    and NaN results are not cached so they get retried next time. ``recompute=True`` drops the cache.
    """
    path = cache_dir / f"{name}.csv"
    keys = df["_result_json_path"].astype(str).map(_run_key)
    if path.exists() and not recompute:
        cache = pd.read_csv(path, index_col="key")[name]
    else:
        cache = pd.Series(dtype=float, name=name)
    todo = np.flatnonzero(~keys.isin(cache.index).to_numpy())
    if len(todo):
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"{name}: {len(df) - len(todo)} cached, computing {len(todo)}")
    for start in range(0, len(todo), chunk_size):
        idx = todo[start:start + chunk_size]
        new = pd.Series(list(fn(df.iloc[idx])), index=keys.iloc[idx].to_numpy(), dtype=float).dropna()
        cache = pd.concat([cache[~cache.index.isin(new.index)], new]).rename(name)
        cache.rename_axis("key").to_csv(path)
    out = df.copy()
    out[name] = keys.map(cache).to_numpy()
    return out


def resolve_original_image_path(folder_value: str) -> Path:
    """Resolve the original (clean) MMM image path for one results row."""
    image_path = _localize(folder_value) / "data_point.JPEG"
    if not image_path.exists():
        image_path = image_path.with_suffix(".jpg")
    return image_path


def _ms_ssim_pair(orig: Path, pth: Path) -> float:
    try:
        with Image.open(orig) as ref_source, Image.open(pth) as dis_source:
            ref_img = ref_source.convert("RGB")
            dis_img = dis_source.convert("RGB").resize(size=ref_img.size)
            ref = np.asarray(ref_img, dtype=np.uint8)
            dis = np.asarray(dis_img, dtype=np.uint8)
        return float(msssim(ref, dis).real)
    except (OSError, ValueError):
        return np.nan


def _nice_worker() -> None:
    os.nice(10)


def _ms_ssim_rows(df: pd.DataFrame) -> list[float]:
    origs = [resolve_original_image_path(str(v)) for v in df["_orig_img_folder"]]
    pths = [_localize(v) for v in df["_best_img_path"]]
    if MS_SSIM_JOBS <= 1:
        return [_ms_ssim_pair(o, p) for o, p in zip(origs, pths)]
    with ProcessPoolExecutor(MS_SSIM_JOBS, initializer=_nice_worker) as ex:
        return list(ex.map(_ms_ssim_pair, origs, pths))


def compute_image_metrics(df: pd.DataFrame, recompute: bool = False) -> pd.DataFrame:
    """Add an ``ms_ssim`` column; values are cached per run in ``cache/ms_ssim.csv``.

    Text-only runs keep the clean image, so they get NaN instead of a (wasted) MS-SSIM of ~1.
    """
    out = df.copy()
    img = out["genome_mode"].isin(["multi", "image"])
    out["ms_ssim"] = np.nan
    out.loc[img, "ms_ssim"] = cached_metric(out[img], "ms_ssim", _ms_ssim_rows, recompute=recompute)["ms_ssim"].to_numpy()
    return out
