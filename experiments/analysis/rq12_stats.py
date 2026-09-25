"""
Statistical tests for RQ1 (effectiveness) and RQ2 (efficiency) of the multi-modal study.

Every seed image of a (model, scene) cell is optimised under all three manipulation regimes
(multi / image / text), so the three regimes are *paired* by seed. The primary tests exploit that:

  binary outcomes (ASR)      Cochran's Q omnibus + exact McNemar pairwise
  continuous outcomes        Friedman omnibus + Wilcoxon signed-rank pairwise
                             (2-regime metrics like MS-SSIM/NED: Wilcoxon only)

Effect sizes: risk difference (ASR), matched-pairs rank-biserial r and Cohen's d_z (continuous).
p-values are Holm-corrected within each metric family (all cells x pairs of that metric).

As a robustness check, Welch t-tests + Cohen's d are also computed from the published
mean / sd / n alone (what a reader could reproduce from Tables 3 and 4).

Usage:  python -m experiments.analysis.rq12_stats [--out DIR]
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2, friedmanchisquare, ttest_ind_from_stats, wilcoxon

sys.path.insert(0, str(Path(__file__).parents[2]))
from experiments.analysis.loader import load_all_results, success_only
from experiments.analysis.metrics import compute_image_metrics, compute_swtd, compute_text_metrics
from experiments.analysis.termination import add_cumulative_asr_outcomes

ALPHA = 0.05
MODELS = ["qwen", "kimi", "intern", "nemotron"]
SCENES = {"multi": "MC", "single/multi": "SC-MI", "single/solo": "SC-SI", "udacity": "Driving"}
MODES = ["multi", "image", "text"]
PAIRS = list(combinations(MODES, 2))
KEY = ["model", "obj_category", "filename"]


# ---------------------------------------------------------------- data

def load() -> pd.DataFrame:
    df = success_only(load_all_results(include_baseline_fail=True))
    df = df[df["model"].isin(MODELS)].copy()
    df = compute_image_metrics(df)
    df = compute_text_metrics(df)
    df.loc[df["genome_mode"] == "image", "ned"] = np.nan
    df = compute_swtd(df)
    df = add_cumulative_asr_outcomes(df, iou_threshold=0.25)
    df.loc[df["termination_type"] == "JSON malformed", "swtd"] = np.nan
    df["total_budget"] = (df["img_budget_used"] + df["txt_budget_used"]).clip(upper=df["budget_max"])
    return df


# ---------------------------------------------------------------- tests

def holm(p: pd.Series) -> pd.Series:
    p = p.astype(float)
    ok = p.notna()
    vals = p[ok].to_numpy()
    order = np.argsort(vals)
    m = len(vals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * vals[i])
        adj[i] = min(1.0, running)
    out = pd.Series(np.nan, index=p.index)
    out[ok] = adj
    return out


def cochran_q(x: np.ndarray) -> float:
    """x: n x k binary matrix. Returns p-value."""
    x = np.asarray(x, dtype=float)
    k = x.shape[1]
    col, row = x.sum(0), x.sum(1)
    denom = k * row.sum() - (row ** 2).sum()
    if denom == 0:
        return np.nan
    q = (k - 1) * (k * (col ** 2).sum() - col.sum() ** 2) / denom
    return chi2.sf(q, k - 1)


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    if n01 + n10 == 0:
        return 1.0
    return binomtest(n01, n01 + n10, 0.5).pvalue


def rank_biserial(d: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation (Kerby): (sum R+ - sum R-) / sum R."""
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = pd.Series(np.abs(d)).rank().to_numpy()
    return (r[d > 0].sum() - r[d < 0].sum()) / r.sum()


def wilcoxon_paired(a: np.ndarray, b: np.ndarray) -> dict:
    d = a - b
    if len(d) < 2 or np.all(d == 0):
        p = np.nan if len(d) < 2 else 1.0
    else:
        p = wilcoxon(a, b, zero_method="wilcox").pvalue
    sd = d.std(ddof=1) if len(d) > 1 else np.nan
    return {"p": p, "r_rb": rank_biserial(d), "d_z": d.mean() / sd if sd and sd > 0 else np.nan}


def friedman(mat: np.ndarray) -> float:
    if len(mat) < 2 or np.allclose(mat, mat[:, :1]):
        return np.nan
    return friedmanchisquare(*mat.T).pvalue


# ---------------------------------------------------------------- per-metric runners

def wide(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return df.pivot_table(index=KEY, columns="genome_mode", values=col, aggfunc="first")


def cells(df: pd.DataFrame):
    for m in MODELS:
        for cat, scene in SCENES.items():
            yield m, scene, df[(df["model"] == m) & (df["obj_category"] == cat)]


def test_binary(df: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = []
    for m, scene, sub in cells(df):
        w = wide(sub, col).dropna()
        if w.empty:
            continue
        w = w.astype(bool)
        rate = w.mean()
        omni = cochran_q(w[MODES].to_numpy())
        for a, b in PAIRS:
            rows.append(dict(metric=col, model=m, scene=scene, a=a, b=b, n=len(w),
                             mean_a=rate[a], mean_b=rate[b], effect=rate[a] - rate[b],
                             effect_name="risk diff", p_omnibus=omni, p=mcnemar_exact(w[a], w[b])))
    return pd.DataFrame(rows)


def test_continuous(df: pd.DataFrame, col: str, modes=MODES) -> pd.DataFrame:
    rows = []
    for m, scene, sub in cells(df):
        w = wide(sub, col)
        w = w[[c for c in modes if c in w]].dropna()
        if len(w) < 2:
            continue
        omni = friedman(w.to_numpy()) if len(modes) == 3 else np.nan
        for a, b in combinations(modes, 2):
            r = wilcoxon_paired(w[a].to_numpy(), w[b].to_numpy())
            rows.append(dict(metric=col, model=m, scene=scene, a=a, b=b, n=len(w),
                             mean_a=w[a].mean(), mean_b=w[b].mean(), effect=r["r_rb"],
                             effect_name="rank-biserial", d_z=r["d_z"], p_omnibus=omni, p=r["p"]))
    return pd.DataFrame(rows)


def welch_from_summary(res: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Welch t + pooled-sd Cohen's d from mean/sd/n only (independent-samples view of the same cells)."""
    out = []
    for _, r in res.iterrows():
        cat = {v: k for k, v in SCENES.items()}[r["scene"]]
        sub = df[(df["model"] == r["model"]) & (df["obj_category"] == cat)]
        xa = sub.loc[sub["genome_mode"] == r["a"], r["metric"]].dropna()
        xb = sub.loc[sub["genome_mode"] == r["b"], r["metric"]].dropna()
        ma, sa, na, mb, sb, nb = xa.mean(), xa.std(), len(xa), xb.mean(), xb.std(), len(xb)
        if na < 2 or nb < 2 or (sa == 0 and sb == 0):
            out.append((np.nan, np.nan))
            continue
        p = ttest_ind_from_stats(ma, sa, na, mb, sb, nb, equal_var=False).pvalue
        sp = np.sqrt(((na - 1) * sa ** 2 + (nb - 1) * sb ** 2) / (na + nb - 2))
        out.append((p, (ma - mb) / sp if sp > 0 else np.nan))
    res = res.copy()
    res["p_welch"], res["cohen_d"] = zip(*out) if out else ([], [])
    return res


# ---------------------------------------------------------------- main

def finalize(res: pd.DataFrame) -> pd.DataFrame:
    res = res.copy()
    res["p_holm"] = holm(res["p"])
    res["sig"] = res["p_holm"] < ALPHA
    if "p_welch" in res:
        res["p_welch_holm"] = holm(res["p_welch"])
    return res


def summarise(res: pd.DataFrame, label: str) -> None:
    print(f"\n=== {label} ===")
    for (a, b), g in res.groupby(["a", "b"], sort=False):
        g = g.dropna(subset=["p"])
        if g.empty:
            continue
        hi = ((g["effect"] > 0) & g["sig"]).sum()
        lo = ((g["effect"] < 0) & g["sig"]).sum()
        print(f"  {a:>5} vs {b:<5}: {len(g):2d} cells | {a} > {b} sig in {hi:2d} | {a} < {b} sig in {lo:2d}"
              + (f" | Welch sig in {(g['p_welch_holm'] < ALPHA).sum():2d}" if "p_welch_holm" in g else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).parents[1] / "stats"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    df = load()

    specs = {
        # RQ1
        "asr_bbox": ("binary", MODES),
        "asr_bbox_empty": ("binary", MODES),
        "asr_bbox_empty_malformed": ("binary", MODES),
        "ms_ssim": ("cont", ["multi", "image"]),
        "ned": ("cont", ["multi", "text"]),
        "swtd": ("cont", MODES),
        # RQ2
        "img_budget_used": ("cont", ["multi", "image"]),
        "txt_budget_used": ("cont", ["multi", "text"]),
        "total_budget": ("cont", MODES),
        "total_evaluations": ("cont", MODES),
        "runtime": ("cont", MODES),
    }
    allres = []
    for col, (kind, modes) in specs.items():
        res = test_binary(df, col) if kind == "binary" else test_continuous(df, col, modes)
        if kind == "cont":
            res = welch_from_summary(res, df)
        res = finalize(res)
        summarise(res, col)
        allres.append(res)
    allres = pd.concat(allres, ignore_index=True)
    allres.to_csv(out / "rq12_tests.csv", index=False)
    print(f"\nwrote {out / 'rq12_tests.csv'} ({len(allres)} comparisons)")


if __name__ == "__main__":
    main()
