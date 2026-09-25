"""
Paired statistical tests for RQ1/RQ2: one-sided, alpha=0.01, H1 = HyNeA better than baseline.

Pairing is by class/target id (see _analysis.class_means / align_by_shared_classes), not by
individual sample: HyNeA's run JSONs carry no seed/sample id (see _hynea_tester.py) and it
silently skips samples on ExceededIterationBudget, so true per-case 1:1 pairing across tools
isn't recoverable from the logged data. Class-level aggregation is the coarsest unit all three
tools share reliably.
"""
import numpy as np
from scipy.stats import wilcoxon, shapiro

from _analysis import extract_class, class_means, align_by_shared_classes, get_trace_diff_per_class

ALPHA = 0.01


def cohens_d(x, y):
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx - 1) * np.std(x, ddof=1) ** 2 + (ny - 1) * np.std(y, ddof=1) ** 2) / (nx + ny - 2))
    return (np.mean(x) - np.mean(y)) / pooled_std


def rank_biserial(x, y):
    """Matched-pairs rank-biserial correlation effect size for Wilcoxon signed-rank."""
    diff = np.asarray(x) - np.asarray(y)
    diff = diff[diff != 0]
    if len(diff) == 0:
        return 0.0
    n_pos = np.sum(diff > 0)
    n_neg = np.sum(diff < 0)
    return (n_pos - n_neg) / len(diff)


def wilcoxon_paired(x, y, alternative="greater"):
    """
    Wilcoxon signed-rank test, paired, one-sided by default.
    alternative="greater": H1 = x (HyNeA) > y (baseline). Use "less" for lower-is-better metrics
    (LPIPS, runtime, budget, trace diff, flip sensitivity).
    """
    x, y = np.asarray(x), np.asarray(y)
    assert len(x) == len(y), "paired arrays must be same length (same class order)"
    stat, p = wilcoxon(x, y, alternative=alternative)
    return {
        "n": len(x),
        "statistic": stat,
        "p": p,
        "rank_biserial_r": rank_biserial(x, y),
        "cohens_d": cohens_d(x, y),
        "significant": p < ALPHA,
    }


def check_normality(*samples, labels=None):
    """Shapiro-Wilk normality check per sample; flags whether a t-test would even be valid."""
    labels = labels or [f"sample_{i}" for i in range(len(samples))]
    for label, s in zip(labels, samples):
        s = np.asarray(s)
        stat, p = shapiro(s) if len(s) >= 3 else (np.nan, np.nan)
        verdict = "normal (t-test admissible)" if p >= 0.05 else "non-normal (Wilcoxon required)"
        print(f"\tShapiro-Wilk {label}: n={len(s)}, mean={s.mean():.3f}, sd={s.std():.3f}, p={p:.1e} -> {verdict}")


def report_wilcoxon(name, hy_vals, baseline_vals, baseline_name, alternative="greater"):
    r = wilcoxon_paired(hy_vals, baseline_vals, alternative=alternative)
    sig = "***" if r["significant"] else "n.s."
    print(f"\t{name}: HyNeA vs {baseline_name} (n classes={r['n']}) p={r['p']:.1e} {sig} "
          f"(rank-biserial r={r['rank_biserial_r']:.3f}, Cohen's d={r['cohens_d']:.3f}, alpha={ALPHA})")
    return r


def report_wilcoxon_by_class(name, hy_vals, hy_paths, baseline_vals, baseline_paths, baseline_name, alternative="greater"):
    """Groups per-case values by class id (extracted from each path/file), then runs a paired
    Wilcoxon test over the classes shared by both methods."""
    hy_cls = class_means(hy_vals, [extract_class(p) for p in hy_paths])
    base_cls = class_means(baseline_vals, [extract_class(p) for p in baseline_paths])
    (hy_aligned, base_aligned), shared = align_by_shared_classes(hy_cls, base_cls)
    if len(shared) < 2:
        print(f"\t{name}: HyNeA vs {baseline_name} - only {len(shared)} shared class(es), skipping")
        return None
    return report_wilcoxon(name, hy_aligned, base_aligned, baseline_name, alternative=alternative)


def report_trace_diff(label, hy_o, hy_t, base_o, base_t, base_name, alternative="less"):
    """Per-class embedding trace-diff (diversity preservation), then paired Wilcoxon over shared classes."""
    hy_tr = get_trace_diff_per_class(hy_o, hy_t)
    base_tr = get_trace_diff_per_class(base_o, base_t)
    (hy_aligned, base_aligned), shared = align_by_shared_classes(hy_tr, base_tr)
    if len(shared) < 2:
        print(f"\tTrace diff {label}: HyNeA vs {base_name} - only {len(shared)} shared class(es), skipping")
        return None
    return report_wilcoxon(f"Trace diff {label}", hy_aligned, base_aligned, base_name, alternative=alternative)
