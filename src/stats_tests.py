"""Statistical analysis of the results (§4.4).

The syllabus explicitly requires "statistical analysis of the results". We use
the **Wilcoxon signed-rank test** (non-parametric, paired) on the per-seed /
per-fold metric values to test whether each pair of methods differs, with a
**Holm-Bonferroni** correction for the multiple pairwise comparisons. A paired
t-test is reported as a secondary check.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon


def holm_bonferroni(pvalues: list[float], alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Holm-Bonferroni step-down correction.

    Returns ``(p_adjusted, reject)`` aligned with the input order. ``p_adjusted``
    are the corrected p-values (monotone, capped at 1); ``reject`` flags those
    significant at level ``alpha``.
    """
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    p_sorted = p[order]
    # Step-down: multiply the k-th smallest by (m - k), enforce monotonicity.
    adj_sorted = np.maximum.accumulate(p_sorted * (m - np.arange(m)))
    adj_sorted = np.minimum(adj_sorted, 1.0)
    p_adj = np.empty_like(adj_sorted)
    p_adj[order] = adj_sorted
    return p_adj, p_adj < alpha


def pairwise_tests(
    values: dict[str, np.ndarray], alpha: float = 0.05, metric_name: str = "metric"
) -> pd.DataFrame:
    """Pairwise Wilcoxon (+ paired t) tests across methods, Holm-corrected.

    Parameters
    ----------
    values : dict[name -> array]
        Per-method paired samples (same length, aligned by seed/fold).
    """
    names = list(values)
    lengths = {len(v) for v in values.values()}
    if len(lengths) != 1:
        raise ValueError("all methods must have the same number of paired samples")

    rows = []
    raw_p = []
    for a, b in combinations(names, 2):
        va, vb = np.asarray(values[a]), np.asarray(values[b])
        diff = va - vb
        if np.allclose(diff, 0.0):
            w_p, t_p, w_stat = 1.0, 1.0, 0.0
        else:
            try:
                w_stat, w_p = wilcoxon(va, vb)
            except ValueError:  # e.g. all-zero differences after ties handling
                w_stat, w_p = 0.0, 1.0
            _, t_p = ttest_rel(va, vb)
        rows.append(
            {
                "metric": metric_name,
                "method_a": a,
                "method_b": b,
                "median_diff": float(np.median(diff)),
                "wilcoxon_stat": float(w_stat),
                "wilcoxon_p": float(w_p),
                "ttest_p": float(t_p),
            }
        )
        raw_p.append(w_p)

    p_adj, reject = holm_bonferroni(raw_p, alpha=alpha)
    df = pd.DataFrame(rows)
    df["wilcoxon_p_holm"] = p_adj
    df["significant"] = reject
    return df


def summary_table(
    values: dict[str, np.ndarray], metric_name: str = "metric"
) -> pd.DataFrame:
    """Mean +/- std (and n) per method, for a results table."""
    rows = []
    for name, v in values.items():
        v = np.asarray(v, dtype=float)
        rows.append(
            {
                "method": name,
                "metric": metric_name,
                "mean": float(np.mean(v)),
                "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                "n": int(len(v)),
            }
        )
    return pd.DataFrame(rows)
