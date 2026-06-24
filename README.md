# Density Estimation Showdown

A controlled empirical comparison of four probability **density estimation**
methods on synthetic data (where the true density is known) and on the
**Old Faithful** geyser benchmark. The "star" method is the **Parzen Neural
Network (PNN)** of Trentin & Gori.

> Project component of the *Artificial Intelligence* course (A.Y. 2025/2026),
> Prof. Edmondo Trentin, University of Siena (MSc in AI & Automation
> Engineering). The deliverable is **experiments + statistics + a report**; this
> repository is the technical substrate that produces the numbers and figures.

## The four methods

| Method | Type | Implemented |
|---|---|---|
| **Parzen window** (Gaussian-kernel KDE) | non-parametric | from scratch (NumPy) |
| **k-NN density** | non-parametric | from scratch (NumPy) |
| **Gaussian Mixture Model** (EM) | parametric | `sklearn` (+ from-scratch EM cross-check) |
| **Parzen Neural Network (PNN)** | neural, compact | from scratch (PyTorch) |

All implement one interface ([src/estimators/base.py](src/estimators/base.py)):

```python
class DensityEstimator:
    def fit(self, X) -> "DensityEstimator": ...
    def score_samples(self, X) -> np.ndarray:   # p(x), linear space (NOT log)
    def log_likelihood(self, X) -> float:        # mean log p(x)
```

## The narrative the experiments support

- **GMM wins** when its Gaussian-mixture assumption matches the data (it does, on
  both the synthetic 3-Gaussian set and bimodal Old Faithful).
- **Parzen / k-NN** are flexible but data-hungry and **slow at query time**
  (each query sums/searches over all training points, O(N)).
- The **PNN recovers most of Parzen's accuracy while being compact and ~O(1) per
  query** — the punchline, reproducing Trentin & Gori's finding.

## Datasets

- **Synthetic 2D** — mixture of 3 bivariate Gaussians with a closed-form
  `p_true`, so **Integrated Squared Error (ISE)** against the truth is available.
- **Synthetic 1D** — mixture of 3 univariate Gaussians, for clean density-curve
  plots.
- **Old Faithful** — 272 observations of eruption `duration` and `waiting` time;
  loaded via `seaborn.load_dataset("geyser")` with a byte-identical bundled CSV
  fallback ([data/old_faithful.csv](data/old_faithful.csv)) so the pipeline runs
  fully offline. True density unknown → evaluated by **held-out log-likelihood**.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
```

Everything is **CPU-only** and pinned. The full run takes a few minutes on a
laptop.

## Run

```bash
# End-to-end: regenerates every table and figure deterministically.
python scripts/run_all.py

# Tests (estimators integrate to ≈1, Parzen matches sklearn KDE, p_true sums to 1,
# PNN validity, EM cross-check).
python -m pytest
```

All knobs (seed, N, grid resolution, folds, hyperparameters) live in
[config.yaml](config.yaml). A single global `seed` makes the run reproducible.

## Experiments (§4 of the spec)

| # | Experiment | Output |
|---|---|---|
| 4.1 | Synthetic accuracy: ISE vs truth + held-out log-likelihood, 20 seeds | `synthetic_accuracy_*.csv` |
| 4.2 | Old Faithful: 10-fold CV log-likelihood + GMM component-mean check | `faithful_cv_*.csv`, `faithful_gmm_means.csv` |
| 4.3 | Query-time vs N scalability benchmark | `timing.csv` |
| 4.4 | Wilcoxon signed-rank tests + Holm correction | `statistical_tests.csv` |
| 4.5 | Parzen bandwidth `h` and k-NN `k` sensitivity sweeps | `parzen_bandwidth_sweep.csv`, `knn_k_sweep.csv` |
| 4.6 | PNN soft-constraint ablation (with vs without unit-integral penalty) | `pnn_ablation_*.csv` |

Tables are written to [results/tables/](results/tables/); figures (PNG @ 300 dpi
**and** vector PDF for LaTeX) to [results/figures/](results/figures/).

## Implementation notes (for the report)

- **Parzen** uses a diagonal per-dimension bandwidth, Silverman's rule by
  default, with k-fold CV selection available. Validated to match
  `sklearn.KernelDensity` to ~1e-6.
- **k-NN density** `p(x) = k / (N·V_k(x))` with the correct d-ball volume. The
  raw estimate does **not** integrate to 1 (a known caveat); we optionally
  renormalize over the evaluation grid and report both behaviours.
- **GMM** uses `sklearn`'s EM; the number of components is set to the known truth
  (3 / 2) and **also** selected by BIC to exercise model selection.
- **PNN** is an MLP regressing the Parzen estimate (MSE), with:
  - a **non-negativity output head** (softplus) → `f_θ(x) ≥ 0`;
  - a **unit-integral soft constraint** `λ(∫f_θ − 1)²` (Trentin's Kolmogorov-axiom
    idea), the integral estimated by trapezoidal quadrature over the support.

  The data (MSE) term is normalized by the mean squared target so that a single
  `λ` is scale-invariant across 1D and 2D (density magnitude scales as 1/area).
  The output head is initialised to a near-uniform density so the integral starts
  near 1 and the penalty acts as a gentle nudge rather than collapsing the fit.

## Results summary

<!-- RESULTS_SUMMARY_START -->
Numbers below are from a full run (`seed=20252026`). See
[results/tables/](results/tables/) for the exact CSVs and
[results/figures/](results/figures/) for all figures.

**Synthetic 2D — accuracy (20 seeds, mean ± std):**

| Method | ISE ↓ | held-out log-lik ↑ |
|---|---|---|
| Parzen | 0.0033 ± 0.0003 | −3.539 ± 0.013 |
| k-NN   | 0.0030 ± 0.0003 | −3.576 ± 0.015 |
| **GMM**    | **0.0006 ± 0.0002** | **−3.465 ± 0.018** |
| PNN    | 0.0032 ± 0.0003 | −3.547 ± 0.014 |

**Old Faithful — 10-fold CV log-likelihood (mean ± std):**

| Method | log-lik ↑ |
|---|---|
| Parzen | −4.467 ± 0.090 |
| k-NN   | −5.000 ± 0.260 |
| **GMM**    | **−4.203 ± 0.173** |
| PNN    | −4.462 ± 0.089 |

**Query time to score 2000 points (seconds):**

| N | Parzen | k-NN | GMM | PNN |
|---|---|---|---|---|
| 100  | 0.0019 | 0.0010 | 0.0006 | 0.0004 |
| 5000 | 0.1127 | 0.0561 | 0.0006 | **0.0004** |

**Headline findings**

- **GMM wins on accuracy** on both datasets (the data really is a Gaussian
  mixture) and recovers the Old Faithful waiting-time component means
  **54.5 / 79.97** vs published **54.6 / 80.1**.
- **The PNN matches Parzen's accuracy** (ISE 0.0032 vs 0.0033; Old Faithful
  −4.462 vs −4.467) — and a Holm-corrected Wilcoxon test finds the **PNN–Parzen
  difference on Old Faithful is not significant** (p = 0.32) — while being
  **~300× faster than Parzen at query time** for N=5000 and ~constant in N.
- **Parzen / k-NN query cost grows linearly in N**; GMM and PNN are ~constant.
- **Soft-constraint ablation:** the unit-integral penalty tightens the learned
  density's integral from **1.045 → 1.007** at no accuracy cost.
<!-- RESULTS_SUMMARY_END -->

## References

- E. Trentin & M. Gori, *Parzen Neural Networks* — the PNN method.
- E. Trentin et al., *Soft-constrained ANNs satisfying the Kolmogorov axioms of
  probability* — the PNN unit-integral constraint.
- E. Parzen (1962), *On estimation of a probability density function and mode*,
  Ann. Math. Statist. 33(3), 1065–1076.
- Duda, Hart & Stork, *Pattern Classification* (2nd ed., 2001) — nonparametric
  techniques (Parzen, k-NN density).
- Bishop, *Pattern Recognition and Machine Learning* (2006) — KDE, GMM/EM.
- Azzalini & Bowman (1990), *A look at some data on the Old Faithful geyser*,
  Applied Statistics 39, 357–365 — the dataset reference.
