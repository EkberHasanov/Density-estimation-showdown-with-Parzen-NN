"""Experiment orchestration (§4.1-4.6): produces every numeric result as a CSV.

Each ``run_*`` function returns its result DataFrame(s) and writes them under
``results/tables/``. ``run_all_experiments`` chains them and returns a dict of
results consumed by :mod:`src.plotting`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from .datasets import (
    Grid,
    make_synthetic_1d,
    make_synthetic_2d,
    old_faithful_xy,
    old_faithful_grid,
    synthetic_2d_grid,
)
from .estimators import (
    GMMDensity,
    KNNDensity,
    ParzenNeuralNetwork,
    ParzenWindow,
    silverman_bandwidth,
)
from .estimators.parzen import _evaluate as _parzen_eval
from .metrics import integrated_squared_error
from .stats_tests import pairwise_tests, summary_table
from .timing import run_timing

# Canonical method ordering used in every table and figure.
METHOD_ORDER = ["Parzen", "k-NN", "GMM", "PNN"]


def make_methods(
    cfg: dict[str, Any], grid: Grid, n_components: int, seed: int
) -> dict[str, Any]:
    """Build the four reported estimators (fresh, unfit) for one run.

    ``grid`` is used for k-NN renormalization; ``n_components`` is the (known)
    GMM component count for the dataset; ``seed`` drives GMM init and PNN
    re-initialisation.
    """
    p = cfg["pnn"]
    return {
        "Parzen": ParzenWindow(
            bandwidth=cfg["parzen"]["bandwidth"],
            cv_folds=cfg["parzen"]["cv_folds"],
            bandwidth_grid=cfg["parzen"]["bandwidth_grid"],
            random_state=seed,
        ),
        "k-NN": KNNDensity(
            k=cfg["knn"]["k"], renormalize=cfg["knn"]["renormalize"], grid=grid
        ),
        "GMM": GMMDensity(
            n_components=n_components,
            reg_covar=cfg["gmm"]["reg_covar"],
            random_state=seed,
        ),
        "PNN": ParzenNeuralNetwork(
            hidden=p["hidden"],
            activation=p["activation"],
            output_head=p["output_head"],
            epochs=p["epochs"],
            lr=p["lr"],
            weight_decay=p["weight_decay"],
            batch_size=p["batch_size"],
            n_grid_inputs=p["n_grid_inputs"],
            integral_penalty=p["integral_penalty"],
            integral_grid_res=p["integral_grid_res"],
            parzen_bandwidth=p["parzen_target_bandwidth"],
            random_state=seed,
        ),
    }


def _write(cfg: dict[str, Any], name: str, df: pd.DataFrame) -> None:
    path = cfg["_paths"]["tables_dir"] / name
    df.to_csv(path, index=False)


# =============================================================================
# 4.1 Accuracy on synthetic data (ISE vs known truth + held-out log-likelihood)
# =============================================================================
def run_synthetic_accuracy(cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    mix = make_synthetic_2d(cfg)
    grid = synthetic_2d_grid(cfg)
    p_true_grid = mix.p_true(grid.points)

    n_seeds = cfg["experiments"]["n_seeds"]
    n_train = cfg["synthetic_2d"]["n_train"]
    n_test = cfg["synthetic_2d"]["n_test"]
    base = cfg["seed"]

    rows = []
    for s in range(n_seeds):
        seed = base + s
        Xtr = mix.sample(n_train, np.random.default_rng(seed))
        Xte = mix.sample(n_test, np.random.default_rng(seed + 100_000))
        methods = make_methods(cfg, grid, cfg["gmm"]["n_components_synthetic"], seed)
        for name, est in methods.items():
            est.fit(Xtr)
            ise = integrated_squared_error(est.score_samples(grid.points), p_true_grid, grid)
            test_ll = est.log_likelihood(Xte)
            rows.append({"seed": seed, "method": name, "ise": ise, "test_loglik": test_ll})

    raw = pd.DataFrame(rows)
    _write(cfg, "synthetic_accuracy_raw.csv", raw)

    summ = (
        raw.groupby("method")[["ise", "test_loglik"]]
        .agg(["mean", "std"])
        .reindex(METHOD_ORDER)
    )
    summ.columns = ["_".join(c) for c in summ.columns]
    summ = summ.reset_index()
    _write(cfg, "synthetic_accuracy_summary.csv", summ)
    return {"raw": raw, "summary": summ}


# =============================================================================
# 4.2 Accuracy on Old Faithful (held-out log-likelihood via k-fold CV)
# =============================================================================
def run_faithful_cv(cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    X = old_faithful_xy(cfg)
    grid = old_faithful_grid(cfg)
    n_folds = cfg["experiments"]["cv_folds"]
    base = cfg["seed"]

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=base)
    rows = []
    for fold, (tr, te) in enumerate(kf.split(X)):
        methods = make_methods(cfg, grid, cfg["gmm"]["n_components_faithful"], base + fold)
        for name, est in methods.items():
            est.fit(X[tr])
            rows.append(
                {"fold": fold, "method": name, "test_loglik": est.log_likelihood(X[te])}
            )
    raw = pd.DataFrame(rows)
    _write(cfg, "faithful_cv_raw.csv", raw)

    summ = (
        raw.groupby("method")["test_loglik"].agg(["mean", "std"]).reindex(METHOD_ORDER)
    )
    summ = summ.reset_index()
    _write(cfg, "faithful_cv_summary.csv", summ)

    # GMM bimodal-structure sanity check (§4.2): waiting-time component means.
    gmm = GMMDensity(n_components=cfg["gmm"]["n_components_faithful"], random_state=base)
    gmm.fit(X)
    wait_means = np.sort(gmm.means_[:, 1])  # column 1 == waiting
    published = np.sort(cfg["old_faithful"]["published_waiting_means"])
    means_df = pd.DataFrame(
        {
            "component": [1, 2],
            "estimated_waiting_mean": wait_means,
            "published_waiting_mean": published,
            "abs_error": np.abs(wait_means - published),
        }
    )
    _write(cfg, "faithful_gmm_means.csv", means_df)
    return {"raw": raw, "summary": summ, "gmm_means": means_df}


# =============================================================================
# 4.4 Statistical significance (Wilcoxon signed-rank + Holm correction)
# =============================================================================
def run_statistical_tests(
    cfg: dict[str, Any], synthetic_raw: pd.DataFrame, faithful_raw: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    alpha = cfg["experiments"]["alpha"]

    def pivot(df: pd.DataFrame, value: str, index: str) -> dict[str, np.ndarray]:
        wide = df.pivot(index=index, columns="method", values=value)
        return {m: wide[m].to_numpy() for m in METHOD_ORDER}

    out = {}
    syn_ise = pairwise_tests(pivot(synthetic_raw, "ise", "seed"), alpha, "synthetic_ISE")
    syn_ll = pairwise_tests(
        pivot(synthetic_raw, "test_loglik", "seed"), alpha, "synthetic_loglik"
    )
    fai_ll = pairwise_tests(
        pivot(faithful_raw, "test_loglik", "fold"), alpha, "faithful_loglik"
    )
    out["synthetic_ise"] = syn_ise
    out["synthetic_loglik"] = syn_ll
    out["faithful_loglik"] = fai_ll

    combined = pd.concat([syn_ise, syn_ll, fai_ll], ignore_index=True)
    _write(cfg, "statistical_tests.csv", combined)

    # Also persist tidy mean +/- std summaries used by the report tables.
    s1 = summary_table(pivot(synthetic_raw, "ise", "seed"), "synthetic_ISE")
    s2 = summary_table(pivot(synthetic_raw, "test_loglik", "seed"), "synthetic_loglik")
    s3 = summary_table(pivot(faithful_raw, "test_loglik", "fold"), "faithful_loglik")
    _write(cfg, "metric_summaries.csv", pd.concat([s1, s2, s3], ignore_index=True))
    return out


# =============================================================================
# 4.5 Bandwidth (Parzen) and k (k-NN) sensitivity sweeps
# =============================================================================
def run_sensitivity_sweeps(cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    mix = make_synthetic_2d(cfg)
    grid = synthetic_2d_grid(cfg)
    p_true_grid = mix.p_true(grid.points)
    n_train = cfg["synthetic_2d"]["n_train"]
    n_test = cfg["synthetic_2d"]["n_test"]
    n_rep = 5  # average a few seeds for smooth curves
    base = cfg["seed"]

    # --- Parzen bandwidth sweep (multipliers of the Silverman bandwidth) ---
    parzen_rows = []
    for mult in cfg["parzen"]["bandwidth_grid"]:
        for s in range(n_rep):
            seed = base + s
            Xtr = mix.sample(n_train, np.random.default_rng(seed))
            Xte = mix.sample(n_test, np.random.default_rng(seed + 100_000))
            h = mult * silverman_bandwidth(Xtr)
            p_grid = _parzen_eval(grid.points, Xtr, h)
            p_test = _parzen_eval(Xte, Xtr, h)
            parzen_rows.append(
                {
                    "multiplier": mult,
                    "bandwidth_mean": float(np.mean(h)),
                    "seed": seed,
                    "ise": integrated_squared_error(p_grid, p_true_grid, grid),
                    "test_loglik": float(np.mean(np.log(np.maximum(p_test, 1e-300)))),
                }
            )
    parzen_df = pd.DataFrame(parzen_rows)
    _write(cfg, "parzen_bandwidth_sweep.csv", parzen_df)

    # --- k-NN k sweep ---
    knn_rows = []
    for k in cfg["knn"]["k_grid"]:
        for s in range(n_rep):
            seed = base + s
            Xtr = mix.sample(n_train, np.random.default_rng(seed))
            Xte = mix.sample(n_test, np.random.default_rng(seed + 100_000))
            est = KNNDensity(k=k, renormalize=True, grid=grid).fit(Xtr)
            knn_rows.append(
                {
                    "k": k,
                    "seed": seed,
                    "ise": integrated_squared_error(
                        est.score_samples(grid.points), p_true_grid, grid
                    ),
                    "test_loglik": est.log_likelihood(Xte),
                }
            )
    knn_df = pd.DataFrame(knn_rows)
    _write(cfg, "knn_k_sweep.csv", knn_df)
    return {"parzen": parzen_df, "knn": knn_df}


# =============================================================================
# 4.6 PNN soft-constraint ablation (with vs without the unit-integral penalty)
# =============================================================================
def run_pnn_ablation(cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    mix = make_synthetic_2d(cfg)
    grid = synthetic_2d_grid(cfg)
    p_true_grid = mix.p_true(grid.points)
    n_train = cfg["synthetic_2d"]["n_train"]
    n_test = cfg["synthetic_2d"]["n_test"]
    n_seeds = cfg["experiments"]["ablation_seeds"]
    base = cfg["seed"]
    p = cfg["pnn"]

    variants = {
        "with_constraint": p["integral_penalty"],
        "without_constraint": 0.0,
    }
    rows = []
    for s in range(n_seeds):
        seed = base + s
        Xtr = mix.sample(n_train, np.random.default_rng(seed))
        Xte = mix.sample(n_test, np.random.default_rng(seed + 100_000))
        for label, penalty in variants.items():
            est = ParzenNeuralNetwork(
                hidden=p["hidden"],
                activation=p["activation"],
                output_head=p["output_head"],
                epochs=p["epochs"],
                lr=p["lr"],
                weight_decay=p["weight_decay"],
                batch_size=p["batch_size"],
                n_grid_inputs=p["n_grid_inputs"],
                integral_penalty=penalty,
                integral_grid_res=p["integral_grid_res"],
                parzen_bandwidth=p["parzen_target_bandwidth"],
                random_state=seed,
            ).fit(Xtr)
            rows.append(
                {
                    "seed": seed,
                    "variant": label,
                    "ise": integrated_squared_error(
                        est.score_samples(grid.points), p_true_grid, grid
                    ),
                    "test_loglik": est.log_likelihood(Xte),
                    "integral": est.integral_value(),
                }
            )
    raw = pd.DataFrame(rows)
    _write(cfg, "pnn_ablation_raw.csv", raw)

    summ = (
        raw.groupby("variant")[["ise", "test_loglik", "integral"]]
        .agg(["mean", "std"])
    )
    summ.columns = ["_".join(c) for c in summ.columns]
    summ = summ.reset_index()
    _write(cfg, "pnn_ablation_summary.csv", summ)
    return {"raw": raw, "summary": summ}


# =============================================================================
# Top-level driver
# =============================================================================
def run_all_experiments(cfg: dict[str, Any], verbose: bool = True) -> dict[str, Any]:
    def log(msg: str) -> None:
        if verbose:
            print(f"[experiments] {msg}", flush=True)

    results: dict[str, Any] = {}

    log("4.1 synthetic accuracy (ISE + held-out log-likelihood)...")
    results["synthetic"] = run_synthetic_accuracy(cfg)

    log("4.2 Old Faithful 10-fold cross-validation...")
    results["faithful"] = run_faithful_cv(cfg)

    log("4.4 statistical significance (Wilcoxon + Holm)...")
    results["stats"] = run_statistical_tests(
        cfg, results["synthetic"]["raw"], results["faithful"]["raw"]
    )

    log("4.5 bandwidth / k sensitivity sweeps...")
    results["sweeps"] = run_sensitivity_sweeps(cfg)

    log("4.3 query-time / scalability benchmark...")
    timing_df = run_timing(cfg)
    _write(cfg, "timing.csv", timing_df)
    results["timing"] = timing_df

    log("4.6 PNN soft-constraint ablation...")
    results["ablation"] = run_pnn_ablation(cfg)

    log("done.")
    return results
