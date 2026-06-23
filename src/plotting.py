"""All report figures (§5). Every figure is saved as 300-dpi PNG and vector PDF.

Qualitative figures (dataset scatters, 1D curves, 2D heatmaps) fit the
estimators on the canonical data with the global seed; quantitative figures
(bars, sweeps, timing, significance) read the experiment DataFrames.
"""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless / file-only rendering

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .datasets import (
    make_synthetic_1d,
    make_synthetic_2d,
    old_faithful_xy,
    old_faithful_grid,
    synthetic_1d_grid,
    synthetic_2d_grid,
)
from .experiments import METHOD_ORDER, make_methods

METHOD_COLORS = {
    "Parzen": "#1f77b4",
    "k-NN": "#ff7f0e",
    "GMM": "#2ca02c",
    "PNN": "#d62728",
    "True": "#000000",
}


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
def _save(cfg: dict[str, Any], fig: plt.Figure, name: str) -> None:
    fig_dir = cfg["_paths"]["figures_dir"]
    fig.savefig(fig_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _fit_methods_2d(cfg: dict[str, Any], X: np.ndarray, grid, n_components: int):
    methods = make_methods(cfg, grid, n_components, cfg["seed"])
    for est in methods.values():
        est.fit(X)
    return methods


# -----------------------------------------------------------------------------
# Fig 1: dataset scatters with marginal histograms
# -----------------------------------------------------------------------------
def _scatter_with_marginals(X, xlabel, ylabel, title, color, hue=None, hue_name=None):
    fig = plt.figure(figsize=(5.5, 5.5))
    gs = fig.add_gridspec(
        2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
        left=0.12, right=0.97, bottom=0.1, top=0.93, wspace=0.05, hspace=0.05,
    )
    ax = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax)

    if hue is None:
        ax.scatter(X[:, 0], X[:, 1], s=12, alpha=0.5, color=color, edgecolor="none")
    else:
        for level in np.unique(hue):
            m = hue == level
            ax.scatter(X[m, 0], X[m, 1], s=12, alpha=0.6, label=str(level), edgecolor="none")
        ax.legend(title=hue_name, fontsize=8, frameon=False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax_top.hist(X[:, 0], bins=30, color=color, alpha=0.7)
    ax_right.hist(X[:, 1], bins=30, orientation="horizontal", color=color, alpha=0.7)
    ax_top.axis("off")
    ax_right.axis("off")
    ax_top.set_title(title)
    return fig


def fig_dataset_scatters(cfg: dict[str, Any]) -> None:
    mix = make_synthetic_2d(cfg)
    X = mix.sample(cfg["synthetic_2d"]["n_train"], np.random.default_rng(cfg["seed"]))
    fig = _scatter_with_marginals(
        X, "$x_1$", "$x_2$", "Synthetic 3-Gaussian mixture (N=1000)", METHOD_COLORS["GMM"]
    )
    _save(cfg, fig, "fig01a_synthetic_scatter")

    from .datasets import load_old_faithful

    df = load_old_faithful(cfg)
    feats = cfg["old_faithful"]["features"]
    Xf = df[feats].to_numpy(float)
    hue = df["kind"].to_numpy() if "kind" in df.columns else None
    fig = _scatter_with_marginals(
        Xf, "eruption duration (min)", "waiting time (min)",
        "Old Faithful geyser (N=272)", METHOD_COLORS["Parzen"],
        hue=hue, hue_name="kind",
    )
    _save(cfg, fig, "fig01b_faithful_scatter")


# -----------------------------------------------------------------------------
# Fig 2: 1D true density vs each estimator
# -----------------------------------------------------------------------------
def fig_1d_density_curves(cfg: dict[str, Any]) -> None:
    mix = make_synthetic_1d(cfg)
    grid = synthetic_1d_grid(cfg)
    X = mix.sample(cfg["synthetic_1d"]["n_train"], np.random.default_rng(cfg["seed"]))
    methods = _fit_methods_2d(cfg, X, grid, len(cfg["synthetic_1d"]["weights"]))

    xs = grid.axes[0]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(xs, mix.p_true(grid.points), color=METHOD_COLORS["True"], lw=2.5, label="True density")
    for name in METHOD_ORDER:
        ax.plot(xs, methods[name].score_samples(grid.points), lw=1.6,
                color=METHOD_COLORS[name], label=name)
    ax.plot(X[:, 0], np.full(len(X), -0.005), "|", color="gray", alpha=0.4, ms=8)
    ax.set_xlabel("$x$")
    ax.set_ylabel("density")
    ax.set_title("1D synthetic: true density vs estimators")
    ax.legend(frameon=False, ncol=2)
    _save(cfg, fig, "fig02_1d_density_curves")


# -----------------------------------------------------------------------------
# Fig 3: 2D density heatmaps per method (synthetic + Old Faithful)
# -----------------------------------------------------------------------------
def _heatmap_panels(cfg, X, grid, panels, suptitle, name, vmax=None):
    res = grid.shape
    xs, ys = grid.axes
    if vmax is None:
        vmax = max(float(np.max(v)) for v in panels.values())
    levels = np.linspace(0, vmax, 25)

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 3.4), constrained_layout=True)
    if n == 1:
        axes = [axes]
    cf = None
    for ax, (label, vals) in zip(axes, panels.items()):
        Z = vals.reshape(res)
        cf = ax.contourf(xs, ys, Z, levels=levels, cmap="viridis", extend="max")
        ax.scatter(X[:, 0], X[:, 1], s=4, c="white", alpha=0.25, edgecolor="none")
        ax.set_title(label)
        ax.set_xlabel("$x_1$")
    axes[0].set_ylabel("$x_2$")
    fig.colorbar(cf, ax=axes, shrink=0.85, label="density")
    fig.suptitle(suptitle)
    _save(cfg, fig, name)


def fig_2d_heatmaps(cfg: dict[str, Any]) -> None:
    # Synthetic: include the true-density panel.
    mix = make_synthetic_2d(cfg)
    grid = synthetic_2d_grid(cfg)
    X = mix.sample(cfg["synthetic_2d"]["n_train"], np.random.default_rng(cfg["seed"]))
    methods = _fit_methods_2d(cfg, X, grid, cfg["gmm"]["n_components_synthetic"])
    panels = {"True": mix.p_true(grid.points)}
    for name in METHOD_ORDER:
        panels[name] = methods[name].score_samples(grid.points)
    _heatmap_panels(cfg, X, grid, panels, "Synthetic: estimated density by method",
                    "fig03a_synthetic_heatmaps", vmax=float(np.max(panels["True"])) * 1.1)

    # Old Faithful: no true density.
    Xf = old_faithful_xy(cfg)
    gridf = old_faithful_grid(cfg)
    methodsf = _fit_methods_2d(cfg, Xf, gridf, cfg["gmm"]["n_components_faithful"])
    panelsf = {name: methodsf[name].score_samples(gridf.points) for name in METHOD_ORDER}
    _heatmap_panels(cfg, Xf, gridf, panelsf, "Old Faithful: estimated density by method",
                    "fig03b_faithful_heatmaps")


# -----------------------------------------------------------------------------
# Fig 4 & 5: ISE and log-likelihood bar charts
# -----------------------------------------------------------------------------
def fig_ise_bar(cfg: dict[str, Any], summary: pd.DataFrame) -> None:
    s = summary.set_index("method").reindex(METHOD_ORDER)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(s.index, s["ise_mean"], yerr=s["ise_std"], capsize=4,
           color=[METHOD_COLORS[m] for m in s.index], alpha=0.85)
    ax.set_ylabel("ISE  (lower is better)")
    ax.set_title("Synthetic 2D: Integrated Squared Error (mean ± std, 20 seeds)")
    _save(cfg, fig, "fig04_synthetic_ise_bar")


def fig_loglik_bar(cfg: dict[str, Any], summary: pd.DataFrame) -> None:
    s = summary.set_index("method").reindex(METHOD_ORDER)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(s.index, s["mean"], yerr=s["std"], capsize=4,
           color=[METHOD_COLORS[m] for m in s.index], alpha=0.85)
    ax.set_ylabel("mean held-out log-likelihood (higher is better)")
    ax.set_title("Old Faithful: 10-fold CV log-likelihood (mean ± std)")
    _save(cfg, fig, "fig05_faithful_loglik_bar")


# -----------------------------------------------------------------------------
# Fig 6: query time vs N (log-log)
# -----------------------------------------------------------------------------
def fig_timing(cfg: dict[str, Any], timing_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name in METHOD_ORDER:
        sub = timing_df[timing_df["method"] == name].sort_values("n_train")
        ax.plot(sub["n_train"], sub["query_time_s"], "o-", color=METHOD_COLORS[name], label=name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("training set size N")
    ax.set_ylabel(f"time to score {int(timing_df['n_query'].iloc[0])} queries (s)")
    ax.set_title("Query-time scalability: PNN is ~constant in N")
    ax.legend(frameon=False)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    _save(cfg, fig, "fig06_query_time_vs_N")


# -----------------------------------------------------------------------------
# Fig 7: bandwidth and k sensitivity sweeps
# -----------------------------------------------------------------------------
def _sweep_panel(ax, df, xcol, ycol, xlabel, ylabel, logx=False):
    g = df.groupby(xcol)[ycol].agg(["mean", "std"]).reset_index()
    ax.errorbar(g[xcol], g["mean"], yerr=g["std"], marker="o", capsize=3, color="#444")
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, ls=":", alpha=0.4)


def fig_sweeps(cfg: dict[str, Any], parzen_df: pd.DataFrame, knn_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    _sweep_panel(axes[0], parzen_df, "bandwidth_mean", "ise",
                 "Parzen bandwidth $h$ (mean over dims)", "ISE", logx=True)
    axes[0].set_title("Parzen: bandwidth sensitivity")
    _sweep_panel(axes[1], parzen_df, "bandwidth_mean", "test_loglik",
                 "Parzen bandwidth $h$ (mean over dims)", "held-out log-lik", logx=True)
    axes[1].set_title("Parzen: bandwidth vs log-likelihood")
    fig.suptitle("Parzen bandwidth sweep (overfit ↔ oversmooth)")
    _save(cfg, fig, "fig07a_parzen_bandwidth_sweep")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    _sweep_panel(axes[0], knn_df, "k", "ise", "k (neighbours)", "ISE", logx=True)
    axes[0].set_title("k-NN: k sensitivity")
    _sweep_panel(axes[1], knn_df, "k", "test_loglik", "k (neighbours)", "held-out log-lik", logx=True)
    axes[1].set_title("k-NN: k vs log-likelihood")
    fig.suptitle("k-NN neighbour-count sweep")
    _save(cfg, fig, "fig07b_knn_k_sweep")


# -----------------------------------------------------------------------------
# Fig 8: Wilcoxon (Holm-corrected) p-value matrix
# -----------------------------------------------------------------------------
def fig_wilcoxon_matrix(cfg: dict[str, Any], tests: pd.DataFrame, title: str, name: str) -> None:
    mat = pd.DataFrame(np.nan, index=METHOD_ORDER, columns=METHOD_ORDER)
    for _, r in tests.iterrows():
        mat.loc[r["method_a"], r["method_b"]] = r["wilcoxon_p_holm"]
        mat.loc[r["method_b"], r["method_a"]] = r["wilcoxon_p_holm"]

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    data = mat.to_numpy(dtype=float)
    im = ax.imshow(np.ma.masked_invalid(data), cmap="RdYlGn_r", vmin=0, vmax=0.1)
    ax.set_xticks(range(len(METHOD_ORDER)), METHOD_ORDER)
    ax.set_yticks(range(len(METHOD_ORDER)), METHOD_ORDER)
    for i in range(len(METHOD_ORDER)):
        for j in range(len(METHOD_ORDER)):
            if not np.isnan(data[i, j]):
                star = "*" if data[i, j] < cfg["experiments"]["alpha"] else ""
                ax.text(j, i, f"{data[i, j]:.3f}{star}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="Holm-adjusted Wilcoxon p")
    ax.set_title(title)
    _save(cfg, fig, name)


# -----------------------------------------------------------------------------
# Fig 9: PNN soft-constraint ablation
# -----------------------------------------------------------------------------
def fig_pnn_ablation(cfg: dict[str, Any], raw: pd.DataFrame) -> None:
    order = ["without_constraint", "with_constraint"]
    labels = ["without\nconstraint", "with\nconstraint"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    g = raw.groupby("variant")
    ise_mean = [g.get_group(v)["ise"].mean() for v in order]
    ise_std = [g.get_group(v)["ise"].std() for v in order]
    axes[0].bar(labels, ise_mean, yerr=ise_std, capsize=4, color=["#999", METHOD_COLORS["PNN"]])
    axes[0].set_ylabel("ISE (lower is better)")
    axes[0].set_title("Accuracy")

    dev_mean = [np.abs(g.get_group(v)["integral"] - 1.0).mean() for v in order]
    dev_std = [np.abs(g.get_group(v)["integral"] - 1.0).std() for v in order]
    axes[1].bar(labels, dev_mean, yerr=dev_std, capsize=4, color=["#999", METHOD_COLORS["PNN"]])
    axes[1].set_ylabel(r"$|\int f_\theta - 1|$ (lower is better)")
    axes[1].set_title("Probabilistic validity (unit integral)")

    fig.suptitle("PNN soft-constraint ablation")
    _save(cfg, fig, "fig09_pnn_ablation")


# -----------------------------------------------------------------------------
# driver
# -----------------------------------------------------------------------------
def make_all_figures(cfg: dict[str, Any], results: dict[str, Any], verbose: bool = True) -> None:
    def log(msg: str) -> None:
        if verbose:
            print(f"[plotting] {msg}", flush=True)

    log("fig 1: dataset scatters")
    fig_dataset_scatters(cfg)
    log("fig 2: 1D density curves")
    fig_1d_density_curves(cfg)
    log("fig 3: 2D density heatmaps")
    fig_2d_heatmaps(cfg)
    log("fig 4: synthetic ISE bar")
    fig_ise_bar(cfg, results["synthetic"]["summary"])
    log("fig 5: Old Faithful log-likelihood bar")
    fig_loglik_bar(cfg, results["faithful"]["summary"])
    log("fig 6: query-time vs N")
    fig_timing(cfg, results["timing"])
    log("fig 7: sensitivity sweeps")
    fig_sweeps(cfg, results["sweeps"]["parzen"], results["sweeps"]["knn"])
    log("fig 8: Wilcoxon p-value matrices")
    fig_wilcoxon_matrix(cfg, results["stats"]["synthetic_ise"],
                        "Synthetic ISE: Holm-adjusted Wilcoxon p", "fig08a_wilcoxon_synthetic_ise")
    fig_wilcoxon_matrix(cfg, results["stats"]["faithful_loglik"],
                        "Old Faithful log-lik: Holm-adjusted Wilcoxon p", "fig08b_wilcoxon_faithful_loglik")
    log("fig 9: PNN ablation")
    fig_pnn_ablation(cfg, results["ablation"]["raw"])
    log("all figures saved.")
