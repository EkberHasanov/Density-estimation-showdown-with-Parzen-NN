"""Datasets: synthetic Gaussian mixtures (with closed-form truth) and Old Faithful.

The synthetic generators expose ``p_true`` so that Integrated Squared Error
(ISE) against the ground-truth density can be computed (§2.1, §4.1). The Old
Faithful loader tries seaborn first and falls back to the bundled CSV so the
pipeline runs fully offline (§2.2, §6).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, norm


# =============================================================================
# Evaluation grid (used for plotting and for numerical integration / ISE)
# =============================================================================
@dataclass(frozen=True)
class Grid:
    """A regular evaluation grid over a 1D or 2D box.

    Attributes
    ----------
    points : np.ndarray, shape (M, d)
        Flattened grid coordinates, one row per node.
    shape : tuple[int, ...]
        Mesh shape so values can be reshaped back for contour plots.
    cell_volume : float
        Volume (length in 1D, area in 2D) of one grid cell. Kept for reference;
        prefer ``weights`` for integration.
    weights : np.ndarray, shape (M,)
        Trapezoidal-rule quadrature weights aligned with ``points`` so that
        ``∫ f ≈ Σ_i w_i f_i`` is exact for affine ``f`` (endpoints half-weighted).
    axes : list[np.ndarray]
        The 1D coordinate vectors along each dimension (for plotting).
    """

    points: np.ndarray
    shape: tuple[int, ...]
    cell_volume: float
    weights: np.ndarray
    axes: list[np.ndarray]

    @property
    def ndim(self) -> int:
        return self.points.shape[1]


def _trapz_weights(coords: np.ndarray) -> np.ndarray:
    """1D trapezoidal weights for nodes ``coords`` (uniform spacing assumed)."""
    h = (coords[-1] - coords[0]) / (len(coords) - 1)
    w = np.full(len(coords), h)
    w[0] = w[-1] = 0.5 * h
    return w


def make_grid_1d(x_min: float, x_max: float, res: int) -> Grid:
    """Uniform 1D grid of ``res`` points on ``[x_min, x_max]``."""
    xs = np.linspace(x_min, x_max, res)
    dx = (x_max - x_min) / (res - 1)
    return Grid(
        points=xs.reshape(-1, 1),
        shape=(res,),
        cell_volume=dx,
        weights=_trapz_weights(xs),
        axes=[xs],
    )


def make_grid_2d(
    x_min: float, x_max: float, y_min: float, y_max: float, res: int
) -> Grid:
    """Uniform ``res`` x ``res`` 2D grid on the given box.

    ``points`` are ordered to match ``np.meshgrid(..., indexing='xy')`` raveled
    in C order, i.e. reshape to ``(res, res)`` gives rows indexed by ``y``.
    """
    xs = np.linspace(x_min, x_max, res)
    ys = np.linspace(y_min, y_max, res)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    points = np.column_stack([xx.ravel(), yy.ravel()])
    dx = (x_max - x_min) / (res - 1)
    dy = (y_max - y_min) / (res - 1)
    # Outer product of 1D trapezoidal weights, raveled to match `points` (y outer).
    weights = np.outer(_trapz_weights(ys), _trapz_weights(xs)).ravel()
    return Grid(
        points=points,
        shape=(res, res),
        cell_volume=dx * dy,
        weights=weights,
        axes=[xs, ys],
    )


# =============================================================================
# Synthetic mixtures of Gaussians (ground-truth density known)
# =============================================================================
@dataclass(frozen=True)
class GaussianMixture2D:
    """Hard-coded mixture of bivariate Gaussians with a closed-form density."""

    weights: np.ndarray  # (K,)
    means: np.ndarray  # (K, 2)
    covs: np.ndarray  # (K, 2, 2)

    def __post_init__(self) -> None:
        w = np.asarray(self.weights, dtype=float)
        if not np.isclose(w.sum(), 1.0):
            raise ValueError("mixture weights must sum to 1")

    @property
    def n_components(self) -> int:
        return len(self.weights)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``n`` samples from the mixture."""
        comps = rng.choice(self.n_components, size=n, p=self.weights)
        out = np.empty((n, 2))
        for k in range(self.n_components):
            mask = comps == k
            n_k = int(mask.sum())
            if n_k:
                out[mask] = rng.multivariate_normal(self.means[k], self.covs[k], n_k)
        return out

    def p_true(self, X: np.ndarray) -> np.ndarray:
        """Closed-form density ``p(x) = Σ_k w_k N(x; μ_k, Σ_k)`` at rows of ``X``."""
        X = np.atleast_2d(X)
        total = np.zeros(X.shape[0])
        for k in range(self.n_components):
            total += self.weights[k] * multivariate_normal.pdf(
                X, mean=self.means[k], cov=self.covs[k]
            )
        return total


@dataclass(frozen=True)
class GaussianMixture1D:
    """Hard-coded mixture of univariate Gaussians with a closed-form density."""

    weights: np.ndarray  # (K,)
    means: np.ndarray  # (K,)
    stds: np.ndarray  # (K,)

    def __post_init__(self) -> None:
        w = np.asarray(self.weights, dtype=float)
        if not np.isclose(w.sum(), 1.0):
            raise ValueError("mixture weights must sum to 1")

    @property
    def n_components(self) -> int:
        return len(self.weights)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        comps = rng.choice(self.n_components, size=n, p=self.weights)
        out = rng.normal(self.means[comps], self.stds[comps])
        return out.reshape(-1, 1)

    def p_true(self, X: np.ndarray) -> np.ndarray:
        """Closed-form density at the rows (or scalars) of ``X``."""
        x = np.asarray(X, dtype=float).reshape(-1)
        total = np.zeros_like(x)
        for k in range(self.n_components):
            total += self.weights[k] * norm.pdf(x, loc=self.means[k], scale=self.stds[k])
        return total


def make_synthetic_2d(cfg: dict[str, Any]) -> GaussianMixture2D:
    s = cfg["synthetic_2d"]
    return GaussianMixture2D(
        weights=np.asarray(s["weights"], dtype=float),
        means=np.asarray(s["means"], dtype=float),
        covs=np.asarray(s["covs"], dtype=float),
    )


def make_synthetic_1d(cfg: dict[str, Any]) -> GaussianMixture1D:
    s = cfg["synthetic_1d"]
    return GaussianMixture1D(
        weights=np.asarray(s["weights"], dtype=float),
        means=np.asarray(s["means"], dtype=float),
        stds=np.asarray(s["stds"], dtype=float),
    )


def synthetic_2d_grid(cfg: dict[str, Any]) -> Grid:
    g = cfg["synthetic_2d"]["grid"]
    return make_grid_2d(g["x_min"], g["x_max"], g["y_min"], g["y_max"], g["res"])


def synthetic_1d_grid(cfg: dict[str, Any]) -> Grid:
    g = cfg["synthetic_1d"]["grid"]
    return make_grid_1d(g["x_min"], g["x_max"], g["res"])


# =============================================================================
# Old Faithful geyser (real benchmark; true density unknown)
# =============================================================================
def load_old_faithful(cfg: dict[str, Any]) -> pd.DataFrame:
    """Load Old Faithful (duration, waiting[, kind]).

    Tries ``seaborn.load_dataset('geyser')`` first; on any failure (e.g. no
    network/cache) falls back to the bundled CSV. The two are byte-for-byte
    equivalent in the numeric columns (verified at build time).
    """
    features = cfg["old_faithful"]["features"]
    try:  # network/cache may be unavailable -> fall back silently
        import seaborn as sns

        df = sns.load_dataset("geyser")
    except Exception:
        csv = cfg["_root"] / cfg["old_faithful"]["csv_fallback"]
        df = pd.read_csv(csv)

    missing = [c for c in features if c not in df.columns]
    if missing:
        raise KeyError(f"Old Faithful is missing columns {missing}")
    return df.reset_index(drop=True)


def old_faithful_xy(cfg: dict[str, Any]) -> np.ndarray:
    """Return the (272, 2) feature matrix [duration, waiting] as float."""
    df = load_old_faithful(cfg)
    return df[cfg["old_faithful"]["features"]].to_numpy(dtype=float)


def old_faithful_grid(cfg: dict[str, Any], pad: float = 0.15, res: int = 200) -> Grid:
    """Evaluation grid spanning the data with a relative ``pad`` margin."""
    X = old_faithful_xy(cfg)
    x_min, y_min = X.min(axis=0)
    x_max, y_max = X.max(axis=0)
    rx, ry = (x_max - x_min) * pad, (y_max - y_min) * pad
    return make_grid_2d(x_min - rx, x_max + rx, y_min - ry, y_max + ry, res)
