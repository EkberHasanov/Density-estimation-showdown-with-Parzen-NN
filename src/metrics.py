"""Evaluation metrics and numerical-integration helpers (§4.1).

* ``integrate_on_grid`` -- Riemann sum of values sampled on a :class:`Grid`.
* ``integrated_squared_error`` -- ISE between an estimate and the truth, the
  synthetic-data ruler (closed-form truth required).
* ``mean_log_likelihood`` -- the real-data metric (mean log p(x)).
"""

from __future__ import annotations

import numpy as np

from .datasets import Grid

_LOG_FLOOR = 1e-300


def integrate_on_grid(values: np.ndarray, grid: Grid) -> float:
    """Approximate ``∫ f`` over the grid's box by the trapezoidal rule.

    ``values`` are ``f`` evaluated at ``grid.points`` (same order). Uses the
    grid's trapezoidal quadrature weights, so the result is exact for affine
    integrands and second-order accurate in general.
    """
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.shape[0] != grid.points.shape[0]:
        raise ValueError("values length does not match number of grid points")
    return float(np.dot(values, grid.weights))


def integrated_squared_error(
    p_hat: np.ndarray, p_true: np.ndarray, grid: Grid
) -> float:
    """ISE ``= ∫ (p_hat(x) − p_true(x))² dx`` evaluated on the grid."""
    p_hat = np.asarray(p_hat, dtype=float).reshape(-1)
    p_true = np.asarray(p_true, dtype=float).reshape(-1)
    return integrate_on_grid((p_hat - p_true) ** 2, grid)


def mean_log_likelihood(p: np.ndarray) -> float:
    """Mean log-density given linear-space densities ``p`` (floored at >0)."""
    p = np.asarray(p, dtype=float).reshape(-1)
    return float(np.mean(np.log(np.maximum(p, _LOG_FLOOR))))


def grid_integral_of_estimator(estimator, grid: Grid) -> float:
    """Numerically integrate an estimator's density over the grid (≈1 check)."""
    p = estimator.score_samples(grid.points)
    return integrate_on_grid(p, grid)
