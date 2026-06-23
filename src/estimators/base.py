"""Common density-estimator interface (§3).

Every estimator returns ``score_samples(X) = p(x)`` in *linear* (not log) space,
matching the AGENT.md contract. ``log_likelihood`` is the mean log-density,
the metric used on real data.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

# Floor applied before taking logs so a zero density estimate does not produce
# ``-inf`` log-likelihood. Small enough not to perturb well-supported points.
_LOG_FLOOR = 1e-300


class DensityEstimator(ABC):
    """Abstract base class shared by all four estimators."""

    @abstractmethod
    def fit(self, X: np.ndarray) -> "DensityEstimator":
        """Fit the estimator to data ``X`` of shape (N, d). Returns ``self``."""

    @abstractmethod
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return the estimated density ``p(x)`` (linear space) at rows of ``X``."""

    def log_likelihood(self, X: np.ndarray) -> float:
        """Mean log-density ``(1/M) Σ log p(x_m)`` over rows of ``X``."""
        p = np.asarray(self.score_samples(X), dtype=float)
        return float(np.mean(np.log(np.maximum(p, _LOG_FLOOR))))

    @staticmethod
    def _as_2d(X: np.ndarray) -> np.ndarray:
        """Coerce input to a 2D float array of shape (N, d)."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X
