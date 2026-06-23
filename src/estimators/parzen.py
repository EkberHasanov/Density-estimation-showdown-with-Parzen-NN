"""Parzen-window (kernel density) estimator with a Gaussian kernel (§3.1).

Implemented from scratch in NumPy (the instructor wants the algorithm, not a
library black box). Supports a diagonal per-dimension bandwidth, a Silverman
rule-of-thumb default, and bandwidth selection by k-fold cross-validation
maximizing held-out log-likelihood.

Reference: E. Parzen (1962), *On estimation of a probability density function
and mode*, Ann. Math. Statist. 33(3), 1065-1076.
"""

from __future__ import annotations

import numpy as np

from .base import DensityEstimator

# Chunk size (number of query rows) for the pairwise kernel evaluation, to keep
# the (n_query x n_train) distance block bounded in memory.
_QUERY_CHUNK = 512


def silverman_bandwidth(X: np.ndarray) -> np.ndarray:
    """Silverman's rule-of-thumb diagonal bandwidth (one value per dimension).

    ``h_j = σ_j · (4 / ((d + 2) N))^{1/(d+4)}`` (Silverman 1986, eq. 4.14
    generalized per-dimension), with ``σ_j`` the per-dimension standard
    deviation.
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    factor = (4.0 / ((d + 2) * n)) ** (1.0 / (d + 4))
    sigma = X.std(axis=0, ddof=1)
    sigma = np.where(sigma > 0, sigma, 1.0)  # guard against a degenerate column
    return factor * sigma


class ParzenWindow(DensityEstimator):
    """Gaussian-kernel Parzen-window density estimator.

    Parameters
    ----------
    bandwidth : {"silverman", "cv"} or float or array-like, default "silverman"
        ``"silverman"`` uses the rule of thumb; ``"cv"`` selects a multiplier of
        the Silverman bandwidth by cross-validation; a float/array fixes the
        (isotropic/diagonal) bandwidth directly.
    cv_folds, bandwidth_grid :
        Used only when ``bandwidth == "cv"``. ``bandwidth_grid`` are multipliers
        of the Silverman bandwidth.
    random_state : int | None
        Seed for the CV fold shuffling.
    """

    def __init__(
        self,
        bandwidth: str | float | np.ndarray = "silverman",
        cv_folds: int = 5,
        bandwidth_grid: list[float] | None = None,
        random_state: int | None = None,
    ) -> None:
        self.bandwidth = bandwidth
        self.cv_folds = cv_folds
        self.bandwidth_grid = bandwidth_grid or [0.5, 0.7, 1.0, 1.4, 2.0]
        self.random_state = random_state
        self.X_: np.ndarray | None = None
        self.h_: np.ndarray | None = None  # resolved diagonal bandwidth (d,)

    # -- fitting -------------------------------------------------------------
    def fit(self, X: np.ndarray) -> "ParzenWindow":
        X = self._as_2d(X)
        self.X_ = X
        self.h_ = self._resolve_bandwidth(X)
        return self

    def _resolve_bandwidth(self, X: np.ndarray) -> np.ndarray:
        n, d = X.shape
        if isinstance(self.bandwidth, str):
            if self.bandwidth == "silverman":
                return silverman_bandwidth(X)
            if self.bandwidth == "cv":
                return self._select_bandwidth_cv(X)
            raise ValueError(f"unknown bandwidth rule {self.bandwidth!r}")
        h = np.asarray(self.bandwidth, dtype=float)
        if h.ndim == 0:
            h = np.full(d, float(h))
        if h.shape != (d,):
            raise ValueError(f"bandwidth shape {h.shape} != ({d},)")
        return h

    def _select_bandwidth_cv(self, X: np.ndarray) -> np.ndarray:
        """Pick the Silverman multiplier maximizing mean held-out log-likelihood."""
        base = silverman_bandwidth(X)
        rng = np.random.default_rng(self.random_state)
        n = X.shape[0]
        folds = np.array_split(rng.permutation(n), self.cv_folds)
        best_mult, best_ll = self.bandwidth_grid[0], -np.inf
        for mult in self.bandwidth_grid:
            h = mult * base
            fold_lls = []
            for i in range(self.cv_folds):
                test_idx = folds[i]
                train_idx = np.concatenate(
                    [folds[j] for j in range(self.cv_folds) if j != i]
                )
                est = _evaluate(X[test_idx], X[train_idx], h)
                fold_lls.append(float(np.mean(np.log(np.maximum(est, 1e-300)))))
            mean_ll = float(np.mean(fold_lls))
            if mean_ll > best_ll:
                best_ll, best_mult = mean_ll, mult
        self.cv_best_multiplier_ = best_mult
        self.cv_best_loglik_ = best_ll
        return best_mult * base

    # -- scoring -------------------------------------------------------------
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        if self.X_ is None or self.h_ is None:
            raise RuntimeError("estimator must be fit before scoring")
        X = self._as_2d(X)
        return _evaluate(X, self.X_, self.h_)


def _evaluate(query: np.ndarray, train: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Vectorized Gaussian-kernel Parzen estimate, chunked over query rows.

    ``p(x) = (1/N) Σ_i (2π)^{-d/2} (Π_j h_j)^{-1} exp(-½ Σ_j ((x_j - x_ij)/h_j)²)``
    """
    n, d = train.shape
    h = np.asarray(h, dtype=float)
    log_norm = -0.5 * d * np.log(2.0 * np.pi) - np.sum(np.log(h))
    train_scaled = train / h  # (N, d)

    out = np.empty(query.shape[0])
    for start in range(0, query.shape[0], _QUERY_CHUNK):
        q = query[start : start + _QUERY_CHUNK] / h  # (B, d)
        # squared distances (B, N) via ||a||² + ||b||² - 2 a·b
        sq = (
            np.sum(q**2, axis=1)[:, None]
            + np.sum(train_scaled**2, axis=1)[None, :]
            - 2.0 * q @ train_scaled.T
        )
        np.maximum(sq, 0.0, out=sq)  # clip tiny negatives from round-off
        log_kernels = log_norm - 0.5 * sq
        # log-mean-exp over the N training kernels for numerical stability
        m = log_kernels.max(axis=1, keepdims=True)
        out[start : start + _QUERY_CHUNK] = (
            np.exp(m[:, 0]) * np.mean(np.exp(log_kernels - m), axis=1)
        )
    return out
