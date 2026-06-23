"""Gaussian Mixture Model density estimator (§3.3).

Thin wrapper over ``sklearn.mixture.GaussianMixture`` (EM is a standard library
routine, explicitly permitted). Supports fitting with a fixed number of
components or selecting it via BIC. A compact from-scratch EM is also provided
(stretch goal) and validated against sklearn in the tests.

Reference: Bishop, *Pattern Recognition and Machine Learning* (2006), §9.2.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture

from .base import DensityEstimator


class GMMDensity(DensityEstimator):
    """GMM density via sklearn EM, optionally selecting components by BIC.

    Parameters
    ----------
    n_components : int | None
        Fixed component count. If ``None``, select in ``[1, bic_max_components]``
        by minimizing BIC.
    bic_max_components : int
        Upper bound for the BIC search (used only when ``n_components is None``).
    reg_covar : float
        Non-negative regularization added to the covariance diagonals.
    random_state : int | None
        Seed forwarded to sklearn for reproducible EM initialisation.
    """

    def __init__(
        self,
        n_components: int | None = 3,
        bic_max_components: int = 8,
        reg_covar: float = 1e-6,
        random_state: int | None = None,
    ) -> None:
        self.n_components = n_components
        self.bic_max_components = bic_max_components
        self.reg_covar = reg_covar
        self.random_state = random_state
        self.model_: GaussianMixture | None = None
        self.selected_n_components_: int | None = None
        self.bic_curve_: dict[int, float] = {}

    def fit(self, X: np.ndarray) -> "GMMDensity":
        X = self._as_2d(X)
        if self.n_components is not None:
            self.model_ = self._fit_single(X, self.n_components)
            self.selected_n_components_ = self.n_components
        else:
            self.model_ = self._fit_by_bic(X)
            self.selected_n_components_ = self.model_.n_components
        return self

    def _fit_single(self, X: np.ndarray, k: int) -> GaussianMixture:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            reg_covar=self.reg_covar,
            max_iter=500,
            n_init=5,
            random_state=self.random_state,
        )
        gmm.fit(X)
        return gmm

    def _fit_by_bic(self, X: np.ndarray) -> GaussianMixture:
        best_model, best_bic = None, np.inf
        for k in range(1, self.bic_max_components + 1):
            gmm = self._fit_single(X, k)
            bic = gmm.bic(X)
            self.bic_curve_[k] = float(bic)
            if bic < best_bic:
                best_bic, best_model = bic, gmm
        assert best_model is not None
        return best_model

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("estimator must be fit before scoring")
        X = self._as_2d(X)
        # sklearn returns log-density; convert to linear space for the interface.
        return np.exp(self.model_.score_samples(X))

    def log_likelihood(self, X: np.ndarray) -> float:
        # Use sklearn's exact log-density (avoids the exp/log round-trip).
        if self.model_ is None:
            raise RuntimeError("estimator must be fit before scoring")
        X = self._as_2d(X)
        return float(np.mean(self.model_.score_samples(X)))

    # -- convenience for the report's GMM sanity check (§4.2) ----------------
    @property
    def means_(self) -> np.ndarray:
        assert self.model_ is not None
        return self.model_.means_


# =============================================================================
# From-scratch EM (stretch goal; validated against sklearn in tests)
# =============================================================================
class GMMDensityScratch(DensityEstimator):
    """Minimal full-covariance GMM fit by Expectation-Maximization, in NumPy.

    Kept simple and not used as the reported estimator -- it exists to show the
    EM algorithm explicitly and to cross-check :class:`GMMDensity`.
    """

    def __init__(
        self,
        n_components: int = 3,
        max_iter: int = 300,
        tol: float = 1e-6,
        reg_covar: float = 1e-6,
        random_state: int | None = None,
    ) -> None:
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state

    def fit(self, X: np.ndarray) -> "GMMDensityScratch":
        X = self._as_2d(X)
        n, d = X.shape
        rng = np.random.default_rng(self.random_state)
        K = self.n_components

        # k-means-free init: random distinct points as means, global cov, uniform weights.
        self.weights_ = np.full(K, 1.0 / K)
        self.means_ = X[rng.choice(n, size=K, replace=False)].copy()
        global_cov = np.cov(X, rowvar=False) + self.reg_covar * np.eye(d)
        self.covs_ = np.stack([global_cov] * K)

        prev_ll = -np.inf
        for _ in range(self.max_iter):
            # E-step: responsibilities
            log_resp = self._log_responsibilities(X)  # (n, K)
            log_norm = _logsumexp(log_resp, axis=1)  # (n,)
            ll = float(np.mean(log_norm))
            resp = np.exp(log_resp - log_norm[:, None])

            # M-step
            nk = resp.sum(axis=0) + 1e-12  # (K,)
            self.weights_ = nk / n
            self.means_ = (resp.T @ X) / nk[:, None]
            for k in range(K):
                diff = X - self.means_[k]
                self.covs_[k] = (
                    (resp[:, k][:, None] * diff).T @ diff
                ) / nk[k] + self.reg_covar * np.eye(d)

            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll
        self.final_loglik_ = prev_ll
        return self

    def _log_responsibilities(self, X: np.ndarray) -> np.ndarray:
        K = self.n_components
        log_comp = np.empty((X.shape[0], K))
        for k in range(K):
            log_comp[:, k] = np.log(self.weights_[k]) + multivariate_normal.logpdf(
                X, mean=self.means_[k], cov=self.covs_[k], allow_singular=True
            )
        return log_comp

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        X = self._as_2d(X)
        return np.exp(_logsumexp(self._log_responsibilities(X), axis=1))


def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    return np.squeeze(m, axis=axis) + np.log(np.sum(np.exp(a - m), axis=axis))
