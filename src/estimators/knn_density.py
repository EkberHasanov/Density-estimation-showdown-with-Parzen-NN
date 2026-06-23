"""k-Nearest-Neighbour density estimation (§3.2), implemented from scratch.

``p(x) = k / (N · V_k(x))`` where ``V_k(x)`` is the volume of the smallest
d-ball centred at ``x`` that contains its ``k`` nearest training points, i.e. a
ball of radius equal to the distance to the k-th neighbour.

The raw k-NN density does **not** integrate to 1 (a known caveat, Duda-Hart-
Stork §4.5); an optional renormalization over an evaluation grid is provided.

Reference: Duda, Hart & Stork, *Pattern Classification* (2nd ed., 2001), §4.5.
"""

from __future__ import annotations

from math import gamma

import numpy as np

from .base import DensityEstimator
from ..datasets import Grid
from ..metrics import integrate_on_grid

_QUERY_CHUNK = 512


def unit_ball_volume(d: int) -> float:
    """Volume of the unit d-ball: ``π^{d/2} / Γ(d/2 + 1)``."""
    return np.pi ** (d / 2.0) / gamma(d / 2.0 + 1.0)


class KNNDensity(DensityEstimator):
    """k-NN density estimator with an optional grid renormalization.

    Parameters
    ----------
    k : int
        Number of neighbours (the analogue of Parzen's bandwidth).
    renormalize : bool, default False
        If True and a grid is supplied via :meth:`set_normalization_grid` (or the
        ``grid`` constructor argument), divide the estimate by its numerical
        integral so it integrates to 1 over the grid box.
    grid : Grid | None
        Convenience: grid used for renormalization (equivalent to calling
        :meth:`set_normalization_grid`).
    """

    def __init__(
        self, k: int = 20, renormalize: bool = False, grid: Grid | None = None
    ) -> None:
        self.k = k
        self.renormalize = renormalize
        self._grid = grid
        self.X_: np.ndarray | None = None
        self.Z_: float = 1.0  # normalization constant (1.0 => raw estimate)

    def set_normalization_grid(self, grid: Grid) -> "KNNDensity":
        self._grid = grid
        if self.X_ is not None:
            self._compute_normalizer()
        return self

    def fit(self, X: np.ndarray) -> "KNNDensity":
        X = self._as_2d(X)
        if self.k > X.shape[0]:
            raise ValueError(f"k={self.k} exceeds number of points {X.shape[0]}")
        self.X_ = X
        self.Z_ = 1.0
        if self.renormalize and self._grid is not None:
            self._compute_normalizer()
        return self

    def _compute_normalizer(self) -> None:
        raw = self._raw_density(self._grid.points)
        self.Z_ = integrate_on_grid(raw, self._grid)

    def _raw_density(self, query: np.ndarray) -> np.ndarray:
        """Unnormalized k-NN density at the query rows."""
        assert self.X_ is not None
        train = self.X_
        n, d = train.shape
        c_d = unit_ball_volume(d)
        train_sq = np.sum(train**2, axis=1)  # (N,)

        out = np.empty(query.shape[0])
        for start in range(0, query.shape[0], _QUERY_CHUNK):
            q = query[start : start + _QUERY_CHUNK]  # (B, d)
            sq = (
                np.sum(q**2, axis=1)[:, None]
                + train_sq[None, :]
                - 2.0 * q @ train.T
            )
            np.maximum(sq, 0.0, out=sq)
            # k-th smallest squared distance (index k-1 after a partial sort)
            kth_sq = np.partition(sq, self.k - 1, axis=1)[:, self.k - 1]
            r = np.sqrt(kth_sq)
            vol = c_d * r**d
            with np.errstate(divide="ignore"):
                p = self.k / (n * vol)
            out[start : start + q.shape[0]] = p
        return out

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        if self.X_ is None:
            raise RuntimeError("estimator must be fit before scoring")
        X = self._as_2d(X)
        return self._raw_density(X) / self.Z_
