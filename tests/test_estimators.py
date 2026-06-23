"""Estimator sanity tests (§6): integrate to ≈1, and cross-check against sklearn.

``sklearn.KernelDensity`` / ``NearestNeighbors`` are used here **only** as
validation oracles for our from-scratch implementations -- never as the
reported estimators.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.neighbors import KernelDensity, NearestNeighbors

from src.config import load_config
from src.datasets import (
    make_grid_1d,
    make_grid_2d,
    make_synthetic_1d,
    make_synthetic_2d,
)
from src.estimators import (
    GMMDensity,
    GMMDensityScratch,
    KNNDensity,
    ParzenNeuralNetwork,
    ParzenWindow,
    unit_ball_volume,
)
from src.metrics import grid_integral_of_estimator, integrate_on_grid


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def data2d(cfg):
    mix = make_synthetic_2d(cfg)
    rng = np.random.default_rng(0)
    X = mix.sample(800, rng)
    grid = make_grid_2d(-6, 6, -5, 7, 160)
    return X, grid


@pytest.fixture(scope="module")
def data1d(cfg):
    mix = make_synthetic_1d(cfg)
    rng = np.random.default_rng(0)
    X = mix.sample(800, rng)
    grid = make_grid_1d(-8, 8, 1500)
    return X, grid


# -- Parzen ------------------------------------------------------------------
def test_parzen_integrates_to_one(data2d):
    X, grid = data2d
    est = ParzenWindow(bandwidth="silverman").fit(X)
    mass = grid_integral_of_estimator(est, grid)
    assert abs(mass - 1.0) < 0.05, f"Parzen mass = {mass}"


def test_parzen_matches_sklearn_kde_2d(data2d):
    X, grid = data2d
    h = 0.6
    ours = ParzenWindow(bandwidth=h).fit(X)
    pts = grid.points[::37]  # subsample to keep the test fast
    skl = KernelDensity(kernel="gaussian", bandwidth=h).fit(X)
    p_ours = ours.score_samples(pts)
    p_skl = np.exp(skl.score_samples(pts))
    np.testing.assert_allclose(p_ours, p_skl, rtol=1e-6, atol=1e-9)


def test_parzen_matches_sklearn_kde_1d(data1d):
    X, _ = data1d
    h = 0.4
    ours = ParzenWindow(bandwidth=h).fit(X)
    q = np.linspace(-8, 8, 200).reshape(-1, 1)
    skl = KernelDensity(kernel="gaussian", bandwidth=h).fit(X)
    np.testing.assert_allclose(
        ours.score_samples(q), np.exp(skl.score_samples(q)), rtol=1e-6, atol=1e-9
    )


def test_parzen_cv_selects_a_bandwidth(data1d):
    X, _ = data1d
    est = ParzenWindow(
        bandwidth="cv", cv_folds=5, bandwidth_grid=[0.3, 0.5, 1.0, 2.0], random_state=0
    ).fit(X)
    assert est.h_ is not None and np.all(est.h_ > 0)
    assert hasattr(est, "cv_best_multiplier_")


# -- k-NN --------------------------------------------------------------------
def test_unit_ball_volume_known_values():
    assert abs(unit_ball_volume(1) - 2.0) < 1e-12  # length of [-1,1]
    assert abs(unit_ball_volume(2) - np.pi) < 1e-12  # area of unit disk


def test_knn_raw_matches_bruteforce_oracle(data2d):
    X, grid = data2d
    k = 15
    est = KNNDensity(k=k, renormalize=False).fit(X)
    pts = grid.points[::53]
    # Oracle: distance to k-th neighbour via sklearn, same formula.
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    dist, _ = nn.kneighbors(pts)
    r = dist[:, -1]
    n, d = X.shape
    p_oracle = k / (n * unit_ball_volume(d) * r**d)
    np.testing.assert_allclose(est.score_samples(pts), p_oracle, rtol=1e-9, atol=1e-12)


def test_knn_renormalized_integrates_to_one(data2d):
    X, grid = data2d
    est = KNNDensity(k=20, renormalize=True, grid=grid).fit(X)
    mass = grid_integral_of_estimator(est, grid)
    assert abs(mass - 1.0) < 1e-6  # renormalized over the same grid -> exactly 1


def test_knn_raw_does_not_integrate_to_one(data2d):
    # The documented caveat: the raw k-NN density is not normalized.
    X, grid = data2d
    est = KNNDensity(k=20, renormalize=False).fit(X)
    mass = grid_integral_of_estimator(est, grid)
    assert mass > 1.0  # heavy tails make the raw integral exceed 1 here


# -- GMM ---------------------------------------------------------------------
def test_gmm_integrates_to_one(data2d):
    X, grid = data2d
    est = GMMDensity(n_components=3, random_state=0).fit(X)
    mass = grid_integral_of_estimator(est, grid)
    assert abs(mass - 1.0) < 0.02


def test_gmm_bic_selects_reasonable_components(data2d):
    X, _ = data2d
    est = GMMDensity(n_components=None, bic_max_components=6, random_state=0).fit(X)
    assert 2 <= est.selected_n_components_ <= 5


def test_scratch_em_matches_sklearn_loglik(data1d):
    X, _ = data1d
    skl = GMMDensity(n_components=3, random_state=0).fit(X)
    scratch = GMMDensityScratch(n_components=3, random_state=0).fit(X)
    q = np.linspace(-8, 8, 400).reshape(-1, 1)
    ll_skl = skl.log_likelihood(q)
    ll_scratch = scratch.log_likelihood(q)
    assert abs(ll_skl - ll_scratch) < 0.1


# -- PNN ---------------------------------------------------------------------
def test_pnn_outputs_are_nonnegative(data1d):
    X, _ = data1d
    pnn = ParzenNeuralNetwork(
        hidden=[32, 32], epochs=120, n_grid_inputs=400, random_state=0
    ).fit(X)
    q = np.linspace(-8, 8, 300).reshape(-1, 1)
    assert np.all(pnn.score_samples(q) >= 0.0)


def test_pnn_approximates_parzen_target(data1d):
    # Pure regression (no integral constraint) must track the Parzen teacher.
    X, _ = data1d
    pnn = ParzenNeuralNetwork(
        hidden=[64, 64],
        epochs=300,
        n_grid_inputs=800,
        integral_penalty=0.0,
        random_state=0,
    ).fit(X)
    parzen = ParzenWindow(bandwidth="silverman").fit(X)
    q = np.linspace(-8, 8, 400).reshape(-1, 1)
    p_pnn = pnn.score_samples(q)
    p_par = parzen.score_samples(q)
    corr = np.corrcoef(p_pnn, p_par)[0, 1]
    assert corr > 0.97, f"PNN/Parzen correlation only {corr}"


def test_pnn_constraint_pulls_integral_to_one(data1d):
    X, _ = data1d
    constrained = ParzenNeuralNetwork(
        hidden=[64, 64],
        epochs=250,
        n_grid_inputs=800,
        integral_penalty=5.0,
        random_state=0,
    ).fit(X)
    assert abs(constrained.integral_value() - 1.0) < 0.1
