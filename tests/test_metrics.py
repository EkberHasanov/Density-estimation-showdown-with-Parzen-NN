"""Tests for the numerical-integration and ISE/log-likelihood helpers."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from src.datasets import make_grid_1d, make_grid_2d
from src.metrics import (
    integrate_on_grid,
    integrated_squared_error,
    mean_log_likelihood,
)


def test_integrate_constant_gives_area():
    grid = make_grid_2d(0.0, 2.0, 0.0, 3.0, 100)
    ones = np.ones(grid.points.shape[0])
    # ∫∫ 1 over [0,2]x[0,3] = 6
    assert abs(integrate_on_grid(ones, grid) - 6.0) < 1e-6


def test_integrate_standard_normal_pdf():
    grid = make_grid_1d(-10.0, 10.0, 4000)
    vals = norm.pdf(grid.points.ravel())
    assert abs(integrate_on_grid(vals, grid) - 1.0) < 1e-3


def test_ise_zero_for_identical_functions():
    grid = make_grid_1d(-5.0, 5.0, 500)
    f = norm.pdf(grid.points.ravel())
    assert integrated_squared_error(f, f, grid) == 0.0


def test_ise_matches_known_gaussian_shift():
    # ISE between N(0,1) and N(mu,1) has closed form:
    #   ∫(p-q)^2 = 2*N(0;0,2) - 2*N(mu;0,2)  (with N the normal pdf)
    grid = make_grid_1d(-12.0, 12.0, 6000)
    mu = 1.0
    p = norm.pdf(grid.points.ravel(), 0.0, 1.0)
    q = norm.pdf(grid.points.ravel(), mu, 1.0)
    ise = integrated_squared_error(p, q, grid)
    closed = 2 * norm.pdf(0.0, 0.0, np.sqrt(2.0)) - 2 * norm.pdf(mu, 0.0, np.sqrt(2.0))
    assert abs(ise - closed) < 1e-3


def test_mean_log_likelihood_floor_handles_zero():
    p = np.array([0.0, 1.0])  # a zero must not yield -inf
    val = mean_log_likelihood(p)
    assert np.isfinite(val)
